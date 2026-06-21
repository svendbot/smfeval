"""smfeval CLI: validate and score commands."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np

from smfeval.align import (
  align_mode_for_gauge,
  apply_body_transform,
  fit_alignment,
  propagate_step,
)
from smfeval.align.fit import AlignmentFit
from smfeval.format import (
  FormatError,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
  WeightFormat,
)
from smfeval.io import (
  iter_steps,
  load_estimate,
  load_square,
  load_tum,
  looks_like_tum,
  parse_header,
)
from smfeval.report import (
  build_report,
  diagnose,
  recommendations,
  render_report,
)
from smfeval.report.verdict import (
  nees_verdict,
  pair_verdict_dict,
  render_nees_verdict,
  render_pair_verdict,
)
from smfeval.scoring import (
  ScoreSummary,
  anees_consistency,
  bias_variance,
  calibrate,
  energy_score,
  ensemble_diagnostics,
  gaussian_log_score,
  gaussian_log_score_components,
  relative_calibration,
  relative_translation_crps,
  student_t_logscore_sweep,
  summarize,
  translation_crps,
  translation_magnitude_interval_score,
)
from smfeval.scoring.pairwise import (
  PROPRIETY_CAVEAT,
  PairInputError,
  pair_translation_nees,
)
from smfeval.se3.lie import homogeneous
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step
from smfeval.sync import (
  MatchResult,
  SyncMode,
  interpolate_ref_at,
  match_timestamps,
  sync_risk,
)


def _add_common_args(p: argparse.ArgumentParser) -> None:
  """Flags shared by the verbs that pair an estimate with a reference."""
  p.add_argument("--t_max_diff", type=float, default=0.01)
  p.add_argument("--t_offset", type=float, default=0.0)
  p.add_argument(
    "--align", default=None, choices=["none", "se3", "gravity_yaw", "sim3"]
  )
  p.add_argument("--n_to_align", type=int, default=None)
  p.add_argument(
    "--ref-body-frame",
    default=None,
    help="body frame of the reference file (required when it is plain TUM)",
  )
  p.add_argument(
    "--ref-pose-frame",
    default="world",
    help="pose frame (outer container) of the reference file when it is "
    'plain TUM. Default "world", matching the common TUM convention. '
    "Must equal the estimate's POSE_FRAME — no in-tool transform.",
  )
  p.add_argument(
    "--body-frame-transform",
    type=Path,
    default=None,
    help='JSON file {"R": [9 floats row-major], "t": [3 floats]} giving '
    "T_est_body__ref_body (the new body frame's pose in the old body "
    "frame). Required when est and reference declare different BODY_FRAMEs.",
  )
  p.add_argument(
    "--sync",
    type=SyncMode,
    choices=list(SyncMode),
    default=SyncMode.NEAREST,
    help="reference-matching strategy. 'nearest' (default) picks the nearest "
    "reference timestamp; 'interpolate_ref' fits a piecewise GP on SE(3) over "
    "a local window of reference samples and queries it at each est timestamp "
    "(Zhang & Scaramuzza 2019, §IV.B).",
  )
  p.add_argument(
    "--sync_window",
    type=int,
    default=10,
    help="number of reference samples per GP window when --sync=interpolate_ref",
  )
  p.add_argument(
    "--sync_length_scale",
    type=float,
    default=0.1,
    help="SE-kernel length scale in seconds when --sync=interpolate_ref",
  )
  # escape hatch: estimates without a SQUARE header --------------------------
  p.add_argument(
    "--cov",
    type=Path,
    default=None,
    help="sidecar covariance file for a bare-TUM estimate: rows "
    "'timestamp c11 c21 c22 ... c66' (21 row-major lower-triangle entries "
    "of the 6x6 tangent covariance, same packing as SQUARE), matched 1:1 "
    "to the pose rows by timestamp",
  )
  p.add_argument(
    "--est-body-frame",
    default=None,
    help="body frame of the estimate when it is a bare TUM file (required "
    "in that case; SQUARE estimates declare it in the header)",
  )
  p.add_argument(
    "--est-pose-frame",
    default="world",
    help='pose frame of a bare-TUM estimate (default "world")',
  )
  p.add_argument(
    "--tangent-convention",
    type=TangentConvention,
    choices=list(TangentConvention),
    default=TangentConvention.RIGHT,
    help="tangent perturbation convention of a bare-TUM estimate's "
    "covariance (default right_perturbation)",
  )
  p.add_argument(
    "--tangent-order",
    type=TangentOrder,
    choices=list(TangentOrder),
    default=TangentOrder.TRANS_ROT,
    help="tangent block order of a bare-TUM estimate's covariance "
    "(default translation_rotation)",
  )
  p.add_argument(
    "--gauge",
    type=Gauge,
    choices=list(Gauge),
    default=Gauge.SE3,
    help="gauge freedom of a bare-TUM estimate (default se3: full SE(3) "
    "alignment is fitted before scoring)",
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="smfeval")
  sub = parser.add_subparsers(dest="cmd", required=True)

  pv = sub.add_parser("validate", help="header and row sanity checks")
  pv.add_argument("file", type=Path)
  pv.add_argument(
    "--strict",
    action="store_true",
    help="per-row exporter checks: covariance SPD, plausible magnitude, "
    "not degenerate-zero, finite poses (the gate for contributed "
    "exporters; no reference needed)",
  )

  pn = sub.add_parser(
    "nees",
    help="three-line calibration verdict: median NEES, covariance scale "
    "gap k, coverage",
  )
  pn.add_argument("est", type=Path, help="SQUARE-format estimate file")
  pn.add_argument("ref", type=Path, help="reference file (SQUARE or TUM)")
  _add_common_args(pn)
  pn.add_argument(
    "--alpha",
    type=float,
    default=0.05,
    help="significance level of the two-sided ANEES chi2 verdict",
  )
  pn.add_argument(
    "--json", action="store_true", help="print the verdict as JSON"
  )

  pp = sub.add_parser(
    "pair",
    help="no-reference verdict: score two filters against each other; "
    "an elevated pairwise NEES certifies overconfidence with no "
    "reference consulted (lower bound)",
  )
  pp.add_argument("a", type=Path, help="SQUARE-format trajectory A (scored)")
  pp.add_argument("b", type=Path, help="SQUARE-format trajectory B (reference)")
  pp.add_argument("--t_max_diff", type=float, default=0.01)
  pp.add_argument("--alpha", type=float, default=0.05)
  pp.add_argument(
    "--min-matched",
    type=int,
    default=10,
    help="minimum timestamp matches required to score the pair",
  )
  pp.add_argument(
    "--body-frame-transform",
    type=Path,
    default=None,
    help='JSON file {"R": [...], "t": [...]} re-expressing A in B\'s body '
    "frame; required when the two files declare different BODY_FRAMEs.",
  )
  pp.add_argument(
    "--json", action="store_true", help="print the verdict as JSON"
  )

  ps = sub.add_parser("score", help="produce a full scoring report")
  ps.add_argument("est", type=Path, help="SQUARE-format estimate file")
  ps.add_argument("ref", type=Path, help="reference file (SQUARE or TUM)")
  _add_common_args(ps)
  ps.add_argument("--alpha", type=float, default=0.1)
  ps.add_argument("--n_samples", type=int, default=128)
  ps.add_argument("--seed", type=int, default=0)
  ps.add_argument(
    "--rpe-window",
    type=str,
    default=None,
    help="comma-separated relative-pose windows in seconds (e.g. "
    "'0.1,1,10') for short-window relative translation CRPS. Restores "
    "sensitivity to local sigma calibration that absolute-pose CRPS "
    "loses in the overconfident regime. Requires gaussian_se3 input.",
  )
  ps.add_argument(
    "--student-t",
    type=str,
    default=None,
    help="comma-separated dof values (e.g. '3,5,10,30') for the Student-t "
    "belief-transform intervention: re-score the reference under a covariance-"
    "matched heavy-tailed predictive and report the mean proper log-score vs "
    "ν. A finite ν* with lower score than Gaussian ⇒ the errors are heavy-"
    "tailed and a robust likelihood would help (cross-filter, no re-run). "
    "Requires gaussian_se3 input.",
  )
  ps.add_argument(
    "--ess-inflate",
    type=Path,
    default=None,
    help="ESS intervention: per-scan covariance inflation. A text file with "
    "'timestamp c' rows; each step's Σ is scaled to c·Σ before scoring. "
    "c = (n_eff/n_eff_ref)^β captures the correlated-measurement effect "
    "(overconfidence ∝ point density, c∝n_eff). Source-agnostic / cross-filter "
    "— generate the c-file from any per-scan density source. Composes with "
    "--consume-ref-cov as Σ_eff = c·Σ_pred + Σ_ref.",
  )
  ps.add_argument(
    "--consume-ref-cov",
    action="store_true",
    help="fold the reference observer covariance into the predictive before "
    "scoring: Σ_eff = Σ_pred + Σ_ref (proof-of-concept; the score then accounts "
    "for reference uncertainty, not just the filter's). Requires "
    "--sync=interpolate_ref, which supplies Σ_ref as the GP predictive "
    "covariance — a stand-in for the per-sample reference covariance datasets "
    "do not yet ship (the call-to-action).",
  )
  ps.add_argument(
    "--calibration",
    action="store_true",
    help="print the translation log-score calibration/sharpness split + ANEES "
    "χ² consistency verdict (dof-3). The calibration term (½·NEES) stays "
    "graded where CRPS/coverage saturate; the χ² interval gives an "
    "optimistic|consistent|conservative verdict. Requires gaussian_se3 input.",
  )
  ps.add_argument(
    "--json-out",
    type=Path,
    default=None,
    help="if set, also write the structured Report as JSON to this path",
  )
  ps.add_argument(
    "--json",
    action="store_true",
    help="print the structured Report as JSON to stdout (schema in "
    "docs/report.schema.json) instead of the text report",
  )

  args = parser.parse_args(argv)

  if args.cmd == "validate":
    return _validate(args.file, strict=args.strict)
  if args.cmd == "nees":
    return _nees(args)
  if args.cmd == "pair":
    return _pair(args)
  if args.cmd == "score":
    return _score(args)
  return 1


# --strict per-row covariance magnitude plausibility bounds (m^2 / rad^2):
# below the floor the covariance is numerically degenerate-zero; above the
# ceiling it is physically implausible for a pose belief.
_STRICT_DIAG_FLOOR = 1e-12
_STRICT_DIAG_CEIL = 1e4
_STRICT_MAX_REPORTED = 5


def _strict_row_problems(k: int, s: Step) -> list[str]:
  """Mechanical exporter checks for one step (the contributor-facing gate)."""
  problems: list[str] = []
  if isinstance(s, GaussianStep):
    cov = s.covariance
    if not np.all(np.isfinite(cov)):
      return [f"row {k}: non-finite covariance entries"]
    diag = np.diag(cov)
    if np.all(diag == 0.0):
      problems.append(f"row {k}: covariance is all-zero (degenerate)")
    else:
      try:
        np.linalg.cholesky(cov)
      except np.linalg.LinAlgError:
        problems.append(f"row {k}: covariance is not positive definite")
      if np.any(diag < _STRICT_DIAG_FLOOR) or np.any(diag > _STRICT_DIAG_CEIL):
        problems.append(
          f"row {k}: covariance diagonal outside plausible range "
          f"[{_STRICT_DIAG_FLOOR:g}, {_STRICT_DIAG_CEIL:g}]: "
          f"min {diag.min():.3g}, max {diag.max():.3g}"
        )
  if isinstance(s, GaussianStep | DeterministicStep):
    if not (
      np.all(np.isfinite(s.translation)) and np.all(np.isfinite(s.quat_xyzw))
    ):
      problems.append(f"row {k}: non-finite pose")
  return problems


def _validate(path: Path, strict: bool = False) -> int:
  problems: list[str] = []
  n_bad = 0
  try:
    with path.open() as f:
      header = parse_header(f)
      n = 0
      for k, s in enumerate(iter_steps(f, header)):
        n += 1
        if not strict:
          continue
        row_problems = _strict_row_problems(k, s)
        if row_problems:
          n_bad += 1
          problems.extend(row_problems)
  except (FormatError, OSError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2

  if problems:
    for p in problems[:_STRICT_MAX_REPORTED]:
      print(f"strict: {p}", file=sys.stderr)
    print(
      f"error: {path.name} failed strict validation "
      f"({n_bad}/{n} rows with problems)",
      file=sys.stderr,
    )
    return 2
  suffix = ", strict checks passed" if strict else ""
  print(
    f"ok: {path.name} ({header.representation.value}, {n} timesteps{suffix})"
  )
  return 0


def _json_default(obj):
  if isinstance(obj, Enum):
    return obj.value
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.generic):
    return obj.item()
  raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _write_report_json(rep, path: Path) -> None:
  with path.open("w") as f:
    json.dump(asdict(rep), f, indent=2, default=_json_default)
    f.write("\n")


def _load_ess_cfile(path: Path) -> tuple[np.ndarray, np.ndarray]:
  """Read a per-scan ESS inflation file: ``timestamp c`` rows (# comments)."""
  ts: list[float] = []
  c: list[float] = []
  for raw in path.read_text().splitlines():
    if raw.startswith("#") or not raw.strip():
      continue
    toks = raw.split()
    ts.append(float(toks[0]))
    c.append(float(toks[1]))
  return np.asarray(ts), np.asarray(c)


def _load_body_frame_transform(path: Path) -> np.ndarray:
  """Read an SE(3) ``T_est_body__ref_body`` from a JSON file with ``R``, ``t``.

  Matches the Spires ``cam-lidar-imu.yaml`` convention: ``R`` is the 3×3
  rotation row-major, ``t`` is the 3-vector. The semantics are that of a ROS
  transform ``target_T_source`` where ``target`` is the estimate's body frame
  and ``source`` is the reference's body frame.
  """
  data = json.loads(path.read_text())
  R = np.array(data["R"], dtype=float).reshape(3, 3)
  t = np.array(data["t"], dtype=float).reshape(3)
  return homogeneous(R, t)


def _resolve_sync(
  args: argparse.Namespace,
  est_steps: list,
  ref_steps: list,
  tangent_order: TangentOrder | None,
) -> (
  tuple[
    MatchResult, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
  ]
  | None
):
  """Pair estimate and reference poses by --sync mode.

  Sixth element is the matched reference covariance (Q, 6, 6) under interpolate_ref
  (the GP predictive Σ_ref), else None.

  Prints to stderr and returns None on error (empty match / no overlap).
  """
  est_ts = np.array([s.timestamp for s in est_steps])
  ref_ts = np.array([s.timestamp for s in ref_steps])
  ref_positions = np.array([s.translation for s in ref_steps])
  ref_quats = np.array([s.quat_xyzw for s in ref_steps])

  match args.sync:
    case SyncMode.INTERPOLATE_REF:
      query_t = est_ts + args.t_offset
      interp_t, interp_q, interp_cov, keep = interpolate_ref_at(
        query_t,
        ref_ts,
        ref_positions,
        ref_quats,
        window=args.sync_window,
        length_scale_s=args.sync_length_scale,
      )
      if not keep.any():
        print(
          "error: no estimate timestamps fall within the reference range",
          file=sys.stderr,
        )
        return None
      est_idx = np.where(keep)[0]
      # ref_indices = -1: the interpolated reference is not a row in ref_steps.
      match_res = MatchResult(
        est_indices=est_idx,
        ref_indices=np.full(est_idx.size, -1, dtype=int),
        n_total=len(est_ts),
        n_matched=int(est_idx.size),
        n_dropped=int(len(est_ts) - est_idx.size),
        gap_seconds=np.zeros(est_idx.size),
      )
      matched_est = [est_steps[i] for i in est_idx]
      matched_ref_t = interp_t[est_idx]
      matched_ref_q = interp_q[est_idx]
      # Risk surrogate: GP predictive sigma - interpolation hits the query exactly,
      # but the predictive variance bounds how trustworthy that hit is.
      risks = np.sqrt(np.maximum(interp_cov[est_idx, 0, 0], 0.0))
      return (
        match_res,
        matched_est,
        matched_ref_t,
        matched_ref_q,
        risks,
        interp_cov[est_idx],
      )

    case SyncMode.NEAREST:
      match_res = match_timestamps(
        est_ts, ref_ts, args.t_max_diff, args.t_offset
      )
      if match_res.n_matched == 0:
        print("error: no matched pairs", file=sys.stderr)
        return None
      matched_est = [est_steps[i] for i in match_res.est_indices]
      matched_ref_t = ref_positions[match_res.ref_indices]
      matched_ref_q = ref_quats[match_res.ref_indices]
      risks = sync_risk(
        est_steps,
        ref_ts,
        ref_positions,
        match_res.est_indices,
        match_res.ref_indices,
        est_ts=est_ts,
        t_offset=args.t_offset,
        tangent_order=tangent_order,
      )
      return match_res, matched_est, matched_ref_t, matched_ref_q, risks, None


def _compute_scores(
  aligned_est: list,
  matched_ref_t: np.ndarray,
  matched_ref_q: np.ndarray,
  order: TangentOrder,
  n_samples: int,
  alpha: float,
  seed: int,
) -> dict[str, ScoreSummary]:
  rng = np.random.default_rng(seed)
  crps_t: list[float] = []
  es: list[float] = []
  is_t: list[float] = []
  log_trans: list[float] = []
  for s, ref_t, ref_q in zip(
    aligned_est, matched_ref_t, matched_ref_q, strict=True
  ):
    crps_t.append(translation_crps(s, ref_t, order))
    es.append(energy_score(s, ref_t, order, n_samples, rng))

    if isinstance(s, GaussianStep):
      ls = gaussian_log_score(s, ref_t, ref_q, order)
      log_trans.append(ls.translation)
    if not isinstance(s, DeterministicStep):
      is_t.append(
        translation_magnitude_interval_score(
          s, ref_t, alpha, n_samples, rng, order
        )
      )

  boot_rng = np.random.default_rng(seed + 2)
  scores: dict[str, ScoreSummary] = {
    "translation_crps": summarize(crps_t, rng=boot_rng),
    "energy_score": summarize(es, rng=boot_rng),
  }
  if log_trans:
    scores["log_score_translation"] = summarize(log_trans, rng=boot_rng)
  if is_t:
    scores["interval_score"] = summarize(is_t, rng=boot_rng)
  return scores


@dataclass
class PreparedRun:
  """Estimate/reference pair after loading, frame checks, sync, and alignment."""

  est_header: SquareHeader
  matched_est: list[Step]
  aligned_est: list[Step]
  matched_ref_t: np.ndarray
  matched_ref_q: np.ndarray
  match: MatchResult
  fit: AlignmentFit
  order: TangentOrder
  risks: np.ndarray
  ref_cov: np.ndarray | None  # GP predictive covariance under interpolate_ref


def _load_estimate(args: argparse.Namespace) -> tuple[SquareHeader, list[Step]]:
  """io.load_estimate plus a stderr note when the escape hatch kicked in."""
  was_tum = looks_like_tum(args.est)
  header, steps = load_estimate(
    args.est,
    cov=args.cov,
    pose_frame=args.est_pose_frame,
    body_frame=args.est_body_frame,
    tangent_convention=args.tangent_convention,
    tangent_order=args.tangent_order,
    gauge=args.gauge,
  )
  if was_tum:
    suffix = ""
    if header.representation is Representation.GAUSSIAN_SE3:
      conv = header.tangent_convention or TangentConvention.RIGHT
      order = header.tangent_order or TangentOrder.TRANS_ROT
      suffix = f", {conv.value}, {order.value}"
    print(
      f"note: bare-TUM estimate read as {header.representation.value} "
      f"(body_frame={header.body_frame!r}, pose_frame={header.pose_frame!r}, "
      f"gauge={header.gauge.value}{suffix})",
      file=sys.stderr,
    )
  return header, steps


def _rebody(
  header: SquareHeader,
  steps: list[Step],
  target_body_frame: str,
  transform_path: Path,
) -> tuple[SquareHeader, list[Step]]:
  """Re-express a trajectory in another body frame and relabel its header."""
  T_off = _load_body_frame_transform(transform_path)
  steps = [
    apply_body_transform(
      s,
      T_off,
      tangent_convention=header.tangent_convention,
      tangent_order=header.tangent_order,
    )
    for s in steps
  ]
  return replace(header, body_frame=target_body_frame), steps


def _prepare(args: argparse.Namespace) -> PreparedRun | None:
  """Load est+ref, check frames, sync, and align (shared by score/nees).

  Prints to stderr and returns None on any input error.
  """
  try:
    est_header, est_steps = _load_estimate(args)
    if looks_like_tum(args.ref):
      if args.ref_body_frame is None:
        print(
          "error: reference file is plain TUM but its body frame "
          "is not declared. Pass --ref-body-frame <name> matching the "
          f"estimate's BODY_FRAME (estimate declares "
          f"{est_header.body_frame!r}).",
          file=sys.stderr,
        )
        return None
      ref_tum, ref_steps = load_tum(
        args.ref,
        pose_frame=args.ref_pose_frame,
        body_frame=args.ref_body_frame,
      )
      ref_header: SquareHeader = ref_tum.to_square()
    else:
      ref_header, ref_steps = load_square(args.ref)
  except (FormatError, OSError) as e:
    print(f"error: {e}", file=sys.stderr)
    return None

  if est_header.pose_frame != ref_header.pose_frame:
    print(
      f"error: pose frames differ — estimate is {est_header.pose_frame!r}, "
      f"reference is {ref_header.pose_frame!r}. The scoring tool does "
      "not transform between outer reference frames; pre-align both "
      "trajectories into a common pose frame, or pass --ref-pose-frame "
      "if the reference file is plain TUM and the default 'world' is wrong.",
      file=sys.stderr,
    )
    return None

  if est_header.body_frame != ref_header.body_frame:
    if args.body_frame_transform is None:
      print(
        f"error: body frames differ — estimate is {est_header.body_frame!r}, "
        f"reference is {ref_header.body_frame!r}. Provide "
        "--body-frame-transform PATH with the SE(3) "
        "T_est_body__ref_body, or rewrite both trajectories in a common "
        "frame.",
        file=sys.stderr,
      )
      return None
    est_header, est_steps = _rebody(
      est_header, est_steps, ref_header.body_frame, args.body_frame_transform
    )

  resolved = _resolve_sync(args, est_steps, ref_steps, est_header.tangent_order)
  if resolved is None:
    return None
  match, matched_est, matched_ref_t, matched_ref_q, risks, ref_cov = resolved

  align_mode = args.align or align_mode_for_gauge(est_header.gauge)
  n_align = args.n_to_align if args.n_to_align else len(matched_est)
  n_align = min(n_align, len(matched_est))
  fit_t = np.array([s.translation for s in matched_est[:n_align]])
  fit = fit_alignment(fit_t, matched_ref_t[:n_align], mode=align_mode)

  aligned_est: list[Step] = [
    propagate_step(
      s,
      fit.transform,
      scale=fit.scale,
      tangent_convention=est_header.tangent_convention,
      tangent_order=est_header.tangent_order,
    )
    for s in matched_est
  ]

  return PreparedRun(
    est_header=est_header,
    matched_est=matched_est,
    aligned_est=aligned_est,
    matched_ref_t=matched_ref_t,
    matched_ref_q=matched_ref_q,
    match=match,
    fit=fit,
    order=est_header.tangent_order or TangentOrder.TRANS_ROT,
    risks=risks,
    ref_cov=ref_cov,
  )


def _nees(args: argparse.Namespace) -> int:
  """Three-line calibration verdict on one estimate vs a reference."""
  pr = _prepare(args)
  if pr is None:
    return 2
  if pr.est_header.representation is not Representation.GAUSSIAN_SE3:
    print(
      "error: nees needs gaussian_se3 (a published covariance); got "
      f"{pr.est_header.representation.value}. Use 'score' for "
      "deterministic or ensemble trajectories.",
      file=sys.stderr,
    )
    return 2

  vals: list[float] = []
  for s, ref_t, ref_q in zip(
    pr.aligned_est, pr.matched_ref_t, pr.matched_ref_q, strict=True
  ):
    if not isinstance(s, GaussianStep):
      continue
    dec = gaussian_log_score_components(s, ref_t, ref_q, pr.order)
    vals.append(dec.translation.nees)

  v = nees_verdict(np.asarray(vals), dof=3, alpha=args.alpha)
  if args.json:
    print(json.dumps(v.to_dict(), indent=2, default=_json_default))
  else:
    print(render_nees_verdict(v))
  return 0


def _pair(args: argparse.Namespace) -> int:
  """No-reference pairwise verdict: filter A scored against filter B."""
  try:
    header_a, steps_a = load_square(args.a)
    header_b, steps_b = load_square(args.b)
  except (FormatError, OSError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2

  if (
    header_a.body_frame != header_b.body_frame
    and args.body_frame_transform is not None
  ):
    header_a, steps_a = _rebody(
      header_a, steps_a, header_b.body_frame, args.body_frame_transform
    )

  try:
    res = pair_translation_nees(
      header_a,
      steps_a,
      header_b,
      steps_b,
      t_max_diff=args.t_max_diff,
      min_matched=args.min_matched,
      alpha=args.alpha,
    )
  except PairInputError as e:
    print(f"error: {e}", file=sys.stderr)
    return 2

  v = nees_verdict(res.nees, dof=3, alpha=args.alpha, anees=res.anees)
  if args.json:
    out = pair_verdict_dict(res, v, PROPRIETY_CAVEAT)
    print(json.dumps(out, indent=2, default=_json_default))
  else:
    print(render_pair_verdict(res, v, caveat=PROPRIETY_CAVEAT))
  return 0


def _adjust_covariance(
  args: argparse.Namespace, aligned_est: list, ref_cov
) -> list | None:
  """Apply ESS inflation and reference-covariance folding to the predictive steps.

  Returns the adjusted steps, or None (after printing an error) when
  --consume-ref-cov is requested without an interpolated reference covariance. ref_cov is
  the isotropic GP predictive covariance, so the tangent-order of the sum is
  immaterial; the gaussian-validity check stays on the raw predictive cov.
  """
  ess_c = None
  if args.ess_inflate is not None:
    c_ts, c_val = _load_ess_cfile(args.ess_inflate)
    step_ts = np.array([s.timestamp for s in aligned_est])
    # nearest c per step; the c-file may be sparse (piecewise-constant over a
    # range), so this is a nearest lookup, not a 1:1 match
    j = np.clip(np.searchsorted(c_ts, step_ts), 0, len(c_ts) - 1)
    jl = np.maximum(j - 1, 0)
    j = np.where(np.abs(c_ts[jl] - step_ts) < np.abs(c_ts[j] - step_ts), jl, j)
    ess_c = c_val[j]
    n_extrap = int(
      np.count_nonzero((step_ts < c_ts.min()) | (step_ts > c_ts.max()))
    )
    if n_extrap:
      print(
        f"note: {n_extrap} step(s) fall outside the ESS c-file range "
        f"[{c_ts.min():.3f}, {c_ts.max():.3f}]; using the nearest edge factor",
        file=sys.stderr,
      )

  if args.consume_ref_cov and ref_cov is None:
    print(
      "error: --consume-ref-cov requires --sync=interpolate_ref (it supplies "
      "Σ_ref as the GP predictive covariance)",
      file=sys.stderr,
    )
    return None

  if ess_c is None and not args.consume_ref_cov:
    return aligned_est

  scored_est: list = []
  for k, s in enumerate(aligned_est):
    if not isinstance(s, GaussianStep):
      scored_est.append(s)
      continue
    cov = s.covariance
    if ess_c is not None:  # cov *= ess_c  (ESS inflation)
      cov = ess_c[k] * cov
    if args.consume_ref_cov:  # cov += ref_cov  (observer uncertainty)
      cov = cov + ref_cov[k]
    scored_est.append(replace(s, covariance=cov))
  return scored_est


def _emit_report(args: argparse.Namespace, rep) -> None:
  """Write and/or print the report per --json-out / --json flags."""
  if args.json_out is not None:
    _write_report_json(rep, args.json_out)
  if args.json:
    print(json.dumps(asdict(rep), indent=2, default=_json_default))
  else:
    print(render_report(rep))


def _emit_side_reports(
  args: argparse.Namespace,
  aligned_est: list,
  scored_est: list,
  matched_ref_t: np.ndarray,
  matched_ref_q: np.ndarray,
  order: TangentOrder,
  est_header: SquareHeader,
  rpe_windows: list[float] | None,
  split: dict | None,
) -> None:
  """Print the optional relative-CRPS, calibration, and student-t side reports."""
  if rpe_windows:
    _report_relative_crps(
      args, aligned_est, matched_ref_t, order, est_header, rpe_windows
    )
  if split is not None:
    _emit_calibration_machine_lines(split)
  if args.student_t:
    _report_student_t(
      args, scored_est, matched_ref_t, matched_ref_q, order, est_header
    )


def _score(args: argparse.Namespace) -> int:
  pr = _prepare(args)
  if pr is None:
    return 2
  est_header = pr.est_header
  matched_est = pr.matched_est
  aligned_est = pr.aligned_est
  matched_ref_t = pr.matched_ref_t
  matched_ref_q = pr.matched_ref_q
  match, fit, risks, ref_cov = pr.match, pr.fit, pr.risks, pr.ref_cov
  order = pr.order
  rpe_windows = (
    [float(x) for x in args.rpe_window.split(",") if x.strip()]
    if args.rpe_window
    else None
  )

  ensemble_diag = None
  if est_header.representation is Representation.ENSEMBLE_SE3:
    ensemble_diag = ensemble_diagnostics(
      [s for s in matched_est if isinstance(s, EnsembleStep)],
      est_header.weight_format or WeightFormat.LINEAR,
      normalized=bool(est_header.weights_normalized),
    )

  scored_est = _adjust_covariance(args, aligned_est, ref_cov)
  if scored_est is None:
    return 2

  scores = _compute_scores(
    scored_est,
    matched_ref_t,
    matched_ref_q,
    order=order,
    n_samples=args.n_samples,
    alpha=args.alpha,
    seed=args.seed,
  )

  cal = None
  if est_header.representation is not Representation.DETERMINISTIC:
    cal = calibrate(
      scored_est,
      matched_ref_t,
      matched_ref_q,
      tangent_order=order,
      alpha=args.alpha,
      n_samples=args.n_samples,
      rng=np.random.default_rng(args.seed + 1),
    )

  traj_len = (
    float(np.sum(np.linalg.norm(np.diff(matched_ref_t, axis=0), axis=1)))
    if len(matched_ref_t) > 1
    else 0.0
  )

  split = None
  if args.calibration:
    if est_header.representation is Representation.GAUSSIAN_SE3:
      split = _calibration_split_dict(
        args, scored_est, matched_ref_t, matched_ref_q, order, rpe_windows
      )
    else:
      print(
        "\ncalibration split: skipped — needs gaussian_se3 (a published "
        f"covariance), got {est_header.representation.value}",
        file=sys.stderr,
      )

  rep = build_report(
    match,
    fit,
    est_header.gauge,
    sync_risks=risks,
    sync_risk_threshold=0.3,
    ensemble=ensemble_diag,
    scores=scores,
    calibration=cal,
    trajectory_length_m=traj_len,
    sync_mode=args.sync,
  )
  rep.calibration_split = split
  if args.calibration and len(matched_ref_t) > 2:
    bv_windows = rpe_windows or [0.1, 1.0, 10.0]
    rep.bias_variance = [
      r.to_dict()
      for r in bias_variance(aligned_est, matched_ref_t, windows_s=bv_windows)
    ]
  rep.recommendations = recommendations(rep)
  rep.diagnoses = diagnose(rep)
  _emit_report(args, rep)
  _emit_side_reports(
    args,
    aligned_est,
    scored_est,
    matched_ref_t,
    matched_ref_q,
    order,
    est_header,
    rpe_windows,
    split,
  )
  return 0


def _report_relative_crps(
  args: argparse.Namespace,
  aligned_est: list,
  matched_ref_t: np.ndarray,
  order: TangentOrder,
  est_header: SquareHeader,
  windows: list[float],
) -> None:
  r"""Compute and print short-window relative translation CRPS.

  Restores sensitivity to local σ calibration that absolute-pose CRPS
  loses once the filter is overconfident (|z| ≫ 1 → CRPS → |error|).
  Requires a Gaussian predictive; deterministic/ensemble inputs are
  skipped with a notice.
  """
  if est_header.representation is not Representation.GAUSSIAN_SE3:
    print(
      "\nrelative CRPS: skipped — needs gaussian_se3 (a published "
      f"position covariance), got {est_header.representation.value}",
      file=sys.stderr,
    )
    return
  results = relative_translation_crps(
    aligned_est,
    matched_ref_t,
    windows_s=windows,
    tangent_order=order,
    rng=np.random.default_rng(args.seed + 3),
  )
  print("\nRelative translation CRPS (short-window RPE-CRPS):")
  hdr = (
    f"  {'window_s':>9} {'n_pairs':>8} {'crps_mean_m':>13} "
    f"{'crps_median_m':>14} {'rpe_rmse_m':>11} {'mean_z2':>9} "
    f"{'sigma_rel_m':>12}"
  )
  print(hdr)
  print("  " + "-" * (len(hdr) - 2))
  for r in results:
    print(
      f"  {r.window_s:>9.3g} {r.n_pairs:>8d} {r.crps.mean:>13.6f} "
      f"{r.crps.median:>14.6f} {r.rpe_rmse_m:>11.6f} {r.mean_z2:>9.3f} "
      f"{r.sigma_rel_median_m:>12.6f}"
    )
    # machine-readable line (stable key=value form for downstream parsers)
    print(
      f"  RELATIVE_CRPS_TRANS window={r.window_s:g} n={r.n_pairs} "
      f"crps_mean={r.crps.mean:.8f} crps_median={r.crps.median:.8f} "
      f"rpe_rmse={r.rpe_rmse_m:.8f} mean_z2={r.mean_z2:.6f}"
    )
  if not results:
    print("  (no valid relative-pose pairs at the requested windows)")


def _calibration_split_dict(
  args: argparse.Namespace,
  aligned_est: list,
  matched_ref_t: np.ndarray,
  matched_ref_q: np.ndarray,
  order: TangentOrder,
  windows: list[float] | None,
) -> dict | None:
  r"""Build the calibration/sharpness split dict consumed by the report + diagnose.

  ``-log p`` splits exactly into ``calibration = ½·NEES`` and ``sharpness =
  ½(log|Σ| + d·log2π)``; the ANEES χ² interval gives the translation
  ``optimistic|consistent|conservative`` verdict (dof-3). Returns
  ``{"absolute": {...}, "windowed": [...]}`` or ``None`` if there are no
  Gaussian steps. ``windowed`` is present only when ``--rpe-window`` is set.
  """
  nees: list[float] = []
  calib: list[float] = []
  sharp: list[float] = []
  for s, ref_t, ref_q in zip(
    aligned_est, matched_ref_t, matched_ref_q, strict=True
  ):
    if not isinstance(s, GaussianStep):
      continue
    comp = gaussian_log_score_components(s, ref_t, ref_q, order).translation
    nees.append(comp.nees)
    calib.append(comp.calibration)
    sharp.append(comp.sharpness)

  if not nees:
    return None

  def _median_finite(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

  res = anees_consistency(
    np.asarray(nees, dtype=float), dof=3, alpha=args.alpha
  )
  d = res.to_dict()
  d["nees_median"] = d.pop("median")
  absolute: dict[str, dict] = {
    "translation": {
      **d,
      "calibration_median": _median_finite(calib),
      "sharpness_median": _median_finite(sharp),
    }
  }

  split: dict = {"absolute": absolute}
  if windows:
    rows = relative_calibration(
      aligned_est,
      matched_ref_t,
      windows_s=windows,
      tangent_order=order,
      alpha=args.alpha,
    )
    split["windowed"] = [
      {
        "window_s": r.window_s,
        "anees": r.anees.anees,
        "nees_median": r.anees.median,
        "dof": r.anees.dof,
        "n": r.anees.n,
        "lo": r.anees.lo,
        "hi": r.anees.hi,
        "verdict": r.anees.verdict,
        "calibration_median": r.calibration_median,
        "sharpness_median": r.sharpness_median,
        "sigma_rel_m": r.sigma_rel_median_m,
      }
      for r in rows
    ]
  return split


def _emit_calibration_machine_lines(split: dict) -> None:
  """Stable key=value lines for downstream parsers (printed after the report)."""
  for name, s in (split.get("absolute") or {}).items():
    print(
      f"CALIBRATION_SPLIT slice={name} dof={s['dof']} n={s['n']} "
      f"anees={s['anees']:.6f} nees_median={s['nees_median']:.6f} "
      f"chi2_lo={s['lo']:.6f} chi2_hi={s['hi']:.6f} "
      f"verdict={s['verdict']} calib_median={s['calibration_median']:.6f} "
      f"sharp_median={s['sharpness_median']:.6f}"
    )
  for w in split.get("windowed") or []:
    print(
      f"WINDOWED_CALIBRATION window={w['window_s']:g} n={w['n']} "
      f"anees={w['anees']:.6f} chi2_lo={w['lo']:.6f} chi2_hi={w['hi']:.6f} "
      f"verdict={w['verdict']} calib_median={w['calibration_median']:.6f} "
      f"sigma_rel_m={w['sigma_rel_m']:.6f}"
    )


def _report_student_t(
  args: argparse.Namespace,
  aligned_est: list,
  matched_ref_t: np.ndarray,
  matched_ref_q: np.ndarray,
  order: TangentOrder,
  est_header: SquareHeader,
) -> None:
  r"""Student-t belief-transform intervention: mean proper log-score vs ν.

  Re-scores the reference under a covariance-matched Student-t belief (heavier
  tails, same mean/Σ) and reports the mean negative log density vs ν, with the
  Gaussian (ν→∞) baseline. The mean log-score is dominated by the overconfident
  tail under a Gaussian; a heavy-tailed belief bounds the tail's contribution,
  so a finite ν* with a lower mean ⇒ the errors are heavy-tailed and a robust
  likelihood would help. Cross-filter, no filter re-run.
  """
  if est_header.representation is not Representation.GAUSSIAN_SE3:
    print(
      "\nstudent-t: skipped — needs gaussian_se3, got "
      f"{est_header.representation.value}",
      file=sys.stderr,
    )
    return
  nus = [float(x) for x in args.student_t.split(",") if x.strip()]
  gauss, tcols = student_t_logscore_sweep(
    aligned_est, matched_ref_t, matched_ref_q, nus, tangent_order=order
  )

  if not gauss:
    print("\nstudent-t: no Gaussian steps after matching")
    return

  def _mean(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

  g = _mean(gauss)
  print("\nStudent-t belief transform (mean proper log-score, lower=better):")
  print(f"  {'nu':>10} {'mean -log p':>14} {'delta vs Gauss':>16}")
  print(f"  {'inf(Gauss)':>10} {g:>14.3f} {0.0:>16.3f}")
  best_nu, best = None, g
  for nu in nus:
    t = _mean(tcols[nu])
    print(f"  {nu:>10.1f} {t:>14.3f} {t - g:>16.3f}")
    print(
      f"  STUDENT_T nu={nu:g} mean_neglogp={t:.6f} delta_vs_gauss={t - g:.6f}"
    )
    if t < best:
      best, best_nu = t, nu
  if best_nu is not None:
    print(
      f"  -> nu*={best_nu:g} improves the proper score by {g - best:.3f} "
      "nat/scan: errors are heavy-tailed, a robust (Student-t) likelihood "
      "would help."
    )
  else:
    print("  -> no finite nu beats Gaussian: errors are not heavy-tailed.")


if __name__ == "__main__":
  sys.exit(main())
