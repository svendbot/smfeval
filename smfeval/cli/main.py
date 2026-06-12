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
  load_square,
  load_tum,
  load_tum_gaussian,
  load_tum_with_sidecar,
  looks_like_tum,
  parse_header,
  sniff_tum_columns,
)
from smfeval.report import (
  build_report,
  diagnose,
  recommendations,
  render_report,
)
from smfeval.report.verdict import (
  nees_verdict,
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
  gaussian_rotation_validity,
  relative_calibration,
  relative_translation_crps,
  rotation_crps,
  student_t_neg_log_density,
  summarize,
  translation_crps,
  translation_magnitude_interval_score,
)
from smfeval.scoring.pairwise import (
  PROPRIETY_CAVEAT,
  PairInputError,
  pair_translation_nees,
)
from smfeval.se3.lie import homogeneous, pose_matrix, relative, se3_log
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step
from smfeval.sync import (
  MatchResult,
  SyncMode,
  interpolate_gt_at,
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
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--gt-body-frame",
    default=None,
    help="body frame of the ground-truth file (required when GT is plain TUM)",
  )
  p.add_argument(
    "--gt-pose-frame",
    default="world",
    help="pose frame (outer container) of the ground-truth file when GT is "
    'plain TUM. Default "world", matching the common TUM convention. '
    "Must equal the estimate's POSE_FRAME — no in-tool transform.",
  )
  p.add_argument(
    "--body-frame-transform",
    type=Path,
    default=None,
    help='JSON file {"R": [9 floats row-major], "t": [3 floats]} giving '
    "T_est_body__gt_body (the new body frame's pose in the old body "
    "frame). Required when est and gt declare different BODY_FRAMEs.",
  )
  p.add_argument(
    "--sync",
    type=SyncMode,
    choices=list(SyncMode),
    default=SyncMode.NEAREST,
    help="GT-matching strategy. 'nearest' (default) picks the nearest GT "
    "timestamp; 'interpolate_gt' fits a piecewise GP on SE(3) over a "
    "local window of GT samples and queries it at each est timestamp "
    "(Zhang & Scaramuzza 2019, §IV.B).",
  )
  p.add_argument(
    "--sync_window",
    type=int,
    default=10,
    help="number of GT samples per GP window when --sync=interpolate_gt",
  )
  p.add_argument(
    "--sync_length_scale",
    type=float,
    default=0.1,
    help="SE-kernel length scale in seconds when --sync=interpolate_gt",
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

  pn = sub.add_parser(
    "nees",
    help="three-line calibration verdict: median NEES, covariance scale "
    "gap k, coverage",
  )
  pn.add_argument("est", type=Path, help="SQUARE-format estimate file")
  pn.add_argument("gt", type=Path, help="ground-truth file (SQUARE or TUM)")
  _add_common_args(pn)
  pn.add_argument(
    "--slice",
    choices=["translation", "joint", "rotation"],
    default="translation",
    help="which tangent block to score (default: translation, dof 3)",
  )
  pn.add_argument(
    "--json", action="store_true", help="print the verdict as JSON"
  )

  pp = sub.add_parser(
    "pair",
    help="no-reference verdict: score two filters against each other; "
    "an elevated pairwise NEES certifies overconfidence with no ground "
    "truth consulted (lower bound)",
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
  ps.add_argument("gt", type=Path, help="ground-truth file (SQUARE or TUM)")
  _add_common_args(ps)
  ps.add_argument("--alpha", type=float, default=0.1)
  ps.add_argument("--n_samples", type=int, default=128)
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
    "belief-transform intervention: re-score the truth under a covariance-"
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
    "--consume-gt-cov as Σ_eff = c·Σ_pred + Σ_gt.",
  )
  ps.add_argument(
    "--consume-gt-cov",
    action="store_true",
    help="fold the GT observer covariance into the predictive before scoring: "
    "Σ_eff = Σ_pred + Σ_gt (proof-of-concept; the score then accounts for "
    "ground-truth uncertainty, not just the filter's). Requires "
    "--sync=interpolate_gt, which supplies Σ_gt as the GP predictive "
    "covariance — a stand-in for the per-sample GT covariance datasets do not "
    "yet ship (the call-to-action).",
  )
  ps.add_argument(
    "--calibration",
    action="store_true",
    help="print the per-slice log-score calibration/sharpness split + ANEES "
    "χ² consistency verdict (joint dof-6, translation/rotation dof-3). The "
    "calibration term (½·NEES) stays graded where CRPS/coverage saturate; "
    "the χ² interval gives an optimistic|consistent|conservative verdict. "
    "Requires gaussian_se3 input.",
  )
  ps.add_argument(
    "--json-out",
    type=Path,
    default=None,
    help="if set, also write the structured Report as JSON to this path",
  )

  args = parser.parse_args(argv)

  if args.cmd == "validate":
    return _validate(args.file)
  if args.cmd == "nees":
    return _nees(args)
  if args.cmd == "pair":
    return _pair(args)
  if args.cmd == "score":
    return _score(args)
  return 1


def _validate(path: Path) -> int:
  try:
    with path.open() as f:
      header = parse_header(f)
      n = sum(1 for _ in iter_steps(f, header))
  except (FormatError, OSError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2
  print(f"ok: {path.name} ({header.representation.value}, {n} timesteps)")
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
  """Read an SE(3) ``T_est_body__gt_body`` from a JSON file with ``R``, ``t``.

  Matches the Spires ``cam-lidar-imu.yaml`` convention: ``R`` is the 3×3
  rotation row-major, ``t`` is the 3-vector. The semantics are that of a ROS
  transform ``target_T_source`` where ``target`` is the estimate's body frame
  and ``source`` is the GT's body frame.
  """
  data = json.loads(path.read_text())
  R = np.array(data["R"], dtype=float).reshape(3, 3)
  t = np.array(data["t"], dtype=float).reshape(3)
  return homogeneous(R, t)


def _resolve_sync(
  args: argparse.Namespace,
  est_steps: list,
  gt_steps: list,
  tangent_order: TangentOrder | None,
) -> (
  tuple[
    MatchResult, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
  ]
  | None
):
  """Pair estimate and GT poses by --sync mode.

  Sixth element is the matched GT covariance (Q, 6, 6) under interpolate_gt
  (the GP predictive Σ_gt), else None.

  Prints to stderr and returns None on error (empty match / no overlap).
  """
  est_ts = np.array([s.timestamp for s in est_steps])
  gt_ts = np.array([s.timestamp for s in gt_steps])
  gt_positions = np.array([s.translation for s in gt_steps])
  gt_quats = np.array([s.quat_xyzw for s in gt_steps])

  match args.sync:
    case SyncMode.INTERPOLATE_GT:
      query_t = est_ts + args.t_offset
      interp_t, interp_q, interp_cov, keep = interpolate_gt_at(
        query_t,
        gt_ts,
        gt_positions,
        gt_quats,
        window=args.sync_window,
        length_scale_s=args.sync_length_scale,
      )
      if not keep.any():
        print(
          "error: no estimate timestamps fall within the GT range",
          file=sys.stderr,
        )
        return None
      est_idx = np.where(keep)[0]
      # gt_indices = -1 — the interpolated GT is not a row in gt_steps.
      match_res = MatchResult(
        est_indices=est_idx,
        gt_indices=np.full(est_idx.size, -1, dtype=int),
        n_total=len(est_ts),
        n_matched=int(est_idx.size),
        n_dropped=int(len(est_ts) - est_idx.size),
        gap_seconds=np.zeros(est_idx.size),
      )
      matched_est = [est_steps[i] for i in est_idx]
      matched_gt_t = interp_t[est_idx]
      matched_gt_q = interp_q[est_idx]
      # Risk surrogate: GP predictive σ — interpolation hits the query exactly,
      # but the predictive variance bounds how trustworthy that hit is.
      risks = np.sqrt(np.maximum(interp_cov[est_idx, 0, 0], 0.0))
      return (
        match_res,
        matched_est,
        matched_gt_t,
        matched_gt_q,
        risks,
        interp_cov[est_idx],
      )

    case SyncMode.NEAREST:
      match_res = match_timestamps(
        est_ts, gt_ts, args.t_max_diff, args.t_offset
      )
      if match_res.n_matched == 0:
        print("error: no matched pairs", file=sys.stderr)
        return None
      matched_est = [est_steps[i] for i in match_res.est_indices]
      matched_gt_t = gt_positions[match_res.gt_indices]
      matched_gt_q = gt_quats[match_res.gt_indices]
      risks = sync_risk(
        est_steps,
        gt_ts,
        gt_positions,
        match_res.est_indices,
        match_res.gt_indices,
        est_ts=est_ts,
        t_offset=args.t_offset,
        tangent_order=tangent_order,
      )
      return match_res, matched_est, matched_gt_t, matched_gt_q, risks, None


def _compute_scores(
  aligned_est: list,
  matched_gt_t: np.ndarray,
  matched_gt_q: np.ndarray,
  order: TangentOrder,
  n_samples: int,
  alpha: float,
  seed: int,
) -> dict[str, ScoreSummary]:
  rng = np.random.default_rng(seed)
  crps_t: list[float] = []
  crps_r: list[float] = []
  es: list[float] = []
  is_t: list[float] = []
  log_joint: list[float] = []
  log_trans: list[float] = []
  log_rot: list[float] = []
  for s, gt_t, gt_q in zip(
    aligned_est, matched_gt_t, matched_gt_q, strict=False
  ):
    crps_t.append(translation_crps(s, gt_t, order))
    crps_r.append(rotation_crps(s, gt_q, order, rng=rng))
    es.append(energy_score(s, gt_t, gt_q, order, n_samples, rng))

    match s:
      case GaussianStep():
        ls = gaussian_log_score(s, gt_t, gt_q, order)
        log_joint.append(ls.joint)
        log_trans.append(ls.translation)
        log_rot.append(ls.rotation)
        is_t.append(
          translation_magnitude_interval_score(
            s, gt_t, alpha, n_samples, rng, order
          )
        )
      case EnsembleStep():
        is_t.append(
          translation_magnitude_interval_score(
            s, gt_t, alpha, n_samples, rng, order
          )
        )
      case DeterministicStep():
        pass

  boot_rng = np.random.default_rng(seed + 2)
  scores: dict[str, ScoreSummary] = {
    "translation_crps": summarize(crps_t, rng=boot_rng),
    "rotation_crps": summarize(crps_r, rng=boot_rng),
    "energy_score": summarize(es, rng=boot_rng),
  }
  if log_joint:
    scores["log_score"] = summarize(log_joint, rng=boot_rng)
    scores["log_score_translation"] = summarize(log_trans, rng=boot_rng)
    scores["log_score_rotation"] = summarize(log_rot, rng=boot_rng)
  if is_t:
    scores["interval_score"] = summarize(is_t, rng=boot_rng)
  return scores


@dataclass
class PreparedRun:
  """Estimate/GT pair after loading, frame checks, sync, and alignment."""

  est_header: SquareHeader
  matched_est: list[Step]
  aligned_est: list[Step]
  matched_gt_t: np.ndarray
  matched_gt_q: np.ndarray
  match: MatchResult
  fit: AlignmentFit
  order: TangentOrder
  risks: np.ndarray
  gt_cov: np.ndarray | None  # GP predictive Σ_gt under interpolate_gt


def _load_estimate(args: argparse.Namespace) -> tuple[SquareHeader, list[Step]]:
  """Load the estimate: SQUARE natively, or the bare-TUM escape hatches.

  A header-less estimate may be wide TUM (29 columns: pose + the 21
  lower-triangle covariance entries) or plain TUM (8 columns) with a
  ``--cov`` sidecar; either way ``--est-body-frame`` must declare what the
  missing header would have. Raises FormatError on undeclared/unknown input.
  """
  if not looks_like_tum(args.est):
    return load_square(args.est)

  if args.est_body_frame is None:
    raise FormatError(
      "estimate file is header-less (bare TUM) so its body frame is "
      "unknown; pass --est-body-frame <name>"
    )
  ncols = sniff_tum_columns(args.est)
  common = dict(
    pose_frame=args.est_pose_frame,
    body_frame=args.est_body_frame,
    tangent_convention=args.tangent_convention,
    tangent_order=args.tangent_order,
    gauge=args.gauge,
  )
  if ncols == 29:
    header, steps = load_tum_gaussian(args.est, **common)
  elif ncols == 8 and args.cov is not None:
    header, steps = load_tum_with_sidecar(args.est, args.cov, **common)
  elif ncols == 8:
    tum, steps = load_tum(
      args.est, pose_frame=args.est_pose_frame, body_frame=args.est_body_frame
    )
    header = replace(tum.to_square(), gauge=args.gauge)
  else:
    raise FormatError(
      f"header-less estimate has {ncols} columns; expected 8 (TUM, "
      "optionally with --cov sidecar) or 29 (TUM + 21 lower-triangle "
      "covariance entries)"
    )
  print(
    f"note: bare-TUM estimate read as {header.representation.value} "
    f"(body_frame={header.body_frame!r}, pose_frame={header.pose_frame!r}, "
    f"gauge={header.gauge.value}"
    + (
      f", {header.tangent_convention.value}, {header.tangent_order.value})"
      if header.representation is Representation.GAUSSIAN_SE3
      and header.tangent_convention is not None
      and header.tangent_order is not None
      else ")"
    ),
    file=sys.stderr,
  )
  return header, steps


def _prepare(args: argparse.Namespace) -> PreparedRun | None:
  """Load est+gt, check frames, sync, and align (shared by score/nees).

  Prints to stderr and returns None on any input error.
  """
  try:
    est_header, est_steps = _load_estimate(args)
    if looks_like_tum(args.gt):
      if args.gt_body_frame is None:
        print(
          "error: ground-truth file is plain TUM but its body frame "
          "is not declared. Pass --gt-body-frame <name> matching the "
          f"estimate's BODY_FRAME (estimate declares "
          f"{est_header.body_frame!r}).",
          file=sys.stderr,
        )
        return None
      gt_tum, gt_steps = load_tum(
        args.gt,
        pose_frame=args.gt_pose_frame,
        body_frame=args.gt_body_frame,
      )
      gt_header: SquareHeader = gt_tum.to_square()
    else:
      gt_header, gt_steps = load_square(args.gt)
  except (FormatError, OSError) as e:
    print(f"error: {e}", file=sys.stderr)
    return None

  if est_header.pose_frame != gt_header.pose_frame:
    print(
      f"error: pose frames differ — estimate is {est_header.pose_frame!r}, "
      f"ground truth is {gt_header.pose_frame!r}. The scoring tool does "
      "not transform between outer reference frames; pre-align both "
      "trajectories into a common pose frame, or pass --gt-pose-frame "
      "if the GT file is plain TUM and the default 'world' is wrong.",
      file=sys.stderr,
    )
    return None

  if est_header.body_frame != gt_header.body_frame:
    if args.body_frame_transform is None:
      print(
        f"error: body frames differ — estimate is {est_header.body_frame!r}, "
        f"ground truth is {gt_header.body_frame!r}. Provide "
        "--body-frame-transform PATH with the SE(3) "
        "T_est_body__gt_body, or rewrite both trajectories in a common "
        "frame.",
        file=sys.stderr,
      )
      return None
    T_body_off = _load_body_frame_transform(args.body_frame_transform)
    est_steps = [
      apply_body_transform(
        s,
        T_body_off,
        tangent_convention=est_header.tangent_convention,
        tangent_order=est_header.tangent_order,
      )
      for s in est_steps
    ]

  resolved = _resolve_sync(args, est_steps, gt_steps, est_header.tangent_order)
  if resolved is None:
    return None
  match, matched_est, matched_gt_t, matched_gt_q, risks, gt_cov = resolved

  align_mode = args.align or align_mode_for_gauge(est_header.gauge)
  n_align = args.n_to_align if args.n_to_align else len(matched_est)
  n_align = min(n_align, len(matched_est))
  fit_t = np.array([s.translation for s in matched_est[:n_align]])
  fit = fit_alignment(fit_t, matched_gt_t[:n_align], mode=align_mode)

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
    matched_gt_t=matched_gt_t,
    matched_gt_q=matched_gt_q,
    match=match,
    fit=fit,
    order=est_header.tangent_order or TangentOrder.TRANS_ROT,
    risks=risks,
    gt_cov=gt_cov,
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
  for s, gt_t, gt_q in zip(
    pr.aligned_est, pr.matched_gt_t, pr.matched_gt_q, strict=False
  ):
    if not isinstance(s, GaussianStep):
      continue
    dec = gaussian_log_score_components(s, gt_t, gt_q, pr.order)
    vals.append(getattr(dec, args.slice).nees)

  dof = 6 if args.slice == "joint" else 3
  v = nees_verdict(np.asarray(vals), dof=dof)
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
    T_off = _load_body_frame_transform(args.body_frame_transform)
    steps_a = [
      apply_body_transform(
        s,
        T_off,
        tangent_convention=header_a.tangent_convention,
        tangent_order=header_a.tangent_order,
      )
      for s in steps_a
    ]
    header_a = replace(header_a, body_frame=header_b.body_frame)

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

  v = nees_verdict(res.nees, dof=3, alpha=args.alpha)
  if args.json:
    out = {
      "n_matched": res.n_matched,
      "n_scored": res.n_scored,
      "join_frac": res.join_frac,
      "gap_median_ms": res.gap_median_ms,
      "med_d_norm_m": res.med_d_norm,
      "log_score_mean": res.log_score_mean,
      "k_pair_lower_bound": res.k_pair,
      "k_axis_lower_bound": list(res.k_axis),
      "verdict": v.to_dict(),
      "caveat": PROPRIETY_CAVEAT,
    }
    print(json.dumps(out, indent=2, default=_json_default))
  else:
    print(
      f"matched {res.n_matched} pose pairs, scored {res.n_scored}  "
      f"(join {res.join_frac:.2f}, median gap {res.gap_median_ms:.1f} ms)"
    )
    print(render_pair_verdict(v, caveat=PROPRIETY_CAVEAT))
  return 0


def _score(args: argparse.Namespace) -> int:
  pr = _prepare(args)
  if pr is None:
    return 2
  est_header = pr.est_header
  matched_est = pr.matched_est
  aligned_est = pr.aligned_est
  matched_gt_t = pr.matched_gt_t
  matched_gt_q = pr.matched_gt_q
  match, fit, risks, gt_cov = pr.match, pr.fit, pr.risks, pr.gt_cov

  ensemble_diag = None
  if est_header.representation is Representation.ENSEMBLE_SE3:
    ensemble_diag = ensemble_diagnostics(
      [s for s in matched_est if isinstance(s, EnsembleStep)],
      est_header.weight_format or WeightFormat.LINEAR,
      normalized=bool(est_header.weights_normalized),
    )

  order = est_header.tangent_order or TangentOrder.TRANS_ROT

  # A5: fold the GT observer covariance into the predictive — Σ_eff = Σ_pred +
  # Σ_gt — so the score consumes ground-truth uncertainty, not just the
  # filter's. Σ_gt is the GP predictive covariance (interpolate_gt); isotropic,
  # so the tangent-order of the addition is immaterial. The gaussian-validity
  # check stays on the raw predictive (it audits the filter's own Σ).
  ess_c = None
  if args.ess_inflate is not None:
    c_ts, c_val = _load_ess_cfile(args.ess_inflate)
    step_ts = np.array([s.timestamp for s in aligned_est])
    j = np.clip(np.searchsorted(c_ts, step_ts), 0, len(c_ts) - 1)
    jl = np.maximum(j - 1, 0)
    j = np.where(np.abs(c_ts[jl] - step_ts) < np.abs(c_ts[j] - step_ts), jl, j)
    ess_c = c_val[j]

  scored_est = aligned_est
  if args.consume_gt_cov and gt_cov is None:
    print(
      "error: --consume-gt-cov requires --sync=interpolate_gt (it supplies "
      "Σ_gt as the GP predictive covariance)",
      file=sys.stderr,
    )
    return 2
  if args.ess_inflate is not None or args.consume_gt_cov:
    scored_est = []
    for k, s in enumerate(aligned_est):
      if not isinstance(s, GaussianStep):
        scored_est.append(s)
        continue
      cov = s.covariance
      if ess_c is not None:  # Σ → c·Σ  (ESS inflation)
        cov = ess_c[k] * cov
      if args.consume_gt_cov:  # + Σ_gt   (observer uncertainty)
        cov = cov + gt_cov[k]
      scored_est.append(replace(s, covariance=cov))

  gaussian_validity = None
  if est_header.representation is Representation.GAUSSIAN_SE3:
    gaussian_validity = gaussian_rotation_validity(
      [s for s in aligned_est if isinstance(s, GaussianStep)],
      tangent_order=order,
    )
  scores = _compute_scores(
    scored_est,
    matched_gt_t,
    matched_gt_q,
    order=order,
    n_samples=args.n_samples,
    alpha=args.alpha,
    seed=args.seed,
  )

  cal = None
  if est_header.representation is not Representation.DETERMINISTIC:
    cal = calibrate(
      scored_est,
      matched_gt_t,
      matched_gt_q,
      tangent_order=order,
      alpha=args.alpha,
      n_samples=args.n_samples,
      rng=np.random.default_rng(args.seed + 1),
    )

  traj_len = (
    float(np.sum(np.linalg.norm(np.diff(matched_gt_t, axis=0), axis=1)))
    if len(matched_gt_t) > 1
    else 0.0
  )

  split = None
  if args.calibration:
    if est_header.representation is Representation.GAUSSIAN_SE3:
      split = _calibration_split_dict(
        args, scored_est, matched_gt_t, matched_gt_q, order
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
    gaussian_validity=gaussian_validity,
    scores=scores,
    calibration=cal,
    trajectory_length_m=traj_len,
    sync_mode=args.sync,
  )
  rep.calibration_split = split
  if args.calibration and len(matched_gt_t) > 2:
    bv_windows = (
      [float(x) for x in args.rpe_window.split(",") if x.strip()]
      if args.rpe_window
      else [0.1, 1.0, 10.0]
    )
    rep.bias_variance = [
      r.to_dict()
      for r in bias_variance(aligned_est, matched_gt_t, windows_s=bv_windows)
    ]
  rep.recommendations = recommendations(rep)
  rep.diagnoses = diagnose(rep)
  if args.json_out is not None:
    _write_report_json(rep, args.json_out)
  print(render_report(rep))

  if args.rpe_window:
    _report_relative_crps(args, aligned_est, matched_gt_t, order, est_header)
  if split is not None:
    _emit_calibration_machine_lines(split)
  if args.student_t:
    _report_student_t(
      args, scored_est, matched_gt_t, matched_gt_q, order, est_header
    )
  return 0


def _report_relative_crps(
  args: argparse.Namespace,
  aligned_est: list,
  matched_gt_t: np.ndarray,
  order: TangentOrder,
  est_header: SquareHeader,
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
  windows = [float(x) for x in args.rpe_window.split(",") if x.strip()]
  results = relative_translation_crps(
    aligned_est,
    matched_gt_t,
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
  matched_gt_t: np.ndarray,
  matched_gt_q: np.ndarray,
  order: TangentOrder,
) -> dict | None:
  r"""Build the calibration/sharpness split dict consumed by the report + diagnose.

  ``-log p`` splits exactly into ``calibration = ½·NEES`` and ``sharpness =
  ½(log|Σ| + d·log2π)``; the ANEES χ² interval gives the per-slice
  ``optimistic|consistent|conservative`` verdict (joint dof-6, trans/rot dof-3).
  Returns ``{"absolute": {...}, "windowed": [...]}`` or ``None`` if there are no
  Gaussian steps. ``windowed`` is present only when ``--rpe-window`` is set.
  """
  nees = {"joint": [], "translation": [], "rotation": []}
  calib = {"joint": [], "translation": [], "rotation": []}
  sharp = {"joint": [], "translation": [], "rotation": []}
  for s, gt_t, gt_q in zip(
    aligned_est, matched_gt_t, matched_gt_q, strict=False
  ):
    if not isinstance(s, GaussianStep):
      continue
    dec = gaussian_log_score_components(s, gt_t, gt_q, order)
    for name, comp in (
      ("joint", dec.joint),
      ("translation", dec.translation),
      ("rotation", dec.rotation),
    ):
      nees[name].append(comp.nees)
      calib[name].append(comp.calibration)
      sharp[name].append(comp.sharpness)

  if not nees["joint"]:
    return None

  def _median_finite(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

  dof = {"joint": 6, "translation": 3, "rotation": 3}
  absolute: dict[str, dict] = {}
  for name in ("joint", "translation", "rotation"):
    res = anees_consistency(
      np.asarray(nees[name], dtype=float), dof=dof[name], alpha=args.alpha
    )
    absolute[name] = {
      "anees": res.anees,
      "nees_median": res.median,
      "dof": res.dof,
      "n": res.n,
      "lo": res.lo,
      "hi": res.hi,
      "verdict": res.verdict,
      "calibration_median": _median_finite(calib[name]),
      "sharpness_median": _median_finite(sharp[name]),
    }

  split: dict = {"absolute": absolute}
  if args.rpe_window:
    windows = [float(x) for x in args.rpe_window.split(",") if x.strip()]
    rows = relative_calibration(
      aligned_est,
      matched_gt_t,
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
  matched_gt_t: np.ndarray,
  matched_gt_q: np.ndarray,
  order: TangentOrder,
  est_header: SquareHeader,
) -> None:
  r"""Student-t belief-transform intervention: mean proper log-score vs ν.

  Re-scores the truth under a covariance-matched Student-t belief (heavier
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
  gauss: list[float] = []
  tcols: dict[float, list[float]] = {nu: [] for nu in nus}
  for s, gt_t, gt_q in zip(
    aligned_est, matched_gt_t, matched_gt_q, strict=False
  ):
    if not isinstance(s, GaussianStep):
      continue
    xi = se3_log(
      relative(
        pose_matrix(s.translation, s.quat_xyzw), pose_matrix(gt_t, gt_q)
      ),
      order=order,
    )
    gauss.append(gaussian_log_score(s, gt_t, gt_q, order).joint)
    for nu in nus:
      tcols[nu].append(student_t_neg_log_density(xi, s.covariance, nu))

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
