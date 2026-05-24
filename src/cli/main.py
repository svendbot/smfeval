"""smfeval CLI: validate and score commands."""

import argparse
import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import numpy as np

from src.align import (
  align_mode_for_gauge,
  apply_body_transform,
  fit_alignment,
  propagate_step,
)
from src.format import (
  FormatError,
  Representation,
  SquareHeader,
  TangentOrder,
  WeightFormat,
)
from src.io import (
  iter_steps,
  load_square,
  load_tum,
  looks_like_tum,
  parse_header,
)
from src.report import build_report, recommendations, render_report
from src.scoring import (
  ScoreSummary,
  calibrate,
  energy_score,
  ensemble_diagnostics,
  gaussian_log_score,
  gaussian_rotation_validity,
  rotation_crps,
  summarize,
  translation_crps,
  translation_magnitude_interval_score,
)
from src.se3.lie import homogeneous
from src.steps import DeterministicStep, EnsembleStep, GaussianStep, Step
from src.sync import (
  MatchResult,
  SyncMode,
  interpolate_gt_at,
  match_timestamps,
  sync_risk,
)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="smfeval")
  sub = parser.add_subparsers(dest="cmd", required=True)

  pv = sub.add_parser("validate", help="header and row sanity checks")
  pv.add_argument("file", type=Path)

  ps = sub.add_parser("score", help="produce a scoring report")
  ps.add_argument("est", type=Path, help="smfeval-format estimate file")
  ps.add_argument("gt", type=Path, help="ground-truth file (smfeval or TUM)")
  ps.add_argument("--t_max_diff", type=float, default=0.01)
  ps.add_argument("--t_offset", type=float, default=0.0)
  ps.add_argument(
    "--align", default=None, choices=["none", "se3", "gravity_yaw", "sim3"]
  )
  ps.add_argument("--n_to_align", type=int, default=None)
  ps.add_argument("--alpha", type=float, default=0.1)
  ps.add_argument("--n_samples", type=int, default=128)
  ps.add_argument("--seed", type=int, default=0)
  ps.add_argument(
    "--gt-body-frame",
    default=None,
    help="body frame of the ground-truth file (required when GT is plain TUM)",
  )
  ps.add_argument(
    "--gt-pose-frame",
    default="world",
    help="pose frame (outer container) of the ground-truth file when GT is "
    'plain TUM. Default "world", matching the common TUM convention. '
    "Must equal the estimate's POSE_FRAME — no in-tool transform.",
  )
  ps.add_argument(
    "--body-frame-transform",
    type=Path,
    default=None,
    help='JSON file {"R": [9 floats row-major], "t": [3 floats]} giving '
    "T_est_body__gt_body (the new body frame's pose in the old body "
    "frame). Required when est and gt declare different BODY_FRAMEs.",
  )
  ps.add_argument(
    "--sync",
    type=SyncMode,
    choices=list(SyncMode),
    default=SyncMode.NEAREST,
    help="GT-matching strategy. 'nearest' (default) picks the nearest GT "
    "timestamp; 'interpolate_gt' fits a piecewise GP on SE(3) over a "
    "local window of GT samples and queries it at each est timestamp "
    "(Zhang & Scaramuzza 2019, §IV.B).",
  )
  ps.add_argument(
    "--sync_window",
    type=int,
    default=10,
    help="number of GT samples per GP window when --sync=interpolate_gt",
  )
  ps.add_argument(
    "--sync_length_scale",
    type=float,
    default=0.1,
    help="SE-kernel length scale in seconds when --sync=interpolate_gt",
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
) -> tuple[MatchResult, list, np.ndarray, np.ndarray, np.ndarray] | None:
  """Pair estimate and GT poses by --sync mode.

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
      return match_res, matched_est, matched_gt_t, matched_gt_q, risks

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
      return match_res, matched_est, matched_gt_t, matched_gt_q, risks


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


def _score(args: argparse.Namespace) -> int:
  try:
    est_header, est_steps = load_square(args.est)
    if looks_like_tum(args.gt):
      if args.gt_body_frame is None:
        print(
          "error: ground-truth file is plain TUM but its body frame "
          "is not declared. Pass --gt-body-frame <name> matching the "
          f"estimate's BODY_FRAME (estimate declares "
          f"{est_header.body_frame!r}).",
          file=sys.stderr,
        )
        return 2
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
    return 2

  if est_header.pose_frame != gt_header.pose_frame:
    print(
      f"error: pose frames differ — estimate is {est_header.pose_frame!r}, "
      f"ground truth is {gt_header.pose_frame!r}. The scoring tool does "
      "not transform between outer reference frames; pre-align both "
      "trajectories into a common pose frame, or pass --gt-pose-frame "
      "if the GT file is plain TUM and the default 'world' is wrong.",
      file=sys.stderr,
    )
    return 2

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
      return 2
    T_body_off = _load_body_frame_transform(args.body_frame_transform)
    new_est: list[Step] = []
    for s in est_steps:
      match s:
        case GaussianStep():
          new_est.append(
            apply_body_transform(
              s,
              T_body_off,
              tangent_convention=est_header.tangent_convention,
              tangent_order=est_header.tangent_order,
            )
          )
        case EnsembleStep():
          new_est.append(
            apply_body_transform(
              s,
              T_body_off,
              tangent_convention=est_header.tangent_convention,
              tangent_order=est_header.tangent_order,
            )
          )
        case DeterministicStep():
          new_est.append(
            apply_body_transform(
              s,
              T_body_off,
              tangent_convention=est_header.tangent_convention,
              tangent_order=est_header.tangent_order,
            )
          )
    est_steps = new_est

  resolved = _resolve_sync(args, est_steps, gt_steps, est_header.tangent_order)
  if resolved is None:
    return 2
  match, matched_est, matched_gt_t, matched_gt_q, risks = resolved

  align_mode = args.align or align_mode_for_gauge(est_header.gauge)
  n_align = args.n_to_align if args.n_to_align else len(matched_est)
  n_align = min(n_align, len(matched_est))
  fit_t = np.array([s.translation for s in matched_est[:n_align]])
  fit = fit_alignment(fit_t, matched_gt_t[:n_align], mode=align_mode)

  aligned_est: list[Step] = []
  for s in matched_est:
    match s:
      case GaussianStep():
        aligned_est.append(
          propagate_step(
            s,
            fit.transform,
            scale=fit.scale,
            tangent_convention=est_header.tangent_convention,
            tangent_order=est_header.tangent_order,
          )
        )
      case EnsembleStep():
        aligned_est.append(
          propagate_step(
            s,
            fit.transform,
            scale=fit.scale,
            tangent_convention=est_header.tangent_convention,
            tangent_order=est_header.tangent_order,
          )
        )
      case DeterministicStep():
        aligned_est.append(
          propagate_step(
            s,
            fit.transform,
            scale=fit.scale,
            tangent_convention=est_header.tangent_convention,
            tangent_order=est_header.tangent_order,
          )
        )

  ensemble_diag = None
  if est_header.representation is Representation.ENSEMBLE_SE3:
    ensemble_diag = ensemble_diagnostics(
      [s for s in matched_est if isinstance(s, EnsembleStep)],
      est_header.weight_format or WeightFormat.LINEAR,
      normalized=bool(est_header.weights_normalized),
    )

  order = est_header.tangent_order or TangentOrder.TRANS_ROT

  gaussian_validity = None
  if est_header.representation is Representation.GAUSSIAN_SE3:
    gaussian_validity = gaussian_rotation_validity(
      [s for s in aligned_est if isinstance(s, GaussianStep)],
      tangent_order=order,
    )
  scores = _compute_scores(
    aligned_est,
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
      aligned_est,
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
  rep.recommendations = recommendations(rep)
  if args.json_out is not None:
    _write_report_json(rep, args.json_out)
  print(render_report(rep))
  return 0


if __name__ == "__main__":
  sys.exit(main())
