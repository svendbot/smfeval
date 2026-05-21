"""smfeval CLI: validate and score commands."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.align import (
    align_mode_for_gauge,
    apply_body_transform,
    fit_alignment,
    propagate_step,
)
from src.io import iter_steps, parse_header
from src.io.reader import _iter_deterministic
from src.report import build_report, recommendations, render_report
from src.scoring import (
    ScoreSummary,
    calibrate,
    energy_score,
    ensemble_diagnostics,
    gaussian_log_score,
    rotation_crps,
    summarize,
    translation_crps,
    translation_magnitude_interval_score,
)
from src.steps import EnsembleStep, GaussianStep
from src.sync import interpolate_gt_at, match_timestamps, sync_risk
from src.types import (
    FormatError,
    Gauge,
    Header,
    Representation,
    TangentOrder,
    WeightFormat,
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
    ps.add_argument("--align", default=None,
                    choices=["none", "se3", "gravity_yaw", "sim3"])
    ps.add_argument("--n_to_align", type=int, default=None)
    ps.add_argument("--alpha", type=float, default=0.1)
    ps.add_argument("--n_samples", type=int, default=128)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument(
        "--gt-body-frame", default=None,
        help="body frame of the ground-truth file (required when GT is plain TUM)",
    )
    ps.add_argument(
        "--body-frame-transform", type=Path, default=None,
        help='JSON file {"R": [9 floats row-major], "t": [3 floats]} giving '
             "T_est_body__gt_body (the new body frame's pose in the old body "
             "frame). Required when est and gt declare different BODY_FRAMEs.",
    )
    ps.add_argument(
        "--sync", choices=["nearest", "interpolate_gt"], default="nearest",
        help="GT-matching strategy. 'nearest' (default) picks the nearest GT "
             "timestamp; 'interpolate_gt' fits a piecewise GP on SE(3) over a "
             "local window of GT samples and queries it at each est timestamp "
             "(Zhang & Scaramuzza 2019, §IV.B).",
    )
    ps.add_argument(
        "--sync_window", type=int, default=10,
        help="number of GT samples per GP window when --sync=interpolate_gt",
    )
    ps.add_argument(
        "--sync_length_scale", type=float, default=0.1,
        help="SE-kernel length scale in seconds when --sync=interpolate_gt",
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


def _looks_like_tum(path: Path) -> bool:
    """A file lacking any `#%FORMAT` declaration is treated as a bare TUM
    trajectory (deterministic). Matches the CLI help: GT may be smfeval or TUM.
    """
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#%FORMAT"):
                return False
            if s.startswith("#"):
                continue
            return True
    return True


def _load(path: Path, tum_body_frame: str | None = None) -> tuple[Header, list]:
    if _looks_like_tum(path):
        header = Header(
            format_version="SQUARE/0.3",
            representation=Representation.DETERMINISTIC,
            pose_frame="world",
            body_frame=tum_body_frame or "unknown",
            gauge=Gauge.FIXED,
            timestamp_unit="seconds",
            algorithm="tum",
            algorithm_version="0",
        )
        with path.open() as f:
            steps = list(_iter_deterministic(f))
        return header, steps

    with path.open() as f:
        header = parse_header(f)
        steps = list(iter_steps(f, header))
    return header, steps


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
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _score(args: argparse.Namespace) -> int:
    try:
        est_header, est_steps = _load(args.est)
        gt_header, gt_steps = _load(args.gt, tum_body_frame=args.gt_body_frame)
    except (FormatError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if gt_header.body_frame == "unknown":
        print(
            "error: ground-truth file is plain TUM but its body frame is not "
            "declared. Pass --gt-body-frame <name> matching the estimate's "
            f"BODY_FRAME (estimate declares {est_header.body_frame!r}).",
            file=sys.stderr,
        )
        return 2

    T_body_off: np.ndarray | None = None
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
        est_steps = [
            apply_body_transform(
                s, T_body_off,
                tangent_convention=est_header.tangent_convention,
                tangent_order=est_header.tangent_order,
            )
            for s in est_steps
        ]

    est_ts = np.array([s.timestamp for s in est_steps])
    gt_ts = np.array([s.timestamp for s in gt_steps])
    gt_positions = np.array([s.translation for s in gt_steps])
    gt_quats = np.array([s.quat_xyzw for s in gt_steps])

    gt_interp_var = None
    if args.sync == "interpolate_gt":
        query_t = est_ts + args.t_offset
        interp_t, interp_q, interp_cov, keep = interpolate_gt_at(
            query_t, gt_ts, gt_positions, gt_quats,
            window=args.sync_window,
            length_scale_s=args.sync_length_scale,
        )
        if not keep.any():
            print("error: no estimate timestamps fall within the GT range", file=sys.stderr)
            return 2
        est_idx = np.where(keep)[0]
        match = match_timestamps(est_ts[est_idx], est_ts[est_idx],
                                 args.t_max_diff, 0.0)
        # GP interpolation hits the query exactly — fabricate a MatchResult with
        # zero gaps, est_indices pointing into the original est_steps, and
        # gt_indices left as -1 (the interpolated GT is not a row in gt_steps).
        match.est_indices = est_idx
        match.gt_indices = np.full(est_idx.size, -1, dtype=int)
        match.n_total = len(est_ts)
        match.n_matched = int(est_idx.size)
        match.n_dropped = int(len(est_ts) - est_idx.size)
        match.gap_seconds = np.zeros(est_idx.size)
        matched_est = [est_steps[i] for i in est_idx]
        matched_gt_t = interp_t[est_idx]
        matched_gt_q = interp_q[est_idx]
        gt_interp_var = np.array([interp_cov[i, 0, 0] for i in est_idx])
    else:
        match = match_timestamps(est_ts, gt_ts, args.t_max_diff, args.t_offset)
        if match.n_matched == 0:
            print("error: no matched pairs", file=sys.stderr)
            return 2
        matched_est = [est_steps[i] for i in match.est_indices]
        matched_gt_t = gt_positions[match.gt_indices]
        matched_gt_q = gt_quats[match.gt_indices]

    mode = args.align or align_mode_for_gauge(est_header.gauge)
    n_align = args.n_to_align if args.n_to_align else len(matched_est)
    n_align = min(n_align, len(matched_est))
    fit_t = np.array([s.translation for s in matched_est[:n_align]])
    fit = fit_alignment(fit_t, matched_gt_t[:n_align], mode=mode)

    aligned_est = [
        propagate_step(
            s, fit.transform, scale=fit.scale,
            tangent_convention=est_header.tangent_convention,
            tangent_order=est_header.tangent_order,
        )
        for s in matched_est
    ]

    if args.sync == "interpolate_gt":
        # The GP hits each query exactly; the natural "risk" surrogate is the
        # GP predictive σ — when it's large, the interpolated GT is itself
        # uncertain and downstream calibration findings should be tempered.
        assert gt_interp_var is not None
        risks = np.sqrt(np.maximum(gt_interp_var, 0.0))
    else:
        risks = sync_risk(
            est_steps, gt_ts, gt_positions,
            match.est_indices, match.gt_indices,
            est_ts=est_ts, t_offset=args.t_offset,
            tangent_order=est_header.tangent_order,
        )

    ensemble_diag = None
    if est_header.representation is Representation.ENSEMBLE_SE3:
        ensemble_diag = ensemble_diagnostics(
            [s for s in matched_est if isinstance(s, EnsembleStep)],
            est_header.weight_format or WeightFormat.LINEAR,
            normalized=bool(est_header.weights_normalized),
        )

    rng = np.random.default_rng(args.seed)
    order = est_header.tangent_order or TangentOrder.TRANS_ROT
    scores: dict[str, ScoreSummary] = {}
    crps_t = []
    crps_r = []
    es = []
    is_t = []
    log_joint = []
    log_trans = []
    log_rot = []
    for s, gt_t, gt_q in zip(aligned_est, matched_gt_t, matched_gt_q):
        crps_t.append(translation_crps(s, gt_t, order, args.n_samples, rng))
        crps_r.append(rotation_crps(s, gt_q, order, min(args.n_samples, 32), rng))
        es.append(energy_score(s, gt_t, gt_q, order, args.n_samples, rng))

        if isinstance(s, GaussianStep):
            ls = gaussian_log_score(s, gt_t, gt_q, order)
            log_joint.append(ls.joint)
            log_trans.append(ls.translation)
            log_rot.append(ls.rotation)
        if isinstance(s, (GaussianStep, EnsembleStep)):
            is_t.append(
                translation_magnitude_interval_score(
                    s, gt_t, args.alpha, args.n_samples, rng, order
                )
            )

    boot_rng = np.random.default_rng(args.seed + 2)
    scores["translation_crps"] = summarize(crps_t, rng=boot_rng)
    scores["rotation_crps"] = summarize(crps_r, rng=boot_rng)
    scores["energy_score"] = summarize(es, rng=boot_rng)
    if log_joint:
        scores["log_score"] = summarize(log_joint, rng=boot_rng)
        scores["log_score_translation"] = summarize(log_trans, rng=boot_rng)
        scores["log_score_rotation"] = summarize(log_rot, rng=boot_rng)
    if is_t:
        scores["interval_score"] = summarize(is_t, rng=boot_rng)

    cal = None
    if est_header.representation is not Representation.DETERMINISTIC:
        cal = calibrate(
            aligned_est, matched_gt_t, matched_gt_q,
            tangent_order=order,
            alpha=args.alpha,
            n_samples=args.n_samples,
            rng=np.random.default_rng(args.seed + 1),
        )

    traj_len = float(
        np.sum(np.linalg.norm(np.diff(matched_gt_t, axis=0), axis=1))
    ) if len(matched_gt_t) > 1 else 0.0

    rep = build_report(
        match, fit, est_header.gauge,
        sync_risks=risks,
        sync_risk_threshold=0.3,
        ensemble=ensemble_diag,
        scores=scores,
        calibration=cal,
        trajectory_length_m=traj_len,
        sync_mode=args.sync,
    )
    rep.recommendations = recommendations(rep)
    print(render_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
