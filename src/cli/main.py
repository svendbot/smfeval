"""smfeval CLI: validate and score commands."""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.align import (
    align_mode_for_gauge,
    fit_alignment,
    propagate_step,
)
from src.io import iter_steps, parse_header
from src.io.reader import _iter_deterministic
from src.report import build_report, recommendations, render_report
from src.scoring import (
    calibrate,
    energy_score,
    ensemble_diagnostics,
    gaussian_log_score,
    rotation_crps,
    translation_crps,
    translation_magnitude_interval_score,
)
from src.steps import EnsembleStep, GaussianStep
from src.sync import match_timestamps, sync_risk
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


def _load(path: Path) -> tuple[Header, list]:
    if _looks_like_tum(path):
        header = Header(
            format_version="smfeval/0.2",
            representation=Representation.DETERMINISTIC,
            pose_frame="world",
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


def _score(args: argparse.Namespace) -> int:
    try:
        est_header, est_steps = _load(args.est)
        _, gt_steps = _load(args.gt)
    except (FormatError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    est_ts = np.array([s.timestamp for s in est_steps])
    gt_ts = np.array([s.timestamp for s in gt_steps])
    gt_positions = np.array([s.translation for s in gt_steps])
    gt_quats = np.array([s.quat_xyzw for s in gt_steps])

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
    scores: dict[str, float] = {}
    crps_t = []
    crps_r = []
    es = []
    is_t = []
    log_scores = []
    for s, gt_t, gt_q in zip(aligned_est, matched_gt_t, matched_gt_q):
        crps_t.append(translation_crps(s, gt_t, order, args.n_samples, rng))
        crps_r.append(rotation_crps(s, gt_q, order, min(args.n_samples, 32), rng))
        es.append(energy_score(s, gt_t, gt_q, order, args.n_samples, rng))

        if isinstance(s, GaussianStep):
            log_scores.append(gaussian_log_score(s, gt_t, gt_q, order))
        if isinstance(s, (GaussianStep, EnsembleStep)):
            is_t.append(
                translation_magnitude_interval_score(
                    s, gt_t, args.alpha, args.n_samples, rng, order
                )
            )

    scores["translation_crps"] = float(np.mean(crps_t))
    scores["rotation_crps"] = float(np.mean(crps_r))
    scores["energy_score"] = float(np.mean(es))
    if log_scores:
        scores["log_score"] = float(np.mean(log_scores))
    if is_t:
        scores["interval_score"] = float(np.mean(is_t))

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
    )
    rep.recommendations = recommendations(rep)
    print(render_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
