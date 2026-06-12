#!/usr/bin/env python3
"""Decompose rotation residual into alignment-absorbed vs per-pair drift.

Given a SQUARE estimate file and a TUM ground-truth, prints the geodesic
rotation residual (median, p95) at three stages:
  1. raw — no alignment applied
  2. gravity_yaw — yaw rotation fitted via 2D Procrustes on (x, y)
  3. se3 — full Kabsch–Umeyama on translation positions
alongside the rotation_crps for stages 2 and 3.

Useful for separating "global frame mismatch the alignment ate" from
"per-pair drift that no global rotation can fix". On Offroad1_alpha, FAST-LIO
publishes gravity-aligned body in its own world frame: ~80° yaw is global
initial-frame mismatch (absorbed by both modes), and ~1.2 rad rotation CRPS
is real drift on rough terrain that nothing global can absorb.

Usage:
  python scripts/diagnose_rotation_alignment.py <est.SQUARE> <gt.tum>
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from smfeval.align import fit_alignment, propagate_step
from smfeval.io import iter_steps, parse_header
from smfeval.io.reader import _iter_deterministic
from smfeval.scoring import rotation_crps
from smfeval.sync import match_timestamps


def _angle(R1: np.ndarray, R2: np.ndarray) -> float:
  return float(np.linalg.norm(Rotation.from_matrix(R1.T @ R2).as_rotvec()))


def main(est_path: Path, gt_path: Path) -> None:
  with est_path.open() as f:
    eh = parse_header(f)
    est_steps = list(iter_steps(f, eh))
  with gt_path.open() as f:
    gt_steps = list(_iter_deterministic(f))

  est_ts = np.array([s.timestamp for s in est_steps])
  gt_ts = np.array([s.timestamp for s in gt_steps])
  gt_t = np.array([s.translation for s in gt_steps])
  gt_q = np.array([s.quat_xyzw for s in gt_steps])

  m = match_timestamps(est_ts, gt_ts, 0.01, 0.0)
  matched = [est_steps[i] for i in m.est_indices]
  mt = gt_t[m.gt_indices]
  mq = gt_q[m.gt_indices]
  et = np.array([s.translation for s in matched])

  raw = [
    _angle(
      Rotation.from_quat(s.quat_xyzw).as_matrix(),
      Rotation.from_quat(q).as_matrix(),
    )
    for s, q in zip(matched, mq, strict=False)
  ]
  print(
    f"{'raw':12s}: geodesic median {np.median(raw):.4f} rad  "
    f"p95 {np.quantile(raw, 0.95):.4f}"
  )

  for mode in ("gravity_yaw", "se3"):
    fit = fit_alignment(et, mt, mode=mode)
    aligned = [
      propagate_step(
        s,
        fit.transform,
        scale=fit.scale,
        tangent_convention=eh.tangent_convention,
        tangent_order=eh.tangent_order,
      )
      for s in matched
    ]
    ang = [
      _angle(
        Rotation.from_quat(s.quat_xyzw).as_matrix(),
        Rotation.from_quat(q).as_matrix(),
      )
      for s, q in zip(aligned, mq, strict=False)
    ]
    rng = np.random.default_rng(0)
    crps = [
      rotation_crps(s, q, eh.tangent_order, 32, rng)
      for s, q in zip(aligned, mq, strict=False)
    ]
    print(
      f"{mode:12s}: geodesic median {np.median(ang):.4f} rad  "
      f"p95 {np.quantile(ang, 0.95):.4f}  "
      f"CRPS mean {np.mean(crps):.4f}"
    )


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
  main(Path(sys.argv[1]), Path(sys.argv[2]))
