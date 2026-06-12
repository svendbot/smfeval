r"""Alignment transform fitting on matched mean trajectories.

Modes mirror the four GAUGE values:

- ``none``: identity (no DoF removed)
- ``se3``: rigid Kabsch (1976), translation + rotation, no scale
- ``gravity_yaw``: 2D Procrustes in xy + yaw rotation, full 3D translation
- ``sim3``: Umeyama (1991) closed-form similarity (7 DoF)

References:
-----------
Kabsch, W. (1976). *A solution for the best rotation to relate two sets
of vectors*. Acta Crystallographica A 32(5), 922–923.

Umeyama, S. (1991). *Least-squares estimation of transformation
parameters between two point patterns*. IEEE TPAMI 13(4), 376–380.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from smfeval.format import Gauge
from smfeval.se3.lie import homogeneous

AlignMode = Literal["none", "se3", "gravity_yaw", "sim3"]


@dataclass
class AlignmentFit:
  mode: AlignMode
  transform: np.ndarray  # 4x4
  scale: float  # 1.0 unless sim3
  dof_removed: int
  residuals: np.ndarray  # per-pair Euclidean residual after alignment

  @property
  def fitted_translation(self) -> np.ndarray:
    return self.transform[:3, 3]

  @property
  def fitted_rotation(self) -> np.ndarray:
    return self.transform[:3, :3]


_DOF: dict[AlignMode, int] = {
  "none": 0,
  "se3": 6,
  "gravity_yaw": 4,
  "sim3": 7,
}

_GAUGE_TO_MODE: dict[Gauge, AlignMode] = {
  Gauge.FIXED: "none",
  Gauge.SE3: "se3",
  Gauge.GRAVITY_YAW: "gravity_yaw",
  Gauge.SIM3: "sim3",
}


def align_mode_for_gauge(gauge: Gauge) -> AlignMode:
  return _GAUGE_TO_MODE[gauge]


def fit_alignment(
  est_positions: np.ndarray, gt_positions: np.ndarray, mode: AlignMode
) -> AlignmentFit:
  """Fit T (and scale) so that scale·R·est + t ≈ gt for matched mean positions."""
  if est_positions.shape != gt_positions.shape:
    raise ValueError("position arrays must have identical shape")
  if est_positions.ndim != 2 or est_positions.shape[1] != 3:
    raise ValueError("expected (N, 3) position arrays")

  if mode == "none":
    residuals = np.linalg.norm(est_positions - gt_positions, axis=1)
    return AlignmentFit("none", np.eye(4), 1.0, 0, residuals)

  if mode == "se3":
    R, t, s = _kabsch_umeyama(est_positions, gt_positions, with_scale=False)
  elif mode == "sim3":
    R, t, s = _kabsch_umeyama(est_positions, gt_positions, with_scale=True)
  elif mode == "gravity_yaw":
    R, t, s = _gravity_yaw_fit(est_positions, gt_positions)

  T = homogeneous(R, t)
  aligned = (s * (R @ est_positions.T)).T + t
  residuals = np.linalg.norm(aligned - gt_positions, axis=1)
  return AlignmentFit(mode, T, float(s), _DOF[mode], residuals)


def _kabsch_umeyama(
  src: np.ndarray, dst: np.ndarray, with_scale: bool
) -> tuple[np.ndarray, np.ndarray, float]:
  """Kabsch–Umeyama. Returns R, t, s such that s·R·src + t ≈ dst.

  Kabsch (1976) gives the optimal rotation; Umeyama (1991) extends it with
  a closed-form scale. With `with_scale=False` this is the rigid Kabsch
  algorithm; with `with_scale=True` it is the full Sim(3) Umeyama fit.
  """
  n = src.shape[0]
  mu_src = src.mean(axis=0)
  mu_dst = dst.mean(axis=0)
  src_c = src - mu_src
  dst_c = dst - mu_dst
  cov = (dst_c.T @ src_c) / n
  U, D, Vt = np.linalg.svd(cov)
  S = np.eye(3)
  if np.linalg.det(U) * np.linalg.det(Vt) < 0:
    S[2, 2] = -1.0
  R = U @ S @ Vt
  if with_scale:
    var_src = (src_c**2).sum() / n
    s = float((D * np.diag(S)).sum() / var_src) if var_src > 0 else 1.0
  else:
    s = 1.0
  t = mu_dst - s * R @ mu_src
  return R, t, s


def _gravity_yaw_fit(
  src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
  """Closed-form: solve yaw via 2D Procrustes on (x,y), then translation in 3D."""
  src_xy = src[:, :2]
  dst_xy = dst[:, :2]
  mu_src = src_xy.mean(axis=0)
  mu_dst = dst_xy.mean(axis=0)
  src_c = src_xy - mu_src
  dst_c = dst_xy - mu_dst
  cov = dst_c.T @ src_c  # 2x2
  U, _, Vt = np.linalg.svd(cov)
  S = np.eye(2)
  if np.linalg.det(U) * np.linalg.det(Vt) < 0:
    S[1, 1] = -1.0
  R2 = U @ S @ Vt
  yaw = float(np.arctan2(R2[1, 0], R2[0, 0]))
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  src_mean = src.mean(axis=0)
  dst_mean = dst.mean(axis=0)
  t = dst_mean - R @ src_mean
  return R, t, 1.0
