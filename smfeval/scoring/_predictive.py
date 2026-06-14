"""Shared predictive samplers used by CRPS, energy score, calibration, and the CLI.

Returns sample translations as (n, 3) and rotation perturbations as (n, 3) tangent
vectors plus the reference rotation matrix; downstream rules pick which scalar
to compute. Single source of truth for predictive sample generation.
"""

import numpy as np

from smfeval.format import TangentOrder
from smfeval.scoring._kernel import sample_gaussian_tangent
from smfeval.se3.lie import rot_slice, so3_log, so3_mean, trans_slice
from smfeval.se3.quat import quat_xyzw_to_rot
from smfeval.steps import EnsembleStep, GaussianStep, Step


def translation_samples(
  step: Step, n: int, rng: np.random.Generator, order: TangentOrder
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (samples, mu_t)."""
  if isinstance(step, GaussianStep):
    cov_t = step.covariance[trans_slice(order), trans_slice(order)]
    s = sample_gaussian_tangent(step.translation, cov_t, n, rng)
    return s, step.translation
  if isinstance(step, EnsembleStep):
    s = step.particles[:, :3]
    return s, s.mean(axis=0)
  return step.translation[None, :], step.translation


def rotation_samples(
  step: Step, n: int, rng: np.random.Generator, order: TangentOrder
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (omegas, R_mean) where omegas are tangent perturbations of R_mean."""
  if isinstance(step, GaussianStep):
    cov_w = step.covariance[rot_slice(order), rot_slice(order)]
    omegas = sample_gaussian_tangent(np.zeros(3), cov_w, n, rng)
    return omegas, quat_xyzw_to_rot(step.quat_xyzw)
  if isinstance(step, EnsembleStep):
    rots = np.array([quat_xyzw_to_rot(p[3:]) for p in step.particles])
    if len(rots) == 0:
      return np.zeros((0, 3)), np.eye(3)
    R_mean = so3_mean(rots)
    omegas = np.array([so3_log(R_mean.T @ R) for R in rots])
    return omegas, R_mean
  return np.zeros((1, 3)), quat_xyzw_to_rot(step.quat_xyzw)
