r"""Sync risk per match pair.

:math:`v \cdot \Delta t / \sigma` quantifies how much of the residual a
sync gap could plausibly explain. Big values mean the gap could account
for the position error, so miscalibration findings should be tempered.
"""

import numpy as np

from smfeval.format import TangentOrder, WeightFormat
from smfeval.se3.lie import trans_slice
from smfeval.steps import (
  DeterministicStep,
  EnsembleStep,
  GaussianStep,
  Step,
  log_normalized_weights,
)

# Risk above which a matched pair is counted as excess in the report; the
# single source for the report builder, renderer, and diagnosis layer.
DEFAULT_SYNC_RISK_THRESHOLD = 0.3


def _ref_velocity(ref_ts: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
  n = len(ref_ts)
  v = np.zeros_like(ref_pos)
  if n < 2:
    return v
  v[1:-1] = (ref_pos[2:] - ref_pos[:-2]) / (ref_ts[2:] - ref_ts[:-2])[:, None]
  v[0] = (ref_pos[1] - ref_pos[0]) / (ref_ts[1] - ref_ts[0])
  v[-1] = (ref_pos[-1] - ref_pos[-2]) / (ref_ts[-1] - ref_ts[-2])
  return v


def _ensemble_weighted_mean_var(
  positions: np.ndarray,
  weights: np.ndarray,
  weight_format: WeightFormat,
  normalized: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Weighted mean/variance under the header's declared weight format."""
  w = np.exp(log_normalized_weights(weights, weight_format, normalized))
  mean = w @ positions
  diff = positions - mean
  var = (w[:, None] * diff * diff).sum(axis=0)
  return mean, var


def _trans_sigma(
  step: Step,
  order: TangentOrder | None,
  weight_format: WeightFormat,
  weights_normalized: bool,
) -> float:
  """Predictive translation 1-sigma in metres, or nan when there is none."""
  match step:
    case GaussianStep():
      ti = trans_slice(order or TangentOrder.TRANS_ROT)
      cov_t = step.covariance[ti, ti]
      return float(np.sqrt(max(np.trace(cov_t) / 3.0, 0.0)))
    case EnsembleStep():
      positions = step.particles[:, :3]
      if step.weights is not None:
        _, var = _ensemble_weighted_mean_var(
          positions, step.weights, weight_format, weights_normalized
        )
      else:
        var = positions.var(axis=0)
      return float(np.sqrt(max(var.mean(), 0.0)))
    case DeterministicStep():
      return float("nan")


def sync_risk(
  est_steps: list,
  ref_ts: np.ndarray,
  ref_positions: np.ndarray,
  est_indices: np.ndarray,
  ref_indices: np.ndarray,
  est_ts: np.ndarray,
  t_offset: float = 0.0,
  tangent_order: TangentOrder | None = None,
  weight_format: WeightFormat = WeightFormat.LINEAR,
  weights_normalized: bool = True,
) -> np.ndarray:
  r"""Per-pair sync risk :math:`\lVert v_\mathrm{ref}\rVert \cdot |\Delta t| / \sigma_\mathrm{trans}`.

  ``nan`` where the pair has no usable :math:`\sigma_\mathrm{trans}` — a
  deterministic step, or a degenerate zero covariance. The ratio is undefined
  without a predictive spread to measure the gap against, so those pairs are
  excluded from the report rather than counted as infinitely risky.
  """
  velocities = _ref_velocity(ref_ts, ref_positions)
  speeds = np.linalg.norm(velocities, axis=1)
  out = np.full(len(est_indices), np.nan)
  for k, (ei, gi) in enumerate(zip(est_indices, ref_indices, strict=True)):
    sigma = _trans_sigma(
      est_steps[ei], tangent_order, weight_format, weights_normalized
    )
    if not np.isfinite(sigma) or sigma <= 0:
      continue
    dt = abs((est_ts[ei] + t_offset) - ref_ts[gi])
    out[k] = float(speeds[gi] * dt / sigma)
  return out
