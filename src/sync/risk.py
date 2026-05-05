"""Sync risk: per-pair `v · Δt / σ` quantifies how much sync gap could explain
the residual. Big values mean the gap could plausibly account for the position
error and miscalibration findings should be tempered."""

import numpy as np
from scipy.special import logsumexp

from src.se3.lie import trans_slice
from src.steps import EnsembleStep, GaussianStep
from src.types import TangentOrder


def _gt_velocity(gt_ts: np.ndarray, gt_pos: np.ndarray) -> np.ndarray:
    n = len(gt_ts)
    v = np.zeros_like(gt_pos)
    if n < 2:
        return v
    v[1:-1] = (gt_pos[2:] - gt_pos[:-2]) / (gt_ts[2:] - gt_ts[:-2])[:, None]
    v[0] = (gt_pos[1] - gt_pos[0]) / (gt_ts[1] - gt_ts[0])
    v[-1] = (gt_pos[-1] - gt_pos[-2]) / (gt_ts[-1] - gt_ts[-2])
    return v


def _ensemble_weighted_mean_var(
    positions: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (weights < 0).any():
        log_w = weights - logsumexp(weights)
        w = np.exp(log_w)
    else:
        s = float(weights.sum())
        w = weights / s if s > 0 else np.full_like(weights, 1.0 / len(weights))
    mean = w @ positions
    diff = positions - mean
    var = (w[:, None] * diff * diff).sum(axis=0)
    return mean, var


def _trans_sigma(step, order: TangentOrder | None) -> float:
    """Predictive translation 1-sigma in metres."""
    if isinstance(step, GaussianStep):
        ti = trans_slice(order or TangentOrder.TRANS_ROT)
        cov_t = step.covariance[ti, ti]
        return float(np.sqrt(max(np.trace(cov_t) / 3.0, 0.0)))
    if isinstance(step, EnsembleStep):
        positions = step.particles[:, :3]
        if step.weights is not None:
            _, var = _ensemble_weighted_mean_var(positions, step.weights)
        else:
            var = positions.var(axis=0)
        return float(np.sqrt(max(var.mean(), 0.0)))
    return 0.0


def sync_risk(
    est_steps: list,
    gt_ts: np.ndarray,
    gt_positions: np.ndarray,
    est_indices: np.ndarray,
    gt_indices: np.ndarray,
    est_ts: np.ndarray,
    t_offset: float = 0.0,
    tangent_order: TangentOrder | None = None,
) -> np.ndarray:
    """Returns the per-pair sync risk r = ||v_gt|| · |Δt| / σ_trans."""
    velocities = _gt_velocity(gt_ts, gt_positions)
    speeds = np.linalg.norm(velocities, axis=1)
    out = np.zeros(len(est_indices))
    for k, (ei, gi) in enumerate(zip(est_indices, gt_indices)):
        sigma = _trans_sigma(est_steps[ei], tangent_order)
        if sigma <= 0:
            out[k] = np.inf if speeds[gi] > 0 else 0.0
            continue
        dt = abs((est_ts[ei] + t_offset) - gt_ts[gi])
        out[k] = float(speeds[gi] * dt / sigma)
    return out
