r"""Bias-vs-variance decomposition of the windowed increment error (B1).

The calibration term says *whether* a filter is over-confident; this says *what
kind* of error drives it — a systematic offset (bias → extrinsic / time-offset /
scale / gravity) or random spread (variance → noise) — and *in which track-frame
channel* (along / cross / vertical). Increments are alignment-invariant, so
unlike absolute-pose residuals the bias is not killed by the SE(3) alignment.

For window Δt and pair (i, i+Δt) the increment error is

    e = (p_est(i+Δt) − p_est(i)) − (p_gt(i+Δt) − p_gt(i))

in the per-pair track frame (along = horizontally-projected GT heading, vertical
= world up, cross = up × along). Over the pairs,

    MSE = ‖bias‖² + tr(cov),   bias = mean_t e,

so ``bias_fraction = ‖bias‖²/MSE`` ∈ [0,1]: →1 systematic, →0 random. The
dominant bias axis localises the channel (vertical → gravity/init; along →
scale/time-offset; cross → lateral extrinsic/heading). Mirrors the benchmark's
``scripts/bias_variance_drift.py`` so smfeval's ``diagnose`` can read it.
"""

from dataclasses import asdict, dataclass

import numpy as np

from src.scoring.relative import _window_pairs

_AXES = ("along", "cross", "vertical")


@dataclass
class BiasVarianceResult:
  window_s: float
  n_pairs: int
  mse: float
  bias_fraction: float
  bias: list[float]   # [along, cross, vertical] (m)
  std: list[float]    # [along, cross, vertical] (m)
  dominant_axis: str

  def to_dict(self) -> dict:
    return asdict(self)


def _track_frame_errors(
  est_inc: np.ndarray, gt_inc: np.ndarray, min_horiz_m: float
) -> np.ndarray:
  """Project world-frame increment errors into the per-pair track frame."""
  e = est_inc - gt_inc
  up = np.array([0.0, 0.0, 1.0])
  horiz = gt_inc.copy()
  horiz[:, 2] = 0.0
  hmag = np.linalg.norm(horiz, axis=1)
  keep = hmag > min_horiz_m
  if not keep.any():
    return np.empty((0, 3))
  a = horiz[keep] / hmag[keep, None]
  c = np.cross(np.broadcast_to(up, a.shape), a)
  ek = e[keep]
  return np.column_stack([
    np.einsum("ij,ij->i", ek, a),
    np.einsum("ij,ij->i", ek, c),
    ek[:, 2],
  ])


def bias_variance(
  steps: list,
  gt_translations: np.ndarray,
  *,
  windows_s: list[float],
  tolerance_s: float | None = None,
  min_horiz_m: float = 0.01,
) -> list[BiasVarianceResult]:
  """Windowed track-frame bias/variance per Δt (skips windows with no pairs)."""
  ts = np.array([s.timestamp for s in steps], dtype=float)
  mu = np.array([s.translation for s in steps], dtype=float)
  gt = np.asarray(gt_translations, dtype=float)
  dt_med = float(np.median(np.diff(np.sort(ts)))) if ts.size > 1 else 0.0
  tol = tolerance_s if tolerance_s is not None else 0.5 * dt_med

  out: list[BiasVarianceResult] = []
  for w in windows_s:
    i, j = _window_pairs(ts, w, tol)
    if i.size == 0:
      continue
    ef = _track_frame_errors(mu[j] - mu[i], gt[j] - gt[i], min_horiz_m)
    if ef.shape[0] == 0:
      continue
    bias = ef.mean(axis=0)
    var = ef.var(axis=0)
    mse = float((ef ** 2).sum(axis=1).mean())
    bias_sq = float((bias ** 2).sum())
    out.append(
      BiasVarianceResult(
        window_s=float(w),
        n_pairs=int(ef.shape[0]),
        mse=mse,
        bias_fraction=bias_sq / mse if mse > 0 else float("nan"),
        bias=bias.tolist(),
        std=np.sqrt(var).tolist(),
        dominant_axis=_AXES[int(np.argmax(np.abs(bias)))],
      )
    )
  return out
