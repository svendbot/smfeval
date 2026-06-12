r"""Sample-based unbiased estimators for kernel scores (Gneiting & Raftery, 2007, §5).

Uniform code path for ensembles (native) and Gaussians (sampled): both
end up as point clouds, then the same estimator runs.

References:
-----------
Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

import numpy as np


def sample_gaussian_tangent(
  mean: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
  """Return n samples from N(mean, cov) (no manifold-Exp; caller decides)."""
  L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
  z = rng.standard_normal(size=(n, cov.shape[0]))
  return mean + z @ L.T


def energy_score_estimator(
  samples: np.ndarray, observation: np.ndarray
) -> float:
  r"""Unbiased Monte-Carlo estimator of the energy score.

  :math:`\mathrm{ES}(F, y) = \mathbb{E}\lVert X-y\rVert - \tfrac12\mathbb{E}\lVert X-X'\rVert`.

  For samples :math:`x_1, \ldots, x_m \stackrel{iid}{\sim} F`,

  .. math::

     \widehat{\mathrm{ES}}
     = \frac{1}{m}\sum_{i=1}^{m} \lVert x_i - y\rVert
       - \frac{1}{m(m-1)}\sum_{i \neq j} \lVert x_i - x_j\rVert.

  The :math:`\tfrac{1}{m(m-1)}` weighting (rather than :math:`1/m^2`)
  yields the unbiased U-statistic estimator.

  References:
  -----------
  Gneiting & Raftery (2007), eqs. (21)–(22).
  """
  m = samples.shape[0]
  if m == 0:
    return float("nan")
  diffs = samples - observation
  term1 = float(np.linalg.norm(diffs, axis=1).mean())
  if m == 1:
    return term1
  pairwise = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
  # exclude diagonal
  sum_pairs = pairwise.sum() - np.trace(pairwise)
  term2 = float(sum_pairs / (m * (m - 1)))
  return term1 - 0.5 * term2


def crps_estimator(samples: np.ndarray, observation: float) -> float:
  r"""Unbiased Monte-Carlo estimator of the univariate CRPS.

  :math:`\mathrm{CRPS}(F, y) = \mathbb{E}|X-y| - \tfrac12\mathbb{E}|X-X'|`
  (the 1-D specialisation of the energy score).

  References:
  -----------
  Gneiting & Raftery (2007), eq. (20).
  """
  m = samples.size
  if m == 0:
    return float("nan")
  term1 = float(np.abs(samples - observation).mean())
  if m == 1:
    return term1
  diffs = np.abs(samples[:, None] - samples[None, :])
  sum_pairs = diffs.sum() - np.trace(diffs)
  term2 = float(sum_pairs / (m * (m - 1)))
  return term1 - 0.5 * term2
