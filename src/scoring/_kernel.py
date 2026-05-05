"""Sample-based estimators for kernel scores.

Uniform code path for ensembles (native) and Gaussians (sampled): both end up
as point clouds, then the same estimator runs.
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
    """Energy score: E‖X-y‖ - 0.5·E‖X-X'‖ (unbiased estimator).

    samples: (m, d), observation: (d,). Uses the standard plug-in estimator
    Σ‖x_i-y‖/m - Σ_{i<j}‖x_i-x_j‖ / (m(m-1)).
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
    """Univariate CRPS via the same identity: E|X-y| - 0.5·E|X-X'|."""
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
