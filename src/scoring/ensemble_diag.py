"""Ensemble degeneracy diagnostics: N_eff and unique-particle counts."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from src.steps import EnsembleStep
from src.types import WeightFormat


@dataclass
class EnsembleDiagnostic:
    n_nominal: int  # max ensemble size across timesteps
    n_eff: np.ndarray  # per-timestep effective sample size
    n_unique: np.ndarray  # per-timestep unique-particle count
    degeneracy_fraction: float  # fraction of timesteps with N_eff < N/10


def _log_normalized(weights: np.ndarray, fmt: WeightFormat, normalized: bool) -> np.ndarray:
    """Return log-normalized weights regardless of input form."""
    if fmt is WeightFormat.LOG:
        log_w = weights.astype(float)
    else:
        with np.errstate(divide="ignore"):
            log_w = np.where(weights > 0, np.log(np.maximum(weights, 1e-300)), -np.inf)
    if not normalized or fmt is WeightFormat.LOG:
        log_w = log_w - logsumexp(log_w)
    return log_w


def _n_eff_from_log_weights(log_w: np.ndarray) -> float:
    """N_eff = (Σw)² / Σw² with Σw = 1 ⇒ 1 / Σw² = exp(-logsumexp(2·log_w))."""
    if log_w.size == 0:
        return 0.0
    return float(np.exp(-logsumexp(2.0 * log_w)))


def _unique_count(particles: np.ndarray, tol: float) -> int:
    """Count distinct rows under tolerance-quantized hashing."""
    if particles.size == 0:
        return 0
    quantized = np.round(particles / tol).astype(np.int64)
    return len({tuple(row) for row in quantized})


def ensemble_diagnostics(
    steps: list[EnsembleStep],
    weight_format: WeightFormat,
    normalized: bool,
    tol: float = 1e-6,
) -> EnsembleDiagnostic:
    n_eff = np.zeros(len(steps))
    n_unique = np.zeros(len(steps), dtype=int)
    n_max = 0
    for k, step in enumerate(steps):
        n = step.particles.shape[0]
        n_max = max(n_max, n)
        if step.weights is None:
            n_eff[k] = float(n)
        else:
            log_w = _log_normalized(step.weights, weight_format, normalized)
            n_eff[k] = _n_eff_from_log_weights(log_w)
        n_unique[k] = _unique_count(step.particles, tol)
    if n_max == 0:
        deg = 0.0
    else:
        threshold = n_max / 10.0
        deg = float((n_eff < threshold).mean()) if n_eff.size else 0.0
    return EnsembleDiagnostic(
        n_nominal=n_max,
        n_eff=n_eff,
        n_unique=n_unique,
        degeneracy_fraction=deg,
    )
