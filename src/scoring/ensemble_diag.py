r"""Ensemble degeneracy diagnostics: effective sample size
:math:`N_\mathrm{eff}` and unique-particle counts.

The effective sample size for self-normalised importance / particle
weights was introduced by Kong, Liu & Wong (1994) and popularised in
sequential Monte Carlo by Liu & Chen (1995):

.. math::

   N_\mathrm{eff} = \frac{1}{\sum_{i=1}^{N} w_i^2},
   \qquad \sum_i w_i = 1.

References
----------
Kong, A., Liu, J. S. & Wong, W. H. (1994). *Sequential imputations and
Bayesian missing data problems*. JASA 89(425), 278–288.

Liu, J. S. & Chen, R. (1995). *Blind deconvolution via sequential
imputations*. JASA 90(430), 567–576.
"""

from dataclasses import dataclass
from typing import cast

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
    r""":math:`N_\mathrm{eff} = (\sum w)^2 / \sum w^2`; with :math:`\sum w = 1`
    this reduces to :math:`1 / \sum w^2 = \exp(-\mathrm{logsumexp}(2\log w))`
    (Kong, Liu & Wong, 1994)."""
    if log_w.size == 0:
        return 0.0
    lse = cast(float, logsumexp(2.0 * log_w))
    return float(np.exp(-lse))


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
