r"""Prequential score-series summary statistics with a stationary block
bootstrap (Politis & Romano, 1994) on the mean, calibrated by an
automatic block-length estimator (Politis & White, 2004).

A SLAM scoring pipeline produces a *prequential* score series
:math:`s_1, \ldots, s_N` obtained by scoring each one-step-ahead
predictive against the realised pose (Dawid, 1984). The series is
almost always autocorrelated — drift accumulates, regime changes
(loop closure, degeneracy exit) persist for many frames — so the iid
percentile bootstrap on the mean (Efron, 1979) under-covers, and the
plain sample mean has an effective sample size below :math:`N`. We
therefore replace the iid bootstrap with the stationary bootstrap of
Politis & Romano (1994), whose mean block length :math:`\ell` is
selected automatically from the data via the flat-top lag-window
estimator of Politis & White (2004).

Reporting :math:`\ell` alongside the CI is itself a diagnostic: a
jump in :math:`\ell` across windows of the trajectory is a signal that
the score series' effective sample rate just dropped (often the
filter just entered a degenerate regime). The TUM-style point
summaries (mean / median / std / min / max) are kept (Sturm et al.,
2012); only the CI aggregation changes.

References
----------
Dawid, A. P. (1984). *Statistical theory: The prequential approach*.
JRSS A 147(2), 278–292.

Efron, B. (1979). *Bootstrap methods: another look at the jackknife*.
Annals of Statistics 7(1), 1–26.

Politis, D. N. & Romano, J. P. (1994). *The stationary bootstrap*.
JASA 89(428), 1303–1313.

Politis, D. N. & White, H. (2004). *Automatic block-length selection
for the dependent bootstrap*. Econometric Reviews 23(1), 53–70.

Sturm, J., Engelhard, N., Endres, F., Burgard, W. & Cremers, D. (2012).
*A benchmark for the evaluation of RGB-D SLAM systems*. IROS.
"""

from dataclasses import asdict, dataclass

import numpy as np

_DEFAULT_N_BOOTSTRAP = 2000
_DEFAULT_CI_LEVEL = 0.95


@dataclass
class ScoreSummary:
    n: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    ci_low: float
    ci_high: float
    ci_level: float
    block_length: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _empty_summary(ci_level: float) -> ScoreSummary:
    nan = float("nan")
    return ScoreSummary(
        n=0, mean=nan, median=nan, std=nan, min=nan, max=nan,
        ci_low=nan, ci_high=nan, ci_level=ci_level, block_length=nan,
    )


def _autocov(x: np.ndarray, max_lag: int) -> np.ndarray:
    r"""Biased sample autocovariance :math:`\hat R(k) = N^{-1}\sum_{t} x_t x_{t+k}`
    for :math:`k = 0, \ldots, M`. ``x`` is assumed already centred."""
    n = x.size
    R = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        R[k] = float(np.dot(x[: n - k], x[k:])) / n
    return R


def _flat_top_window(z: np.ndarray) -> np.ndarray:
    r"""Flat-top lag window :math:`\lambda(z) = 1` for :math:`|z| \le 1/2`,
    :math:`2(1 - |z|)` for :math:`1/2 < |z| \le 1`, else 0 (Politis & White,
    2004, eq. 4)."""
    az = np.abs(z)
    out = np.zeros_like(az, dtype=float)
    out[az <= 0.5] = 1.0
    taper = (az > 0.5) & (az <= 1.0)
    out[taper] = 2.0 * (1.0 - az[taper])
    return out


def politis_white_block_length(values: np.ndarray | list[float]) -> float:
    r"""Politis & White (2004) optimal mean block length for the stationary
    bootstrap.

    Algorithm (PW §3.2):

    1. Locate the smallest lag :math:`k^\star` such that
       :math:`|\hat\rho(k)| < 2\sqrt{\log_{10} N / N}` for
       :math:`K_N = \max(5, \lceil\sqrt{\log_{10} N}\rceil)` consecutive
       lags starting at :math:`k^\star`.
    2. Set the lag-window cutoff :math:`M = 2 k^\star`.
    3. With the flat-top kernel :math:`\lambda` of eq. (4), estimate

       .. math::

          \hat G = \sum_{k = -M}^{M} \lambda(k/M)\,|k|\,\hat R(k),
          \qquad
          \hat\sigma_\infty^{2} = \sum_{k = -M}^{M} \lambda(k/M)\,\hat R(k).

    4. The stationary-bootstrap optimum (PW eq. 9, with
       :math:`\hat D_{SB} = 2\,\hat\sigma_\infty^{4}`) is

       .. math::

          \hat\ell_{SB} = \Bigl(\frac{2\,\hat G^{2}}{\hat D_{SB}}\Bigr)^{1/3}
                         N^{1/3}.

    Returns 1.0 for series too short or with no usable autocovariance
    signal (constant series, zero variance, indefinite spectral
    estimate). Clamped to :math:`[1, N]`.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 8:
        return 1.0
    scale = float(max(np.max(np.abs(x)), 1.0))
    if float(np.ptp(x)) <= 1e-12 * scale:
        return 1.0
    x = x - x.mean()
    var0 = float(np.dot(x, x) / n)
    if var0 <= 0.0:
        return 1.0

    log10n = np.log10(max(n, 2))
    K_n = max(5, int(np.ceil(np.sqrt(log10n))))
    M_max = max(K_n + 1, min(int(np.ceil(np.sqrt(n))) + K_n, n - 1))
    R = _autocov(x, M_max)
    rho = R / R[0]
    threshold = 2.0 * np.sqrt(log10n / n)

    k_star = 0
    for k in range(1, M_max - K_n + 2):
        if np.all(np.abs(rho[k : k + K_n]) < threshold):
            k_star = k
            break
    if k_star == 0:
        k_star = max(1, M_max // 2)

    M = int(min(max(2 * k_star, 2), n - 1))
    ks = np.arange(-M, M + 1)
    R_k = R[np.abs(ks)]
    w = _flat_top_window(ks / M)
    G_hat = float(np.sum(w * np.abs(ks) * R_k))
    sigma_inf_sq = float(np.sum(w * R_k))

    if sigma_inf_sq <= 0.0 or G_hat <= 0.0:
        return 1.0
    D_SB = 2.0 * sigma_inf_sq**2
    b = (2.0 * G_hat**2 / D_SB) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, float(n)))


def _stationary_bootstrap_indices(
    n: int, block_length: float, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    r"""Index draws for the stationary bootstrap of Politis & Romano (1994).

    Each replicate is built by picking a uniform start index, then at each
    subsequent step either continuing one position forward (wrapping mod
    :math:`N`) with probability :math:`1 - 1/\ell` or restarting from a
    fresh uniform index with probability :math:`1/\ell`. Block lengths are
    geometric with mean :math:`\ell`, which makes the bootstrap series
    stationary (whereas the moving-block bootstrap is not).
    """
    p = 1.0 / max(block_length, 1.0)
    starts = rng.integers(0, n, size=(n_resamples, n))
    restart = rng.random(size=(n_resamples, n)) < p
    restart[:, 0] = True

    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def summarize(
    values: np.ndarray | list[float],
    *,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> ScoreSummary:
    r"""TUM-style summary statistics with a stationary-bootstrap CI on the
    mean.

    Parameters
    ----------
    values
        Prequential per-timestep score series.
    n_bootstrap
        Number of bootstrap resamples for the percentile CI on the mean.
    ci_level
        Confidence level for the percentile interval (e.g. ``0.95``).
    block_length
        Mean geometric block length for the stationary bootstrap. When
        ``None`` (the default) it is estimated from the data with
        :func:`politis_white_block_length`. Pass ``1.0`` to recover the
        iid percentile bootstrap.
    rng
        Random generator for the resampling.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return _empty_summary(ci_level)

    rng = rng if rng is not None else np.random.default_rng(0)
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    smin = float(arr.min())
    smax = float(arr.max())

    if block_length is None:
        ell = politis_white_block_length(arr)
    else:
        ell = float(block_length)
    ell = float(np.clip(ell, 1.0, float(n)))

    if n == 1 or n_bootstrap <= 0:
        ci_low, ci_high = mean, mean
    else:
        idx = _stationary_bootstrap_indices(n, ell, n_bootstrap, rng)
        boot_means = arr[idx].mean(axis=1)
        alpha = 1.0 - ci_level
        ci_low = float(np.quantile(boot_means, alpha / 2.0))
        ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    return ScoreSummary(
        n=n, mean=mean, median=median, std=std, min=smin, max=smax,
        ci_low=ci_low, ci_high=ci_high, ci_level=ci_level,
        block_length=ell,
    )
