r"""Calibration:

- Probability Integral Transform (PIT)
- Kolmogorov–Smirnov (KS) test on the PIT
- Mahalanobis coverage of the translation residual under the predictive Σ_t
  (proper ellipsoidal credible region, χ²_3 threshold)
- standardized residuals (Mahalanobis form)
- TODO(Ola): add HDI?

Orthogonal to scoring rules: scoring rules (e.g. CRPS, log score) reward
sharpness conditional on the truth landing in support, while calibration
tests the *shape* of the predictive CDF independent of sharpness. A sharp
but overconfident predictor scores well on CRPS yet fails calibration; a
wide but well-shaped one is the reverse.

PIT here uses two univariate scalarisations of the SE(3) residual:

- translation magnitude :math:`\lVert t - \mu_t\rVert`
- rotation angle :math:`d_\mathrm{geo}(R, R_\mu)`

For each scalarisation the PIT value is
:math:`p = F_\mathrm{pred}(y_\mathrm{obs})`, computed empirically from
predictive samples; under a well-calibrated belief :math:`p \sim U(0, 1)`
(Dawid, 1984; Diebold, Gunther & Tay, 1998).

References
----------
Dawid, A. P. (1984). *Statistical theory: The prequential approach*.
JRSS A 147(2), 278–292.

Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). *Evaluating density
forecasts with applications to financial risk management*. International
Economic Review 39(4), 863–883.

Kolmogorov, A. N. (1933). *Sulla determinazione empirica di una legge di
distribuzione*. Giornale dell'Istituto Italiano degli Attuari 4, 83–91.

Smirnov, N. V. (1948). *Table for estimating the goodness of fit of
empirical distributions*. Annals of Mathematical Statistics 19(2),
279–281.

Mahalanobis, P. C. (1936). *On the generalised distance in statistics*.
Proc. National Institute of Sciences of India 2(1), 49–55.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, kstest

from src.scoring._predictive import rotation_samples, translation_samples
from src.se3.lie import pose_matrix, relative, se3_log, so3_log, trans_slice
from src.se3.quat import quat_xyzw_to_rot
from src.steps import EnsembleStep, GaussianStep, Step
from src.types import TangentOrder

# Empirical-CDF resolution for PIT is 1/N; 256 keeps it well below KS sensitivity
# at the trial counts we evaluate. Bump if PIT histograms look quantised.
_DEFAULT_N_SAMPLES = 256


@dataclass
class CalibrationResult:
    pit_translation: np.ndarray
    pit_rotation: np.ndarray
    ks_p_translation: float
    ks_p_rotation: float
    coverage: float
    nominal_coverage: float
    z_translation_mean: float
    z_translation_std: float


def _translation_mahalanobis_sq(
    step: Step, gt_t: np.ndarray, gt_q: np.ndarray, order: TangentOrder
) -> float:
    r"""Squared Mahalanobis distance of the translation residual under the
    predictive covariance.

    Under a Gaussian predictive, :math:`d^2 = \rho^\top \Sigma_t^{-1} \rho`
    is :math:`\chi^2_3`-distributed, and the proper :math:`(1-\alpha)` credible
    region is :math:`\{x : d^2(x) \le \chi^2_{3, 1-\alpha}\}` — the ellipsoidal
    ball whose orientation and aspect track the off-diagonal entries of
    :math:`\Sigma_t`. For ensembles the predictive may not be Gaussian; the
    sample translation covariance plays the role of :math:`\Sigma_t`, which is
    the Gaussian fit to the support of the particles (exact under Gaussian
    ensembles, defensible otherwise).

    Returns ``nan`` for deterministic predictives or when :math:`\Sigma_t` is
    singular.
    """
    if isinstance(step, GaussianStep):
        T_mean = pose_matrix(step.translation, step.quat_xyzw)
        T_obs = pose_matrix(gt_t, gt_q)
        xi = se3_log(relative(T_mean, T_obs), order=order)
        ti = trans_slice(order)
        rho = xi[ti]
        cov_t = step.covariance[ti, ti]
    elif isinstance(step, EnsembleStep):
        positions = step.particles[:, :3]
        if positions.shape[0] < 4:
            return float("nan")
        mu = positions.mean(axis=0)
        rho = gt_t - mu
        cov_t = np.cov(positions, rowvar=False)
    else:
        return float("nan")
    cov_t = (cov_t + cov_t.T) / 2
    try:
        L = np.linalg.cholesky(cov_t + 1e-12 * np.eye(3))
    except np.linalg.LinAlgError:
        return float("nan")
    z = np.linalg.solve(L, rho)
    return float(z @ z)


def _z_translation(step: Step, gt_t: np.ndarray, gt_q: np.ndarray, order: TangentOrder) -> float:
    r"""Mahalanobis-normalised translation residual (Mahalanobis, 1936):

    .. math::

       z = \frac{\lVert L^{-1}\rho\rVert}{\sqrt{d}},
       \qquad \Sigma_t = L L^\top,\ \rho = (T_\mathrm{mean}^{-1} T_\mathrm{obs})_{\mathrm{trans}},

    with :math:`d = \dim\rho`. Gaussian only; NaN otherwise.
    """
    if not isinstance(step, GaussianStep):
        return float("nan")
    T_mean = pose_matrix(step.translation, step.quat_xyzw)
    T_obs = pose_matrix(gt_t, gt_q)
    xi = se3_log(relative(T_mean, T_obs), order=order)
    ti = trans_slice(order)
    rho = xi[ti]
    cov_t = step.covariance[ti, ti]
    cov_t = (cov_t + cov_t.T) / 2  # ensure symmetric
    try:
        L = np.linalg.cholesky(cov_t + 1e-12 * np.eye(3))
    except np.linalg.LinAlgError:
        return float("nan")
    z = np.linalg.solve(L, rho)
    # ‖z‖ is chi-distributed with rho.size dof, so E[‖z‖] ≈ √dof under H0;
    # dividing makes the statistic ≈ 1 when calibrated, regardless of dim.
    return float(np.linalg.norm(z) / np.sqrt(rho.size))


def _pit(samples: np.ndarray, observation: float) -> float:
    if samples.size == 0:
        return float("nan")
    return float((samples <= observation).mean())


def calibrate(
    pred_steps: list[Step],
    gt_translations: np.ndarray,
    gt_quats: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
    alpha: float = 0.1,
    n_samples: int = _DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> CalibrationResult:
    rng = rng if rng is not None else np.random.default_rng(0)
    pit_t: list[float] = []
    pit_r: list[float] = []
    z_t: list[float] = []
    inside: list[bool] = []
    # Proper 90% ball is the Mahalanobis ellipsoid {ρ : ρᵀ Σ_t⁻¹ ρ ≤ χ²_{3,1-α}}.
    chi2_threshold = float(chi2.ppf(1.0 - alpha, df=3))
    for step, gt_t, gt_q in zip(pred_steps, gt_translations, gt_quats):
        t_samples, mu_t = translation_samples(step, n_samples, rng, tangent_order)
        sm = np.linalg.norm(t_samples - mu_t, axis=1)
        om = float(np.linalg.norm(gt_t - mu_t))
        pit_t.append(_pit(sm, om))

        omegas, R_mean = rotation_samples(step, n_samples, rng, tangent_order)
        sa = np.linalg.norm(omegas, axis=1)
        oa = float(np.linalg.norm(so3_log(R_mean.T @ quat_xyzw_to_rot(gt_q))))
        pit_r.append(_pit(sa, oa))

        z_t.append(_z_translation(step, gt_t, gt_q, tangent_order))
        d2 = _translation_mahalanobis_sq(step, gt_t, gt_q, tangent_order)
        if not np.isnan(d2):
            inside.append(d2 <= chi2_threshold)

    pit_t_arr = np.array(pit_t)
    pit_r_arr = np.array(pit_r)
    z_t_arr = np.array([z for z in z_t if not np.isnan(z)])

    ks_t = float(kstest(pit_t_arr, "uniform").pvalue) if pit_t_arr.size else float("nan")
    ks_r = float(kstest(pit_r_arr, "uniform").pvalue) if pit_r_arr.size else float("nan")
    cov = float(np.mean(inside)) if inside else float("nan")
    z_mean = float(z_t_arr.mean()) if z_t_arr.size else float("nan")
    z_std = float(z_t_arr.std(ddof=1)) if z_t_arr.size > 1 else float("nan")

    return CalibrationResult(
        pit_translation=pit_t_arr,
        pit_rotation=pit_r_arr,
        ks_p_translation=ks_t,
        ks_p_rotation=ks_r,
        coverage=cov,
        nominal_coverage=1.0 - alpha,
        z_translation_mean=z_mean,
        z_translation_std=z_std,
    )
