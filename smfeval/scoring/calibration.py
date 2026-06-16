r"""Calibration diagnostics.

- Probability Integral Transform (PIT)
- Kolmogorov–Smirnov (KS) test on the PIT
- Mahalanobis coverage of the translation residual under the predictive Σ_t
  (proper ellipsoidal credible region, χ²_3 threshold)
- standardized residuals (Mahalanobis form)
- TODO(Ola): add HDI?

Orthogonal to scoring rules: scoring rules (e.g. CRPS, log score) reward
sharpness conditional on the reference landing in support, while calibration
tests the *shape* of the predictive CDF independent of sharpness. A sharp
but overconfident predictor scores well on CRPS yet fails calibration; a
wide but well-shaped one is the reverse.

PIT here uses the translation-magnitude scalarisation of the residual,
:math:`\lVert t - \mu_t\rVert`. The PIT value is
:math:`p = F_\mathrm{pred}(y_\mathrm{obs})`, computed empirically from
predictive samples; under a well-calibrated belief :math:`p \sim U(0, 1)`
(Dawid, 1984; Diebold, Gunther & Tay, 1998). Orientation is not scored
(proper scores on SO(3) carry intractable normalisers; see the paper).

References:
-----------
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

from smfeval.format import TangentOrder
from smfeval.scoring._predictive import translation_samples
from smfeval.se3.lie import pose_matrix, relative, se3_log, trans_slice
from smfeval.steps import EnsembleStep, GaussianStep, Step

# Empirical-CDF resolution for PIT is 1/N; 256 keeps it well below KS sensitivity
# at the trial counts we evaluate. Bump if PIT histograms look quantised.
_DEFAULT_N_SAMPLES = 256


@dataclass
class CalibrationResult:
  pit_translation: np.ndarray
  ks_p_translation: float
  coverage: float
  nominal_coverage: float
  z_translation_mean: float
  z_translation_std: float


def _whitened_translation_residual(
  step: Step, gt_t: np.ndarray, gt_q: np.ndarray, order: TangentOrder
) -> np.ndarray | None:
  r"""Cholesky-whitened translation residual.

  Returns :math:`z = L^{-1}\rho` where :math:`\Sigma_t = L L^\top` is the
  predictive translation covariance.

  Under Gaussian predictives :math:`\Sigma_t` is read from the step; for
  ensembles the sample translation covariance plays its role (the Gaussian fit
  to the support — exact under Gaussian ensembles, defensible otherwise).
  From the same ``z`` callers derive both the squared Mahalanobis distance
  :math:`z^\top z \sim \chi^2_3` (used for the ellipsoidal coverage check) and
  the dim-normalised z-score :math:`\lVert z\rVert / \sqrt{d}` (≈1 under H0).

  Returns ``None`` for deterministic predictives, undersized ensembles, or
  singular :math:`\Sigma_t`.
  """
  match step:
    case GaussianStep():
      T_mean = pose_matrix(step.translation, step.quat_xyzw)
      T_obs = pose_matrix(gt_t, gt_q)
      xi = se3_log(relative(T_mean, T_obs), order=order)
      ti = trans_slice(order)
      rho = xi[ti]
      cov_t = step.covariance[ti, ti]
    case EnsembleStep() if step.particles.shape[0] >= 4:
      positions = step.particles[:, :3]
      mu = positions.mean(axis=0)
      rho = gt_t - mu
      cov_t = np.cov(positions, rowvar=False)
    case _:
      return None
  cov_t = (cov_t + cov_t.T) / 2
  try:
    L = np.linalg.cholesky(cov_t + 1e-12 * np.eye(rho.size))
  except np.linalg.LinAlgError:
    return None
  return np.linalg.solve(L, rho)


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
  z_t: list[float] = []
  inside: list[bool] = []
  chi2_threshold = float(chi2.ppf(1.0 - alpha, df=3))
  for step, gt_t, gt_q in zip(
    pred_steps, gt_translations, gt_quats, strict=True
  ):
    t_samples, mu_t = translation_samples(step, n_samples, rng, tangent_order)
    sm = np.linalg.norm(t_samples - mu_t, axis=1)
    om = float(np.linalg.norm(gt_t - mu_t))
    pit_t.append(_pit(sm, om))

    z = _whitened_translation_residual(step, gt_t, gt_q, tangent_order)
    if z is None:
      z_t.append(float("nan"))
    else:
      inside.append(float(z @ z) <= chi2_threshold)
      # z-score reported for Gaussian only (chi-distribution interpretation).
      z_t.append(
        float(np.linalg.norm(z) / np.sqrt(z.size))
        if isinstance(step, GaussianStep)
        else float("nan")
      )

  pit_t_arr = np.array(pit_t)
  z_t_arr = np.array([z for z in z_t if not np.isnan(z)])

  ks_t = (
    float(kstest(pit_t_arr, "uniform").pvalue)
    if pit_t_arr.size
    else float("nan")
  )
  cov = float(np.mean(inside)) if inside else float("nan")
  z_mean = float(z_t_arr.mean()) if z_t_arr.size else float("nan")
  z_std = float(z_t_arr.std(ddof=1)) if z_t_arr.size > 1 else float("nan")

  return CalibrationResult(
    pit_translation=pit_t_arr,
    ks_p_translation=ks_t,
    coverage=cov,
    nominal_coverage=1.0 - alpha,
    z_translation_mean=z_mean,
    z_translation_std=z_std,
  )
