r"""Calibration diagnostics.

- Mahalanobis coverage of the translation residual under the predictive
  :math:`\Sigma_t` (proper ellipsoidal credible region, :math:`\chi^2_3`
  threshold), with an exact binomial test of the realised hit rate against
  the nominal level
- standardized residuals (Mahalanobis form)

Orthogonal to scoring rules: scoring rules (e.g. CRPS, log score) reward
sharpness conditional on the reference landing in support, while calibration
tests whether the stated uncertainty is the right *size* independent of
sharpness. A sharp but overconfident predictor scores well on CRPS yet fails
calibration; a wide but well-shaped one is the reverse.

Coverage is the fraction of poses whose reference lands inside the nominal
:math:`1-\alpha` credible ellipsoid. Under a calibrated belief each pose is an
independent Bernoulli trial with success probability :math:`1-\alpha`, so the
hit count is Binomial and ``coverage_p`` is the two-sided exact binomial
p-value against that null (Clopper & Pearson, 1934) — a distribution-free
check on the size of the stated uncertainty, over the quantity the report
already displays. Orientation is not scored (proper scores on SO(3) carry
intractable normalisers; the full argument is in a forthcoming paper).

References:
-----------
Clopper, C. J. & Pearson, E. S. (1934). *The use of confidence or fiducial
limits illustrated in the case of the binomial*. Biometrika 26(4), 404-413.

Mahalanobis, P. C. (1936). *On the generalised distance in statistics*.
Proc. National Institute of Sciences of India 2(1), 49-55.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, chi2

from smfeval.format import TangentConvention, TangentOrder
from smfeval.se3.lie import pose_residual, trans_slice
from smfeval.steps import EnsembleStep, GaussianStep, Step


@dataclass
class CalibrationResult:
  coverage: float
  nominal_coverage: float
  n_coverage: int  # poses with a usable Sigma_t (the binomial n)
  coverage_p: float  # two-sided exact binomial p vs the nominal level
  z_translation_mean: float
  z_translation_std: float


def _whitened_translation_residual(
  step: Step,
  ref_t: np.ndarray,
  ref_q: np.ndarray,
  order: TangentOrder,
  convention: TangentConvention = TangentConvention.RIGHT,
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
      xi = pose_residual(
        step.translation, step.quat_xyzw, ref_t, ref_q, order, convention
      )
      ti = trans_slice(order)
      rho = xi[ti]
      cov_t = step.covariance[ti, ti]
    case EnsembleStep() if step.particles.shape[0] >= 4:
      positions = step.particles[:, :3]
      mu = positions.mean(axis=0)
      rho = ref_t - mu
      cov_t = np.cov(positions, rowvar=False)
    case _:
      return None
  cov_t = (cov_t + cov_t.T) / 2
  try:
    L = np.linalg.cholesky(cov_t + 1e-12 * np.eye(rho.size))
  except np.linalg.LinAlgError:
    return None
  return np.linalg.solve(L, rho)


def _coverage_p(n_inside: int, n: int, nominal: float) -> float:
  """Two-sided exact binomial p for ``n_inside`` hits out of ``n`` at ``nominal``."""
  if n == 0:
    return float("nan")
  return float(binomtest(n_inside, n, nominal).pvalue)


def calibrate(
  pred_steps: list[Step],
  ref_translations: np.ndarray,
  ref_quats: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  tangent_convention: TangentConvention = TangentConvention.RIGHT,
  alpha: float = 0.1,
) -> CalibrationResult:
  r"""Ellipsoidal coverage and standardized residuals over matched poses.

  Poses whose predictive :math:`\Sigma_t` is unusable (deterministic steps,
  undersized ensembles, singular covariance) contribute to neither the
  coverage nor the z-score.
  """
  z_t: list[float] = []
  inside: list[bool] = []
  chi2_threshold = float(chi2.ppf(1.0 - alpha, df=3))
  for step, ref_t, ref_q in zip(
    pred_steps, ref_translations, ref_quats, strict=True
  ):
    z = _whitened_translation_residual(
      step, ref_t, ref_q, tangent_order, tangent_convention
    )
    if z is None:
      continue
    inside.append(float(z @ z) <= chi2_threshold)
    # z-score reported for Gaussian only (chi-distribution interpretation).
    if isinstance(step, GaussianStep):
      z_t.append(float(np.linalg.norm(z) / np.sqrt(z.size)))

  z_t_arr = np.array(z_t)
  n_cov = len(inside)
  nominal = 1.0 - alpha
  cov = float(np.mean(inside)) if inside else float("nan")
  z_mean = float(z_t_arr.mean()) if z_t_arr.size else float("nan")
  z_std = float(z_t_arr.std(ddof=1)) if z_t_arr.size > 1 else float("nan")

  return CalibrationResult(
    coverage=cov,
    nominal_coverage=nominal,
    n_coverage=n_cov,
    coverage_p=_coverage_p(int(sum(inside)), n_cov, nominal),
    z_translation_mean=z_mean,
    z_translation_std=z_std,
  )
