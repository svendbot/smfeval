r"""Continuous Ranked Probability Score (CRPS) on the translation marginal.

Translation is scored per-axis. Orientation is not scored: proper scores
on SO(3) need a belief density whose normaliser is intractable for the
natural rotation families, so the tool scores translation only (the full
argument is in a forthcoming paper).

CRPS was introduced by Matheson & Winkler (1976); the kernel-score and
energy-form characterisation we use here is from Gneiting & Raftery
(2007).

References:
-----------
Matheson, J. E. & Winkler, R. L. (1976). *Scoring rules for continuous
probability distributions*. Management Science 22(10), 1087–1096.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

import numpy as np
from scipy.stats import norm

from smfeval.format import TangentOrder
from smfeval.scoring._kernel import crps_estimator
from smfeval.se3.lie import trans_slice
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _gaussian_crps(
  mu: np.ndarray, sigma: np.ndarray, y: np.ndarray
) -> np.ndarray:
  r"""Closed-form per-axis CRPS for :math:`N(\mu, \sigma^2)` at :math:`y`.

  .. math::

     \mathrm{CRPS}(N(\mu,\sigma^2), y)
     = \sigma\left[\omega\bigl(2\Phi(\omega)-1\bigr)
                   + 2\phi(\omega) - 1/\sqrt{\pi}\right],
     \quad \omega = (y-\mu)/\sigma.

  Gneiting & Raftery (2007), closed form following eq. (20).
  """
  z = (y - mu) / sigma
  return sigma * (
    z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI
  )


def translation_crps(
  pred_step: Step,
  ref_translation: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> float:
  r"""Mean per-axis CRPS over the three translation components.

  Gaussian predictive: closed form (Gneiting & Raftery 2007, after eq. 20).
  Ensemble predictive: per-axis U-statistic estimator over the full
  ensemble (the filter sets the sample count; we do not subsample).
  Deterministic: per-axis absolute error.
  """
  match pred_step:
    case GaussianStep():
      sl = trans_slice(tangent_order)
      sigma = np.sqrt(np.diag(pred_step.covariance)[sl])
      return float(
        _gaussian_crps(pred_step.translation, sigma, ref_translation).mean()
      )
    case EnsembleStep():
      samples = pred_step.particles[:, :3]
      axis_scores = [
        crps_estimator(samples[:, i], float(ref_translation[i]))
        for i in range(3)
      ]
      return float(np.mean(axis_scores))
    case DeterministicStep():
      return float(np.abs(pred_step.translation - ref_translation).mean())
    case _:
      raise TypeError(f"unsupported step type {type(pred_step).__name__}")
