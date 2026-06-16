r"""Interval score for a univariate predictive (Gneiting & Raftery, 2007).

Nominal coverage :math:`1-\alpha`.
"""

import numpy as np

from smfeval.format import TangentOrder
from smfeval.scoring._predictive import translation_samples
from smfeval.steps import Step


def interval_score(
  lower: float, upper: float, observation: float, alpha: float = 0.1
) -> float:
  r"""Proper interval score at nominal coverage :math:`1-\alpha`.

  For a central prediction interval :math:`[l, u]` and observation
  :math:`y`,

  .. math::

     \mathrm{IS}_\alpha(l, u; y)
     = (u - l)
     + \frac{2}{\alpha}(l - y)\,\mathbf{1}\{y < l\}
     + \frac{2}{\alpha}(y - u)\,\mathbf{1}\{y > u\}.

  Smaller is better. The score sums a sharpness term (the width
  :math:`u-l`) and a calibration penalty whose :math:`2/\alpha`
  coefficient renders the rule strictly proper: the expected score is
  minimised iff :math:`(l, u)` are the true predictive quantiles
  :math:`\bigl(F^{-1}(\alpha/2),\ F^{-1}(1-\alpha/2)\bigr)`.

  References:
  -----------
  Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
  prediction, and estimation*. JASA 102(477), 359–378.
  """
  width = upper - lower
  pen = 0.0
  if observation < lower:
    pen += (2.0 / alpha) * (lower - observation)
  elif observation > upper:
    pen += (2.0 / alpha) * (observation - upper)
  return float(width + pen)


def interval_from_samples(
  samples: np.ndarray, alpha: float = 0.1
) -> tuple[float, float]:
  r"""Equal-tailed empirical :math:`(\alpha/2,\ 1-\alpha/2)` interval."""
  lo = float(np.quantile(samples, alpha / 2.0))
  hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
  return lo, hi


def translation_magnitude_interval_score(
  pred_step: Step,
  gt_translation: np.ndarray,
  alpha: float = 0.1,
  n_samples: int = 128,
  rng: np.random.Generator | None = None,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> float:
  r"""Interval score on translation magnitude :math:`\lVert t - \mu_t\rVert`.

  Predictive samples from the step's belief yield the equal-tailed
  interval :math:`[l, u]`; the observation is the reference
  translation's distance from the predictive mean,
  :math:`y = \lVert t_\mathrm{gt} - \mu_t\rVert`.
  """
  rng = rng if rng is not None else np.random.default_rng(0)
  samples, mu = translation_samples(pred_step, n_samples, rng, tangent_order)
  mags = np.linalg.norm(samples - mu, axis=1)
  lo, hi = interval_from_samples(mags, alpha)
  obs = float(np.linalg.norm(gt_translation - mu))
  return interval_score(lo, hi, obs, alpha)
