r"""Energy score on the translation marginal (:math:`\mathbb{R}^3`).

The energy score generalises CRPS to multivariate forecasts. It was
introduced by Székely (2003) and characterised as a strictly proper
multivariate kernel score by Gneiting & Raftery (2007).

We score the translation marginal only. Orientation needs a proper score
on SO(3), whose natural belief families carry intractable normalisers
(paper, §II.b / §V.d), so it is out of scope. At the reported covariance
the translation energy score reduces to the per-pose error norm that APE
aggregates (paper, §I / §III).

References:
-----------
Székely, G. J. (2003). *E-statistics: The energy of statistical
samples*. Bowling Green State University, Tech. Rep. 03-05.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

import numpy as np

from smfeval.format import TangentOrder
from smfeval.scoring._kernel import (
  energy_score_estimator,
  sample_gaussian_tangent,
)
from smfeval.se3.lie import trans_slice
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step


def energy_score(
  pred_step: Step,
  ref_translation: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  n_samples: int = 128,
  rng: np.random.Generator | None = None,
) -> float:
  r"""Translation energy score at the predictive position.

  For samples :math:`X, X' \stackrel{iid}{\sim} F` and observation
  :math:`y \in \mathbb{R}^3`,

  .. math::

     \mathrm{ES}(F, y) = \mathbb{E}\,\lVert X - y\rVert
         - \tfrac12\,\mathbb{E}\,\lVert X - X'\rVert.

  Strictly proper for the Euclidean norm (Gneiting & Raftery, 2007, §4.2).
  """
  rng = rng if rng is not None else np.random.default_rng(0)

  match pred_step:
    case GaussianStep():
      sl = trans_slice(tangent_order)
      cov_t = pred_step.covariance[sl, sl]
      samples = sample_gaussian_tangent(
        pred_step.translation, cov_t, n_samples, rng
      )
      return energy_score_estimator(samples, ref_translation)
    case EnsembleStep():
      n = pred_step.particles.shape[0]
      if n == 0:
        return float("nan")
      return energy_score_estimator(pred_step.particles[:, :3], ref_translation)
    case DeterministicStep():
      return float(np.linalg.norm(pred_step.translation - ref_translation))
