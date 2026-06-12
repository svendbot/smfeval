r"""Tangent-Gaussian validity check for :class:`GaussianStep` on SO(3).

Flags steps whose worst-axis rotation std :math:`\sqrt{\lambda_\max(\Sigma_{rr})}`
exceeds soft (0.3 rad, ~17°) and hard (0.5 rad, ~29°) limits. Past the soft
limit the concentrated-normal approximation drifts from the manifold pushforward
(missing BCH curvature corrections); past the hard limit the Exp map is no
longer bijective on the support and the predictive cannot meaningfully be
called a Gaussian on SO(3) — switch to matrix-Fisher or particles.
"""

from dataclasses import dataclass

import numpy as np

from smfeval.format import TangentOrder
from smfeval.se3.lie import rot_slice
from smfeval.steps import GaussianStep

_SOFT_LIMIT_RAD = 0.3
_HARD_LIMIT_RAD = 0.5

# Fraction of steps past each limit that triggers a recommendation. Hard
# violations are catastrophic per-step, so even a small fraction matters; soft
# violations only matter at scale.
HARD_LIMIT_REPORT_FRACTION = 0.01
SOFT_LIMIT_REPORT_FRACTION = 0.05


@dataclass
class GaussianValidityDiagnostic:
  n_total: int
  max_sigma_r: float
  soft_limit_rad: float
  hard_limit_rad: float
  n_exceeding_soft: int
  n_exceeding_hard: int


def gaussian_rotation_validity(
  steps: list[GaussianStep],
  tangent_order: TangentOrder,
) -> GaussianValidityDiagnostic | None:
  r"""Counts of steps past soft/hard tangent-Gaussian limits on SO(3).

  Per-step scale is :math:`\sqrt{\lambda_\max(\Sigma_{rr})}` (worst-axis
  rotation std). Returns ``None`` when ``steps`` is empty.
  """
  if not steps:
    return None
  sl = rot_slice(tangent_order)
  sigmas = np.sqrt(
    np.maximum(
      0.0,
      np.array([np.linalg.eigvalsh(s.covariance[sl, sl])[-1] for s in steps]),
    )
  )
  return GaussianValidityDiagnostic(
    n_total=len(steps),
    max_sigma_r=float(sigmas.max()),
    soft_limit_rad=_SOFT_LIMIT_RAD,
    hard_limit_rad=_HARD_LIMIT_RAD,
    n_exceeding_soft=int((sigmas > _SOFT_LIMIT_RAD).sum()),
    n_exceeding_hard=int((sigmas > _HARD_LIMIT_RAD).sum()),
  )
