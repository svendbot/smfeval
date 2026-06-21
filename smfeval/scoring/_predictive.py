"""Shared predictive translation sampler used by CRPS, energy, calibration, CLI.

Returns sample translations as (n, 3); downstream rules pick which scalar to
compute. Single source of truth for predictive sample generation.
"""

import numpy as np

from smfeval.format import TangentOrder
from smfeval.scoring._kernel import sample_gaussian_tangent
from smfeval.se3.lie import trans_slice
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step


def translation_samples(
  step: Step, n: int, rng: np.random.Generator, order: TangentOrder
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (samples, mu_t)."""
  match step:
    case GaussianStep():
      sl = trans_slice(order)
      cov_t = step.covariance[sl, sl]
      s = sample_gaussian_tangent(step.translation, cov_t, n, rng)
      return s, step.translation
    case EnsembleStep():
      s = step.particles[:, :3]
      return s, s.mean(axis=0)
    case DeterministicStep():
      return step.translation[None, :], step.translation
