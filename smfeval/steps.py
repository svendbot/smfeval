from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from smfeval.format import WeightFormat


@dataclass
class GaussianStep:
  timestamp: float
  translation: np.ndarray
  quat_xyzw: np.ndarray
  covariance: np.ndarray


@dataclass
class EnsembleStep:
  timestamp: float
  particles: np.ndarray
  weights: np.ndarray | None


@dataclass
class DeterministicStep:
  timestamp: float
  translation: np.ndarray
  quat_xyzw: np.ndarray


Step = GaussianStep | EnsembleStep | DeterministicStep


def log_normalized_weights(
  weights: np.ndarray, fmt: WeightFormat, normalized: bool
) -> np.ndarray:
  """Return log-normalized ensemble weights regardless of input form.

  The declared ``WEIGHT_FORMAT`` decides how to read them; the sign of the
  values does not. Unnormalized log weights are routinely all-positive, so
  guessing from the sign silently misreads them as linear.
  """
  if fmt is WeightFormat.LOG:
    log_w = weights.astype(float)
  else:
    with np.errstate(divide="ignore"):
      log_w = np.where(
        weights > 0, np.log(np.maximum(weights, 1e-300)), -np.inf
      )
  if not normalized or fmt is WeightFormat.LOG:
    log_w = log_w - logsumexp(log_w)
  return log_w
