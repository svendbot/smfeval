"""Shared builders for SQUARE headers and steps.

One home for the full header field list and the identity-orientation
Gaussian step, so a header/step schema change touches one file instead
of a private copy per test module.
"""

import numpy as np

from smfeval.format import (
  FORMAT_VERSION,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
)
from smfeval.steps import DeterministicStep, GaussianStep

Q_ID = np.array([0.0, 0.0, 0.0, 1.0])


def square_header(representation: Representation, **overrides) -> SquareHeader:
  """SQUARE header with test defaults; override any field per test."""
  fields: dict = {
    "format_version": FORMAT_VERSION,
    "representation": representation,
    "pose_frame": "world",
    "body_frame": "imu",
    "gauge": Gauge.SE3,
    "timestamp_unit": "seconds",
    "algorithm": "testbot",
    "algorithm_version": "1.0",
  }
  if representation is Representation.GAUSSIAN_SE3:
    fields |= {
      "tangent_convention": TangentConvention.RIGHT,
      "tangent_order": TangentOrder.TRANS_ROT,
      "rotation_param": "axis_angle",
    }
  fields |= overrides
  return SquareHeader(**fields)


def gauss_step(
  timestamp: float,
  translation: np.ndarray,
  cov: float | np.ndarray = 1.0,
) -> GaussianStep:
  """GaussianStep at the identity orientation.

  ``cov`` may be a scalar (isotropic), a length-6 diagonal, or a full
  (6, 6) covariance.
  """
  c = np.asarray(cov, dtype=float)
  if c.ndim == 0:
    c = np.eye(6) * float(c)
  elif c.ndim == 1:
    c = np.diag(c)
  return GaussianStep(
    timestamp=timestamp,
    translation=np.asarray(translation, dtype=float),
    quat_xyzw=Q_ID,
    covariance=c,
  )


def det_step(timestamp: float, translation: np.ndarray) -> DeterministicStep:
  """DeterministicStep at the identity orientation."""
  return DeterministicStep(
    timestamp=timestamp,
    translation=np.asarray(translation, dtype=float),
    quat_xyzw=Q_ID,
  )
