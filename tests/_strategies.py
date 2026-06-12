"""Shared hypothesis strategies for the property-based tests."""

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

FINITE = {"allow_nan": False, "allow_infinity": False}


def floats(lo: float, hi: float):
  return st.floats(min_value=lo, max_value=hi, **FINITE)


def vec(n: int, lo: float = -10.0, hi: float = 10.0):
  return st.lists(floats(lo, hi), min_size=n, max_size=n).map(np.array)


@st.composite
def spd6(draw, scale_lo: float = 1e-2, scale_hi: float = 10.0):
  """Well-conditioned SPD 6x6 via a Gram matrix plus identity ridge."""
  a = draw(vec(36, -1.0, 1.0)).reshape(6, 6)
  s = draw(floats(scale_lo, scale_hi))
  return s * (a @ a.T + np.eye(6))


@st.composite
def pose(draw, t_lo: float = -10.0, t_hi: float = 10.0):
  """Random SE(3) pose as (translation, quat_xyzw), angle < pi."""
  rotvec = draw(vec(3, -1.0, 1.0))
  norm = np.linalg.norm(rotvec)
  assume(norm < np.pi - 1e-3)
  t = draw(vec(3, t_lo, t_hi))
  q = Rotation.from_rotvec(rotvec).as_quat()
  return t, q
