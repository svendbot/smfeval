import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from smfeval.format import TangentOrder
from smfeval.se3 import (
  adjoint,
  invert,
  quat_xyzw_to_rot,
  relative,
  rot_to_quat_xyzw,
  se3_exp,
  se3_log,
  so3_exp,
  so3_log,
  so3_mean,
)

RNG = np.random.default_rng(0)


def _random_rotvec(rng):
  axis = rng.normal(size=3)
  axis /= np.linalg.norm(axis)
  angle = rng.uniform(0.0, np.pi - 1e-3)
  return axis * angle


def _random_se3(rng):
  w = _random_rotvec(rng)
  t = rng.normal(size=3)
  return se3_exp(np.concatenate([t, w]))


def test_so3_exp_log_round_trip():
  for _ in range(20):
    w = _random_rotvec(RNG)
    R = so3_exp(w)
    w2 = so3_log(R)
    # rotvec is unique modulo direction at θ=π; use the rotation it produces
    assert np.allclose(so3_exp(w2), R, atol=1e-9)


def test_so3_exp_zero_is_identity():
  assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


def test_se3_exp_log_round_trip():
  for _ in range(20):
    T = _random_se3(RNG)
    xi = se3_log(T)
    T2 = se3_exp(xi)
    assert np.allclose(T2, T, atol=1e-9)


def test_se3_exp_log_round_trip_rot_trans_order():
  for _ in range(20):
    T = _random_se3(RNG)
    xi = se3_log(T, order=TangentOrder.ROT_TRANS)
    T2 = se3_exp(xi, order=TangentOrder.ROT_TRANS)
    assert np.allclose(T2, T, atol=1e-9)


def test_se3_exp_small_angle_taylor():
  xi = np.array([1e-10, 2e-10, 3e-10, 1e-10, 2e-10, 3e-10])
  T = se3_exp(xi)
  assert np.allclose(T[:3, 3], xi[:3], atol=1e-15)
  assert np.allclose(T[:3, :3], np.eye(3), atol=1e-9)


def test_relative_identity():
  T = _random_se3(RNG)
  assert np.allclose(relative(T, T), np.eye(4), atol=1e-12)


def test_adjoint_identity():
  Ad = adjoint(np.eye(4))
  assert np.allclose(Ad, np.eye(6))


def test_adjoint_property():
  """T·exp(ξ)·T⁻¹ = exp(Ad_T · ξ)"""
  for _ in range(10):
    T = _random_se3(RNG)
    xi = np.concatenate([RNG.normal(size=3) * 0.1, RNG.normal(size=3) * 0.1])
    lhs = T @ se3_exp(xi) @ invert(T)
    rhs = se3_exp(adjoint(T) @ xi)
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_adjoint_orders_consistent():
  """Adjoint in rot_trans order is the [ω,ρ] permutation of the trans_rot one."""
  T = _random_se3(RNG)
  Ad_tr = adjoint(T, order=TangentOrder.TRANS_ROT)
  Ad_rt = adjoint(T, order=TangentOrder.ROT_TRANS)
  P = np.block(
    [
      [np.zeros((3, 3)), np.eye(3)],
      [np.eye(3), np.zeros((3, 3))],
    ]
  )
  assert np.allclose(Ad_rt, P @ Ad_tr @ P.T, atol=1e-12)


def test_quat_round_trip():
  for _ in range(10):
    w = _random_rotvec(RNG)
    R = so3_exp(w)
    q = rot_to_quat_xyzw(R)
    R2 = quat_xyzw_to_rot(q)
    assert np.allclose(R, R2, atol=1e-12)


def test_quat_xyzw_layout():
  R = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
  q = rot_to_quat_xyzw(R)
  assert q.shape == (4,)
  assert abs(q[3] - np.cos(np.pi / 4)) < 1e-9
  assert abs(q[2] - np.sin(np.pi / 4)) < 1e-9


@given(
  st.lists(
    st.floats(min_value=-2, max_value=2, allow_nan=False),
    min_size=6,
    max_size=6,
  )
)
@settings(max_examples=50, deadline=None)
def test_se3_round_trip_property(xs):
  xi = np.array(xs)
  if np.linalg.norm(xi[3:]) >= np.pi - 1e-3:
    return
  T = se3_exp(xi)
  xi2 = se3_log(T)
  T2 = se3_exp(xi2)
  assert np.allclose(T2, T, atol=1e-9)


def test_so3_mean_identity():
  R = so3_exp(np.array([0.3, -0.2, 0.5]))
  assert np.allclose(so3_mean(np.stack([R, R, R])), R, atol=1e-9)


def test_so3_mean_left_equivariance():
  rng = np.random.default_rng(3)
  rots = np.stack([so3_exp(rng.normal(scale=0.5, size=3)) for _ in range(8)])
  Q = so3_exp(np.array([0.4, -0.1, 0.7]))
  left = so3_mean(np.einsum("ij,njk->nik", Q, rots))
  right = Q @ so3_mean(rots)
  assert np.allclose(left, right, atol=1e-8)


def test_so3_mean_is_stationary():
  rng = np.random.default_rng(4)
  rots = np.stack([so3_exp(rng.normal(scale=0.5, size=3)) for _ in range(10)])
  rbar = so3_mean(rots)
  grad = np.sum([so3_log(rbar.T @ ri) for ri in rots], axis=0)
  assert np.linalg.norm(grad) < 1e-8


def test_ensemble_rotation_reference_is_permutation_invariant():
  from smfeval.format import TangentOrder
  from smfeval.scoring._predictive import rotation_samples
  from smfeval.steps import EnsembleStep

  rng = np.random.default_rng(5)
  rots = [so3_exp(rng.normal(scale=0.3, size=3)) for _ in range(6)]
  particles = np.array([[0.0, 0.0, 0.0, *rot_to_quat_xyzw(r)] for r in rots])
  perm = [3, 1, 5, 0, 2, 4]
  step = EnsembleStep(timestamp=0.0, particles=particles, weights=None)
  step_perm = EnsembleStep(
    timestamp=0.0, particles=particles[perm], weights=None
  )
  om1, _ = rotation_samples(step, 0, rng, TangentOrder.TRANS_ROT)
  om2, _ = rotation_samples(step_perm, 0, rng, TangentOrder.TRANS_ROT)
  norms1 = np.sort(np.linalg.norm(om1, axis=1))
  norms2 = np.sort(np.linalg.norm(om2, axis=1))
  assert np.allclose(norms1, norms2, atol=1e-9)
