r"""Lie-group exponential / logarithm and adjoint maps for SO(3) and SE(3).

Together with helper slicings for the tangent-order convention.

We follow the conventions of Solà, Deray & Atchuthan (2018) and Barfoot
(2017). The SE(3) exponential / logarithm uses the closed-form Jacobian

.. math::

   V(\omega) = I + \frac{1 - \cos\theta}{\theta^2}\,[\omega]_\times
       + \frac{\theta - \sin\theta}{\theta^3}\,[\omega]_\times^2,
   \qquad \theta = \lVert\omega\rVert,

with a Taylor fallback for :math:`\theta < \texttt{\_EPS}`.

References:
-----------
Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for
state estimation in robotics*. arXiv:1812.01537.

Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge
University Press.

Murray, R. M., Li, Z. & Sastry, S. S. (1994). *A Mathematical
Introduction to Robotic Manipulation*. CRC Press.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from smfeval.format import TangentOrder

# Small-angle threshold below which the closed-form SO(3)/SE(3) Jacobians
# are numerically ill-conditioned; we switch to their Taylor expansion.
# At theta = 1e-2 the naive (1-cos)/theta^2 loses ~eps/theta^2 ~ 2e-12 to
# cancellation while the two-term Taylor truncation error is theta^6/40320
# ~ 2.5e-17; below this the Taylor branch is strictly more accurate.
_EPS = 1e-2


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Pack a (3,3) rotation and (3,) translation into a 4x4 homogeneous transform."""
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t
  return T


def pose_matrix(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
  # Homogeneous transform: [[R, p], [0, 1]] with R in SO(3), p in R^3.
  return homogeneous(Rotation.from_quat(quat_xyzw).as_matrix(), translation)


def trans_slice(order: TangentOrder) -> slice:
  if order is TangentOrder.TRANS_ROT:
    return slice(0, 3)
  else:
    return slice(3, 6)


def rot_slice(order: TangentOrder) -> slice:
  if order is TangentOrder.TRANS_ROT:
    return slice(3, 6)
  else:
    return slice(0, 3)


def hat_so3(w: np.ndarray) -> np.ndarray:
  return np.array(
    [
      [0.0, -w[2], w[1]],
      [w[2], 0.0, -w[0]],
      [-w[1], w[0], 0.0],
    ]
  )


def so3_exp(w: np.ndarray) -> np.ndarray:
  return Rotation.from_rotvec(w).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
  return Rotation.from_matrix(R).as_rotvec()


def so3_mean(
  rots: np.ndarray, max_iter: int = 20, tol: float = 1e-10
) -> np.ndarray:
  """Frechet (geodesic L2) mean of a stack of (n, 3, 3) rotations.

  Seeded from the chordal mean (SVD projection of the average matrix), then
  refined by iterated tangent averaging ``R <- R Exp(mean_i Log(R^T R_i))``.
  Unweighted; the reference center of an ensemble does not depend on particle
  order, unlike picking a single particle.
  """
  rots = np.asarray(rots, dtype=float)
  if rots.ndim != 3 or rots.shape[0] == 0:
    return np.eye(3)
  M = rots.mean(axis=0)
  U, _, Vt = np.linalg.svd(M)
  d = np.sign(np.linalg.det(U @ Vt))
  R = U @ np.diag([1.0, 1.0, d]) @ Vt
  for _ in range(max_iter):
    delta = np.mean([so3_log(R.T @ Ri) for Ri in rots], axis=0)
    R = R @ so3_exp(delta)
    if float(np.linalg.norm(delta)) < tol:
      break
  return R


def _v_jacobian(w: np.ndarray) -> np.ndarray:
  theta = float(np.linalg.norm(w))
  W = hat_so3(w)
  t2 = theta * theta
  if theta < _EPS:
    a = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
    b = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
  else:
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
  return np.eye(3) + a * W + b * (W @ W)


def _v_jacobian_inv(w: np.ndarray) -> np.ndarray:
  theta = float(np.linalg.norm(w))
  W = hat_so3(w)
  t2 = theta * theta
  if theta < _EPS:
    c = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
  else:
    half = theta / 2.0
    c = (1.0 / t2) - (1.0 / (2.0 * theta)) * (np.cos(half) / np.sin(half))
  return np.eye(3) - 0.5 * W + c * (W @ W)


def se3_exp(
  xi: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT
) -> np.ndarray:
  rho, w = _split(xi, order)
  return homogeneous(so3_exp(w), _v_jacobian(w) @ rho)


def se3_log(
  T: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT
) -> np.ndarray:
  R = T[:3, :3]
  t = T[:3, 3]
  w = so3_log(R)
  rho = _v_jacobian_inv(w) @ t
  return _join(rho, w, order)


def adjoint(
  T: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT
) -> np.ndarray:
  R = T[:3, :3]
  t = T[:3, 3]
  tx_R = hat_so3(t) @ R
  Ad = np.zeros((6, 6))
  if order is TangentOrder.TRANS_ROT:
    Ad[:3, :3] = R
    Ad[:3, 3:] = tx_R
    Ad[3:, 3:] = R
  else:
    Ad[:3, :3] = R
    Ad[3:, :3] = tx_R
    Ad[3:, 3:] = R
  return Ad


def invert(T: np.ndarray) -> np.ndarray:
  R = T[:3, :3]
  t = T[:3, 3]
  return homogeneous(R.T, -R.T @ t)


def relative(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
  return invert(T1) @ T2


def pose_residual(
  t1: np.ndarray,
  q1_xyzw: np.ndarray,
  t2: np.ndarray,
  q2_xyzw: np.ndarray,
  order: TangentOrder = TangentOrder.TRANS_ROT,
) -> np.ndarray:
  r"""Tangent residual :math:`\log(T_1^{-1} T_2)` between two poses.

  Each pose is given as a translation and ``xyzw`` quaternion; the result is
  the 6-vector in the convention selected by ``order``.
  """
  return se3_log(
    relative(pose_matrix(t1, q1_xyzw), pose_matrix(t2, q2_xyzw)), order
  )


def _split(
  xi: np.ndarray, order: TangentOrder
) -> tuple[np.ndarray, np.ndarray]:
  if order is TangentOrder.TRANS_ROT:
    return xi[:3], xi[3:]
  return xi[3:], xi[:3]


def _join(rho: np.ndarray, w: np.ndarray, order: TangentOrder) -> np.ndarray:
  if order is TangentOrder.TRANS_ROT:
    return np.concatenate([rho, w])
  return np.concatenate([w, rho])
