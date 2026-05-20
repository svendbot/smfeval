r"""Lie-group exponential / logarithm and adjoint maps for SO(3) and SE(3),
together with helper slicings for the tangent-order convention.

We follow the conventions of Solà, Deray & Atchuthan (2018) and Barfoot
(2017). The SE(3) exponential / logarithm uses the closed-form Jacobian

.. math::

   V(\omega) = I + \frac{1 - \cos\theta}{\theta^2}\,[\omega]_\times
       + \frac{\theta - \sin\theta}{\theta^3}\,[\omega]_\times^2,
   \qquad \theta = \lVert\omega\rVert,

with a Taylor fallback for :math:`\theta < \texttt{\_EPS}`.

References
----------
Solà, J., Deray, J. & Atchuthan, D. (2018). *A micro Lie theory for
state estimation in robotics*. arXiv:1812.01537.

Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge
University Press.

Murray, R. M., Li, Z. & Sastry, S. S. (1994). *A Mathematical
Introduction to Robotic Manipulation*. CRC Press.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.types import TangentOrder

# Small-angle threshold below which the closed-form SO(3)/SE(3) Jacobians
# are numerically ill-conditioned; we switch to their Taylor expansion.
_EPS = 1e-12


def pose_matrix(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    # Homogeneous transform: [[R, p], [0, 1]] with R ∈ SO(3), p ∈ R^3.
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


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


def vee_so3(W: np.ndarray) -> np.ndarray:
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(w).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def _v_jacobian(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat_so3(w)
    if theta < _EPS:
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * W @ W
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * W + b * (W @ W)


def _v_jacobian_inv(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat_so3(w)
    if theta < _EPS:
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    half = theta / 2.0
    c = (1.0 / theta**2) - (1.0 / (2.0 * theta)) * (np.cos(half) / np.sin(half))
    return np.eye(3) - 0.5 * W + c * (W @ W)


def se3_exp(xi: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT) -> np.ndarray:
    rho, w = _split(xi, order)
    R = so3_exp(w)
    V = _v_jacobian(w)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    rho = _v_jacobian_inv(w) @ t
    return _join(rho, w, order)


def adjoint(T: np.ndarray, order: TangentOrder = TangentOrder.TRANS_ROT) -> np.ndarray:
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
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return T1 @ T2


def relative(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return invert(T1) @ T2


def _split(xi: np.ndarray, order: TangentOrder) -> tuple[np.ndarray, np.ndarray]:
    if order is TangentOrder.TRANS_ROT:
        return xi[:3], xi[3:]
    return xi[3:], xi[:3]


def _join(rho: np.ndarray, w: np.ndarray, order: TangentOrder) -> np.ndarray:
    if order is TangentOrder.TRANS_ROT:
        return np.concatenate([rho, w])
    return np.concatenate([w, rho])
