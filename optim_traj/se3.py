"""SE(3) Lie group operations.

Conventions:
- Tangent vector tau = [rho, phi] in R^6, with rho translational (R^3) and phi rotational (R^3).
- Exponential map exp(tau) -> 4x4 homogeneous matrix.
- Logarithm log(T) -> R^6 tangent vector.
- Composition: matrix multiplication.
- Body-frame velocity convention (right-trivialization): xi(t) = xi(0) * exp(t * varpi).
"""
import numpy as np

EPS = 1e-10


def skew(v):
    """3x3 skew-symmetric matrix from 3-vector."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def exp_so3(phi):
    """SO(3) exponential (Rodrigues)."""
    theta = np.linalg.norm(phi)
    if theta < EPS:
        return np.eye(3) + skew(phi)
    a = phi / theta
    K = skew(a)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R):
    """SO(3) logarithm. Returns 3-vector axis-angle."""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < EPS:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if abs(theta - np.pi) < 1e-4:
        # Near pi: standard fix using diagonal.
        M = (R + np.eye(3)) / 2
        # axis is column with largest diagonal
        diag = np.diag(M)
        i = int(np.argmax(diag))
        axis = M[:, i] / np.sqrt(max(diag[i], EPS))
        # disambiguate sign from off-diagonals
        return theta * axis
    return theta / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2],
                                                    R[0, 2] - R[2, 0],
                                                    R[1, 0] - R[0, 1]])


def V_so3(phi):
    """V(phi) matrix used in SE(3) exp: t = V @ rho."""
    theta = np.linalg.norm(phi)
    if theta < EPS:
        return np.eye(3) + 0.5 * skew(phi)
    a = phi / theta
    K = skew(a)
    return (np.eye(3)
            + ((1 - np.cos(theta)) / theta) * K
            + ((theta - np.sin(theta)) / theta) * (K @ K))


def V_inv_so3(phi):
    """Inverse of V(phi)."""
    theta = np.linalg.norm(phi)
    if theta < EPS:
        return np.eye(3) - 0.5 * skew(phi)
    half = theta / 2
    K = skew(phi / theta)
    return (np.eye(3)
            - 0.5 * theta * K
            + (1 - half / np.tan(half)) * (K @ K))


def exp_se3(tau):
    """SE(3) exponential. tau in R^6 -> 4x4 matrix."""
    rho = tau[:3]
    phi = tau[3:]
    T = np.eye(4)
    T[:3, :3] = exp_so3(phi)
    T[:3, 3] = V_so3(phi) @ rho
    return T


def log_se3(T):
    """SE(3) logarithm. 4x4 matrix -> R^6 tangent vector."""
    R = T[:3, :3]
    t = T[:3, 3]
    phi = log_so3(R)
    rho = V_inv_so3(phi) @ t
    return np.concatenate([rho, phi])


def inv_se3(T):
    """Inverse of an SE(3) element."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def compose(A, B):
    """SE(3) composition: A * B."""
    return A @ B


def make_se3(R, t):
    """Build 4x4 SE(3) from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def look_at(eye, target, up=None):
    """Build SE(3) pose with optical axis (+z) pointing from eye to target.

    Camera convention: +z forward (looking direction), +x right, +y down.
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z_axis = target - eye
    z_norm = np.linalg.norm(z_axis)
    if z_norm < EPS:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = z_axis / z_norm
    # Pick x-axis perpendicular to z and up
    x_axis = np.cross(up, z_axis)
    x_n = np.linalg.norm(x_axis)
    if x_n < 1e-6:
        # up parallel to z; pick arbitrary perp
        alt_up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(alt_up, z_axis)
        x_n = np.linalg.norm(x_axis)
    x_axis = x_axis / x_n
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return make_se3(R, eye)


def apply_roll(T, roll_angle):
    """Rotate the pose about its optical axis (local z) by roll_angle."""
    cz = np.cos(roll_angle)
    sz = np.sin(roll_angle)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    Tnew = T.copy()
    Tnew[:3, :3] = T[:3, :3] @ Rz
    return Tnew


def se3_distance(A, B, w_t=1.0, w_r=0.1):
    """Left-invariant distance d(A, B) = || log(A^-1 B) ||_M."""
    tau = log_se3(inv_se3(A) @ B)
    return np.sqrt(w_t * np.sum(tau[:3] ** 2) + w_r * np.sum(tau[3:] ** 2))
