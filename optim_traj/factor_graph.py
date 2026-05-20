"""GP-on-SE(3) trajectory smoothing via factor graph + manifold Gauss-Newton.

State at knot k: (xi_k, varpi_k) where
- xi_k in SE(3) is the pose, stored as 4x4 matrix
- varpi_k in R^6 is the body-frame velocity twist [rho_dot; phi_dot]

Trajectory model (constant-velocity GP prior, Anderson/Barfoot/Mukadam):
  xi(t) = xi_k * exp((t - t_k) * varpi_k)   for t in [t_k, t_{k+1}]
  varpi assumed approximately constant, perturbed by white-noise acceleration.

Discrete-time prior factor between knots k and k+1:
  r_pos = log(exp(dt * varpi_k)^{-1} * xi_k^{-1} * xi_{k+1})      [R^6]
  r_vel = varpi_{k+1} - varpi_k                                    [R^6]
Information matrix Q_k^{-1} derived from continuous-time PSD Q_c:
  Q_k = | dt^3/3 * Q_c   dt^2/2 * Q_c |
        | dt^2/2 * Q_c   dt   * Q_c    |

Waypoint factor at knot k_j with target pose xi_j*:
  r_wp = log((xi_j*)^{-1} * xi_{k_j})                              [R^6]
Information matrix W^{-1}.

Optimization: manifold Gauss-Newton.
At each iteration, linearize residuals in tangent space of current state, solve
sparse normal equations for increment delta in R^{12K}, and retract:
  xi_k <- xi_k * exp(delta_xi_k)
  varpi_k <- varpi_k + delta_v_k

Jacobians here are computed by finite differences in the tangent space, which is
correct (Lie-group derivative) and avoids hand-coded matrix calculus on SE(3).
For K~30 knots this is fast enough; analytical Jacobians would be O(10x) faster.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import se3


# ---------- State container ----------------------------------------------

class TrajectoryState:
    """K knots; state at each knot is (pose, body velocity)."""

    def __init__(self, poses, velocities):
        assert len(poses) == len(velocities)
        self.K = len(poses)
        self.poses = [p.copy() for p in poses]                # list of 4x4
        self.velocities = [v.copy() for v in velocities]      # list of R^6

    def retract(self, delta):
        """Apply 12K-dim increment in tangent space. delta layout: [dxi_0, dv_0, dxi_1, dv_1, ...]."""
        for k in range(self.K):
            dxi = delta[12 * k: 12 * k + 6]
            dv = delta[12 * k + 6: 12 * k + 12]
            self.poses[k] = self.poses[k] @ se3.exp_se3(dxi)
            self.velocities[k] = self.velocities[k] + dv

    def copy(self):
        return TrajectoryState(self.poses, self.velocities)

    def get_pose(self, k):
        return self.poses[k]

    def get_velocity(self, k):
        return self.velocities[k]


# ---------- Factor interface ---------------------------------------------

class Factor:
    """Base class. A factor knows which knots it touches and computes its residual."""

    knots = ()         # tuple of knot indices

    def residual(self, state):
        raise NotImplementedError

    def sqrt_information(self):
        """Cholesky factor L such that L^T L = information matrix."""
        raise NotImplementedError


def _finite_diff_jacobian(fn, state, knot, comp, eps=1e-5):
    """Numerical Jacobian of residual function fn w.r.t. perturbation at given knot/component.

    comp == 'pose': perturb pose by exp(eps * e_i), i in 0..5
    comp == 'velocity': perturb velocity by eps * e_i, i in 0..5
    """
    n_dim = 6
    r0 = fn(state)
    R = len(r0)
    J = np.zeros((R, n_dim))
    for i in range(n_dim):
        # Perturb forward
        delta = np.zeros(12 * state.K)
        if comp == 'pose':
            delta[12 * knot + i] = eps
        else:
            delta[12 * knot + 6 + i] = eps
        s_fwd = state.copy()
        s_fwd.retract(delta)
        r_fwd = fn(s_fwd)

        # Perturb backward
        delta = -delta
        s_bwd = state.copy()
        s_bwd.retract(delta)
        r_bwd = fn(s_bwd)

        J[:, i] = (r_fwd - r_bwd) / (2 * eps)
    return J


# ---------- Concrete factors ----------------------------------------------

class WaypointFactor(Factor):
    """Soft constraint pulling knot k_idx toward target pose xi_target."""

    def __init__(self, knot_idx, xi_target, sigma_t=0.02, sigma_r=0.05):
        self.knots = (knot_idx,)
        self.xi_target = xi_target.copy()
        # Diagonal information: information per axis = 1 / sigma^2
        inv = np.array([1 / sigma_t**2] * 3 + [1 / sigma_r**2] * 3)
        self._L = np.diag(np.sqrt(inv))      # sqrt information

    def residual(self, state):
        xi_k = state.get_pose(self.knots[0])
        return se3.log_se3(se3.inv_se3(self.xi_target) @ xi_k)

    def sqrt_information(self):
        return self._L


class ConstantVelocityPriorFactor(Factor):
    """GP prior between consecutive knots under constant-velocity-with-white-noise SDE.

    Residual is 12-dim: [r_pos (6); r_vel (6)].
    Information matrix derived from the discrete-time noise covariance
        Q_k = [[dt^3/3 * Qc, dt^2/2 * Qc], [dt^2/2 * Qc, dt * Qc]]
    """

    def __init__(self, k1, k2, dt, Q_c):
        """Q_c: 6x6 power spectral density. dt: time between knots."""
        self.knots = (k1, k2)
        self.dt = dt
        # Build discrete-time covariance
        Q_c = np.asarray(Q_c)
        Q11 = (dt ** 3 / 3) * Q_c
        Q12 = (dt ** 2 / 2) * Q_c
        Q22 = dt * Q_c
        Q = np.block([[Q11, Q12], [Q12, Q22]])
        # Cholesky of Q^{-1}: L such that L^T L = Q^{-1}
        # Compute as L = inv(chol(Q)).T more stably
        Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(12))
        self._L = np.linalg.solve(Lq, np.eye(12)).T

    def residual(self, state):
        k1, k2 = self.knots
        xi_1 = state.get_pose(k1)
        xi_2 = state.get_pose(k2)
        v_1 = state.get_velocity(k1)
        v_2 = state.get_velocity(k2)

        # Predicted xi_2 from xi_1, v_1, dt
        pred = xi_1 @ se3.exp_se3(self.dt * v_1)
        r_pos = se3.log_se3(se3.inv_se3(pred) @ xi_2)
        r_vel = v_2 - v_1
        return np.concatenate([r_pos, r_vel])

    def sqrt_information(self):
        return self._L


# ---------- Solver --------------------------------------------------------

def build_linear_system(state, factors):
    """Build sparse Jacobian and weighted residual for current state.

    Returns J (sparse), r (dense), where minimizing 0.5 || J delta + r ||^2 gives
    the Gauss-Newton step.
    """
    n_var = 12 * state.K
    rows, cols, vals = [], [], []
    r_blocks = []
    row_offset = 0

    for factor in factors:
        # Evaluate residual at current state
        L = factor.sqrt_information()
        r = L @ factor.residual(state)
        r_dim = len(r)
        r_blocks.append(r)

        # For each knot the factor touches, compute Jacobian numerically
        for kn in factor.knots:
            # Wrap residual fn for finite diff
            def resid_fn(s, _f=factor, _L=L):
                return _L @ _f.residual(s)

            J_pose = _finite_diff_jacobian(resid_fn, state, kn, 'pose')
            J_vel = _finite_diff_jacobian(resid_fn, state, kn, 'velocity')

            # Place into sparse matrix
            for i in range(r_dim):
                for j in range(6):
                    if J_pose[i, j] != 0:
                        rows.append(row_offset + i)
                        cols.append(12 * kn + j)
                        vals.append(J_pose[i, j])
                    if J_vel[i, j] != 0:
                        rows.append(row_offset + i)
                        cols.append(12 * kn + 6 + j)
                        vals.append(J_vel[i, j])

        row_offset += r_dim

    n_resid = row_offset
    J = sp.csr_matrix((vals, (rows, cols)), shape=(n_resid, n_var))
    r_full = np.concatenate(r_blocks)
    return J, r_full


def gauss_newton_solve(state, factors, max_iter=20, tol=1e-6, lambda_init=1e-4,
                      verbose=False):
    """Manifold Levenberg-Marquardt with damping.

    Solves: minimize sum_f || L_f * r_f(x) ||^2 over the trajectory state.
    """
    lam = lambda_init
    prev_cost = None

    for it in range(max_iter):
        J, r = build_linear_system(state, factors)
        cost = 0.5 * float(r @ r)

        # Damped normal equations: (J^T J + lam I) delta = -J^T r
        H = (J.T @ J).toarray()
        H += lam * np.eye(H.shape[0])
        g = J.T @ r

        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        # Trial step
        trial = state.copy()
        trial.retract(delta)
        _, r_trial = build_linear_system(trial, factors)
        new_cost = 0.5 * float(r_trial @ r_trial)

        if verbose:
            print(f'  iter {it}: cost {cost:.6e} -> {new_cost:.6e}, lam={lam:.2e}, ||delta||={np.linalg.norm(delta):.4e}')

        if new_cost < cost:
            # Accept
            state = trial
            lam *= 0.5
            if prev_cost is not None and abs(prev_cost - new_cost) < tol * (1 + new_cost):
                if verbose:
                    print('  converged')
                break
            prev_cost = new_cost
        else:
            # Reject, increase damping
            lam *= 4
            if lam > 1e8:
                if verbose:
                    print('  damping saturated')
                break

    return state


# ---------- Trajectory evaluation ----------------------------------------

def evaluate_trajectory(state, n_samples_per_segment=10):
    """Densely sample the trajectory using the constant-velocity model on each segment.

    For segment [k, k+1] of length dt: xi(s) = xi_k * exp(s * dt * varpi_k), s in [0, 1].
    Linearly interpolate velocity for continuity at knots (not strictly the GP mean,
    but close enough for visualization).
    """
    K = state.K
    poses = []
    for k in range(K - 1):
        xi_k = state.get_pose(k)
        v_k = state.get_velocity(k)
        for s in np.linspace(0, 1, n_samples_per_segment, endpoint=False):
            poses.append(xi_k @ se3.exp_se3(s * v_k))
    poses.append(state.get_pose(K - 1))
    return poses


def trajectory_length(state, n_samples_per_segment=20):
    """Translational arc length of the smoothed trajectory."""
    poses = evaluate_trajectory(state, n_samples_per_segment)
    positions = np.array([p[:3, 3] for p in poses])
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ---------- Convenience: build the full factor graph from a tour ---------

def build_and_solve_trajectory(waypoint_poses, dt=1.0, Q_c=None,
                               sigma_t_wp=0.01, sigma_r_wp=0.05,
                               max_iter=30, verbose=False):
    """Build factor graph from waypoint sequence and solve.

    Each waypoint becomes a knot in the trajectory (1-to-1). Prior factors enforce
    smooth motion between knots. Waypoint factors anchor each knot to its target.
    """
    K = len(waypoint_poses)
    if Q_c is None:
        # Diagonal PSD: translational and rotational
        Q_c = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5

    # Initialize: poses at waypoints, velocities estimated by finite diff
    poses_init = [w.copy() for w in waypoint_poses]
    vels_init = []
    for k in range(K):
        if k < K - 1:
            v = se3.log_se3(se3.inv_se3(poses_init[k]) @ poses_init[k + 1]) / dt
        else:
            v = vels_init[-1].copy()  # last velocity = previous
        vels_init.append(v)

    state = TrajectoryState(poses_init, vels_init)

    factors = []
    # Prior factors
    for k in range(K - 1):
        factors.append(ConstantVelocityPriorFactor(k, k + 1, dt, Q_c))
    # Waypoint factors
    for k in range(K):
        factors.append(WaypointFactor(k, waypoint_poses[k],
                                      sigma_t=sigma_t_wp, sigma_r=sigma_r_wp))

    state = gauss_newton_solve(state, factors, max_iter=max_iter, verbose=verbose)
    return state, factors
