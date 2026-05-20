"""Sensor model q(xi, x, n) and coverage matrix construction.

q is a soft probability of "useful observation" of surface point (x, n) from
camera pose xi. Factorizes as frustum * range * incidence-angle * occlusion.
"""
import numpy as np
import se3


def sigmoid(u, beta=20.0):
    return 1.0 / (1.0 + np.exp(-beta * u))


def sensor_prob(xi, x, n, params):
    """Probability that surface point (x, n) is usefully observed from camera xi.

    Camera convention: +z forward (optical axis), +x right, +y down.
    """
    R = xi[:3, :3]
    t = xi[:3, 3]

    # Point in camera frame
    x_c = R.T @ (x - t)
    z = x_c[2]

    # Must be in front of camera
    if z <= 0:
        return 0.0

    # Frustum (soft)
    theta_h = params['fov_h']
    theta_v = params['fov_v']
    ang_x = np.arctan2(abs(x_c[0]), z)
    ang_y = np.arctan2(abs(x_c[1]), z)
    p_fov = sigmoid(theta_h - ang_x, params['beta_fov']) * sigmoid(theta_v - ang_y, params['beta_fov'])

    # Range with attenuation
    r = np.linalg.norm(x - t)
    p_range = (sigmoid(r - params['r_min'], params['beta_range'])
               * sigmoid(params['r_max'] - r, params['beta_range'])
               * np.exp(-params['kappa'] * r))

    # Incidence angle: optical axis facing surface, ideal when ray hits face-on
    # cos(alpha) = n . (t - x) / |t - x|, want this near 1
    d = (t - x) / max(r, 1e-9)
    cos_alpha = float(n @ d)
    if cos_alpha <= 0:
        return 0.0
    p_angle = cos_alpha ** params['angle_sharpness']

    return p_fov * p_range * p_angle


def default_sensor_params(scale=1.0):
    """Sensible defaults for the BAYES example (letter-scale ~1)."""
    return dict(
        fov_h=np.deg2rad(35),
        fov_v=np.deg2rad(25),
        beta_fov=30.0,
        r_min=0.15 * scale,
        r_max=1.2 * scale,
        beta_range=20.0,
        kappa=0.2,           # underwater attenuation
        angle_sharpness=2.0,
    )


def generate_candidates(points, normals, standoffs, rolls):
    """Generate SE(3) viewpoint candidates by offsetting along normals."""
    candidates = []
    candidate_target = []  # which point each candidate "looks at"
    for i, (x, n) in enumerate(zip(points, normals)):
        for d in standoffs:
            eye = x + d * n
            T = se3.look_at(eye, x)
            for phi in rolls:
                Tr = se3.apply_roll(T, phi)
                candidates.append(Tr)
                candidate_target.append(i)
    return np.array(candidates), np.array(candidate_target)


def voxel_downsample_candidates(candidates, voxel_size=0.05):
    """Keep one candidate per translation voxel (rough deduplication)."""
    keys = {}
    keep = []
    for v, xi in enumerate(candidates):
        t = xi[:3, 3]
        key = tuple(np.round(t / voxel_size).astype(int))
        if key not in keys:
            keys[key] = v
            keep.append(v)
    return np.array(keep)


def build_coverage_matrix(candidates, points, normals, params, q_min=1e-3):
    """Build dense coverage matrix Q[v, i] = q(xi_v, x_i, n_i)."""
    V = len(candidates)
    N = len(points)
    Q = np.zeros((V, N))
    for v, xi in enumerate(candidates):
        for i in range(N):
            q = sensor_prob(xi, points[i], normals[i], params)
            if q > q_min:
                Q[v, i] = q
    return Q


def greedy_weighted_submodular(Q, rho, budget_K):
    """Greedy maximization of sum_i rho_i * [1 - prod_v (1 - Q[v, i])] over subsets S of size K.

    By Lemma 2.1 (monotone submodularity, preserved under non-negative reweighting)
    and Nemhauser-Wolsey-Fisher 1978, this achieves (1 - 1/e) of the optimum.
    """
    V, N = Q.shape
    selected = []
    not_covered = np.ones(N)  # product of (1 - q) over selected
    remaining = set(range(V))
    history = []  # (coverage, weighted_coverage) at each step

    for k in range(budget_K):
        best_v, best_gain = -1, -np.inf
        for v in remaining:
            # Marginal gain: sum_i rho_i * not_covered[i] * Q[v, i]
            gain = float(np.sum(rho * not_covered * Q[v, :]))
            if gain > best_gain:
                best_gain = gain
                best_v = v
        selected.append(best_v)
        not_covered = not_covered * (1.0 - Q[best_v, :])
        remaining.remove(best_v)
        p_cov = 1.0 - not_covered
        history.append((float(p_cov.sum()), float((rho * p_cov).sum())))

    return selected, history
