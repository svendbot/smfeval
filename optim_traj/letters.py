"""Generate a point cloud spelling BAYES with per-point importance weights rho(x).

Each letter is rendered as line strokes in 2D, extruded into a thin slab in z so
each point gets a well-defined outward normal.
"""
import numpy as np


def _line_points(p0, p1, n):
    """n points along a line from p0 to p1."""
    t = np.linspace(0, 1, n)
    return p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None]


def _arc_points(center, r, theta0, theta1, n):
    """n points along a circular arc."""
    th = np.linspace(theta0, theta1, n)
    return center[None, :] + r * np.stack([np.cos(th), np.sin(th)], axis=1)


def letter_B(origin, scale=1.0, density=20):
    pts = []
    o = np.asarray(origin, dtype=float)
    # Vertical bar
    pts.append(_line_points(o + scale * np.array([0, -0.5]),
                            o + scale * np.array([0, 0.5]), density))
    # Upper bump (semicircle)
    pts.append(_arc_points(o + scale * np.array([0, 0.25]), scale * 0.25,
                           -np.pi / 2, np.pi / 2, density))
    # Lower bump
    pts.append(_arc_points(o + scale * np.array([0, -0.25]), scale * 0.25,
                           -np.pi / 2, np.pi / 2, density))
    return np.vstack(pts)


def letter_A(origin, scale=1.0, density=20):
    pts = []
    o = np.asarray(origin, dtype=float)
    # Left diagonal
    pts.append(_line_points(o + scale * np.array([-0.25, -0.5]),
                            o + scale * np.array([0, 0.5]), density))
    # Right diagonal
    pts.append(_line_points(o + scale * np.array([0, 0.5]),
                            o + scale * np.array([0.25, -0.5]), density))
    # Crossbar
    pts.append(_line_points(o + scale * np.array([-0.12, 0]),
                            o + scale * np.array([0.12, 0]), density // 2))
    return np.vstack(pts)


def letter_Y(origin, scale=1.0, density=20):
    pts = []
    o = np.asarray(origin, dtype=float)
    # Upper-left diagonal
    pts.append(_line_points(o + scale * np.array([-0.25, 0.5]),
                            o + scale * np.array([0, 0.1]), density))
    # Upper-right diagonal
    pts.append(_line_points(o + scale * np.array([0.25, 0.5]),
                            o + scale * np.array([0, 0.1]), density))
    # Lower vertical
    pts.append(_line_points(o + scale * np.array([0, 0.1]),
                            o + scale * np.array([0, -0.5]), density))
    return np.vstack(pts)


def letter_E(origin, scale=1.0, density=20):
    pts = []
    o = np.asarray(origin, dtype=float)
    # Vertical bar
    pts.append(_line_points(o + scale * np.array([0, -0.5]),
                            o + scale * np.array([0, 0.5]), density))
    # Top bar
    pts.append(_line_points(o + scale * np.array([0, 0.5]),
                            o + scale * np.array([0.3, 0.5]), density))
    # Middle bar
    pts.append(_line_points(o + scale * np.array([0, 0]),
                            o + scale * np.array([0.22, 0]), density))
    # Bottom bar
    pts.append(_line_points(o + scale * np.array([0, -0.5]),
                            o + scale * np.array([0.3, -0.5]), density))
    return np.vstack(pts)


def letter_S(origin, scale=1.0, density=20):
    pts = []
    o = np.asarray(origin, dtype=float)
    # Top arc (upper half of letter), opening downward-right
    pts.append(_arc_points(o + scale * np.array([0, 0.25]), scale * 0.2,
                           np.pi / 2, 3 * np.pi / 2, density))
    # Top bar to middle
    pts.append(_line_points(o + scale * np.array([-0.2, 0.25]),
                            o + scale * np.array([0.2, 0.0]), density))
    # Bottom arc, opening upper-left
    pts.append(_arc_points(o + scale * np.array([0, -0.25]), scale * 0.2,
                           -np.pi / 2, np.pi / 2, density))
    return np.vstack(pts)


def build_bayes(scale=1.0, density=22, gap=0.7, slab_depth=0.08, n_depth=2,
                rho_prioritized_letter='Y', rho_high=3.0, rho_low=1.0,
                seed=0):
    """Build the BAYES point cloud as a thin 3D slab.

    Returns:
        points: (N, 3) array of 3D points
        normals: (N, 3) outward normals (in +/- z)
        rho: (N,) importance weights
        letter_id: (N,) integer letter index for visualization (0-4)
    """
    letters = [
        ('B', letter_B),
        ('A', letter_A),
        ('Y', letter_Y),
        ('E', letter_E),
        ('S', letter_S),
    ]
    all_pts2d = []
    all_letter_id = []
    x_offset = 0.0
    for i, (name, fn) in enumerate(letters):
        pts2d = fn(np.array([x_offset, 0.0]), scale=scale, density=density)
        all_pts2d.append(pts2d)
        all_letter_id.append(np.full(len(pts2d), i))
        x_offset += scale * gap

    pts2d = np.vstack(all_pts2d)
    letter_id_2d = np.concatenate(all_letter_id)

    # Extrude into thin slab: front face (n=+z) and back face (n=-z)
    rng = np.random.default_rng(seed)
    z_levels = np.linspace(-slab_depth / 2, slab_depth / 2, n_depth)

    points, normals, letter_id = [], [], []
    for j, z in enumerate(z_levels):
        # Add small jitter for realism
        jitter = rng.normal(scale=0.003, size=pts2d.shape)
        slab = np.column_stack([pts2d + jitter, np.full(len(pts2d), z)])
        points.append(slab)
        # Normal: +z if on front face (last z level), -z if on back face (first)
        n_z = 1.0 if z > 0 else -1.0
        nrm = np.zeros_like(slab)
        nrm[:, 2] = n_z
        normals.append(nrm)
        letter_id.append(letter_id_2d)

    points = np.vstack(points)
    normals = np.vstack(normals)
    letter_id = np.concatenate(letter_id)

    # Priorities: prioritized letter gets rho_high, others rho_low
    letter_names = [name for name, _ in letters]
    prioritized_idx = letter_names.index(rho_prioritized_letter)
    rho = np.where(letter_id == prioritized_idx, rho_high, rho_low)

    return points, normals, rho, letter_id


if __name__ == '__main__':
    points, normals, rho, lid = build_bayes()
    print(f'Generated {len(points)} points')
    print(f'  Letter B: {(lid==0).sum()} pts')
    print(f'  Letter A: {(lid==1).sum()} pts')
    print(f'  Letter Y: {(lid==2).sum()} pts (priority {rho[lid==2][0]})')
    print(f'  Letter E: {(lid==3).sum()} pts')
    print(f'  Letter S: {(lid==4).sum()} pts')
    print(f'  Total weighted mass: {rho.sum():.1f} (would be {len(points)} if uniform)')
