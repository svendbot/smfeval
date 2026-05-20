"""End-to-end pipeline: plan an inspection trajectory for the BAYES point cloud
with the 'Y' letter prioritized 3x.

Pipeline stages:
0. Generate BAYES point cloud with priorities rho.
1. Generate SE(3) viewpoint candidates by offsetting along normals.
2. Build coverage matrix Q with the soft sensor model.
3. Greedy weighted submodular viewpoint selection (Nemhauser 1-1/e).
4. SE(3) TSP ordering via nearest-neighbor + 2-opt.
5. GP-on-SE(3) factor graph smoothing.
6. Visualize.
"""
import argparse
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import se3
import letters
import sensor
import tsp
import factor_graph as fg


def plan(budget_K=20, scale=1.0, density=22, slab_depth=0.08,
         standoffs=(0.25, 0.40, 0.6), rolls=(0.0, np.pi / 2),
         rho_high=3.0, prioritized_letter='Y', seed=0,
         dt=1.0, smoothness=0.4, verbose=True):
    """Run the full planning pipeline and return all artifacts."""
    t0 = time.time()

    # Stage 0: BAYES point cloud
    points, normals, rho, letter_id = letters.build_bayes(
        scale=scale, density=density, slab_depth=slab_depth,
        rho_prioritized_letter=prioritized_letter, rho_high=rho_high,
        rho_low=1.0, seed=seed)
    if verbose:
        print(f'[0] {len(points)} surface points, prioritized {prioritized_letter} '
              f'with weight {rho_high}, total weighted mass = {rho.sum():.1f}')

    # Stage 1: viewpoint candidates
    cand_poses, cand_target = sensor.generate_candidates(
        points, normals, standoffs=standoffs, rolls=rolls)
    if verbose:
        print(f'[1] {len(cand_poses)} raw candidates')

    keep = sensor.voxel_downsample_candidates(cand_poses, voxel_size=0.05)
    cand_poses = cand_poses[keep]
    cand_target = cand_target[keep]
    if verbose:
        print(f'    downsampled to {len(cand_poses)} candidates')

    # Stage 2: coverage matrix
    params = sensor.default_sensor_params(scale=scale)
    Q = sensor.build_coverage_matrix(cand_poses, points, normals, params)
    if verbose:
        density_pct = 100 * (Q > 0).mean()
        print(f'[2] coverage matrix {Q.shape}, density {density_pct:.2f}%')

    # Stage 3: greedy weighted submodular selection
    selected_idx, history = sensor.greedy_weighted_submodular(Q, rho, budget_K)
    if verbose:
        final_cov, final_weighted = history[-1]
        print(f'[3] greedy selected K={budget_K} viewpoints')
        print(f'    expected coverage: {final_cov:.1f}/{len(points)} points '
              f'({100*final_cov/len(points):.1f}%)')
        print(f'    weighted coverage: {final_weighted:.1f}/{rho.sum():.1f} '
              f'({100*final_weighted/rho.sum():.1f}%)')

    selected_poses = [cand_poses[i] for i in selected_idx]

    # Stage 4: TSP ordering
    order, length_pre, D = tsp.solve_tsp(selected_poses, w_t=1.0, w_r=0.05)
    ordered_poses = [selected_poses[i] for i in order]
    if verbose:
        print(f'[4] TSP tour length (waypoints only): {length_pre:.3f}')

    # Stage 5: GP smoothing
    Q_c = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * smoothness
    state, _ = fg.build_and_solve_trajectory(
        ordered_poses, dt=dt, Q_c=Q_c,
        sigma_t_wp=0.015, sigma_r_wp=0.08,
        max_iter=20, verbose=False)

    smoothed_traj = fg.evaluate_trajectory(state, n_samples_per_segment=15)
    traj_len = fg.trajectory_length(state)
    if verbose:
        print(f'[5] smoothed trajectory: {len(smoothed_traj)} samples, '
              f'arc length {traj_len:.3f}')

    # Recompute coverage achieved by smoothed trajectory (not just waypoints)
    # We evaluate the sensor at each sampled pose along the smoothed path.
    not_cov = np.ones(len(points))
    for xi in smoothed_traj:
        for i in range(len(points)):
            q = sensor.sensor_prob(xi, points[i], normals[i], params)
            not_cov[i] *= (1 - q)
    p_cov = 1 - not_cov
    smoothed_coverage = float(p_cov.sum())
    smoothed_weighted = float((rho * p_cov).sum())
    if verbose:
        print(f'    actual smoothed coverage: {smoothed_coverage:.1f}/{len(points)} '
              f'({100*smoothed_coverage/len(points):.1f}%)')
        print(f'    actual smoothed weighted: {smoothed_weighted:.1f}/{rho.sum():.1f} '
              f'({100*smoothed_weighted/rho.sum():.1f}%)')

    if verbose:
        print(f'\nTotal pipeline time: {time.time() - t0:.2f} s')

    return dict(
        points=points, normals=normals, rho=rho, letter_id=letter_id,
        cand_poses=cand_poses, selected_poses=selected_poses,
        ordered_poses=ordered_poses, smoothed_traj=smoothed_traj,
        state=state, history=history,
        traj_len=traj_len, smoothed_coverage=smoothed_coverage,
        smoothed_weighted=smoothed_weighted, budget_K=budget_K,
        per_point_coverage_prob=p_cov,
    )


def _draw_scene_3d(ax, result, view_elev=30, view_azim=-70, title=''):
    """Render a single 3D scene panel."""
    pts = result['points']
    rho = result['rho']
    is_priority = rho > 1.1

    ax.scatter(pts[~is_priority, 0], pts[~is_priority, 1], pts[~is_priority, 2],
               c='#444444', s=10, alpha=0.75, depthshade=False)
    ax.scatter(pts[is_priority, 0], pts[is_priority, 1], pts[is_priority, 2],
               c='#ff7f00', s=22, alpha=0.95, depthshade=False)

    # Smoothed trajectory
    traj_pos = np.array([p[:3, 3] for p in result['smoothed_traj']])
    ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
            color='#1f77b4', lw=2.5, alpha=0.9)

    # Viewpoints with their optical axes
    sel = np.array([p[:3, 3] for p in result['ordered_poses']])
    ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c='#d62728', s=50,
               marker='^', edgecolors='black', linewidths=0.5, zorder=10, depthshade=False)
    for p in result['ordered_poses']:
        eye = p[:3, 3]
        axis = p[:3, 2]
        end = eye + 0.18 * axis
        ax.plot([eye[0], end[0]], [eye[1], end[1]], [eye[2], end[2]],
                '-', color='#d62728', lw=0.9, alpha=0.6)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    if title:
        ax.set_title(title)

    all_pos = np.vstack([pts, traj_pos])
    mid = all_pos.mean(0)
    span = np.max(all_pos.max(0) - all_pos.min(0)) * 0.55
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.view_init(elev=view_elev, azim=view_azim)


def _draw_scene_topdown(ax, result, title=''):
    """Top-down (xy) view, ignoring z. Letters are nearly in z=0 plane."""
    pts = result['points']
    rho = result['rho']
    is_priority = rho > 1.1

    ax.scatter(pts[~is_priority, 0], pts[~is_priority, 1],
               c='#444444', s=12, alpha=0.7, edgecolors='none')
    ax.scatter(pts[is_priority, 0], pts[is_priority, 1],
               c='#ff7f00', s=26, alpha=0.95, edgecolors='none')

    traj_pos = np.array([p[:3, 3] for p in result['smoothed_traj']])
    ax.plot(traj_pos[:, 0], traj_pos[:, 1], color='#1f77b4', lw=2.0, alpha=0.85)

    sel = np.array([p[:3, 3] for p in result['ordered_poses']])
    ax.scatter(sel[:, 0], sel[:, 1], c='#d62728', s=60, marker='^',
               edgecolors='black', linewidths=0.6, zorder=10)

    for i, p in enumerate(result['ordered_poses']):
        eye = p[:3, 3]
        axis = p[:3, 2]
        end = eye + 0.18 * axis
        ax.plot([eye[0], end[0]], [eye[1], end[1]], '-', color='#d62728', lw=0.8, alpha=0.55)

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_result(result, out_path='inspection_plan.png'):
    """Render: BAYES point cloud (color = priority), selected viewpoints, smoothed trajectory."""
    fig = plt.figure(figsize=(18, 10))
    rho = result['rho']
    pts = result['points']
    p_cov = result['per_point_coverage_prob']
    is_priority = rho > 1.1
    rho_high = rho.max()

    # Top-down view (main, since BAYES lies near z=0)
    ax_td = fig.add_subplot(2, 3, (1, 4))
    _draw_scene_topdown(
        ax_td, result,
        title=f'Top-down view: K={result["budget_K"]} viewpoints\n'
              f'orange = priority (rho={rho_high:.1f}), gray = ordinary (rho=1)\n'
              f'blue = smoothed SE(3) trajectory, red = selected viewpoints'
    )

    # 3D oblique view
    ax_3d = fig.add_subplot(2, 3, 2, projection='3d')
    _draw_scene_3d(ax_3d, result, title='3D oblique view')

    # Greedy progress
    ax_p = fig.add_subplot(2, 3, 3)
    hist = np.array(result['history'])
    K = result['budget_K']
    N = len(pts)
    ax_p.plot(range(1, K + 1), hist[:, 0] / N * 100, 'o-', label='Coverage (uniform)', color='steelblue')
    ax_p.plot(range(1, K + 1), hist[:, 1] / rho.sum() * 100, 's-', label='Weighted coverage', color='darkorange')
    ax_p.set_xlabel('# viewpoints selected')
    ax_p.set_ylabel('Coverage (%)')
    ax_p.set_title('Greedy submodular progress\n(weighted > uniform => priority matters)')
    ax_p.legend()
    ax_p.grid(alpha=0.3)

    # Per-point coverage by priority
    ax_h = fig.add_subplot(2, 3, 5)
    ax_h.hist([p_cov[~is_priority], p_cov[is_priority]],
              bins=np.linspace(0, 1, 21), stacked=False,
              label=['Ordinary (rho=1)', f'Priority (rho={rho_high:.1f})'],
              color=['#888888', '#ff7f00'], edgecolor='k', alpha=0.85)
    ax_h.set_xlabel('Observation probability p_i')
    ax_h.set_ylabel('# points')
    ax_h.set_title('Coverage distribution by priority\n(priority points observed more reliably)')
    ax_h.legend()
    ax_h.grid(alpha=0.3)

    # Summary text
    ax_s = fig.add_subplot(2, 3, 6)
    ax_s.axis('off')
    summary = (
        f"PIPELINE SUMMARY\n"
        f"================\n\n"
        f"Surface points       : {len(pts)}\n"
        f"Priority points      : {is_priority.sum()} (weight {rho_high:.1f}x)\n"
        f"Weighted mass        : {rho.sum():.0f}\n\n"
        f"Candidates evaluated : {len(result['cand_poses'])}\n"
        f"Selected (greedy)    : {result['budget_K']}\n\n"
        f"Waypoint coverage    :\n"
        f"  uniform   : {100*hist[-1,0]/N:.1f}%\n"
        f"  weighted  : {100*hist[-1,1]/rho.sum():.1f}%\n\n"
        f"Smoothed coverage    :\n"
        f"  uniform   : {100*result['smoothed_coverage']/N:.1f}%\n"
        f"  weighted  : {100*result['smoothed_weighted']/rho.sum():.1f}%\n\n"
        f"Trajectory length    : {result['traj_len']:.2f}\n\n"
        f"Priority observation : {p_cov[is_priority].mean():.3f}\n"
        f"Ordinary observation : {p_cov[~is_priority].mean():.3f}"
    )
    ax_s.text(0.0, 1.0, summary, family='monospace', fontsize=10,
              verticalalignment='top', transform=ax_s.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    return out_path


def plot_comparison(results_list, labels, out_path='comparison.png'):
    """Side-by-side comparison: same budget, different priority assignments."""
    n = len(results_list)
    fig = plt.figure(figsize=(7 * n, 6))
    for j, (res, lab) in enumerate(zip(results_list, labels)):
        ax = fig.add_subplot(1, n, j + 1)
        _draw_scene_topdown(ax, res, title=lab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=20)
    p.add_argument('--prioritized', default='Y')
    p.add_argument('--rho-high', type=float, default=3.0)
    p.add_argument('--out-dir', default='/mnt/user-data/outputs')
    p.add_argument('--compare', action='store_true', help='Also run comparison across priorities')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    print(f'MAIN RUN: priority on {args.prioritized}, K={args.budget}')
    print('=' * 60)
    res = plan(budget_K=args.budget,
               prioritized_letter=args.prioritized,
               rho_high=args.rho_high)
    out_main = plot_result(res, out_path=os.path.join(args.out_dir, 'inspection_plan.png'))
    print(f'\nMain figure saved: {out_main}')

    if args.compare:
        print('\n' + '=' * 60)
        print('COMPARISON: how does priority shift the trajectory?')
        print('=' * 60)
        results = []
        labels = []
        for letter in ['B', 'Y', 'S']:
            print(f'\n  -- Priority on {letter} --')
            r = plan(budget_K=args.budget,
                     prioritized_letter=letter,
                     rho_high=args.rho_high,
                     verbose=False)
            results.append(r)
            labels.append(f'Priority on "{letter}" (rho={args.rho_high})\n'
                          f'weighted cov {100*r["smoothed_weighted"]/r["rho"].sum():.1f}%, '
                          f'len {r["traj_len"]:.2f}')
            print(f'    coverage {100*r["smoothed_coverage"]/len(r["points"]):.1f}%, '
                  f'weighted {100*r["smoothed_weighted"]/r["rho"].sum():.1f}%, '
                  f'length {r["traj_len"]:.2f}')

        out_cmp = plot_comparison(results, labels,
                                  out_path=os.path.join(args.out_dir, 'priority_comparison.png'))
        print(f'\nComparison figure saved: {out_cmp}')
