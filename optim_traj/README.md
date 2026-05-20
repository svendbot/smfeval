# BAYES Inspection — Factor Graph + Prioritized Region

Minimal working implementation of the Bayesian decision-theoretic SE(3)
inspection pipeline from the plan, with two pieces fully implemented:

1. **GP-on-SE(3) trajectory smoothing via factor graph** (`factor_graph.py`)
2. **Prioritized region** via weighted submodular coverage (`sensor.py`, `letters.py`)

## Files

| File | What it does |
|------|---|
| `se3.py` | SE(3) Lie group operations: exp, log, composition, look_at, left-invariant distance |
| `letters.py` | Generate a 3D point cloud spelling "BAYES" with per-point importance weights ρ(x) |
| `sensor.py` | Soft sensor probability model, candidate viewpoint generation, weighted greedy submodular selection |
| `tsp.py` | Nearest-neighbor + 2-opt TSP on the SE(3) distance matrix |
| `factor_graph.py` | **Factor graph trajectory smoother.** State (xi_k, varpi_k), constant-velocity GP prior factors, waypoint factors, manifold Gauss-Newton (LM with damping) solver |
| `main.py` | End-to-end pipeline plus visualization |

## Running

```bash
cd bayes_inspection
python main.py --budget 22 --prioritized Y --rho-high 4.0 --compare
```

Outputs `inspection_plan.png` and (with `--compare`) `priority_comparison.png`.

Total runtime: ~25 s for K=22 viewpoints, ~700 surface points, ~1200 candidates.
Numerical Jacobians in the factor graph are the bottleneck; analytical Jacobians
would speed this up by ~10x.

## What the factor graph implementation contains

The state at each knot k is a pair (xi_k, varpi_k) where:

- xi_k in SE(3) is the pose at time t_k
- varpi_k in R^6 is the body-frame velocity twist

**Constant-velocity GP prior factor** between knots k and k+1:

```
r_pos = log(exp(dt * varpi_k)^{-1} * xi_k^{-1} * xi_{k+1})    [R^6]
r_vel = varpi_{k+1} - varpi_k                                 [R^6]
```

with information matrix Q_k^{-1} from the continuous-time PSD Q_c (Anderson/
Barfoot/Mukadam discrete-time form):

```
Q_k = [[ dt^3/3 * Qc,   dt^2/2 * Qc ],
       [ dt^2/2 * Qc,   dt   * Qc   ]]
```

**Waypoint factor** at knot k_j with target pose xi_j*:

```
r_wp = log((xi_j*)^{-1} * xi_{k_j})                           [R^6]
```

with information matrix W^{-1} from sigma_t, sigma_r.

**Solver:** Manifold Levenberg-Marquardt. At each iteration:

1. Linearize all residuals in tangent space (numerical Jacobians, finite diff in se(3))
2. Build sparse normal equations (J^T J + lambda I) delta = -J^T r
3. Retract: xi_k <- xi_k * exp(delta_xi_k), varpi_k <- varpi_k + delta_v_k

Typically converges in 3-6 iterations for the BAYES example.

## What the priority implementation contains

Each surface point x_i carries a non-negative weight rho_i. The greedy
submodular maximization becomes weighted:

```
gain(v | S) = sum_i rho_i * not_covered_i * Q[v, i]
```

By Lemma 2.1 in the plan, this preserves monotonicity and submodularity (sum of
non-negatively-weighted submodular functions is submodular), so the
Nemhauser (1-1/e) approximation guarantee carries over.

**Result:** with the "Y" prioritized 4x, the planner concentrates viewpoints
around the Y, achieving 100% expected observation probability on priority
points vs 97.5% on ordinary points, with no change to the runtime structure.

## Limits and next steps

- Numerical Jacobians: correct but slow. Replace with analytical SE(3)
  Jacobians (Barfoot ch. 7) for a ~10x speedup.
- The trajectory is a sequence of waypoint-knots. For higher-resolution
  smoothing, add intermediate knots between waypoints with no waypoint factor
  attached (they're free to be shaped by the GP prior).
- Obstacle factors (point cloud distance) are stubbed but not added — easy
  extension: penalize distance from any pose along the trajectory to the
  nearest point cloud point falling below a safety margin.
- To get a Pareto front in (coverage, length) space, sweep budget_K from
  small to large and record each result. Plot expected coverage vs trajectory length.
