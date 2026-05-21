# smfeval

Probabilistic SLAM trajectory format and scoring tool: strictly proper
scoring rules and calibration diagnostics for Gaussian, ensemble, and
deterministic pose beliefs on SE(3).

## Why this exists

Conventional SLAM evaluation (ATE/RPE) summarises a deterministic
trajectory error. When the SLAM backend reports an uncertainty over its
pose belief — a 6×6 covariance, a particle ensemble — those numbers are
the whole point: a filter that drifts but knows it drifts is doing
something fundamentally different from one that drifts confidently.
`smfeval` scores the predictive belief, not just its mean, using
strictly proper scoring rules (Gneiting & Raftery, 2007):

- **CRPS** on the three translation axes (Matheson & Winkler, 1976) and
  on rotation via the SO(3) geodesic kernel.
- **Energy score** on the SE(3) tangent (Székely, 2003; Gneiting &
  Raftery, 2007).
- **Log score** (Good, 1952), reported in three pieces:
  - joint SE(3) negative log density;
  - translation marginal (3-D sub-block of the joint covariance);
  - rotation marginal (3-D sub-block).
  The 6×6 covariance couples translation and rotation, so a single
  scalar hides pathologies that target only one block — e.g. a LiDAR
  filter that is well-calibrated in translation but overconfident in
  yaw when the geometry degenerates along the motion direction.
- **Interval score** on translation magnitude (Gneiting & Raftery, 2007).
- **Calibration diagnostics**: PIT + KS test (Dawid, 1984), interval
  coverage, Mahalanobis-normalised translation residuals.

## Prequential scoring, not iid averaging

Scoring runs in *walk-forward* / prequential mode (Dawid, 1984): at
each matched timestep we score the SLAM backend's one-step-ahead
predictive against the realised ground-truth pose. The resulting score
series is **not** an iid sample. SLAM error is autocorrelated by
construction — drift accumulates, regime changes (loop closure,
entering or leaving a degenerate corridor) persist for many frames —
so the textbook iid percentile bootstrap on the mean under-covers and
hides the most diagnostically interesting structure of the time
series.

`smfeval` therefore aggregates per-step scores with the **stationary
bootstrap** of Politis & Romano (1994), whose mean geometric block
length `ℓ` is selected automatically from the data by the **Politis &
White (2004)** flat-top lag-window estimator (hand-rolled in
`src/scoring/summary.py`, no `arch` dependency). Every score row in
the report carries the estimated `ℓ` next to its CI: `ℓ ≈ 1` means
the series behaves like white noise; `ℓ ≫ 1` flags a strong temporal
dependence that the scalar mean is summarising over.

Pass `block_length=1.0` to `summarize()` to fall back to the iid
percentile bootstrap when you want to compare.

## Install

```sh
nix develop --command uv sync
```

The dev shell pins NumPy ≥ 1.24 and SciPy ≥ 1.10; no other runtime
dependencies. Docs and tests live under `docs/` and `tests/`.

## Use

Validate a trajectory file:

```sh
nix develop --command uv run smfeval validate path/to/est.SQUARE
```

Score an estimate against a ground-truth trajectory (SQUARE format or
plain TUM):

```sh
nix develop --command uv run smfeval score est.SQUARE gt.tum
```

The report has six sections — synchronisation, alignment, ensemble
diagnostics (when applicable), scores, calibration, recommendations —
following the layout in `SQUARE_spec.md`.

## Layout

| Module | Role |
| --- | --- |
| `src/scoring/` | proper scoring rules + calibration |
| `src/scoring/summary.py` | Politis–White block-length + stationary bootstrap |
| `src/scoring/logscore.py` | joint + trans-marginal + rot-marginal log scores |
| `src/se3/` | SE(3) / SO(3) Lie group machinery |
| `src/align/` | gauge-aware alignment of estimate to ground truth |
| `src/sync/` | timestamp matching and sync-risk computation |
| `src/io/` | header parser and reader/writer for the SQUARE text format |
| `src/report/` | report dataclass, text renderer, recommendations |
| `src/cli/` | `smfeval validate` / `smfeval score` |

## References

- Dawid, A. P. (1984). *Statistical theory: the prequential approach.* JRSS A 147(2), 278–292.
- Efron, B. (1979). *Bootstrap methods: another look at the jackknife.* Annals of Statistics 7(1), 1–26.
- Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation.* JASA 102(477), 359–378.
- Good, I. J. (1952). *Rational decisions.* JRSS B 14(1), 107–114.
- Matheson, J. E. & Winkler, R. L. (1976). *Scoring rules for continuous probability distributions.* Management Science 22(10), 1087–1096.
- Politis, D. N. & Romano, J. P. (1994). *The stationary bootstrap.* JASA 89(428), 1303–1313.
- Politis, D. N. & White, H. (2004). *Automatic block-length selection for the dependent bootstrap.* Econometric Reviews 23(1), 53–70.
- Sturm, J., Engelhard, N., Endres, F., Burgard, W. & Cremers, D. (2012). *A benchmark for the evaluation of RGB-D SLAM systems.* IROS.
- Székely, G. J. (2003). *E-statistics: the energy of statistical samples.* BGSU Tech. Rep. 03-05.
