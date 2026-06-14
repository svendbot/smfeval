# smfeval — score the belief, not just the mean

A SLAM filter reports a pose *and* a covariance. ATE/RPE check the pose;
smfeval checks whether the covariance is telling the truth.

<p align="center">
  <img src="https://raw.githubusercontent.com/svendbot/smfeval/main/docs/img/overconfidence.png"
       alt="FAST-LIO2 on Oxford Spires christ-church-03: the estimate tracks truth to 3 cm, but the filter's reported 90% region is millimetres wide, so that 3 cm gap is 37 sigma"
       width="460">
</p>

*FAST-LIO2 on Oxford Spires `christ-church-03`. The estimate (blue) tracks
truth (black) to **3 cm** APE — ATE/RPE call this excellent. But the filter's
reported 90% region is **millimetres** wide, so that same 3 cm gap is **37σ**:
the belief is wrong exactly where the mean is right. That gap, per pose, is
what smfeval scores. (Data: Oxford Spires, CC BY-NC-SA 4.0; figure reproducible
via `notebooks/figure_overconfidence.py`.)*

## Install

```sh
pip install smfeval
```

The only dependencies are NumPy and SciPy (Python ≥ 3.10).

## Thirty seconds to a verdict

```
$ smfeval nees estimate.SQUARE reference.tum --gt-body-frame lidar
median NEES 1.04e3   (calibrated: 2.37)
covariance scale gap k = 441, ~21x too tight per axis
90% coverage: 0.000  (calibrated: 0.900)
```

(FAST-LIO2 on Oxford Spires `christ-church-03` — see
`exporters/fast_lio2/VALIDATION.md` for the full reproduction.)

Under a calibrated belief the per-pose translation NEES is χ²₃-distributed
(median 2.37). The scale gap **k = median NEES / 2.37** is the factor by
which the published covariance is too tight; per axis that is √k. Here the
filter's 90% credible ellipsoid never contains the truth.

## No ground truth? Run two filters and score them against each other

```
$ smfeval pair a.SQUARE b.SQUARE
matched 3101 pose pairs, scored 3101  (join 1.00, median gap 0.0 ms)
propriety caveat: pairwise scores are strictly proper only under a
truthful reference sigma and independent errors; both violations push
conservative, so NEES_pair lower-bounds miscalibration.

pairwise median NEES 56.1   (calibrated: 2.37)
pairwise scale gap k >= 23.7, >=4.87x too tight per axis  (lower bound)
verdict: optimistic  (ANEES 71.7 vs chi2 interval [2.91, 3.09])
```

An elevated pairwise NEES **certifies overconfidence with no reference
consulted**: filter A is aligned to filter B directly and the difference is
scored under the summed covariances. Common-mode error and an understated
reference covariance both push the statistic *down*, so the verdict is a
lower bound on the miscalibration — when it fires, it fires honestly.

## Try it now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/svendbot/smfeval/blob/main/notebooks/figure1_verdict.ipynb)

The notebook reproduces the headline verdict on one Oxford Spires sequence
end to end (install → data → verdict → NEES-vs-χ²₃ plot) in a few seconds.

---

## Your filter doesn't write SQUARE yet?

Two on-ramps, no format adoption required (`SQUARE_spec.md`, appendix):

- **Wide TUM**: standard TUM pose columns plus the 21 row-major
  lower-triangle entries of the 6×6 tangent covariance (29 columns total).
- **Sidecar file**: plain TUM poses plus `--cov cov.txt` with
  `timestamp c11 c21 c22 … c66` rows.

```sh
smfeval nees est.tum gt.tum --cov est.cov --est-body-frame imu --gt-body-frame imu
```

And for four popular LiDAR-inertial filters the export already exists:
[`exporters/`](exporters/) carries the audited few-line diff that makes
**FAST-LIO2, Faster-LIO, Point-LIO, and I2EKF-LO** publish their belief,
each with its pinned upstream commit, a bag→SQUARE converter, and a
validation run on a named public sequence. Contributions follow the PR
template (`smfeval validate --strict` is the mechanical gate).

## The full report

`smfeval score est.SQUARE gt.tum` produces the complete analysis:
synchronization and alignment diagnostics, proper scoring rules
(translation/rotation CRPS, energy score, log score with its exact
calibration/sharpness split, interval score) with stationary-bootstrap
confidence intervals, PIT/coverage calibration, windowed relative-pose
calibration (`--rpe-window`), track-frame bias/variance attribution, and
structured failure-mode diagnoses with recommended actions.
`--json-out` emits the whole report machine-readable.

Why several scores? Each proper scoring rule touches a different part of
the predictive distribution — bulk shape, tails, joint structure, a chosen
coverage level — and no single number is sufficient. `SQUARE_spec.md`
documents the format, the taxonomy, and the theory.

## Commands

| Verb | What it does |
|---|---|
| `smfeval nees est gt` | three-line calibration verdict (median NEES, scale gap k, coverage) |
| `smfeval pair a b` | no-reference pairwise verdict (lower bound on miscalibration) |
| `smfeval score est gt` | full scoring report (`--json-out` for machines) |
| `smfeval validate file` | header/row sanity checks (`--strict`: exporter gate) |

## Development

```sh
uv sync && uv run pytest
```

Docs live under [`docs/`](docs/index.rst) (`make docs`). The test suite
includes property-based invariants (hypothesis) and seeded Monte Carlo
power tests of the verdict machinery itself — see `tests/test_power.py`.

## Provenance

smfeval grew out of a systematic audit of uncertainty calibration in
LiDAR-inertial odometry —
[slam_benchmark](https://github.com/svendbot/slam_benchmark) is the audit
that motivated this tool (DOI forthcoming). The trajectory data used in
fixtures and the notebook derives from the
[Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)
(CC BY-NC-SA 4.0; see the data license notes in those directories).
