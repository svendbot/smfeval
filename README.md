# smfeval — score the belief, not just the mean

[![PyPI](https://img.shields.io/pypi/v/smfeval)](https://pypi.org/project/smfeval/)
[![Python](https://img.shields.io/pypi/pyversions/smfeval)](https://pypi.org/project/smfeval/)
[![Tests](https://github.com/svendbot/smfeval/actions/workflows/test.yml/badge.svg)](https://github.com/svendbot/smfeval/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A SLAM filter reports a pose *and* a covariance. ATE/RPE check the pose;
smfeval checks whether the covariance is telling the truth.

<p align="center">
  <img src="https://raw.githubusercontent.com/svendbot/smfeval/main/docs/img/overconfidence.png"
       alt="FAST-LIO2 on Oxford Spires christ-church-03: the estimate tracks truth to 3 cm, but the filter's reported 90% region is millimetres wide, so that 3 cm gap is 37 sigma"
       width="460">
</p>

*Illustration (built in `notebooks/figure_overconfidence.py`), not smfeval
output — smfeval emits the text verdict below; this is what that verdict means
geometrically. FAST-LIO2 on Oxford Spires `christ-church-03`: the estimate
(blue) tracks truth (black) to **3 cm** APE, which ATE/RPE call excellent, but
the filter's reported 90% region is **millimetres** wide — so that same 3 cm
gap is **37σ**. The belief is wrong exactly where the mean is right, and that
per-pose gap is what smfeval scores. (Data: Oxford Spires, CC BY-NC-SA 4.0.)*

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

> **No `.SQUARE` file?** smfeval scores the *covariance*, so it needs one
> per pose — plain TUM poses are not enough on their own. If your filter
> publishes a covariance, feed plain TUM + a sidecar directly (no format
> adoption); if it doesn't yet, see
> [Your filter doesn't write SQUARE yet?](#your-filter-doesnt-write-square-yet).

Under a calibrated belief the per-pose translation NEES is χ²₃-distributed
(median 2.37 — read NEES as "the error in sigmas, squared"). The scale gap
**k = median NEES / 2.37** is the factor by which the published covariance
is too tight; per axis that is √k. Here the filter's 90% credible ellipsoid
never contains the truth. `smfeval score` goes further — it localizes
*which* block is wrong (translation vs rotation, bulk vs tail) and emits
[structured, actionable diagnoses](#the-full-report).

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

smfeval needs your filter's per-pose covariance, not just its poses — but
you don't have to adopt the SQUARE format to provide it. Two on-ramps
(`SQUARE_spec.md`, appendix):

- **Wide TUM**: standard TUM pose columns plus the 21 row-major
  lower-triangle entries of the 6×6 tangent covariance (29 columns total).
- **Sidecar file**: plain TUM poses plus `--cov cov.txt` with
  `timestamp c11 c21 c22 … c66` rows.

```sh
smfeval nees est.tum gt.tum --cov est.cov --est-body-frame imu --gt-body-frame imu
```

Most filters compute a covariance internally and never publish it. For
four popular LiDAR-inertial filters the export already exists:
[`exporters/`](exporters/) carries the audited few-line diff that makes
**FAST-LIO2, Faster-LIO, Point-LIO, and I2EKF-LO** publish their belief,
each with its pinned upstream commit, a bag→SQUARE converter, and a
validation run on a named public sequence. Contributions follow the PR
template (`smfeval validate --strict` is the mechanical gate).

## The full report

`smfeval score est.SQUARE gt.tum` produces the complete analysis:

- synchronization and alignment diagnostics;
- proper scoring rules — translation/rotation CRPS, energy score, Gaussian
  log score with its exact calibration/sharpness split, interval score —
  each with a stationary-bootstrap confidence interval;
- PIT/coverage calibration and windowed relative-pose calibration
  (`--rpe-window`);
- track-frame bias/variance attribution;
- structured failure-mode diagnoses with recommended actions.

`--json-out` emits the whole report machine-readable.

Why several scores? Each proper rule touches a different part of the
predictive distribution — bulk shape, tails, joint structure, a chosen
coverage level — so no single number suffices.
[**docs/metrics.rst**](docs/metrics.rst) explains every metric and how to
read it; `SQUARE_spec.md` documents the format and the theory.

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

## Citation

If you use smfeval, please cite the software — GitHub's **Cite this
repository** button reads [`CITATION.cff`](CITATION.cff):

> Rønning, O. *smfeval: probabilistic SLAM trajectory scoring.* 2026.
> https://github.com/svendbot/smfeval

The methodology and the audit behind it are described in the accompanying
paper, released with
[slam_benchmark](https://github.com/svendbot/slam_benchmark) (DOI
forthcoming) — please cite that as well once it is available.

## Provenance

smfeval grew out of a systematic audit of uncertainty calibration in
LiDAR-inertial odometry —
[slam_benchmark](https://github.com/svendbot/slam_benchmark) is the audit
that motivated this tool. The trajectory data used in
fixtures and the notebook derives from the
[Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)
(CC BY-NC-SA 4.0; see the data license notes in those directories).
