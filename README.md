# smfeval

Score SLAM trajectories that report uncertainty. Works on Gaussian-,
ensemble-, or deterministic-pose estimates on SE(3) and gives back
proper scoring rules plus calibration diagnostics.

Full documentation lives under [`docs/`](docs/index.rst); run
`make docs` to build the HTML version and `make test` to run the
test suite.

## Install

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Runtime depends only on NumPy ≥ 1.24 and SciPy ≥ 1.10.

## Use

Validate a trajectory file:

```sh
uv run smfeval validate path/to/est.SQUARE
```

Score an estimate against a ground-truth trajectory (SQUARE format or
plain TUM):

```sh
uv run smfeval score est.SQUARE gt.TUM
```

Example output:

```
Synchronization
  Mode:                   interpolate_gt
  Pairs matched:          5,738 / 5,738
  Dropped:                0
  GP σ (m):   median 0.0003, p95 0.0004, p99 0.0011

Alignment
  Gauge (declared):       gravity_yaw
  Mode applied:           se3   (6 DoF)
  Fitted Δxyz:            (-28.4562, 28.0381, -4.7399) m
  Fit residual (m):       median 0.9007, p95 1.2460
                          6 DoF removed over 649 m of trajectory

Scores
  Translation CRPS:           mean 0.281 m   [95% CI 0.224, 0.334]   (n=5738)
                              median 0.313, std 0.124, min 0.014, max 0.503
                              block length (Politis–White): 176.5
  Rotation CRPS:              mean 0.036 rad   [95% CI 0.036, 0.037]   (n=5738)
                              median 0.036, std 0.002, min 0.030, max 0.043
                              block length (Politis–White): 173.8
  Energy score (SE(3)):       mean 0.802   [95% CI 0.637, 0.960]   (n=5738)
                              median 0.901, std 0.365, min 0.044, max 1.486
                              block length (Politis–White): 176.5
  Log score (joint):          mean 1401860.038   [95% CI 1050457.394, 1733644.908]
                              median 1317643.220, std 848368.184
                              block length (Politis–White): 175.1
  Log score (translation):    mean 1008309.207   [95% CI 690401.685, 1341320.364]
                              median 903201.643, std 777138.180
                              block length (Politis–White): 175.5
  Log score (rotation):       mean 271085.256   [95% CI 244685.182, 299495.521]
                              median 271799.807, std 83672.340
                              block length (Politis–White): 172.7
  Interval score:             mean 15.996   [95% CI 12.522, 19.090]   (n=5738)
                              median 17.973, std 7.325
                              block length (Politis–White): 176.5

Calibration
  PIT uniformity (KS):    p = 0.000  ⚠ possible miscalibration
  90% Mahalanobis coverage:  0.0%     (nominal 90.0%)
  Translation z-score:    mean 741.74, std 349.38   (over-confident)

Recommendations
  - Coverage below nominal combined with KS p < 0.05 — the filter is
    over-confident (claimed Σ too tight, truth falls outside the
    predicted intervals); widen process noise. Miscalibration is
    unlikely to be explained by sync error alone.
```

The report has six sections — synchronisation, alignment, ensemble
diagnostics (when applicable), scores, calibration, recommendations —
following the layout in `SQUARE_spec.md`. See [`docs/`](docs/index.rst)
for what each statistic means and how it's computed.
