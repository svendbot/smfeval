# Changelog

## 0.4.0 - 2026-06-12

The release that matches the paper. Everything needed to score a filter's
own belief against reference. Scoring is on the **translation** marginal;
orientation is left to future work, since a proper score on SO(3) needs a
belief density whose normaliser is intractable for the natural rotation
families (paper §II.b / §V.d).

### New verbs

- `smfeval nees est ref`. Three-line calibration verdict (median NEES
  against the χ² reference 2.37 for dof 3, the covariance scale gap
  `k = median NEES / 2.37` with √k per axis, and a nominal-coverage check).
- `smfeval pair a b`. No-reference calibration. Scores two filters against
  each other (A→B Umeyama, difference under summed covariances, χ²₃). An
  elevated pairwise NEES certifies overconfidence with no reference
  consulted, and the statistic lower-bounds the miscalibration (propriety
  caveat printed with every verdict). Ported from the slam_benchmark audit
  and verified exact against it.
- `smfeval validate --strict`. Mechanical exporter gate (per-row covariance
  SPD, plausible magnitude, not degenerate-zero, finite poses).

### On-ramps

- TUM + covariance escape hatch for estimates. Wide TUM (29 columns, pose
  plus lower-triangle covariance, SQUARE packing) or plain TUM with a
  `--cov` sidecar file. Conventions are declared by flags and echoed.
- `exporters/`, the four audited belief exporters (FAST-LIO2, Faster-LIO,
  Point-LIO, I2EKF-LO). Each carries the diff, the pinned upstream commit, a
  bag→SQUARE converter, and a validation run. All declare GAUGE se3, so the
  verdict fits a full SE(3) alignment before scoring.
- Colab notebook (`notebooks/figure1_verdict.ipynb`) reproducing the
  headline verdict on an Oxford Spires sequence end to end.

### Scoring

- Short-window relative translation CRPS and windowed NEES calibration
  (`--rpe-window`). Restores sensitivity to local σ calibration that
  absolute-pose CRPS loses in the overconfident regime.
- Exact translation log-score calibration/sharpness split with a two-sided
  ANEES χ² verdict (`--calibration`).
- Track-frame bias/variance decomposition (along/cross/vertical) and
  structured failure-mode diagnoses with recommended actions.
- Student-t belief-transform intervention (`--student-t`), per-scan ESS
  covariance inflation (`--ess-inflate`), reference-covariance folding
  (`--consume-ref-cov`).
- Fixed catastrophic cancellation in the SE(3) V-jacobian small-angle
  branch (exp/log round-trip error 5e-9 to <5e-15).

### Reports

- `smfeval score --json` prints the structured report to stdout; `--json-out`
  writes it to a file. Both follow `docs/report.schema.json` (schema_version 2.0).
- `docs/metrics.rst` and its sub-pages (proper scores, calibration, alignment,
  diagnostics) explain how to read each metric, equation-first.

### Testing

- Property-based invariant tests (hypothesis) across packing, CRPS,
  NEES frame invariance, pairing, alignment, and summaries.
- Seeded Monte Carlo power tests of the verdict machinery: ANEES
  false-positive rate at nominal α, power against covariance
  understatement, pairwise dilution laws, coverage separation.
- Real-data regression fixtures from Oxford Spires christ-church-03
  (CC BY-NC-SA 4.0, attributed), including a no-reference pair scenario.
- CI test and lint workflow (previously only tag-publish existed).

### Packaging

- Package renamed `src` to `smfeval` (a PyPI wheel must not install a
  top-level `src`).
- Version 0.4.0 follows 0.3.0 directly; the distributional-reference
  work formerly tagged 0.4 in planning docs is renumbered to 0.5
  (`plans/0.5-distributional-ref.md`).
- Dropped unused dev dependencies (pandas, natsort, colorama, pygments,
  brotli).

## 0.3.0

Closed-form Gaussian CRPS, adaptive SO(3) CRPS with jackknife SE,
tangent-Gaussian validity diagnostic.
