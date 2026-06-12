# Changelog

## 0.4.0 — 2026-06-12

The paper-matching release: everything a stranger needs to score their
own filter's belief.

### New verbs

- `smfeval nees est gt` — three-line calibration verdict: median NEES
  against the χ² reference (2.37 for dof 3), the covariance scale gap
  `k = median NEES / 2.37` (√k per axis), and nominal-coverage check.
- `smfeval pair a b` — **no-reference calibration**: score two filters
  against each other (A→B Umeyama, difference under summed covariances,
  χ²₃). An elevated pairwise NEES certifies overconfidence with no
  ground truth consulted; the statistic lower-bounds the miscalibration
  (propriety caveat printed with every verdict). Ported from the
  slam_benchmark audit and verified exact against it.
- `smfeval validate --strict` — mechanical exporter gate: per-row
  covariance SPD, plausible magnitude, not degenerate-zero, finite poses.

### On-ramps

- TUM + covariance escape hatch for estimates: wide TUM (29 columns,
  pose + lower-triangle covariance, SQUARE packing) or plain TUM with a
  `--cov` sidecar file; conventions declared by flags and echoed.
- `exporters/` — the four audited belief exporters (FAST-LIO2,
  Faster-LIO, Point-LIO, I2EKF-LO): the diff, the pinned upstream
  commit, a bag→SQUARE converter, and a validation run each.
- Colab notebook (`notebooks/figure1_verdict.ipynb`) reproducing the
  headline verdict on an Oxford Spires sequence end to end.

### Scoring

- Short-window relative translation CRPS and windowed NEES calibration
  (`--rpe-window`): restores sensitivity to local σ calibration that
  absolute-pose CRPS loses in the overconfident regime.
- Exact log-score calibration/sharpness split with two-sided ANEES χ²
  verdicts per slice (`--calibration`).
- Track-frame bias/variance decomposition (along/cross/vertical) and
  structured failure-mode diagnoses with recommended actions.
- Student-t belief-transform intervention (`--student-t`), per-scan ESS
  covariance inflation (`--ess-inflate`), GT-covariance folding
  (`--consume-gt-cov`).
- Fixed catastrophic cancellation in the SE(3) V-jacobian small-angle
  branch (exp/log round-trip error 5e-9 → <5e-15).

### Testing

- Property-based invariant tests (hypothesis) across packing, CRPS,
  NEES frame invariance, pairing, alignment, and summaries.
- Seeded Monte Carlo power tests of the verdict machinery: ANEES
  false-positive rate at nominal α, power against covariance
  understatement, pairwise dilution laws, coverage separation.
- Real-data regression fixtures from Oxford Spires christ-church-03
  (CC BY-NC-SA 4.0, attributed), including a no-reference pair scenario.
- CI test + lint workflow (previously only tag-publish existed).

### Packaging

- Package renamed `src` → `smfeval` (a PyPI wheel must not install a
  top-level `src`).
- Version 0.4.0 follows 0.3.0 directly; the distributional-ground-truth
  work formerly tagged 0.4 in planning docs is renumbered to 0.5
  (`plans/0.5-distributional-gt.md`).

## 0.3.0

Closed-form Gaussian CRPS, adaptive SO(3) CRPS with jackknife SE,
tangent-Gaussian validity diagnostic.
