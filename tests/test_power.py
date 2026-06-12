"""Statistical power of the calibration verdicts.

These tests verify the *detection machinery itself*: that the ANEES chi2
verdict, the pairwise no-reference NEES, and the coverage test (a) hold
their false-positive rate at the nominal alpha when the belief is
calibrated, and (b) detect covariance understatement of a given factor k
with high probability at realistic trajectory lengths.

Generative model: errors e_t ~ N(0, Sigma_true) scored under a published
Sigma_pub = Sigma_true / k, so the per-pose NEES is k * chi2_d. All
Monte Carlo loops are seeded and the acceptance thresholds leave >= 3
binomial sigmas of margin — if a seed change ever trips one, the power
itself regressed (do not widen the bounds).
"""

import numpy as np
import pytest
from scipy.stats import chi2

from smfeval.scoring.logscore import anees_consistency

_DOF = 3
_ALPHA = 0.05
_CHI2_MEDIAN_3 = float(chi2.ppf(0.5, df=_DOF))


def _verdict_rate(
  rng: np.random.Generator, k: float, n: int, reps: int
) -> float:
  """Fraction of reps where the ANEES verdict is 'optimistic'."""
  draws = k * rng.chisquare(_DOF, size=(reps, n))
  hits = sum(
    anees_consistency(row, dof=_DOF, alpha=_ALPHA).verdict == "optimistic"
    for row in draws
  )
  return hits / reps


def test_anees_false_positive_rate_at_nominal_alpha():
  # Two-sided test at alpha=0.05: under k=1 either verdict != consistent
  # counts as a false alarm. 3-sigma binomial band around 0.05 at R=1000.
  rng = np.random.default_rng(20260612)
  reps, n = 1000, 200
  draws = rng.chisquare(_DOF, size=(reps, n))
  false_alarms = sum(
    anees_consistency(row, dof=_DOF, alpha=_ALPHA).verdict != "consistent"
    for row in draws
  )
  rate = false_alarms / reps
  band = 3.0 * np.sqrt(_ALPHA * (1 - _ALPHA) / reps)
  assert abs(rate - _ALPHA) < band, (
    f"FPR {rate:.3f} outside {_ALPHA}+-{band:.3f}"
  )


def test_anees_power_against_understated_covariance():
  # n=200 poses: analytic power is ~0.89 at k=1.2 and ~1 at k=1.5.
  rng = np.random.default_rng(7)
  reps, n = 500, 200
  p11 = _verdict_rate(rng, 1.1, n, reps)
  p12 = _verdict_rate(rng, 1.2, n, reps)
  p15 = _verdict_rate(rng, 1.5, n, reps)
  assert p12 >= 0.8, f"power(k=1.2) = {p12:.3f} < 0.8"
  assert p15 >= 0.99, f"power(k=1.5) = {p15:.3f} < 0.99"
  assert p11 <= p12 <= p15, f"power not monotone: {p11}, {p12}, {p15}"


def _pair_nees(rng: np.random.Generator, k: float, n: int) -> np.ndarray:
  """Simulated pairwise NEES: A understates Sigma by k, B is calibrated."""
  a = rng.normal(size=(6, 6))
  sigma = a @ a.T + np.eye(6)
  sigma = sigma[:_DOF, :_DOF]
  L = np.linalg.cholesky(sigma)
  e_a = (L @ rng.normal(size=(_DOF, n))).T
  e_b = (L @ rng.normal(size=(_DOF, n))).T
  d = e_a - e_b
  sigma_eff = sigma / k + sigma  # published A + published(=true) B
  sol = np.linalg.solve(sigma_eff, d.T)
  return np.einsum("ij,ji->i", d, sol)


def test_pairwise_dilution_matches_analytic_scale():
  # d ~ N(0, 2 Sigma) scored under Sigma(1+k)/k gives NEES distributed as
  # (2k/(k+1)) chi2_3 — the quantitative form of the lower-bound property.
  rng = np.random.default_rng(11)
  n = 200_000
  for k in (1.0, 2.0, 5.0):
    nees = _pair_nees(rng, k, n)
    expected_mean = _DOF * 2.0 * k / (k + 1.0)
    # mean of c*chi2_3 over n draws: sd = c*sqrt(2*3/n)
    sd = (2.0 * k / (k + 1.0)) * np.sqrt(2.0 * _DOF / n)
    assert abs(nees.mean() - expected_mean) < 4.0 * sd


def test_pairwise_false_positive_rate_when_both_calibrated():
  rng = np.random.default_rng(13)
  reps, n = 500, 300
  false_alarms = sum(
    anees_consistency(_pair_nees(rng, 1.0, n), dof=_DOF).verdict != "consistent"
    for _ in range(reps)
  )
  rate = false_alarms / reps
  band = 3.0 * np.sqrt(_ALPHA * (1 - _ALPHA) / reps)
  assert abs(rate - _ALPHA) < band, f"pair FPR {rate:.3f} outside band"


def test_pairwise_power_despite_dilution():
  # k=2 dilutes to effective scale 4/3, yet n=300 gives power ~1; the
  # no-reference verdict survives the common-mode cancellation penalty.
  rng = np.random.default_rng(17)
  reps, n = 500, 300
  hits = sum(
    anees_consistency(_pair_nees(rng, 2.0, n), dof=_DOF).verdict == "optimistic"
    for _ in range(reps)
  )
  assert hits / reps >= 0.9, f"pair power {hits / reps:.3f} < 0.9"


def test_coverage_test_power():
  # 90% ellipsoid coverage under k=2 drops to P(chi2_3 <= q90/2) ~ 0.63;
  # at n=500 the 3-sigma binomial CI around the empirical coverage must
  # exclude 0.9, while the calibrated case must include it.
  rng = np.random.default_rng(19)
  n = 500
  q90 = chi2.ppf(0.9, df=_DOF)

  calibrated = rng.chisquare(_DOF, size=n) <= q90
  cov_cal = calibrated.mean()
  band_cal = 3.0 * np.sqrt(cov_cal * (1 - cov_cal) / n)
  assert abs(cov_cal - 0.9) < band_cal

  overconfident = 2.0 * rng.chisquare(_DOF, size=n) <= q90
  cov_over = overconfident.mean()
  band_over = 3.0 * np.sqrt(cov_over * (1 - cov_over) / n)
  assert cov_over + band_over < 0.9, (
    f"coverage test failed to separate: {cov_over:.3f} + {band_over:.3f}"
  )
  expected = chi2.cdf(q90 / 2.0, df=_DOF)
  assert abs(cov_over - expected) < band_over


@pytest.mark.parametrize("k", [4.0, 100.0])
def test_k_estimate_recovers_understatement_factor(k):
  # The nees verb's headline: k_hat = median NEES / chi2-median(3).
  # Median estimation noise at n=500 is ~4.5% relative; +-15% is >3 sigma.
  rng = np.random.default_rng(int(k))
  nees = k * rng.chisquare(_DOF, size=500)
  k_hat = np.median(nees) / _CHI2_MEDIAN_3
  assert abs(k_hat - k) / k < 0.15, f"k_hat {k_hat:.3g} vs k {k}"
