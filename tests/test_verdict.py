"""Verdict block: the covariance scale gap and its per-axis factor.

k = median NEES / chi2_median(dof). A published Sigma = c*Sigma_true gives
k = 1/c, so the per-axis sigma is off by sqrt(c): sqrt(k) when too tight,
1/sqrt(k) when too loose.
"""

import numpy as np
from scipy.stats import chi2

from smfeval.report.verdict import nees_verdict, render_nees_verdict

_CHI2_MED_3 = float(chi2.ppf(0.5, df=3))


def _verdict_with_k(k: float, n: int = 400):
  """A NEES series whose median is exactly k * chi2_median(3)."""
  nees = np.full(n, k * _CHI2_MED_3)
  return nees_verdict(nees, dof=3)


def test_too_tight_reports_sqrt_k_per_axis():
  v = _verdict_with_k(441.0)
  assert v.scale_direction == "too tight"
  assert v.per_axis_factor == 21.0
  assert "~21x too tight per axis" in render_nees_verdict(v)


def test_too_loose_reports_inverse_sqrt_k_per_axis():
  """Sigma 4x too large is 2x too loose per axis, not 0.5x.

  k = 0.25 means the published variance is 4x the honest one; reporting
  sqrt(k) = 0.5 understated the miscalibration and read as a factor
  *smaller* than one in a direction the same line called "too loose".
  """
  v = _verdict_with_k(0.25)
  assert v.scale_direction == "too loose"
  assert v.per_axis_factor == 2.0
  assert "~2x too loose per axis" in render_nees_verdict(v)


def test_per_axis_factor_is_never_below_one():
  for k in (1e-6, 0.1, 0.5, 1.0, 2.0, 1e6):
    assert _verdict_with_k(k).per_axis_factor >= 1.0


def test_degenerate_k_is_undefined_not_a_factor():
  v = _verdict_with_k(0.0)
  assert v.scale_direction == "undefined"
  assert np.isnan(v.per_axis_factor)


def test_consistent_verdict_prints_no_direction():
  """Within the ANEES interval the gap line reports k without a direction."""
  rng = np.random.default_rng(0)
  v = nees_verdict(rng.chisquare(3, size=2000), dof=3)
  assert v.anees.verdict == "consistent"
  out = render_nees_verdict(v)
  assert "covariance scale consistent" in out
  assert "per axis" not in out
