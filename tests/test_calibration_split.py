"""A1: calibration/sharpness split of the log score + ANEES consistency.

Verifies the per-slice decomposition `-log p = calibration + sharpness`
(calibration = ½·NEES, sharpness = ½(log|Σ| + d·log2π)) and the two-sided χ²
ANEES verdict (optimistic | consistent | conservative).
"""

import numpy as np
from scipy.stats import chi2

from smfeval.scoring.logscore import (
  anees_consistency,
  gaussian_log_score,
  gaussian_log_score_components,
  student_t_neg_log_density,
)
from smfeval.scoring.relative import relative_calibration
from smfeval.steps import GaussianStep


def _gauss_neglogp(xi, cov):
  d = cov.shape[0]
  _, logdet = np.linalg.slogdet(cov)
  return 0.5 * (xi @ np.linalg.solve(cov, xi) + logdet + d * np.log(2 * np.pi))


def test_student_t_approaches_gaussian_as_nu_large():
  xi = np.array([0.3, -0.1, 0.2, 0.01, 0.0, -0.02])
  cov = np.diag([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
  t_big = student_t_neg_log_density(xi, cov, nu=1e7)
  assert np.isclose(t_big, _gauss_neglogp(xi, cov), atol=1e-3)


def test_student_t_lighter_on_tail_outliers():
  # Truth many σ out: the heavy-tailed belief assigns higher density (lower
  # -log p) than the Gaussian — the robustness the intervention exploits.
  cov = np.eye(6) * 0.01
  xi = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # ~10σ per axis
  assert student_t_neg_log_density(xi, cov, nu=3.0) < _gauss_neglogp(xi, cov)


def test_student_t_non_pd_is_inf():
  cov = np.diag([-1.0, 0.05, 0.05, 0.01, 0.01, 0.01])
  assert student_t_neg_log_density(np.ones(6), cov, nu=5.0) == float("inf")


RNG = np.random.default_rng(7)


def _gauss(t: np.ndarray, cov_diag: float | np.ndarray) -> GaussianStep:
  cov = np.eye(6) * cov_diag if np.isscalar(cov_diag) else np.diag(cov_diag)
  return GaussianStep(
    timestamp=0.0,
    translation=t,
    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    covariance=cov,
  )


def test_components_sum_to_log_score_and_match_neg_log_density():
  gt = np.zeros(3)
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  step = _gauss(np.array([0.3, -0.2, 0.1]), 0.05)
  dec = gaussian_log_score_components(step, gt, gt_q)
  ref = gaussian_log_score(step, gt, gt_q)
  slice_ = dec.translation
  assert np.isclose(slice_.calibration + slice_.sharpness, slice_.log_score)
  assert np.isclose(slice_.calibration, 0.5 * slice_.nees)
  # log_score equals the existing neg-log-density on the translation block.
  assert np.isclose(dec.translation.log_score, ref.translation)
  assert dec.translation.dof == 3


def test_calibration_zero_when_truth_at_mean():
  gt = np.zeros(3)
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  dec = gaussian_log_score_components(_gauss(gt, 0.05), gt, gt_q)
  assert np.isclose(dec.translation.nees, 0.0, atol=1e-9)
  assert np.isclose(dec.translation.calibration, 0.0, atol=1e-9)
  # with zero error the score IS the sharpness penalty.
  assert np.isclose(dec.translation.log_score, dec.translation.sharpness)


def test_shrinking_cov_raises_calibration_lowers_sharpness():
  gt = np.zeros(3)
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  off = np.array([0.4, 0.0, 0.0])
  wide = gaussian_log_score_components(_gauss(off, 0.1), gt, gt_q).translation
  tight = gaussian_log_score_components(
    _gauss(off, 0.001), gt, gt_q
  ).translation
  assert tight.calibration > wide.calibration  # overconfident → NEES blows up
  assert tight.sharpness < wide.sharpness  # tighter Σ → smaller log-volume


def test_non_pd_covariance_is_inf():
  gt = np.zeros(3)
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  step = _gauss(np.array([0.1, 0.0, 0.0]), 0.05)
  step.covariance[0, 0] = -1.0  # break positive-definiteness
  comp = gaussian_log_score_components(step, gt, gt_q).translation
  assert comp.log_score == float("inf")
  assert comp.nees == float("inf")


def test_anees_consistent_for_calibrated_draws():
  dof = 6
  nees = chi2.rvs(df=dof, size=4000, random_state=RNG)
  res = anees_consistency(nees, dof=dof)
  assert res.verdict == "consistent"
  assert res.lo < res.anees < res.hi
  assert abs(res.anees - dof) < 0.5  # ANEES ≈ dof
  assert res.n == 4000


def test_anees_optimistic_when_overconfident():
  dof = 3
  # error variance 3× what Σ claims → NEES inflated → over-confident.
  nees = 3.0 * chi2.rvs(df=dof, size=2000, random_state=RNG)
  res = anees_consistency(nees, dof=dof)
  assert res.verdict == "optimistic"
  assert res.anees > res.hi


def test_anees_conservative_when_underconfident():
  dof = 3
  nees = 0.3 * chi2.rvs(df=dof, size=2000, random_state=RNG)
  res = anees_consistency(nees, dof=dof)
  assert res.verdict == "conservative"
  assert res.anees < res.lo


def test_anees_drops_nonfinite_values():
  dof = 6
  nees = chi2.rvs(df=dof, size=100, random_state=RNG)
  contaminated = np.concatenate([nees, [np.inf, np.nan, np.inf]])
  res = anees_consistency(contaminated, dof=dof)
  assert res.n == 100  # the 3 non-finite entries dropped
  assert np.isfinite(res.anees)


def test_anees_undefined_when_empty():
  res = anees_consistency(np.array([np.inf, np.nan]), dof=6)
  assert res.verdict == "undefined"
  assert res.n == 0


def _relative_track(scale: float) -> tuple[list[GaussianStep], np.ndarray]:
  """A track whose per-step Σ_t = 0.01·I; relative increment error injected
  at `scale`·σ_rel so the windowed (dof-3) NEES is controllable."""
  rng = np.random.default_rng(3)
  n = 400
  ts = np.linspace(0.0, 40.0, n)
  gt = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
  # Σ_rel over a 1-step window = Σ_i+Σ_j = 0.02·I → σ_rel = sqrt(0.02) per axis.
  sigma_rel = np.sqrt(0.02)
  est = gt + rng.normal(scale=scale * sigma_rel / np.sqrt(2.0), size=gt.shape)
  steps = [
    GaussianStep(
      timestamp=t,
      translation=p,
      quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
      covariance=np.diag([0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4]),
    )
    for t, p in zip(ts, est, strict=False)
  ]
  return steps, gt


def test_relative_calibration_consistent_at_unit_scale():
  steps, gt = _relative_track(scale=1.0)
  res = relative_calibration(steps, gt, windows_s=[0.1], alpha=0.01)
  assert len(res) == 1
  r = res[0]
  assert r.anees.dof == 3
  assert r.anees.n > 100
  assert r.anees.verdict == "consistent"


def test_relative_calibration_optimistic_when_overconfident():
  steps, gt = _relative_track(scale=3.0)
  res = relative_calibration(steps, gt, windows_s=[0.1], alpha=0.01)
  assert res[0].anees.verdict == "optimistic"
  assert res[0].anees.anees > res[0].anees.hi


def test_relative_calibration_requires_gaussian():
  import pytest

  from smfeval.steps import DeterministicStep

  steps = [
    DeterministicStep(
      timestamp=float(i),
      translation=np.array([i, 0.0, 0.0]),
      quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    for i in range(5)
  ]
  with pytest.raises(TypeError):
    relative_calibration(steps, np.zeros((5, 3)), windows_s=[1.0])
