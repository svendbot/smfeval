import numpy as np
import pytest
from scipy.stats import chi

from smfeval.format import TangentConvention, TangentOrder, WeightFormat
from smfeval.scoring import (
  calibrate,
  energy_score,
  ensemble_diagnostics,
  gaussian_log_score,
  gaussian_log_score_components,
  interval_score,
  translation_components,
  translation_crps,
)
from smfeval.scoring.interval import interval_from_samples
from smfeval.se3.lie import se3_exp
from smfeval.se3.quat import rot_to_quat_xyzw
from smfeval.steps import EnsembleStep, GaussianStep
from smfeval.sync.risk import _trans_sigma
from tests._factories import gauss_step as _gauss

RNG = np.random.default_rng(11)


def test_translation_crps_decreases_with_better_predictive():
  ref = np.array([0.0, 0.0, 0.0])
  step_good = _gauss(0.0, ref, 0.01)
  step_bad = _gauss(0.0, ref + 1.0, 0.01)
  g = translation_crps(step_good, ref)
  b = translation_crps(step_bad, ref)
  assert g < b


def test_energy_score_finite():
  step = _gauss(0.0, np.zeros(3), 0.1)
  s = energy_score(
    step,
    np.array([0.05, 0.0, 0.0]),
    n_samples=64,
    rng=np.random.default_rng(0),
  )
  assert np.isfinite(s)


def test_interval_score_zero_width_when_collapsed():
  s = interval_score(0.5, 0.5, 0.5, alpha=0.1)
  assert s == 0.0


def test_interval_score_penalizes_outside():
  s_inside = interval_score(0.0, 1.0, 0.5, alpha=0.1)
  s_outside = interval_score(0.0, 1.0, 2.0, alpha=0.1)
  assert s_outside > s_inside


def test_interval_from_samples_brackets_central_mass():
  samples = np.linspace(-1.0, 1.0, 1001)
  lo, hi = interval_from_samples(samples, alpha=0.1)
  assert lo < 0 < hi
  assert abs(lo - (-0.9)) < 0.01
  assert abs(hi - 0.9) < 0.01


def test_gaussian_log_score_improves_with_centered_truth():
  ref = np.array([0.1, 0.0, 0.0])
  ref_q = np.array([0.0, 0.0, 0.0, 1.0])
  step_centered = _gauss(0.0, ref, 0.01)
  step_off = _gauss(0.0, ref + 1.0, 0.01)
  s_c = gaussian_log_score(step_centered, ref, ref_q)
  s_o = gaussian_log_score(step_off, ref, ref_q)
  assert s_c.translation < s_o.translation


def test_ensemble_diagnostics_uniform_weights():
  n = 100
  particles = np.zeros((n, 7))
  particles[:, :3] = RNG.normal(size=(n, 3))
  particles[:, 6] = 1.0
  weights = np.ones(n) / n
  steps = [
    EnsembleStep(timestamp=t, particles=particles, weights=weights)
    for t in (0.0, 1.0)
  ]
  diag = ensemble_diagnostics(steps, WeightFormat.LINEAR, normalized=True)
  assert np.allclose(diag.n_eff, n, atol=1e-6)
  assert diag.degeneracy_fraction == 0.0


def test_ensemble_diagnostics_log_weights_collapse():
  """One particle dominant under log weights ⇒ N_eff ≈ 1."""
  n = 50
  particles = np.zeros((n, 7))
  particles[:, 6] = 1.0
  log_w = np.full(n, -100.0)
  log_w[0] = 0.0  # one dominant
  steps = [EnsembleStep(timestamp=0.0, particles=particles, weights=log_w)]
  diag = ensemble_diagnostics(steps, WeightFormat.LOG, normalized=False)
  assert diag.n_eff[0] < 1.5


def test_ensemble_unique_count():
  particles = np.zeros((10, 7))
  particles[:5, :3] = 0.0  # cluster at origin
  particles[5:, :3] = 1.0  # cluster at (1,1,1)
  particles[:, 6] = 1.0
  step = EnsembleStep(timestamp=0.0, particles=particles, weights=np.ones(10))
  diag = ensemble_diagnostics(
    [step], WeightFormat.LINEAR, normalized=True, tol=1e-3
  )
  assert diag.n_unique[0] == 2


def _draw_calibrated_pair(mu_t, mu_q, cov, rng):
  """Sample (ref_t, ref_q) from the predictive defined by (mu_t, mu_q, cov)
  under right-perturbation: T_obs = T_mean · Exp(ξ), ξ ~ N(0, Σ).
  """
  L = np.linalg.cholesky(cov)
  xi = L @ rng.standard_normal(6)
  from smfeval.se3.lie import pose_matrix

  T_mean = pose_matrix(mu_t, mu_q)
  T_obs = T_mean @ se3_exp(xi, order=TangentOrder.TRANS_ROT)
  ref_t = T_obs[:3, 3]
  ref_q = rot_to_quat_xyzw(T_obs[:3, :3])
  return ref_t, ref_q


def test_calibration_matches_nominal_when_data_is_drawn_from_predictive():
  """End-to-end check that calibrate() reports the right thing when the reference
  is sampled from the predictive Gaussian. Failures here indicate a bug in
  smfeval itself (the residual, whitening, or coverage pipeline), not in
  any algorithm being scored."""
  rng = np.random.default_rng(42)
  n = 600
  sigma_t = 0.05  # 5 cm — small enough that V(w)≈I in se3_exp
  sigma_r = 0.01  # ~0.6° — keeps translation/rotation coupling tiny
  cov = np.diag([sigma_t**2] * 3 + [sigma_r**2] * 3)

  steps = []
  ref_ts = []
  ref_qs = []
  for _ in range(n):
    # Predictive mean — random pose, doesn't matter where.
    mu_t = rng.normal(size=3) * 5.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    ref_t, ref_q = _draw_calibrated_pair(mu_t, mu_q, cov, rng)
    ref_ts.append(ref_t)
    ref_qs.append(ref_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, cov.copy()))

  res = calibrate(
    steps,
    np.array(ref_ts),
    np.array(ref_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
  )

  # Coverage: nominal 0.9; binomial SD on n=600 is ≈0.012, so ±4% is loose.
  assert 0.86 < res.coverage < 0.94, f"coverage {res.coverage}"

  # Coverage matches nominal, so the binomial test should not reject.
  assert res.n_coverage == n
  assert res.coverage_p > 0.01, f"coverage p={res.coverage_p}"

  # z_translation = ‖L^-1 ρ‖ / √3. Under correct calibration ρ ~ N(0, Σ_t),
  # so ‖z‖ has chi(df=3) distribution; mean ≈ 0.921, std ≈ 0.390 after /√3.
  expected_mean = chi.mean(3) / np.sqrt(3)
  expected_std = chi.std(3) / np.sqrt(3)
  se_mean = expected_std / np.sqrt(n)
  assert abs(res.z_translation_mean - expected_mean) < 4 * se_mean, (
    f"z_mean {res.z_translation_mean} expected ~{expected_mean}"
  )
  assert 0.7 * expected_std < res.z_translation_std < 1.3 * expected_std, (
    f"z_std {res.z_translation_std} expected ~{expected_std}"
  )


def test_calibration_coverage_is_anisotropic_via_mahalanobis():
  """Calibrated reference drawn from an anisotropic Gaussian (σ_z 100× σ_xy) must
  hit 90% coverage. An isotropic-ball coverage check would over-cover here
  because the ball radius is set by the wide axis, eating typical
  tight-axis residuals; the proper Mahalanobis ellipsoid handles it."""
  rng = np.random.default_rng(7)
  n = 600
  sigma_xy = 0.01
  sigma_z = 1.0
  sigma_r = 0.001
  cov = np.diag(
    [sigma_xy**2, sigma_xy**2, sigma_z**2, sigma_r**2, sigma_r**2, sigma_r**2]
  )

  steps = []
  ref_ts = []
  ref_qs = []
  for _ in range(n):
    mu_t = rng.normal(size=3) * 5.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    ref_t, ref_q = _draw_calibrated_pair(mu_t, mu_q, cov, rng)
    ref_ts.append(ref_t)
    ref_qs.append(ref_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, cov.copy()))

  res = calibrate(
    steps,
    np.array(ref_ts),
    np.array(ref_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
  )
  assert 0.86 < res.coverage < 0.94, f"coverage {res.coverage} (expected ~0.9)"


def test_calibration_collapses_when_predictive_is_overconfident():
  """If we shrink the reported covariance by 1e6 while the truth still
  deviates by σ_t, coverage must drop to ~0 and z must blow up. Mirrors the
  failure mode observed when scoring filters that report Cramér–Rao-tight
  covariance against real-world drift."""
  rng = np.random.default_rng(7)
  n = 200
  sigma_t = 0.1
  sigma_r = 0.01
  truth_cov = np.diag([sigma_t**2] * 3 + [sigma_r**2] * 3)
  reported_cov = truth_cov / 1.0e6  # ~1000x tighter σ

  steps = []
  ref_ts = []
  ref_qs = []
  for _ in range(n):
    mu_t = rng.normal(size=3) * 2.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    ref_t, ref_q = _draw_calibrated_pair(mu_t, mu_q, truth_cov, rng)
    ref_ts.append(ref_t)
    ref_qs.append(ref_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, reported_cov.copy()))

  res = calibrate(
    steps,
    np.array(ref_ts),
    np.array(ref_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
  )
  assert res.coverage < 0.05
  assert res.z_translation_mean > 100  # truth is hundreds of σ away


def test_calibration_runs_end_to_end():
  n = 30
  rng = np.random.default_rng(0)
  steps = []
  ref_t = []
  ref_q = []
  for _ in range(n):
    truth = rng.normal(size=3) * 0.5
    steps.append(_gauss(0.0, truth + rng.normal(size=3) * 0.1, 0.01))
    ref_t.append(truth)
    ref_q.append(np.array([0.0, 0.0, 0.0, 1.0]))
  res = calibrate(
    steps,
    np.array(ref_t),
    np.array(ref_q),
    tangent_order=TangentOrder.TRANS_ROT,
  )
  assert res.n_coverage == n
  assert 0.0 <= res.coverage <= 1.0
  assert np.isfinite(res.coverage_p)


def _draw_perturbed_pair(
  mu_t: np.ndarray,
  mu_q: np.ndarray,
  cov: np.ndarray,
  convention: TangentConvention,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  """Draw an observation from the belief in the given perturbation convention."""
  from smfeval.se3.lie import pose_matrix

  xi = np.linalg.cholesky(cov) @ rng.standard_normal(6)
  T_mean = pose_matrix(mu_t, mu_q)
  E = se3_exp(xi, order=TangentOrder.TRANS_ROT)
  T_obs = E @ T_mean if convention is TangentConvention.LEFT else T_mean @ E
  return T_obs[:3, 3], rot_to_quat_xyzw(T_obs[:3, :3])


@pytest.mark.parametrize(
  "convention", [TangentConvention.RIGHT, TangentConvention.LEFT]
)
def test_nees_is_calibrated_in_both_perturbation_conventions(convention):
  """A calibrated belief scores NEES ~ chi2_3 under either declared convention.

  The residual must be the perturbation the covariance is the covariance of:
  log(T_est^-1 T_ref) for right, log(T_ref T_est^-1) for left. Pairing one
  with the other inflates the NEES by the adjoint mismatch, which reads as
  over-confidence in a filter that is honest.
  """
  rng = np.random.default_rng(7)
  n = 4000
  # Anisotropic translation block and a mean pose well away from the origin,
  # so Ad_T is far from the identity and the two conventions cannot coincide.
  cov = np.diag([0.10**2, 0.01**2, 0.01**2, 1e-6, 1e-6, 1e-6])
  mu_t = np.array([10.0, -3.0, 1.0])
  mu_q = rot_to_quat_xyzw(se3_exp(np.array([0, 0, 0, 0.0, 0.0, 2.0]))[:3, :3])

  steps, ref_ts, ref_qs = [], [], []
  for _ in range(n):
    ref_t, ref_q = _draw_perturbed_pair(mu_t, mu_q, cov, convention, rng)
    ref_ts.append(ref_t)
    ref_qs.append(ref_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, cov.copy()))

  comps = translation_components(
    steps,
    np.array(ref_ts),
    np.array(ref_qs),
    TangentOrder.TRANS_ROT,
    convention,
  )
  nees = np.array([c.nees for c in comps])
  # chi2_3: mean 3, median 2.366. n=4000 keeps the sampling error well under 5%.
  assert 2.8 < nees.mean() < 3.2, f"ANEES {nees.mean()}"
  assert 2.2 < np.median(nees) < 2.6, f"median NEES {np.median(nees)}"


def test_left_perturbation_residual_differs_from_right():
  """Guard the fix: the two conventions must not silently be the same code path."""
  cov = np.diag([0.10**2, 0.01**2, 0.01**2, 1e-6, 1e-6, 1e-6])
  mu_t = np.array([10.0, -3.0, 1.0])
  mu_q = rot_to_quat_xyzw(se3_exp(np.array([0, 0, 0, 0.0, 0.0, 2.0]))[:3, :3])
  step = GaussianStep(0.0, mu_t, mu_q, cov)
  ref_t = mu_t + np.array([0.05, 0.02, -0.01])

  right = gaussian_log_score_components(
    step, ref_t, mu_q, TangentOrder.TRANS_ROT, TangentConvention.RIGHT
  ).translation.nees
  left = gaussian_log_score_components(
    step, ref_t, mu_q, TangentOrder.TRANS_ROT, TangentConvention.LEFT
  ).translation.nees
  assert not np.isclose(right, left)


def test_ensemble_sigma_reads_positive_log_weights_as_log():
  """Unnormalized log weights can be all-positive; the header decides, not the sign.

  Here the weight falls off with distance from the origin, so read as declared
  (LOG) the belief concentrates near the origin and the reported sigma is
  tight. Misread as LINEAR the same numbers normalize to nearly uniform and
  the sigma reverts to the raw particle spread -- the concentration the
  filter reported is thrown away.
  """
  rng = np.random.default_rng(3)
  particles = np.zeros((256, 7))
  particles[:, :3] = rng.normal(scale=0.5, size=(256, 3))
  particles[:, 6] = 1.0
  log_w = 12.0 - 5.0 * np.linalg.norm(particles[:, :3], axis=1)
  assert (log_w > 0).all(), "the point of the test is all-positive log weights"
  step = EnsembleStep(0.0, particles, log_w)

  as_log = _trans_sigma(step, TangentOrder.TRANS_ROT, WeightFormat.LOG, False)
  as_linear = _trans_sigma(
    step, TangentOrder.TRANS_ROT, WeightFormat.LINEAR, False
  )
  unweighted = float(np.sqrt(particles[:, :3].var(axis=0).mean()))

  assert as_log < 0.8 * as_linear
  # Misread as linear, the sigma is within 10% of the unweighted spread.
  assert as_linear == pytest.approx(unweighted, rel=0.1)
