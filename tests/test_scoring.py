import numpy as np
from scipy.stats import chi

from smfeval.format import TangentOrder, WeightFormat
from smfeval.scoring import (
  calibrate,
  energy_score,
  ensemble_diagnostics,
  gaussian_log_score,
  interval_score,
  rotation_crps,
  translation_crps,
)
from smfeval.scoring.interval import interval_from_samples
from smfeval.se3.lie import se3_exp
from smfeval.se3.quat import rot_to_quat_xyzw
from smfeval.steps import EnsembleStep, GaussianStep

RNG = np.random.default_rng(11)


def _gauss(
  ts: float, t: np.ndarray, cov_diag: float | np.ndarray
) -> GaussianStep:
  cov = np.eye(6) * cov_diag if np.isscalar(cov_diag) else np.diag(cov_diag)
  return GaussianStep(
    timestamp=ts,
    translation=t,
    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    covariance=cov,
  )


def test_translation_crps_decreases_with_better_predictive():
  gt = np.array([0.0, 0.0, 0.0])
  step_good = _gauss(0.0, gt, 0.01)
  step_bad = _gauss(0.0, gt + 1.0, 0.01)
  g = translation_crps(step_good, gt)
  b = translation_crps(step_bad, gt)
  assert g < b


def test_rotation_crps_zero_when_concentrated_at_truth():
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  step = GaussianStep(
    timestamp=0.0,
    translation=np.zeros(3),
    quat_xyzw=gt_q,
    covariance=np.eye(6) * 1e-8,
  )
  s = rotation_crps(step, gt_q, rng=np.random.default_rng(0))
  assert s < 1e-3


def test_energy_score_finite():
  step = _gauss(0.0, np.zeros(3), 0.1)
  s = energy_score(
    step,
    np.array([0.05, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0]),
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
  gt = np.array([0.1, 0.0, 0.0])
  gt_q = np.array([0.0, 0.0, 0.0, 1.0])
  step_centered = _gauss(0.0, gt, 0.01)
  step_off = _gauss(0.0, gt + 1.0, 0.01)
  s_c = gaussian_log_score(step_centered, gt, gt_q)
  s_o = gaussian_log_score(step_off, gt, gt_q)
  assert s_c.joint < s_o.joint
  assert s_c.translation < s_o.translation
  # Rotation residual is identical for the two — only translation moved —
  # so the rotation-marginal score should not separate them.
  assert s_c.rotation == s_o.rotation


def test_gaussian_log_score_rotation_marginal_isolates_yaw_miscalibration():
  """Translation block is well-calibrated but rotation block is shrunk 100×.
  The joint score is dragged down; the rotation marginal must spike."""
  gt_t = np.zeros(3)
  # Non-zero rotation residual against an over-confident rotation block:
  # tight Σ_rr makes the marginal NLL spike at xi_r ≠ 0.
  from smfeval.se3.lie import so3_exp
  from smfeval.se3.quat import rot_to_quat_xyzw

  R_obs = so3_exp(np.array([0.1, 0.0, 0.0]))  # ~5.7° error
  gt_q_off = rot_to_quat_xyzw(R_obs)

  cov_good = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
  cov_overconfident_rot = cov_good.copy()
  cov_overconfident_rot[3:, 3:] *= 1e-4

  step_good = GaussianStep(0.0, gt_t, np.array([0.0, 0.0, 0.0, 1.0]), cov_good)
  step_overconfident = GaussianStep(
    0.0, gt_t, np.array([0.0, 0.0, 0.0, 1.0]), cov_overconfident_rot
  )
  s_good = gaussian_log_score(step_good, gt_t, gt_q_off)
  s_bad = gaussian_log_score(step_overconfident, gt_t, gt_q_off)
  assert s_bad.rotation > s_good.rotation
  # Translation cov untouched → translation marginal essentially unchanged.
  assert abs(s_bad.translation - s_good.translation) < 1e-9


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
  """Sample (gt_t, gt_q) from the predictive defined by (mu_t, mu_q, cov)
  under right-perturbation: T_obs = T_mean · Exp(ξ), ξ ~ N(0, Σ).
  """
  L = np.linalg.cholesky(cov)
  xi = L @ rng.standard_normal(6)
  from smfeval.se3.lie import pose_matrix

  T_mean = pose_matrix(mu_t, mu_q)
  T_obs = T_mean @ se3_exp(xi, order=TangentOrder.TRANS_ROT)
  gt_t = T_obs[:3, 3]
  gt_q = rot_to_quat_xyzw(T_obs[:3, :3])
  return gt_t, gt_q


def test_calibration_matches_nominal_when_data_is_drawn_from_predictive():
  """End-to-end check that calibrate() reports the right thing when the GT
  is sampled from the predictive Gaussian. Failures here indicate a bug in
  smfeval itself (sampling, scoring, or the PIT/coverage pipeline), not in
  any algorithm being scored."""
  rng = np.random.default_rng(42)
  n = 600
  sigma_t = 0.05  # 5 cm — small enough that V(w)≈I in se3_exp
  sigma_r = 0.01  # ~0.6° — keeps translation/rotation coupling tiny
  cov = np.diag([sigma_t**2] * 3 + [sigma_r**2] * 3)

  steps = []
  gt_ts = []
  gt_qs = []
  for _ in range(n):
    # Predictive mean — random pose, doesn't matter where.
    mu_t = rng.normal(size=3) * 5.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    gt_t, gt_q = _draw_calibrated_pair(mu_t, mu_q, cov, rng)
    gt_ts.append(gt_t)
    gt_qs.append(gt_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, cov.copy()))

  res = calibrate(
    steps,
    np.array(gt_ts),
    np.array(gt_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
    n_samples=512,
    rng=np.random.default_rng(0),
  )

  # Coverage: nominal 0.9; binomial SD on n=600 is ≈0.012, so ±4% is loose.
  assert 0.86 < res.coverage < 0.94, f"coverage {res.coverage}"

  # PIT under correct calibration is U(0,1); KS p should not be tiny.
  assert res.ks_p_translation > 0.01, f"KS_t p={res.ks_p_translation}"
  assert res.ks_p_rotation > 0.01, f"KS_r p={res.ks_p_rotation}"

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
  """Calibrated GT drawn from an anisotropic Gaussian (σ_z 100× σ_xy) must
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
  gt_ts = []
  gt_qs = []
  for _ in range(n):
    mu_t = rng.normal(size=3) * 5.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    gt_t, gt_q = _draw_calibrated_pair(mu_t, mu_q, cov, rng)
    gt_ts.append(gt_t)
    gt_qs.append(gt_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, cov.copy()))

  res = calibrate(
    steps,
    np.array(gt_ts),
    np.array(gt_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
    n_samples=256,
    rng=np.random.default_rng(0),
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
  gt_ts = []
  gt_qs = []
  for _ in range(n):
    mu_t = rng.normal(size=3) * 2.0
    mu_q = rot_to_quat_xyzw(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    gt_t, gt_q = _draw_calibrated_pair(mu_t, mu_q, truth_cov, rng)
    gt_ts.append(gt_t)
    gt_qs.append(gt_q)
    steps.append(GaussianStep(0.0, mu_t, mu_q, reported_cov.copy()))

  res = calibrate(
    steps,
    np.array(gt_ts),
    np.array(gt_qs),
    tangent_order=TangentOrder.TRANS_ROT,
    alpha=0.1,
    n_samples=128,
    rng=np.random.default_rng(0),
  )
  assert res.coverage < 0.05
  assert res.z_translation_mean > 100  # truth is hundreds of σ away


def test_calibration_runs_end_to_end():
  n = 30
  rng = np.random.default_rng(0)
  steps = []
  gt_t = []
  gt_q = []
  for _ in range(n):
    truth = rng.normal(size=3) * 0.5
    steps.append(_gauss(0.0, truth + rng.normal(size=3) * 0.1, 0.01))
    gt_t.append(truth)
    gt_q.append(np.array([0.0, 0.0, 0.0, 1.0]))
  res = calibrate(
    steps,
    np.array(gt_t),
    np.array(gt_q),
    tangent_order=TangentOrder.TRANS_ROT,
    n_samples=64,
    rng=np.random.default_rng(0),
  )
  assert res.pit_translation.shape == (n,)
  assert 0.0 <= res.coverage <= 1.0
  assert np.isfinite(res.ks_p_translation)
