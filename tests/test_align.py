import numpy as np

from src.align import align_mode_for_gauge, fit_alignment, propagate_step
from src.se3.lie import se3_exp
from src.se3.quat import rot_to_quat_xyzw
from src.steps import DeterministicStep, EnsembleStep, GaussianStep
from src.types import Gauge, TangentConvention, TangentOrder

RNG = np.random.default_rng(7)


def test_mode_for_gauge():
    assert align_mode_for_gauge(Gauge.FIXED) == "none"
    assert align_mode_for_gauge(Gauge.SE3) == "se3"
    assert align_mode_for_gauge(Gauge.GRAVITY_YAW) == "gravity_yaw"
    assert align_mode_for_gauge(Gauge.SIM3) == "sim3"


def test_se3_recovers_known_transform():
    pts = RNG.normal(size=(20, 3))
    T_true = se3_exp(np.array([1.0, -0.5, 0.2, 0.1, 0.2, 0.3]))
    R_true = T_true[:3, :3]
    t_true = T_true[:3, 3]
    gt = (R_true @ pts.T).T + t_true
    fit = fit_alignment(pts, gt, mode="se3")
    assert np.allclose(fit.transform, T_true, atol=1e-9)
    assert fit.scale == 1.0
    assert fit.dof_removed == 6
    assert np.max(fit.residuals) < 1e-9


def test_sim3_recovers_scale():
    pts = RNG.normal(size=(30, 3))
    T_true = se3_exp(np.array([0.5, 0.0, 0.1, 0.0, 0.0, 0.4]))
    R_true = T_true[:3, :3]
    t_true = T_true[:3, 3]
    s_true = 2.5
    gt = s_true * (R_true @ pts.T).T + t_true
    fit = fit_alignment(pts, gt, mode="sim3")
    assert abs(fit.scale - s_true) < 1e-6
    assert np.allclose(fit.fitted_rotation, R_true, atol=1e-6)
    assert fit.dof_removed == 7


def test_gravity_yaw_recovers_yaw_only():
    pts = RNG.normal(size=(40, 3))
    yaw = 0.7
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    d = np.array([0.3, -0.2, 1.1])
    gt = (R @ pts.T).T + d
    fit = fit_alignment(pts, gt, mode="gravity_yaw")
    assert np.allclose(fit.fitted_rotation, R, atol=1e-9)
    assert np.allclose(fit.fitted_translation, d, atol=1e-9)
    assert fit.dof_removed == 4


def test_gravity_yaw_ignores_pitch_roll_in_gt():
    """If gt has stray pitch/roll, gravity_yaw can't represent it; xy still fits."""
    pts = RNG.normal(size=(50, 3))
    yaw = 0.4
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    gt = (R_yaw @ pts.T).T + np.array([0.0, 0.0, 0.0])
    fit = fit_alignment(pts, gt, mode="gravity_yaw")
    assert np.median(fit.residuals) < 1e-6


def test_align_mode_none_identity():
    pts = RNG.normal(size=(5, 3))
    fit = fit_alignment(pts, pts + 0.5, mode="none")
    assert np.allclose(fit.transform, np.eye(4))
    assert fit.dof_removed == 0


def test_propagate_deterministic():
    T = se3_exp(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.5]))
    step = DeterministicStep(
        timestamp=1.0,
        translation=np.array([1.0, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    out = propagate_step(step, T)
    expected = T[:3, :3] @ step.translation + T[:3, 3]
    assert np.allclose(out.translation, expected)


def test_propagate_gaussian_right_perturbation_unchanged():
    T = se3_exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.3]))
    cov = np.diag([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    step = GaussianStep(
        timestamp=1.0,
        translation=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        covariance=cov,
    )
    out = propagate_step(
        step, T,
        tangent_convention=TangentConvention.RIGHT,
        tangent_order=TangentOrder.TRANS_ROT,
    )
    assert np.allclose(out.covariance, cov)


def test_propagate_gaussian_left_perturbation_uses_adjoint():
    T = se3_exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.3]))
    cov = np.diag([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    step = GaussianStep(
        timestamp=1.0,
        translation=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        covariance=cov,
    )
    out = propagate_step(
        step, T,
        tangent_convention=TangentConvention.LEFT,
        tangent_order=TangentOrder.TRANS_ROT,
    )
    # Should not equal cov (Adjoint isn't identity)
    assert not np.allclose(out.covariance, cov)
    # Symmetry preserved
    assert np.allclose(out.covariance, out.covariance.T, atol=1e-12)


def test_propagate_ensemble():
    T = se3_exp(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.5]))
    n = 4
    particles = np.zeros((n, 7))
    particles[:, :3] = RNG.normal(size=(n, 3))
    for i in range(n):
        R_p = se3_exp(np.array([0, 0, 0, 0, 0, 0.1 * i]))[:3, :3]
        particles[i, 3:] = rot_to_quat_xyzw(R_p)
    weights = np.ones(n) / n
    step = EnsembleStep(timestamp=1.0, particles=particles, weights=weights)
    out = propagate_step(step, T)
    expected_t = (T[:3, :3] @ particles[:, :3].T).T + T[:3, 3]
    assert np.allclose(out.particles[:, :3], expected_t)
    assert np.allclose(out.weights, weights)


def test_propagate_sim3_scales_translation():
    T = np.eye(4)
    T[:3, 3] = [10.0, 0.0, 0.0]
    s = 3.0
    step = DeterministicStep(
        timestamp=1.0,
        translation=np.array([1.0, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    out = propagate_step(step, T, scale=s)
    assert np.allclose(out.translation, [13.0, 0.0, 0.0])
