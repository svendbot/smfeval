"""Property-based tests of the scoring/IO/alignment math invariants.

Each test states an exact mathematical invariant of the implementation
(scale equivariance, frame invariance, decomposition identities, packing
round trips) and lets hypothesis search for counterexamples. Tolerances
are float-roundoff-sized, not statistical.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from smfeval.align import fit_alignment, propagate_step
from smfeval.format import TangentConvention, TangentOrder
from smfeval.io.reader import _expand_lower_triangular
from smfeval.scoring.bias_variance import _AXES, bias_variance
from smfeval.scoring.crps import _gaussian_crps
from smfeval.scoring.logscore import (
  _gaussian_neg_log_density,
  anees_consistency,
  gaussian_log_score_components,
  student_t_neg_log_density,
)
from smfeval.scoring.relative import _window_pairs, relative_translation_crps
from smfeval.scoring.summary import summarize
from smfeval.se3.lie import homogeneous, se3_exp
from smfeval.se3.quat import rot_to_quat_xyzw
from smfeval.steps import DeterministicStep, GaussianStep

_FINITE = {"allow_nan": False, "allow_infinity": False}


def _floats(lo: float, hi: float):
  return st.floats(min_value=lo, max_value=hi, **_FINITE)


def _vec(n: int, lo: float = -10.0, hi: float = 10.0):
  return st.lists(_floats(lo, hi), min_size=n, max_size=n).map(np.array)


@st.composite
def spd6(draw, scale_lo: float = 1e-2, scale_hi: float = 10.0):
  """Well-conditioned SPD 6x6 via a Gram matrix plus identity ridge."""
  a = draw(_vec(36, -1.0, 1.0)).reshape(6, 6)
  s = draw(_floats(scale_lo, scale_hi))
  return s * (a @ a.T + np.eye(6))


@st.composite
def pose(draw, t_lo: float = -10.0, t_hi: float = 10.0):
  """Random SE(3) pose as (translation, quat_xyzw), angle < pi."""
  rotvec = draw(_vec(3, -1.0, 1.0))
  norm = np.linalg.norm(rotvec)
  assume(norm < np.pi - 1e-3)
  t = draw(_vec(3, t_lo, t_hi))
  q = Rotation.from_rotvec(rotvec).as_quat()
  return t, q


# P1 — SPD lower-triangle packing round trip ---------------------------------


@given(entries=_vec(21, -5.0, 5.0))
def test_lower_triangle_pack_unpack_round_trip(entries):
  cov = _expand_lower_triangular(list(entries))
  assert np.array_equal(cov, cov.T)
  repacked = [cov[i, j] for i in range(6) for j in range(i + 1)]
  assert np.array_equal(np.array(repacked), entries)


# P2 — Gaussian CRPS closed form ---------------------------------------------


@given(mu=_floats(-50, 50), sigma=_floats(1e-3, 1e3), y=_floats(-50, 50))
def test_gaussian_crps_nonnegative(mu, sigma, y):
  crps = _gaussian_crps(np.array([mu]), np.array([sigma]), np.array([y]))
  assert crps[0] >= -1e-12


@given(
  mu=_floats(-50, 50),
  sigma=_floats(1e-3, 1e3),
  y=_floats(-50, 50),
  c=_floats(1e-2, 1e2),
)
def test_gaussian_crps_scale_equivariant(mu, sigma, y, c):
  base = _gaussian_crps(np.array([mu]), np.array([sigma]), np.array([y]))
  scaled = _gaussian_crps(
    np.array([c * mu]), np.array([c * sigma]), np.array([c * y])
  )
  assert np.isclose(scaled[0], c * base[0], rtol=1e-9, atol=1e-12)


@given(mu=_floats(-50, 50), sigma=_floats(1e-3, 1e3))
def test_gaussian_crps_at_mean_closed_form(mu, sigma):
  crps = _gaussian_crps(np.array([mu]), np.array([sigma]), np.array([mu]))
  expected = sigma * (2.0 / np.sqrt(2.0 * np.pi) - 1.0 / np.sqrt(np.pi))
  assert np.isclose(crps[0], expected, rtol=1e-12)


@given(
  mu=_floats(-50, 50),
  sigma=_floats(1e-3, 1e3),
  d1=_floats(0, 100),
  d2=_floats(0, 100),
)
def test_gaussian_crps_monotone_in_error(mu, sigma, d1, d2):
  lo, hi = sorted([d1, d2])
  c_lo = _gaussian_crps(np.array([mu]), np.array([sigma]), np.array([mu + lo]))
  c_hi = _gaussian_crps(np.array([mu]), np.array([sigma]), np.array([mu + hi]))
  assert c_lo[0] <= c_hi[0] + 1e-12


# P3 — NEES invariance under a common rigid transform ------------------------


@given(
  est=pose(),
  xi_pert=_vec(6, -0.5, 0.5),
  cov=spd6(),
  common=pose(t_lo=-5.0, t_hi=5.0),
)
def test_nees_invariant_under_common_transform(est, xi_pert, cov, common):
  est_t, est_q = est
  step = GaussianStep(
    timestamp=0.0, translation=est_t, quat_xyzw=est_q, covariance=cov
  )
  T_mean = homogeneous(Rotation.from_quat(est_q).as_matrix(), est_t)
  T_obs = T_mean @ se3_exp(xi_pert)
  gt_t, gt_q = T_obs[:3, 3], rot_to_quat_xyzw(T_obs[:3, :3])
  base = gaussian_log_score_components(step, gt_t, gt_q)

  ct, cq = common
  T = homogeneous(Rotation.from_quat(cq).as_matrix(), ct)
  moved_step = propagate_step(
    step,
    T,
    tangent_convention=TangentConvention.RIGHT,
    tangent_order=TangentOrder.TRANS_ROT,
  )
  T_obs2 = T @ T_obs
  moved = gaussian_log_score_components(
    moved_step, T_obs2[:3, 3], rot_to_quat_xyzw(T_obs2[:3, :3])
  )
  for slc in ("joint", "translation", "rotation"):
    got = getattr(moved, slc)
    want = getattr(base, slc)
    assert np.isclose(got.nees, want.nees, rtol=1e-6, atol=1e-9)
    assert np.isclose(got.sharpness, want.sharpness, rtol=1e-9, atol=1e-9)


# P6 — bias/variance decomposition identity ----------------------------------


@st.composite
def trajectory_pair(draw):
  """(timestamps, gt, est) with guaranteed horizontal GT motion per step."""
  n = draw(st.integers(min_value=6, max_value=30))
  dt = 0.1
  ts = np.arange(n) * dt
  inc_x = np.array(draw(st.lists(_floats(0.05, 0.5), min_size=n, max_size=n)))
  inc_yz = draw(_vec(2 * n, -0.2, 0.2)).reshape(n, 2)
  gt = np.cumsum(np.column_stack([inc_x, inc_yz]), axis=0)
  err = draw(_vec(3 * n, -1.0, 1.0)).reshape(n, 3)
  return ts, gt, gt + err


@given(traj=trajectory_pair(), k=st.integers(min_value=1, max_value=3))
def test_bias_variance_decomposition_identity(traj, k):
  ts, gt, est = traj
  steps = [
    DeterministicStep(
      timestamp=t, translation=p, quat_xyzw=np.array([0, 0, 0, 1.0])
    )
    for t, p in zip(ts, est, strict=True)
  ]
  results = bias_variance(steps, gt, windows_s=[k * 0.1])
  for r in results:
    bias_sq = float(np.sum(np.square(r.bias)))
    var = float(np.sum(np.square(r.std)))
    assert np.isclose(r.mse, bias_sq + var, rtol=1e-9, atol=1e-12)
    if r.mse > 0:
      assert -1e-12 <= r.bias_fraction <= 1.0 + 1e-12
    else:
      assert np.isnan(r.bias_fraction)
    assert r.dominant_axis in _AXES


# P7 — window pairing invariants ---------------------------------------------


@given(
  ts=st.lists(_floats(0.0, 100.0), min_size=2, max_size=50, unique=True).map(
    lambda x: np.array(sorted(x))
  ),
  w=_floats(0.01, 50.0),
  tol=_floats(0.0, 5.0),
)
def test_window_pairs_invariants(ts, w, tol):
  i, j = _window_pairs(ts, w, tol)
  assert np.all(j > i)
  assert np.all(np.abs(ts[j] - ts[i] - w) <= tol + 1e-12)


@given(
  n=st.integers(min_value=5, max_value=50),
  dt=_floats(0.01, 1.0),
  k=st.integers(min_value=1, max_value=4),
)
def test_window_pairs_exact_on_uniform_grid(n, dt, k):
  assume(k < n)
  ts = np.arange(n) * dt
  i, j = _window_pairs(ts, k * dt, 1e-9)
  assert np.array_equal(i, np.arange(n - k))
  assert np.array_equal(j, np.arange(k, n))


# P8 — alignment recovers a known SE(3) --------------------------------------


@given(
  pts=_vec(36, -10.0, 10.0).map(lambda v: v.reshape(12, 3)),
  T_parts=pose(t_lo=-5.0, t_hi=5.0),
)
def test_fit_alignment_recovers_known_se3(pts, T_parts):
  centered = pts - pts.mean(axis=0)
  assume(np.linalg.svd(centered, compute_uv=False)[-1] > 1e-2)
  t0, q0 = T_parts
  R0 = Rotation.from_quat(q0).as_matrix()
  gt = (R0 @ pts.T).T + t0
  fit = fit_alignment(pts, gt, mode="se3")
  assert np.max(fit.residuals) < 1e-6
  assert np.allclose(fit.transform, homogeneous(R0, t0), atol=1e-6)


# P9 — summary statistic ordering --------------------------------------------


@given(
  vals=st.lists(_floats(-1e3, 1e3), min_size=3, max_size=100).map(np.array)
)
def test_summarize_bounds(vals):
  s = summarize(vals, rng=np.random.default_rng(0))
  eps = 1e-12 * max(1.0, abs(s.min), abs(s.max))  # mean/CI roundoff headroom
  assert s.min <= s.median <= s.max
  assert s.min - eps <= s.mean <= s.max + eps
  assert s.ci_low <= s.ci_high
  assert s.min - eps <= s.ci_low and s.ci_high <= s.max + eps


# P10 — relative mean(z^2) scales exactly as 1/c -----------------------------


@given(
  traj=trajectory_pair(),
  cov=spd6(scale_lo=1e-2, scale_hi=1.0),
  c=_floats(0.25, 16.0),
)
def test_relative_mean_z2_scales_inverse_with_covariance(traj, cov, c):
  ts, gt, est = traj
  q = np.array([0.0, 0.0, 0.0, 1.0])
  steps = [
    GaussianStep(timestamp=t, translation=p, quat_xyzw=q, covariance=cov)
    for t, p in zip(ts, est, strict=True)
  ]
  scaled = [
    GaussianStep(timestamp=t, translation=p, quat_xyzw=q, covariance=c * cov)
    for t, p in zip(ts, est, strict=True)
  ]
  rng = np.random.default_rng(0)
  base = relative_translation_crps(steps, gt, windows_s=[0.1], rng=rng)
  got = relative_translation_crps(scaled, gt, windows_s=[0.1], rng=rng)
  assert len(base) == len(got) == 1
  assert np.isclose(got[0].mean_z2, base[0].mean_z2 / c, rtol=1e-9)


# P11 — Student-t converges to the Gaussian as nu -> inf ---------------------


@given(xi=_vec(6, -2.0, 2.0), cov=spd6(scale_lo=1.0, scale_hi=10.0))
def test_student_t_matches_gaussian_at_large_nu(xi, cov):
  t_nll = student_t_neg_log_density(xi, cov, nu=1e7)
  g_nll = _gaussian_neg_log_density(xi, cov)
  assert np.isclose(t_nll, g_nll, atol=1e-3)


# P12 — ANEES verdict is monotone in the NEES scale --------------------------

_VERDICT_RANK = {"conservative": 0, "consistent": 1, "optimistic": 2}


@given(
  vals=st.lists(_floats(0.0, 1e3), min_size=1, max_size=200).map(np.array),
  dof=st.integers(min_value=1, max_value=6),
  c=_floats(1.0, 100.0),
)
def test_anees_interval_and_verdict_monotone(vals, dof, c):
  base = anees_consistency(vals, dof=dof)
  assert base.lo < base.hi
  scaled = anees_consistency(c * vals, dof=dof)
  assert _VERDICT_RANK[scaled.verdict] >= _VERDICT_RANK[base.verdict]
