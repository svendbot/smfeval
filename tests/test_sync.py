import numpy as np

from smfeval.format import TangentOrder
from smfeval.steps import GaussianStep
from smfeval.sync import match_timestamps, sync_risk


def test_match_basic():
  est = np.array([0.0, 0.1, 0.2, 0.3])
  gt = np.array([0.0, 0.1, 0.2, 0.3])
  r = match_timestamps(est, gt, t_max_diff=0.001)
  assert r.n_matched == 4
  assert r.n_dropped == 0
  assert list(r.est_indices) == [0, 1, 2, 3]
  assert list(r.gt_indices) == [0, 1, 2, 3]


def test_match_drops_above_tolerance():
  est = np.array([0.0, 1.0, 2.0])
  gt = np.array([0.001, 1.5, 2.001])
  r = match_timestamps(est, gt, t_max_diff=0.01)
  assert r.n_matched == 2
  assert r.n_dropped == 1


def test_match_offset_applied():
  est = np.array([0.0, 1.0])
  gt = np.array([0.5, 1.5])
  r = match_timestamps(est, gt, t_max_diff=0.001, t_offset=0.5)
  assert r.n_matched == 2


def test_match_empty_when_no_overlap():
  est = np.array([0.0, 1.0])
  gt = np.array([10.0, 11.0])
  r = match_timestamps(est, gt, t_max_diff=0.01)
  assert r.n_matched == 0
  assert r.n_dropped == 2


def test_match_iterates_over_shorter():
  """Evo flips so iteration is over the shorter trajectory; check we follow."""
  est = np.array([0.0, 0.05, 0.1, 0.15, 0.2])  # 5 samples
  gt = np.array([0.0, 0.1, 0.2])  # 3 samples — the shorter
  r = match_timestamps(est, gt, t_max_diff=0.01)
  assert r.n_matched == 3
  assert sorted(r.gt_indices.tolist()) == [0, 1, 2]


def test_match_gap_quantiles():
  est = np.array([0.0, 1.0, 2.0])
  gt = np.array([0.001, 1.002, 2.005])
  r = match_timestamps(est, gt, t_max_diff=0.01)
  q = r.gap_quantiles_ms
  assert q["median"] > 0
  assert q["p99"] >= q["median"]


def _gauss_step(ts: float, pos: np.ndarray, cov_diag: float) -> GaussianStep:
  return GaussianStep(
    timestamp=ts,
    translation=pos,
    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    covariance=np.eye(6) * cov_diag,
  )


def test_sync_risk_zero_when_dt_zero():
  gt_ts = np.array([0.0, 1.0, 2.0])
  gt_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
  est_ts = gt_ts.copy()
  est_steps = [
    _gauss_step(t, p, 1.0) for t, p in zip(est_ts, gt_pos, strict=False)
  ]
  r = sync_risk(
    est_steps,
    gt_ts,
    gt_pos,
    est_indices=np.array([0, 1, 2]),
    gt_indices=np.array([0, 1, 2]),
    est_ts=est_ts,
    tangent_order=TangentOrder.TRANS_ROT,
  )
  assert np.allclose(r, 0.0)


def test_sync_risk_grows_with_velocity():
  gt_ts = np.array([0.0, 1.0, 2.0])
  gt_pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
  est_ts = np.array([0.005, 1.005, 2.005])  # 5ms ahead
  est_steps = [
    _gauss_step(t, p, 1.0) for t, p in zip(est_ts, gt_pos, strict=False)
  ]
  r = sync_risk(
    est_steps,
    gt_ts,
    gt_pos,
    est_indices=np.array([0, 1, 2]),
    gt_indices=np.array([0, 1, 2]),
    est_ts=est_ts,
    tangent_order=TangentOrder.TRANS_ROT,
  )
  # |v|=10 m/s, dt=5e-3, σ=1 → r ≈ 0.05
  assert np.allclose(r[1], 10.0 * 0.005 / 1.0, atol=1e-6)
