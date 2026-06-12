"""Unit tests for the track-frame bias/variance decomposition."""

import numpy as np
import pytest

from smfeval.scoring.bias_variance import bias_variance
from smfeval.steps import DeterministicStep

_Q_ID = np.array([0.0, 0.0, 0.0, 1.0])


def _steps(ts: np.ndarray, pos: np.ndarray) -> list[DeterministicStep]:
  return [
    DeterministicStep(timestamp=float(t), translation=p, quat_xyzw=_Q_ID)
    for t, p in zip(ts, pos, strict=True)
  ]


def _straight_track(n: int = 50, dt: float = 0.1, v: float = 1.0):
  """GT moving along +x: track frame is along=+x, cross=+y, vertical=+z."""
  ts = np.arange(n) * dt
  gt = np.column_stack([v * ts, np.zeros(n), np.zeros(n)])
  return ts, gt


@pytest.mark.parametrize(
  ("drift", "axis"),
  [
    (np.array([0.05, 0.0, 0.0]), "along"),
    (np.array([0.0, 0.05, 0.0]), "cross"),
    (np.array([0.0, 0.0, 0.05]), "vertical"),
  ],
)
def test_constant_drift_is_pure_bias_on_the_right_axis(drift, axis):
  ts, gt = _straight_track()
  # est accumulates `drift` per step: every increment error equals drift,
  # so the error is 100% bias and lands on the drifted axis.
  est = gt + np.outer(np.arange(len(ts)), drift)
  results = bias_variance(_steps(ts, est), gt, windows_s=[0.1])
  assert len(results) == 1
  r = results[0]
  assert r.dominant_axis == axis
  assert r.bias_fraction > 0.999999
  assert np.allclose(r.std, 0.0, atol=1e-12)
  assert np.isclose(r.mse, float(drift @ drift), rtol=1e-9)


def test_iid_noise_is_mostly_variance():
  rng = np.random.default_rng(42)
  ts, gt = _straight_track(n=400)
  est = gt + rng.normal(scale=0.05, size=gt.shape)
  results = bias_variance(_steps(ts, est), gt, windows_s=[0.1])
  assert len(results) == 1
  # zero-mean noise: bias contribution shrinks as 1/n_pairs.
  assert results[0].bias_fraction < 0.05


def test_vertical_only_motion_is_gated_out():
  n = 20
  ts = np.arange(n) * 0.1
  gt = np.column_stack([np.zeros(n), np.zeros(n), 0.5 * np.arange(n)])
  est = gt + 0.01
  # No horizontal GT increment exceeds min_horiz_m: no track frame exists.
  assert bias_variance(_steps(ts, est), gt, windows_s=[0.1]) == []


def test_window_longer_than_track_yields_nothing():
  ts, gt = _straight_track(n=10)
  assert bias_variance(_steps(ts, gt), gt, windows_s=[100.0]) == []


def test_multiple_windows_one_result_each():
  ts, gt = _straight_track(n=100)
  est = gt + 0.01
  results = bias_variance(_steps(ts, est), gt, windows_s=[0.1, 1.0])
  assert [r.window_s for r in results] == [0.1, 1.0]
  assert results[0].n_pairs > results[1].n_pairs > 0
