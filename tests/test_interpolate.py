"""Sanity checks for the piecewise GP reference interpolator."""

import numpy as np
import pytest

from smfeval.sync import interpolate_ref_at


def _circular_trajectory(n: int = 60, radius: float = 1.0, omega: float = 1.0):
  """Smooth helix-ish trajectory: x,y on a circle, z linear, identity rotation."""
  t = np.linspace(0.0, 2.0 * np.pi / omega, n)
  pos = np.column_stack(
    [radius * np.cos(omega * t), radius * np.sin(omega * t), 0.1 * t]
  )
  quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
  return t, pos, quat


def test_interpolation_at_sample_time_is_exact():
  t_ref, pos_ref, q_ref = _circular_trajectory()
  # Query exactly at one of the sample times — should reproduce that sample.
  query = np.array([t_ref[25]])
  t_out, q_out, cov_out, keep = interpolate_ref_at(query, t_ref, pos_ref, q_ref)
  assert keep[0]
  np.testing.assert_allclose(t_out[0], pos_ref[25], atol=1e-6)
  # Predictive variance should be ~zero on the sample.
  assert cov_out[0, 0, 0] < 1e-5


def test_interpolation_variance_grows_with_distance():
  # Two reference samples bracketing a gap; variance should be highest in the middle.
  t_ref = np.array([0.0, 1.0])
  pos_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  q_ref = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))
  queries = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
  _, _, cov, keep = interpolate_ref_at(
    queries,
    t_ref,
    pos_ref,
    q_ref,
    length_scale_s=0.5,
    window=2,
  )
  assert keep.all()
  vars_ = cov[:, 0, 0]
  # Endpoints essentially zero; mid-gap higher than near-endpoint.
  assert vars_[2] > vars_[1]
  assert vars_[2] > vars_[3]
  assert vars_[0] < vars_[1]
  assert vars_[4] < vars_[3]


def test_out_of_range_queries_marked():
  t_ref, pos_ref, q_ref = _circular_trajectory(n=10)
  queries = np.array([t_ref[0] - 1.0, t_ref[5], t_ref[-1] + 1.0])
  _, _, _, keep = interpolate_ref_at(queries, t_ref, pos_ref, q_ref)
  assert not keep[0]
  assert keep[1]
  assert not keep[2]


def test_smooth_interpolation_tracks_curve():
  """Between samples, the interpolated position should land near the
  underlying smooth trajectory — within a few percent of segment length."""
  t_ref, pos_ref, q_ref = _circular_trajectory(n=60)
  # Query at midpoints between consecutive samples.
  midpoints = 0.5 * (t_ref[:-1] + t_ref[1:])
  truth = 0.5 * (pos_ref[:-1] + pos_ref[1:])  # crude midpoint baseline
  t_out, _, _, keep = interpolate_ref_at(
    midpoints, t_ref, pos_ref, q_ref, window=8, length_scale_s=0.3
  )
  assert keep.all()
  # Interior of the trajectory only — boundary windows are biased.
  err = np.linalg.norm(t_out[5:-5] - truth[5:-5], axis=1)
  # Smooth curve sampled at 60 pts, query midpoints: GP should hit within a
  # small multiple of the underlying curvature error of linear interpolation.
  assert err.max() < 0.05


def test_too_few_samples_rejected():
  with pytest.raises(ValueError, match="at least 2"):
    interpolate_ref_at(
      np.array([0.0]),
      np.array([0.0]),
      np.array([[0.0, 0.0, 0.0]]),
      np.array([[0.0, 0.0, 0.0, 1.0]]),
    )
