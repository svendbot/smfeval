"""Unit tests for relative_translation_crps (the windowed CRPS path).

relative_calibration verdicts are covered in test_calibration_split.py;
here the CRPS variant gets a hand-computed oracle, edge cases, and the
derived-field definitions (sigma_rel, RPE RMSE).
"""

import numpy as np
import pytest

from smfeval.scoring.crps import _gaussian_crps
from smfeval.scoring.relative import relative_translation_crps
from smfeval.steps import DeterministicStep, GaussianStep

_Q_ID = np.array([0.0, 0.0, 0.0, 1.0])


def _step(t: float, p, var_diag) -> GaussianStep:
  cov = np.zeros((6, 6))
  cov[:3, :3] = np.diag(var_diag)
  cov[3:, 3:] = np.eye(3) * 1e-4
  return GaussianStep(
    timestamp=t,
    translation=np.asarray(p, dtype=float),
    quat_xyzw=_Q_ID,
    covariance=cov,
  )


def test_two_pose_window_matches_hand_computation():
  var0 = np.array([1e-4, 4e-4, 9e-4])
  var1 = np.array([4e-4, 1e-4, 1e-4])
  steps = [
    _step(0.0, [0.0, 0.0, 0.0], var0),
    _step(0.1, [1.0, 0.5, -0.2], var1),
  ]
  ref = np.array([[0.01, -0.02, 0.0], [1.02, 0.49, -0.21]])
  res = relative_translation_crps(steps, ref, windows_s=[0.1])
  assert len(res) == 1
  r = res[0]
  assert r.n_pairs == 1

  de = steps[1].translation - steps[0].translation
  dg = ref[1] - ref[0]
  sigma = np.sqrt(var0 + var1)
  expected_crps = float(_gaussian_crps(de, sigma, dg).mean())
  assert np.isclose(r.crps.mean, expected_crps, rtol=1e-12)
  assert np.isclose(r.crps.median, expected_crps, rtol=1e-12)
  assert np.isclose(r.mean_z2, (((dg - de) / sigma) ** 2).mean(), rtol=1e-12)
  assert np.isclose(r.sigma_rel_median_m, np.sqrt((var0 + var1).sum()))
  assert np.isclose(r.rpe_rmse_m, np.linalg.norm(dg - de), rtol=1e-12)


def test_requires_gaussian_steps():
  steps = [
    DeterministicStep(timestamp=0.0, translation=np.zeros(3), quat_xyzw=_Q_ID),
    DeterministicStep(timestamp=0.1, translation=np.ones(3), quat_xyzw=_Q_ID),
  ]
  with pytest.raises(TypeError, match="GaussianStep"):
    relative_translation_crps(steps, np.zeros((2, 3)), windows_s=[0.1])


def test_window_with_no_pairs_is_skipped():
  steps = [
    _step(0.0, np.zeros(3), np.ones(3)),
    _step(0.1, np.ones(3), np.ones(3)),
  ]
  ref = np.zeros((2, 3))
  # 10 s window over a 0.1 s track: no pairs, no result for that window.
  res = relative_translation_crps(steps, ref, windows_s=[10.0, 0.1])
  assert [r.window_s for r in res] == [0.1]


def test_nonuniform_timestamps_respect_tolerance():
  # Gap between 0.30 and 0.55 means only some 0.1 s pairs exist at the
  # default tolerance (half the median dt = 0.05 s).
  ts = [0.0, 0.1, 0.2, 0.3, 0.55, 0.65]
  steps = [_step(t, [t * 10.0, 0.0, 0.0], np.ones(3) * 1e-4) for t in ts]
  ref = np.array([[t * 10.0, 0.0, 0.0] for t in ts])
  res = relative_translation_crps(steps, ref, windows_s=[0.1])
  assert len(res) == 1
  # pairs: (0,1) (1,2) (2,3) and (4,5); 0.3->0.55 misses the window.
  assert res[0].n_pairs == 4


def test_perfect_increments_have_zero_z2_and_rmse():
  rng = np.random.default_rng(3)
  ts = np.arange(20) * 0.1
  pos = np.cumsum(rng.normal(size=(20, 3)), axis=0)
  steps = [_step(t, p, np.ones(3) * 1e-2) for t, p in zip(ts, pos, strict=True)]
  # reference differs by a constant offset: increments match exactly.
  res = relative_translation_crps(steps, pos + 5.0, windows_s=[0.1])
  assert len(res) == 1
  # not exactly 0: the +5.0 offset perturbs the float increments by ~1 ulp
  assert res[0].mean_z2 < 1e-20
  assert res[0].rpe_rmse_m < 1e-12
