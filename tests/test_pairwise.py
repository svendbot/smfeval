"""No-reference pairwise NEES: properties, dilution pipeline check, errors.

The simulated-NEES power analysis lives in test_power.py; here the full
pair pipeline (timestamp match -> Umeyama A->B -> tangent difference ->
NEES under summed covariance) is exercised end to end.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from smfeval.format import (
  FORMAT_VERSION,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
)
from smfeval.scoring.pairwise import PairInputError, pair_translation_nees
from smfeval.steps import DeterministicStep, GaussianStep
from tests._strategies import spd6, vec

_Q_ID = np.array([0.0, 0.0, 0.0, 1.0])


def _header(
  convention: TangentConvention = TangentConvention.RIGHT,
  representation: Representation = Representation.GAUSSIAN_SE3,
  body_frame: str = "lidar",
) -> SquareHeader:
  return SquareHeader(
    format_version=FORMAT_VERSION,
    representation=representation,
    pose_frame="odom",
    body_frame=body_frame,
    gauge=Gauge.SE3,
    timestamp_unit="seconds",
    algorithm="testbot",
    algorithm_version="1.0",
    tangent_convention=convention,
    tangent_order=TangentOrder.TRANS_ROT,
    rotation_param="axis_angle",
  )


def _gauss_traj(
  rng: np.random.Generator,
  n: int = 30,
  cov: np.ndarray | None = None,
  jitter: float = 0.0,
) -> list[GaussianStep]:
  if cov is None:
    cov = np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
  pos = np.cumsum(rng.normal(scale=0.5, size=(n, 3)), axis=0)
  quats = Rotation.from_rotvec(rng.normal(scale=0.1, size=(n, 3))).as_quat()
  return [
    GaussianStep(
      timestamp=0.1 * i,
      translation=pos[i] + rng.normal(scale=jitter, size=3),
      quat_xyzw=quats[i],
      covariance=cov.copy(),
    )
    for i in range(n)
  ]


# P5 — self-pair is exactly zero ----------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31), cov=spd6())
def test_pair_with_itself_is_zero(seed, cov):
  rng = np.random.default_rng(seed)
  steps = _gauss_traj(rng, n=15, cov=cov)
  res = pair_translation_nees(_header(), steps, _header(), steps)
  assert res.n_matched == 15
  assert res.med_d_norm < 1e-9
  finite = res.nees[np.isfinite(res.nees)]
  assert finite.size == 15
  assert np.all(finite < 1e-12)


# P4 — pair(a, b) and pair(b, a) agree per pose -------------------------------


@given(
  seed=st.integers(min_value=0, max_value=2**31),
  offset=vec(3, -2.0, 2.0),
)
def test_pair_is_symmetric(seed, offset):
  rng = np.random.default_rng(seed)
  a = _gauss_traj(rng, n=25, jitter=0.05)
  b = [
    GaussianStep(
      timestamp=s.timestamp,
      translation=s.translation + offset + rng.normal(scale=0.05, size=3),
      quat_xyzw=s.quat_xyzw,
      covariance=s.covariance,
    )
    for s in a
  ]
  ab = pair_translation_nees(_header(), a, _header(), b)
  ba = pair_translation_nees(_header(), b, _header(), a)
  # the optimal B->A Umeyama is the inverse of A->B, so the tangent
  # difference flips sign and the quadratic form is unchanged.
  np.testing.assert_allclose(ab.nees, ba.nees, rtol=1e-6, atol=1e-10)


# dilution: the full pipeline lands in the analytic 2k/(k+1) band -------------


@pytest.mark.parametrize(
  ("k", "verdict"), [(1.0, "consistent"), (4.0, "optimistic")]
)
def test_pipeline_median_matches_dilution_model(k, verdict):
  # Two filters on a common (latent) trajectory, both publishing the same
  # Sigma; A's true error covariance is k times its published one, B is
  # calibrated. Then d ~ N(0, (k+1) Sigma) scored under 2 Sigma gives a
  # pairwise NEES of ((k+1)/2) chi2_3 — the equal-published-covariance
  # variant of the dilution law (test_power.py covers the equal-true-
  # covariance variant, 2k/(k+1)). Identity rotations keep the translation
  # slice exact; the Umeyama alignment must not disturb the band.
  rng = np.random.default_rng(101)
  n = 400
  sigma = np.diag([4e-2, 9e-2, 1e-2])
  pub_cov = np.zeros((6, 6))
  pub_cov[:3, :3] = sigma
  pub_cov[3:, 3:] = np.eye(3) * 1e-6
  latent = np.cumsum(rng.normal(scale=0.5, size=(n, 3)), axis=0)

  def _traj(scale: float) -> list[GaussianStep]:
    err = rng.multivariate_normal(np.zeros(3), scale * sigma, size=n)
    return [
      GaussianStep(
        timestamp=0.1 * i,
        translation=latent[i] + err[i],
        quat_xyzw=_Q_ID,
        covariance=pub_cov,
      )
      for i in range(n)
    ]

  res = pair_translation_nees(_header(), _traj(k), _header(), _traj(1.0))
  eff = (k + 1.0) / 2.0
  expected_median = eff * chi2.ppf(0.5, df=3)
  # median of eff*chi2_3 over ~n draws: sd ~ eff / (2 f(med) sqrt(n))
  sd = eff / (2.0 * 0.207 * np.sqrt(n))
  assert abs(res.anees.median - expected_median) < 5.0 * sd, (
    f"median {res.anees.median:.2f} vs expected {expected_median:.2f}"
  )
  assert res.anees.verdict == verdict


# error paths ------------------------------------------------------------------


def test_rejects_non_gaussian_input():
  rng = np.random.default_rng(0)
  det = [
    DeterministicStep(
      timestamp=0.1 * i, translation=np.zeros(3), quat_xyzw=_Q_ID
    )
    for i in range(15)
  ]
  gauss = _gauss_traj(rng, n=15)
  det_header = _header(representation=Representation.DETERMINISTIC)
  with pytest.raises(PairInputError, match="gaussian_se3"):
    pair_translation_nees(det_header, det, _header(), gauss)


def test_rejects_tangent_convention_mismatch():
  rng = np.random.default_rng(0)
  a = _gauss_traj(rng, n=15)
  b = _gauss_traj(rng, n=15)
  with pytest.raises(PairInputError, match="convention"):
    pair_translation_nees(
      _header(TangentConvention.RIGHT), a, _header(TangentConvention.LEFT), b
    )


def test_rejects_body_frame_mismatch():
  rng = np.random.default_rng(0)
  a = _gauss_traj(rng, n=15)
  b = _gauss_traj(rng, n=15)
  with pytest.raises(PairInputError, match="body frames"):
    pair_translation_nees(
      _header(body_frame="imu"), a, _header(body_frame="lidar"), b
    )


def test_rejects_too_few_matches():
  rng = np.random.default_rng(0)
  a = _gauss_traj(rng, n=5)
  b = _gauss_traj(rng, n=5)
  with pytest.raises(PairInputError, match="matched poses"):
    pair_translation_nees(_header(), a, _header(), b)


def test_non_pd_rows_are_skipped_not_fatal():
  rng = np.random.default_rng(0)
  a = _gauss_traj(rng, n=15, jitter=0.05)
  b = _gauss_traj(rng, n=15, jitter=0.05)
  bad = a[3]
  a[3] = GaussianStep(
    timestamp=bad.timestamp,
    translation=bad.translation,
    quat_xyzw=bad.quat_xyzw,
    covariance=-np.eye(6),
  )
  res = pair_translation_nees(_header(), a, _header(), b)
  assert res.n_matched == 15
  assert res.n_scored == 14
  assert not np.isfinite(res.nees[3])
