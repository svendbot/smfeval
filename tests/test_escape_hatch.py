"""TUM + covariance escape hatch: wide-TUM and sidecar loaders + CLI.

A user whose filter prints covariance anywhere must get a verdict
without adopting SQUARE: either 29-column wide TUM (pose + 21
lower-triangle covariance entries, SQUARE packing) or plain TUM plus a
--cov sidecar file.
"""

import numpy as np
import pytest

from smfeval.cli.main import main
from smfeval.format import (
  FormatError,
  Representation,
  TangentConvention,
  TangentOrder,
)
from smfeval.io import (
  load_cov_sidecar,
  load_square,
  load_tum_gaussian,
  load_tum_with_sidecar,
  sniff_tum_columns,
)

_RNG = np.random.default_rng(9)


def _pack_lower(cov: np.ndarray) -> list[float]:
  return [cov[i, j] for i in range(6) for j in range(i + 1)]


def _make_case(n: int = 100, scale: float = 1.0):
  """Calibrated (scale=1) gaussian trajectory: rows + gt rows."""
  ts = np.linspace(0.0, 10.0, n)
  gt = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
  sigma = 0.1
  est = gt + _RNG.normal(scale=scale * sigma, size=gt.shape)
  cov = np.diag([sigma**2] * 3 + [1e-4] * 3)
  return ts, est, gt, cov


def _write_wide_tum(path, ts, pos, cov):
  rows = []
  for t, p in zip(ts, pos, strict=True):
    fields = [t, *p, 0.0, 0.0, 0.0, 1.0, *_pack_lower(cov)]
    rows.append(" ".join(f"{x:.17g}" for x in fields))
  path.write_text("\n".join(rows) + "\n")


def _write_tum(path, ts, pos):
  rows = [
    f"{t:.17g} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 0 0 0 1"
    for t, p in zip(ts, pos, strict=True)
  ]
  path.write_text("\n".join(rows) + "\n")


def _write_sidecar(path, ts, cov, header_lines: list[str] | None = None):
  rows = list(header_lines or [])
  for t in ts:
    rows.append(" ".join(f"{x:.17g}" for x in [t, *_pack_lower(cov)]))
  path.write_text("\n".join(rows) + "\n")


def test_sniff_columns(tmp_path):
  ts, est, _, cov = _make_case(5)
  wide = tmp_path / "wide.txt"
  _write_wide_tum(wide, ts, est, cov)
  assert sniff_tum_columns(wide) == 29
  plain = tmp_path / "plain.txt"
  _write_tum(plain, ts, est)
  assert sniff_tum_columns(plain) == 8


def test_wide_tum_is_headerless_square_body(tmp_path):
  """Stripping the header off a SQUARE gaussian file must load identically."""
  ts, est, _, cov = _make_case(20)
  wide = tmp_path / "wide.txt"
  _write_wide_tum(wide, ts, est, cov)

  square = tmp_path / "same.SQUARE"
  square.write_text(
    "#%FORMAT SQUARE/0.3\n#%REPRESENTATION gaussian_se3\n"
    "#%POSE_FRAME world\n#%BODY_FRAME imu\n#%GAUGE se3\n"
    "#%TIMESTAMP_UNIT seconds\n#%ALGORITHM x\n#%ALGORITHM_VERSION 0\n"
    "#%TANGENT_CONVENTION right_perturbation\n"
    "#%TANGENT_ORDER translation_rotation\n#%ROTATION_PARAM axis_angle\n"
    + wide.read_text()
  )

  h_w, s_w = load_tum_gaussian(wide, pose_frame="world", body_frame="imu")
  h_s, s_s = load_square(square)
  assert h_w.representation is Representation.GAUSSIAN_SE3
  assert len(s_w) == len(s_s) == 20
  for a, b in zip(s_w, s_s, strict=True):
    assert a.timestamp == b.timestamp
    np.testing.assert_array_equal(a.translation, b.translation)
    np.testing.assert_array_equal(a.covariance, b.covariance)


def test_sidecar_happy_path(tmp_path):
  ts, est, _, cov = _make_case(15)
  pose_f = tmp_path / "est.tum"
  cov_f = tmp_path / "est.cov"
  _write_tum(pose_f, ts, est)
  _write_sidecar(cov_f, ts, cov)
  header, steps = load_tum_with_sidecar(
    pose_f, cov_f, pose_frame="world", body_frame="imu"
  )
  assert header.representation is Representation.GAUSSIAN_SE3
  assert header.tangent_convention is TangentConvention.RIGHT
  assert len(steps) == 15
  np.testing.assert_array_equal(steps[3].covariance, cov)


def test_sidecar_header_overrides_convention(tmp_path):
  ts, est, _, cov = _make_case(5)
  pose_f = tmp_path / "est.tum"
  cov_f = tmp_path / "est.cov"
  _write_tum(pose_f, ts, est)
  _write_sidecar(
    cov_f,
    ts,
    cov,
    header_lines=[
      "#%TANGENT_CONVENTION left_perturbation",
      "#%TANGENT_ORDER rotation_translation",
    ],
  )
  header, _ = load_tum_with_sidecar(
    pose_f, cov_f, pose_frame="world", body_frame="imu"
  )
  assert header.tangent_convention is TangentConvention.LEFT
  assert header.tangent_order is TangentOrder.ROT_TRANS


def test_sidecar_count_mismatch_rejected(tmp_path):
  ts, est, _, cov = _make_case(10)
  pose_f = tmp_path / "est.tum"
  cov_f = tmp_path / "est.cov"
  _write_tum(pose_f, ts, est)
  _write_sidecar(cov_f, ts[:-2], cov)
  with pytest.raises(FormatError, match="1:1"):
    load_tum_with_sidecar(pose_f, cov_f, pose_frame="world", body_frame="imu")


def test_sidecar_timestamp_mismatch_rejected(tmp_path):
  ts, est, _, cov = _make_case(10)
  pose_f = tmp_path / "est.tum"
  cov_f = tmp_path / "est.cov"
  _write_tum(pose_f, ts, est)
  _write_sidecar(cov_f, ts + 0.01, cov)  # shifted beyond the 1e-6 tolerance
  with pytest.raises(FormatError, match="does not match pose timestamp"):
    load_tum_with_sidecar(pose_f, cov_f, pose_frame="world", body_frame="imu")


def test_sidecar_bad_field_count_rejected(tmp_path):
  cov_f = tmp_path / "est.cov"
  cov_f.write_text("0.0 1.0 2.0\n")
  with pytest.raises(FormatError, match="expected 22"):
    load_cov_sidecar(cov_f)


# --- CLI end to end -----------------------------------------------------------


def test_cli_nees_on_wide_tum(tmp_path, capsys):
  ts, est, gt, cov = _make_case(150)
  est_f = tmp_path / "est.txt"
  gt_f = tmp_path / "gt.tum"
  _write_wide_tum(est_f, ts, est, cov)
  _write_tum(gt_f, ts, gt)
  rc = main(
    [
      "nees",
      str(est_f),
      str(gt_f),
      "--est-body-frame",
      "imu",
      "--ref-body-frame",
      "imu",
      "--align",
      "none",
    ]
  )
  cap = capsys.readouterr()
  assert rc == 0, cap.err
  assert "median NEES" in cap.out
  assert "consistent" in cap.out
  assert "bare-TUM estimate read as gaussian_se3" in cap.err


def test_cli_nees_on_tum_with_sidecar(tmp_path, capsys):
  ts, est, gt, cov = _make_case(150)
  est_f = tmp_path / "est.tum"
  cov_f = tmp_path / "est.cov"
  gt_f = tmp_path / "gt.tum"
  _write_tum(est_f, ts, est)
  _write_sidecar(cov_f, ts, cov)
  _write_tum(gt_f, ts, gt)
  rc = main(
    [
      "nees",
      str(est_f),
      str(gt_f),
      "--cov",
      str(cov_f),
      "--est-body-frame",
      "imu",
      "--ref-body-frame",
      "imu",
      "--align",
      "none",
    ]
  )
  cap = capsys.readouterr()
  assert rc == 0, cap.err
  assert "median NEES" in cap.out
  assert "consistent" in cap.out


def test_cli_bare_tum_estimate_requires_body_frame(tmp_path, capsys):
  ts, est, gt, cov = _make_case(20)
  est_f = tmp_path / "est.txt"
  gt_f = tmp_path / "gt.tum"
  _write_wide_tum(est_f, ts, est, cov)
  _write_tum(gt_f, ts, gt)
  rc = main(["nees", str(est_f), str(gt_f), "--ref-body-frame", "imu"])
  err = capsys.readouterr().err
  assert rc == 2
  assert "--est-body-frame" in err


def test_cli_rejects_unknown_column_count(tmp_path, capsys):
  est_f = tmp_path / "est.txt"
  est_f.write_text("0.0 1.0 2.0 3.0\n")
  gt_f = tmp_path / "gt.tum"
  gt_f.write_text("0.0 1.0 2.0 3.0 0 0 0 1\n")
  rc = main(
    [
      "nees",
      str(est_f),
      str(gt_f),
      "--est-body-frame",
      "imu",
      "--ref-body-frame",
      "imu",
    ]
  )
  err = capsys.readouterr().err
  assert rc == 2
  assert "4 columns" in err
