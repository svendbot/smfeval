from pathlib import Path

import numpy as np
import pytest

from smfeval.cli.main import main
from smfeval.format import (
  FORMAT_VERSION,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
)
from smfeval.io import write_header, write_steps
from smfeval.steps import DeterministicStep, GaussianStep


def _gauss_header(
  gauge: Gauge = Gauge.SE3, body_frame: str = "imu"
) -> SquareHeader:
  return SquareHeader(
    format_version=FORMAT_VERSION,
    representation=Representation.GAUSSIAN_SE3,
    pose_frame="world",
    body_frame=body_frame,
    gauge=gauge,
    timestamp_unit="seconds",
    algorithm="testbot",
    algorithm_version="1.0",
    tangent_convention=TangentConvention.RIGHT,
    tangent_order=TangentOrder.TRANS_ROT,
    rotation_param="axis_angle",
  )


def _det_header(body_frame: str = "imu") -> SquareHeader:
  return SquareHeader(
    format_version=FORMAT_VERSION,
    representation=Representation.DETERMINISTIC,
    pose_frame="world",
    body_frame=body_frame,
    gauge=Gauge.FIXED,
    timestamp_unit="seconds",
    algorithm="gt",
    algorithm_version="1.0",
  )


def _write(path: Path, header: SquareHeader, steps: list) -> None:
  with path.open("w") as f:
    write_header(f, header)
    write_steps(f, steps, header)


def _gauss_step(t: float, pos: np.ndarray) -> GaussianStep:
  cov = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
  return GaussianStep(
    timestamp=t,
    translation=pos,
    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    covariance=cov,
  )


def _det_step(t: float, pos: np.ndarray) -> DeterministicStep:
  return DeterministicStep(
    timestamp=t,
    translation=pos,
    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
  )


def test_validate_ok(tmp_path: Path, capsys: pytest.CaptureFixture):
  p = tmp_path / "f.SQUARE"
  _write(
    p, _det_header(), [_det_step(0.0, np.zeros(3)), _det_step(1.0, np.ones(3))]
  )
  rc = main(["validate", str(p)])
  out = capsys.readouterr().out
  assert rc == 0
  assert "deterministic" in out
  assert "2 timesteps" in out


def test_validate_bad_file(tmp_path: Path):
  p = tmp_path / "broken.SQUARE"
  p.write_text("#%FORMAT SQUARE/0.3\n")  # missing required fields
  rc = main(["validate", str(p)])
  assert rc == 2


def test_score_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture):
  rng = np.random.default_rng(0)
  n = 30
  ts = np.linspace(0.0, 3.0, n)
  gt_pos = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
  est_pos = gt_pos + rng.normal(scale=0.02, size=gt_pos.shape)

  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(
    est_path,
    _gauss_header(),
    [_gauss_step(t, p) for t, p in zip(ts, est_pos, strict=False)],
  )
  _write(
    gt_path,
    _det_header(),
    [_det_step(t, p) for t, p in zip(ts, gt_pos, strict=False)],
  )

  rc = main(["score", str(est_path), str(gt_path), "--n_samples", "32"])
  out = capsys.readouterr().out
  assert rc == 0
  assert "smfeval scoring report" in out
  assert "Synchronization" in out
  assert "Alignment" in out
  assert "Scores" in out
  assert "Translation CRPS" in out
  assert "Calibration" in out


def _write_calibration_case(tmp_path: Path, scale: float) -> tuple[Path, Path]:
  """Gaussian est (Σ from _gauss_step) with truth offset by `scale`·σ."""
  rng = np.random.default_rng(1)
  n = 200
  ts = np.linspace(0.0, 20.0, n)
  gt_pos = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
  # σ_t = 0.1 m (cov diag 0.01); inject N(0, (scale·σ)²) translation error.
  est_pos = gt_pos + rng.normal(scale=scale * 0.1, size=gt_pos.shape)
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(
    est_path,
    _gauss_header(),
    [_gauss_step(t, p) for t, p in zip(ts, est_pos, strict=False)],
  )
  _write(
    gt_path,
    _det_header(),
    [_det_step(t, p) for t, p in zip(ts, gt_pos, strict=False)],
  )
  return est_path, gt_path


def test_calibration_flag_consistent(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # error matched to Σ (scale 1) → ANEES ≈ dof. Use a 99% interval
  # (--alpha 0.01): for calibrated data E[ANEES]=dof sits at the interval
  # centre, so a tight (90%) interval rejects ~10% of draws by construction.
  est_path, gt_path = _write_calibration_case(tmp_path, scale=1.0)
  rc = main(
    [
      "score",
      str(est_path),
      str(gt_path),
      "--align",
      "none",
      "--alpha",
      "0.01",
      "--calibration",
    ]
  )
  out = capsys.readouterr().out
  assert rc == 0
  assert "Log-score calibration / sharpness split" in out
  assert "CALIBRATION_SPLIT slice=translation" in out
  # find the machine-readable translation line and check the verdict.
  line = next(
    ln for ln in out.splitlines() if "CALIBRATION_SPLIT slice=translation" in ln
  )
  assert "verdict=consistent" in line


def test_calibration_flag_optimistic_when_overconfident(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # error 3× what Σ claims → over-confident → optimistic verdict.
  est_path, gt_path = _write_calibration_case(tmp_path, scale=3.0)
  rc = main(
    ["score", str(est_path), str(gt_path), "--align", "none", "--calibration"]
  )
  out = capsys.readouterr().out
  assert rc == 0
  line = next(
    ln for ln in out.splitlines() if "CALIBRATION_SPLIT slice=translation" in ln
  )
  assert "verdict=optimistic" in line


def test_calibration_flag_skipped_for_deterministic(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(
    est_path,
    _det_header(),
    [_det_step(t, np.array([t, 0, 0])) for t in range(5)],
  )
  _write(
    gt_path,
    _det_header(),
    [_det_step(t, np.array([t, 0, 0])) for t in range(5)],
  )
  rc = main(["score", str(est_path), str(gt_path), "--calibration"])
  err = capsys.readouterr().err
  assert rc == 0
  assert "calibration split: skipped" in err


def _joint_anees(out: str) -> float:
  line = next(
    ln for ln in out.splitlines() if "CALIBRATION_SPLIT slice=joint" in ln
  )
  return float(
    next(tok.split("=")[1] for tok in line.split() if tok.startswith("anees="))
  )


def test_consume_gt_cov_requires_interpolate(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  est_path, gt_path = _write_calibration_case(tmp_path, scale=3.0)
  rc = main(["score", str(est_path), str(gt_path), "--consume-gt-cov"])
  assert rc == 2
  assert "requires --sync=interpolate_gt" in capsys.readouterr().err


def test_consume_gt_cov_reduces_calibration_term(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # Dense GT at 10 Hz; est offset half a period so the GP predictive Σ_gt > 0.
  rng = np.random.default_rng(2)
  n = 240
  gt_t = np.linspace(0.0, 24.0, n)
  est_t = gt_t + 0.05
  gt_pos = np.column_stack([gt_t * 0.5, np.zeros(n), np.zeros(n)])
  est_pos = np.column_stack([est_t * 0.5, np.zeros(n), np.zeros(n)])
  est_pos += rng.normal(
    scale=0.3, size=est_pos.shape
  )  # over-confident (Σ tight)
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(
    est_path,
    _gauss_header(),
    [_gauss_step(t, p) for t, p in zip(est_t, est_pos, strict=False)],
  )
  _write(
    gt_path,
    _det_header(),
    [_det_step(t, p) for t, p in zip(gt_t, gt_pos, strict=False)],
  )

  common = [
    "score",
    str(est_path),
    str(gt_path),
    "--align",
    "none",
    "--sync",
    "interpolate_gt",
    "--calibration",
  ]
  rc = main(common)
  base = _joint_anees(capsys.readouterr().out)
  assert rc == 0
  rc = main(common + ["--consume-gt-cov"])
  consumed = _joint_anees(capsys.readouterr().out)
  assert rc == 0
  # Folding a PSD Σ_gt into the predictive can only inflate Σ_eff ⇒ NEES drops.
  assert consumed < base


def test_ess_inflate_scales_anees(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # Σ → c·Σ with constant c=4 ⇒ NEES → NEES/4 ⇒ ANEES drops ~4×.
  est_path, gt_path = _write_calibration_case(tmp_path, scale=3.0)
  cfile = tmp_path / "cfile.txt"
  cfile.write_text("# t c\n0 4.0\n30 4.0\n")  # constant c=4 over the range
  base = [
    "score",
    str(est_path),
    str(gt_path),
    "--align",
    "none",
    "--calibration",
  ]
  rc = main(base)
  a_base = _joint_anees(capsys.readouterr().out)
  assert rc == 0
  rc = main(base + ["--ess-inflate", str(cfile)])
  a_ess = _joint_anees(capsys.readouterr().out)
  assert rc == 0
  assert np.isclose(a_ess, a_base / 4.0, rtol=0.02)


def test_score_no_matches(tmp_path: Path, capsys: pytest.CaptureFixture):
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(est_path, _gauss_header(), [_gauss_step(0.0, np.zeros(3))])
  _write(gt_path, _det_header(), [_det_step(100.0, np.zeros(3))])
  rc = main(["score", str(est_path), str(gt_path)])
  err = capsys.readouterr().err
  assert rc == 2
  assert "no matched pairs" in err


def test_help_smoke():
  with pytest.raises(SystemExit):
    main(["--help"])


def test_score_body_frame_mismatch_rejected(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  n = 10
  ts = np.linspace(0.0, 1.0, n)
  pos = np.column_stack([ts, np.zeros(n), np.zeros(n)])
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  _write(
    est_path,
    _gauss_header(body_frame="imu"),
    [_gauss_step(t, p) for t, p in zip(ts, pos, strict=False)],
  )
  _write(
    gt_path,
    _det_header(body_frame="lidar"),
    [_det_step(t, p) for t, p in zip(ts, pos, strict=False)],
  )
  rc = main(["score", str(est_path), str(gt_path), "--n_samples", "16"])
  err = capsys.readouterr().err
  assert rc == 2
  assert "body frames differ" in err
  assert "'imu'" in err and "'lidar'" in err


def test_score_body_frame_transform_applied(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # Same positions in both; estimate orientations rotated 90° about z relative
  # to GT (which is identity). The body-frame transform R = R_z(-π/2) should
  # bring them into agreement so scoring proceeds without saturating.
  import json as _json

  n = 30
  ts = np.linspace(0.0, 3.0, n)
  pos = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
  est_path = tmp_path / "est.SQUARE"
  gt_path = tmp_path / "gt.SQUARE"
  R_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  q_z90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
  est_steps = []
  cov = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
  for t, p in zip(ts, pos, strict=False):
    est_steps.append(
      GaussianStep(timestamp=t, translation=p, quat_xyzw=q_z90, covariance=cov)
    )
  gt_steps = [_det_step(t, p) for t, p in zip(ts, pos, strict=False)]
  _write(est_path, _gauss_header(body_frame="imu"), est_steps)
  _write(gt_path, _det_header(body_frame="lidar"), gt_steps)
  # T_est_body__gt_body: gt_body (identity-oriented) expressed in est_body
  # (z-90-rotated) coords → R = R_z(-π/2).
  extr = tmp_path / "extrinsic.json"
  extr.write_text(
    _json.dumps({"R": R_z90.T.flatten().tolist(), "t": [0, 0, 0]})
  )
  rc = main(
    [
      "score",
      str(est_path),
      str(gt_path),
      "--body-frame-transform",
      str(extr),
      "--n_samples",
      "16",
    ]
  )
  out = capsys.readouterr().out
  assert rc == 0
  # After the transform, rotation CRPS should be ≈ 0 (well below π/2 saturation).
  assert "Rotation CRPS" in out


# --- nees verb ----------------------------------------------------------------


def test_nees_consistent_three_line_verdict(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  est, gt = _write_calibration_case(tmp_path, scale=1.0)
  # --align none: the SE(3) Umeyama fit would absorb part of the injected
  # error and read exactly-calibrated data as slightly conservative.
  rc = main(["nees", str(est), str(gt), "--align", "none"])
  out = capsys.readouterr().out
  assert rc == 0
  assert "median NEES" in out
  assert "(calibrated: 2.37)" in out
  assert "consistent" in out
  assert "90% coverage" in out


def test_nees_overconfident_reports_scale_gap(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  est, gt = _write_calibration_case(tmp_path, scale=10.0)
  rc = main(["nees", str(est), str(gt)])
  out = capsys.readouterr().out
  assert rc == 0
  assert "covariance scale gap k" in out
  assert "too tight per axis" in out
  # coverage collapses when sigma is 10x understated
  assert "coverage: 0.0" in out


def test_nees_json(tmp_path: Path, capsys: pytest.CaptureFixture):
  import json as _json

  est, gt = _write_calibration_case(tmp_path, scale=3.0)
  rc = main(["nees", str(est), str(gt), "--json"])
  out = capsys.readouterr().out
  assert rc == 0
  v = _json.loads(out)
  assert v["dof"] == 3
  assert v["n"] == 200
  assert v["k"] == pytest.approx(v["median_nees"] / 2.3659738, rel=1e-6)
  assert 0.0 <= v["coverage"] <= 1.0
  assert v["anees"]["verdict"] == "optimistic"


def test_nees_rejects_deterministic(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  p = tmp_path / "det.SQUARE"
  g = tmp_path / "gt.SQUARE"
  steps = [_det_step(0.1 * i, np.array([i * 0.1, 0.0, 0.0])) for i in range(20)]
  _write(p, _det_header(), steps)
  _write(g, _det_header(), steps)
  rc = main(["nees", str(p), str(g)])
  err = capsys.readouterr().err
  assert rc == 2
  assert "gaussian_se3" in err


# --- pair verb ----------------------------------------------------------------


def _write_pair_case(
  tmp_path: Path, scale_a: float, n: int = 200
) -> tuple[Path, Path]:
  """Two gaussian filters on one latent track; A's true error is scale_a^2x."""
  rng = np.random.default_rng(5)
  ts = np.linspace(0.0, 20.0, n)
  latent = np.column_stack([ts * 0.5, np.sin(ts), np.zeros(n)])
  # _gauss_step publishes sigma_t = 0.1 m per axis
  a_pos = latent + rng.normal(scale=scale_a * 0.1, size=latent.shape)
  b_pos = latent + rng.normal(scale=0.1, size=latent.shape)
  a_path = tmp_path / "a.SQUARE"
  b_path = tmp_path / "b.SQUARE"
  _write(
    a_path,
    _gauss_header(),
    [_gauss_step(t, p) for t, p in zip(ts, a_pos, strict=False)],
  )
  _write(
    b_path,
    _gauss_header(),
    [_gauss_step(t, p) for t, p in zip(ts, b_pos, strict=False)],
  )
  return a_path, b_path


def test_pair_end_to_end_consistent(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  a, b = _write_pair_case(tmp_path, scale_a=1.0)
  rc = main(["pair", str(a), str(b)])
  out = capsys.readouterr().out
  assert rc == 0
  assert "propriety caveat" in out
  assert "pairwise median NEES" in out
  assert "lower bound" in out
  assert "verdict: consistent" in out
  assert "matched 200 pose pairs" in out


def test_pair_detects_overconfidence_without_reference(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  # A's true error is 5x its published sigma; B is calibrated. No ground
  # truth is consulted, yet the verdict certifies overconfidence.
  a, b = _write_pair_case(tmp_path, scale_a=5.0)
  rc = main(["pair", str(a), str(b)])
  out = capsys.readouterr().out
  assert rc == 0
  assert "verdict: optimistic" in out
  assert "pairwise scale gap k >=" in out


def test_pair_json(tmp_path: Path, capsys: pytest.CaptureFixture):
  import json as _json

  a, b = _write_pair_case(tmp_path, scale_a=5.0)
  rc = main(["pair", str(a), str(b), "--json"])
  out = capsys.readouterr().out
  assert rc == 0
  v = _json.loads(out)
  assert v["n_matched"] == 200
  assert v["verdict"]["anees"]["verdict"] == "optimistic"
  assert v["k_pair_lower_bound"] > 1.0
  assert "caveat" in v


def test_pair_rejects_deterministic(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  a = tmp_path / "a.SQUARE"
  b = tmp_path / "b.SQUARE"
  steps = [_det_step(0.1 * i, np.array([i * 0.1, 0.0, 0.0])) for i in range(20)]
  _write(a, _det_header(), steps)
  _write(b, _det_header(), steps)
  rc = main(["pair", str(a), str(b)])
  err = capsys.readouterr().err
  assert rc == 2
  assert "gaussian_se3" in err


def test_pair_body_frame_mismatch_needs_transform(
  tmp_path: Path, capsys: pytest.CaptureFixture
):
  rng = np.random.default_rng(5)
  ts = np.linspace(0.0, 5.0, 50)
  pos = np.column_stack([ts, np.zeros(50), np.zeros(50)])
  a = tmp_path / "a.SQUARE"
  b = tmp_path / "b.SQUARE"
  _write(
    a,
    _gauss_header(body_frame="imu"),
    [_gauss_step(t, p) for t, p in zip(ts, pos, strict=False)],
  )
  _write(
    b,
    _gauss_header(body_frame="lidar"),
    [_gauss_step(t, p) for t, p in zip(ts, pos, strict=False)],
  )
  rc = main(["pair", str(a), str(b)])
  err = capsys.readouterr().err
  assert rc == 2
  assert "body frames differ" in err
  del rng


# --- validate --strict ----------------------------------------------------------


def test_validate_strict_passes_on_real_fixture(capsys: pytest.CaptureFixture):
  fixture = (
    Path(__file__).parent / "fixtures/regression/real_point_lio/est.smfeval"
  )
  rc = main(["validate", "--strict", str(fixture)])
  out = capsys.readouterr().out
  assert rc == 0
  assert "strict checks passed" in out


def _zero_cov(cov: np.ndarray) -> None:
  cov[:] = 0.0


def _negate_first(cov: np.ndarray) -> None:
  cov[0, 0] = -1.0


def _inflate(cov: np.ndarray) -> None:
  cov[:] = np.eye(6) * 1e6


@pytest.mark.parametrize(
  ("mutate", "expected"),
  [
    (_zero_cov, "all-zero"),
    (_negate_first, "not positive definite"),
    (_inflate, "plausible range"),
  ],
  ids=["zero", "non-pd", "implausible"],
)
def test_validate_strict_rejects_bad_covariance(
  tmp_path: Path, capsys: pytest.CaptureFixture, mutate, expected
):
  p = tmp_path / "bad.SQUARE"
  step = _gauss_step(0.0, np.zeros(3))
  mutate(step.covariance)
  _write(p, _gauss_header(), [step])
  rc = main(["validate", "--strict", str(p)])
  err = capsys.readouterr().err
  assert rc == 2
  assert expected in err


def test_validate_without_strict_accepts_zero_covariance(tmp_path: Path):
  p = tmp_path / "zero.SQUARE"
  step = _gauss_step(0.0, np.zeros(3))
  step.covariance[:] = 0.0
  _write(p, _gauss_header(), [step])
  assert main(["validate", str(p)]) == 0
