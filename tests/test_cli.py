from pathlib import Path

import numpy as np
import pytest

from src.cli.main import main
from src.io import write_header, write_steps
from src.steps import DeterministicStep, GaussianStep
from src.types import (
    Gauge,
    Header,
    Representation,
    TangentConvention,
    TangentOrder,
)


def _gauss_header(gauge: Gauge = Gauge.SE3) -> Header:
    return Header(
        format_version="smfeval/0.2",
        representation=Representation.GAUSSIAN_SE3,
        pose_frame="world",
        gauge=gauge,
        timestamp_unit="seconds",
        algorithm="testbot",
        algorithm_version="1.0",
        tangent_convention=TangentConvention.RIGHT,
        tangent_order=TangentOrder.TRANS_ROT,
        rotation_param="axis_angle",
    )


def _det_header() -> Header:
    return Header(
        format_version="smfeval/0.2",
        representation=Representation.DETERMINISTIC,
        pose_frame="world",
        gauge=Gauge.FIXED,
        timestamp_unit="seconds",
        algorithm="gt",
        algorithm_version="1.0",
    )


def _write(path: Path, header: Header, steps: list) -> None:
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
    p = tmp_path / "f.smfeval"
    _write(p, _det_header(), [_det_step(0.0, np.zeros(3)), _det_step(1.0, np.ones(3))])
    rc = main(["validate", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "deterministic" in out
    assert "2 timesteps" in out


def test_validate_bad_file(tmp_path: Path):
    p = tmp_path / "broken.smfeval"
    p.write_text("#%FORMAT smfeval/0.2\n")  # missing required fields
    rc = main(["validate", str(p)])
    assert rc == 2


def test_score_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture):
    rng = np.random.default_rng(0)
    n = 30
    ts = np.linspace(0.0, 3.0, n)
    gt_pos = np.column_stack([ts * 0.5, np.zeros(n), np.zeros(n)])
    est_pos = gt_pos + rng.normal(scale=0.02, size=gt_pos.shape)

    est_path = tmp_path / "est.smfeval"
    gt_path = tmp_path / "gt.smfeval"
    _write(est_path, _gauss_header(), [_gauss_step(t, p) for t, p in zip(ts, est_pos)])
    _write(gt_path, _det_header(), [_det_step(t, p) for t, p in zip(ts, gt_pos)])

    rc = main(["score", str(est_path), str(gt_path), "--n_samples", "32"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "smfeval scoring report" in out
    assert "Synchronization" in out
    assert "Alignment" in out
    assert "Scores" in out
    assert "Translation CRPS" in out
    assert "Calibration" in out


def test_score_no_matches(tmp_path: Path, capsys: pytest.CaptureFixture):
    est_path = tmp_path / "est.smfeval"
    gt_path = tmp_path / "gt.smfeval"
    _write(est_path, _gauss_header(), [_gauss_step(0.0, np.zeros(3))])
    _write(gt_path, _det_header(), [_det_step(100.0, np.zeros(3))])
    rc = main(["score", str(est_path), str(gt_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no matched pairs" in err


def test_help_smoke():
    with pytest.raises(SystemExit):
        main(["--help"])
