import io

import numpy as np
import pytest

from src.io import iter_steps, parse_header, write_header, write_steps
from src.steps import DeterministicStep, EnsembleStep, GaussianStep
from src.types import (
    FormatError,
    Gauge,
    Header,
    Representation,
    TangentConvention,
    TangentOrder,
    WeightFormat,
)


RNG = np.random.default_rng(42)


def _gaussian_header() -> Header:
    return Header(
        format_version="smfeval/0.2",
        representation=Representation.GAUSSIAN_SE3,
        pose_frame="world",
        gauge=Gauge.GRAVITY_YAW,
        timestamp_unit="seconds",
        algorithm="testbot",
        algorithm_version="1.0",
        tangent_convention=TangentConvention.RIGHT,
        tangent_order=TangentOrder.TRANS_ROT,
        rotation_param="axis_angle",
    )


def _ensemble_header(weighted: bool = True) -> Header:
    return Header(
        format_version="smfeval/0.2",
        representation=Representation.ENSEMBLE_SE3,
        pose_frame="world",
        gauge=Gauge.SIM3,
        timestamp_unit="seconds",
        algorithm="testbot",
        algorithm_version="1.0",
        weighted=weighted,
        weight_format=WeightFormat.LOG,
        weights_normalized=False,
    )


def _deterministic_header() -> Header:
    return Header(
        format_version="smfeval/0.2",
        representation=Representation.DETERMINISTIC,
        pose_frame="world",
        gauge=Gauge.FIXED,
        timestamp_unit="seconds",
        algorithm="gt",
        algorithm_version="1.0",
    )


def _random_cov() -> np.ndarray:
    A = RNG.normal(size=(6, 6))
    return A @ A.T + np.eye(6)


def _random_gaussian_step(ts: float) -> GaussianStep:
    return GaussianStep(
        timestamp=ts,
        translation=RNG.normal(size=3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        covariance=_random_cov(),
    )


def _random_ensemble_step(ts: float, n: int, weighted: bool) -> EnsembleStep:
    poses = np.zeros((n, 7))
    poses[:, :3] = RNG.normal(size=(n, 3))
    poses[:, 6] = 1.0
    weights = RNG.normal(size=n) if weighted else None
    return EnsembleStep(timestamp=ts, particles=poses, weights=weights)


def _round_trip(header: Header, steps: list) -> list:
    buf = io.StringIO()
    write_header(buf, header)
    write_steps(buf, steps, header)
    buf.seek(0)
    h2 = parse_header(buf)
    assert h2 == header
    return list(iter_steps(buf, h2))


def test_gaussian_round_trip():
    h = _gaussian_header()
    steps = [_random_gaussian_step(t) for t in (1.0, 2.0, 3.0)]
    out = _round_trip(h, steps)
    assert len(out) == 3
    for a, b in zip(steps, out):
        assert abs(a.timestamp - b.timestamp) < 1e-9
        assert np.allclose(a.translation, b.translation)
        assert np.allclose(a.quat_xyzw, b.quat_xyzw)
        assert np.allclose(a.covariance, b.covariance)
        assert np.allclose(b.covariance, b.covariance.T)


def test_gaussian_lower_triangular_packing():
    h = _gaussian_header()
    cov = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            cov[i, j] = cov[j, i] = float(i * 10 + j)
    step = GaussianStep(
        timestamp=1.0,
        translation=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        covariance=cov,
    )
    out = _round_trip(h, [step])[0]
    assert np.allclose(out.covariance, cov)


def test_gaussian_wrong_field_count_raises():
    h = _gaussian_header()
    src = io.StringIO()
    write_header(src, h)
    src.write("1.0 0 0 0 0 0 0 1\n")  # only 8 fields
    src.seek(0)
    parse_header(src)
    with pytest.raises(FormatError, match="expected 29"):
        list(iter_steps(src, h))


def test_ensemble_weighted_round_trip():
    h = _ensemble_header(weighted=True)
    steps = [
        _random_ensemble_step(1.0, 3, True),
        _random_ensemble_step(2.0, 5, True),
        _random_ensemble_step(3.0, 2, True),
    ]
    out = _round_trip(h, steps)
    assert len(out) == 3
    for a, b in zip(steps, out):
        assert abs(a.timestamp - b.timestamp) < 1e-9
        assert np.allclose(a.particles, b.particles)
        assert b.weights is not None
        assert np.allclose(a.weights, b.weights)


def test_ensemble_unweighted_round_trip():
    h = _ensemble_header(weighted=False)
    steps = [_random_ensemble_step(1.0, 4, False)]
    out = _round_trip(h, steps)
    assert out[0].weights is None
    assert np.allclose(steps[0].particles, out[0].particles)


def test_ensemble_grouping_by_string_equality():
    """Particles with the same timestamp string land in one step even if floats differ."""
    h = _ensemble_header(weighted=False)
    src = io.StringIO()
    write_header(src, h)
    src.write("1.000000000 0 0 0 0 0 0 0 1\n")
    src.write("1.000000000 1 1 0 0 0 0 0 1\n")
    src.write("2.000000000 0 2 0 0 0 0 0 1\n")
    src.seek(0)
    parse_header(src)
    out = list(iter_steps(src, h))
    assert len(out) == 2
    assert out[0].particles.shape == (2, 7)
    assert out[1].particles.shape == (1, 7)


def test_ensemble_varying_size_per_step():
    h = _ensemble_header(weighted=True)
    steps = [
        _random_ensemble_step(1.0, 100, True),
        _random_ensemble_step(2.0, 50, True),
        _random_ensemble_step(3.0, 200, True),
    ]
    out = _round_trip(h, steps)
    assert [s.particles.shape[0] for s in out] == [100, 50, 200]


def test_deterministic_round_trip():
    h = _deterministic_header()
    steps = [
        DeterministicStep(
            timestamp=t,
            translation=RNG.normal(size=3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        for t in (1.0, 2.0, 3.0)
    ]
    out = _round_trip(h, steps)
    assert len(out) == 3


def test_truncated_final_gaussian_row_tolerated():
    h = _gaussian_header()
    src = io.StringIO()
    write_header(src, h)
    write_steps(src, [_random_gaussian_step(1.0), _random_gaussian_step(2.0)], h)
    text = src.getvalue()
    truncated = text.rstrip("\n")[:-20]  # mid-line crash → no trailing newline
    f = io.StringIO(truncated)
    parse_header(f)
    out = list(iter_steps(f, h))
    assert len(out) == 1
    assert abs(out[0].timestamp - 1.0) < 1e-9


def test_truncated_final_ensemble_row_tolerated():
    h = _ensemble_header(weighted=False)
    src = io.StringIO()
    write_header(src, h)
    src.write("1.000000000 0 0 0 0 0 0 0 1\n")
    src.write("1.000000000 1 1 0 0 0 0 0 1\n")
    src.write("2.000000000 0 5 0 0 0 0 0 1\n")
    src.write("2.000000000 1 6 0")  # truncated, no trailing newline
    src.seek(0)
    parse_header(src)
    out = list(iter_steps(src, h))
    assert len(out) == 2
    assert out[0].particles.shape[0] == 2
    assert out[1].particles.shape[0] == 1


def test_blank_lines_in_data_skipped():
    h = _deterministic_header()
    src = io.StringIO()
    write_header(src, h)
    src.write("1.0 0 0 0 0 0 0 1\n")
    src.write("\n")
    src.write("2.0 1 0 0 0 0 0 1\n")
    src.seek(0)
    parse_header(src)
    out = list(iter_steps(src, h))
    assert len(out) == 2


def test_writer_rejects_mismatched_step_type():
    h = _gaussian_header()
    buf = io.StringIO()
    with pytest.raises(ValueError, match="ensemble_se3"):
        write_steps(buf, [_random_ensemble_step(1.0, 2, True)], h)
