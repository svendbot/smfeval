from typing import TextIO

from src.types import (
    FormatError,
    Gauge,
    Header,
    Representation,
    TangentConvention,
    TangentOrder,
    WeightFormat,
)

_COMMON_REQUIRED = (
    "FORMAT",
    "REPRESENTATION",
    "POSE_FRAME",
    "GAUGE",
    "TIMESTAMP_UNIT",
    "ALGORITHM",
    "ALGORITHM_VERSION",
)
_GAUSSIAN_REQUIRED = ("TANGENT_CONVENTION", "TANGENT_ORDER", "ROTATION_PARAM")
_ENSEMBLE_REQUIRED = ("WEIGHTED", "WEIGHT_FORMAT", "WEIGHTS_NORMALIZED")


def parse_header(f: TextIO) -> Header:
    raw: dict[str, str] = {}
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        s = line.strip()
        if not s:
            continue
        if s.startswith("#%"):
            key, _, val = s[2:].strip().partition(" ")
            if not key:
                raise FormatError(f"empty key in header line: {line!r}")
            up = key.upper()
            if up in raw:
                raise FormatError(f"duplicate header field: {up}")
            raw[up] = val.strip()
            continue
        if s.startswith("#"):
            continue
        f.seek(pos)
        break
    return _build(raw)


def write_header(f: TextIO, h: Header) -> None:
    def emit(key: str, val: str) -> None:
        f.write(f"#%{key} {val}\n")

    emit("FORMAT", h.format_version)
    emit("REPRESENTATION", h.representation.value)
    emit("POSE_FRAME", h.pose_frame)
    emit("GAUGE", h.gauge.value)
    emit("TIMESTAMP_UNIT", h.timestamp_unit)
    emit("ALGORITHM", h.algorithm)
    emit("ALGORITHM_VERSION", h.algorithm_version)
    if h.representation is Representation.GAUSSIAN_SE3:
        assert h.tangent_convention and h.tangent_order and h.rotation_param
        emit("TANGENT_CONVENTION", h.tangent_convention.value)
        emit("TANGENT_ORDER", h.tangent_order.value)
        emit("ROTATION_PARAM", h.rotation_param)
    elif h.representation is Representation.ENSEMBLE_SE3:
        assert h.weighted is not None and h.weight_format and h.weights_normalized is not None
        emit("WEIGHTED", "true" if h.weighted else "false")
        emit("WEIGHT_FORMAT", h.weight_format.value)
        emit("WEIGHTS_NORMALIZED", "true" if h.weights_normalized else "false")
        if h.particle_count_hint is not None:
            emit("PARTICLE_COUNT_HINT", str(h.particle_count_hint))


def _build(raw: dict[str, str]) -> Header:
    for k in _COMMON_REQUIRED:
        if k not in raw:
            raise FormatError(f"missing required header field: {k}")
    fmt = raw["FORMAT"]
    if not fmt.startswith("smfeval/"):
        raise FormatError(f"unrecognized FORMAT: {fmt!r}")

    rep = _enum(Representation, "REPRESENTATION", raw["REPRESENTATION"])
    gauge = _enum(Gauge, "GAUGE", raw["GAUGE"])

    h = Header(
        format_version=fmt,
        representation=rep,
        pose_frame=raw["POSE_FRAME"],
        gauge=gauge,
        timestamp_unit=raw["TIMESTAMP_UNIT"],
        algorithm=raw["ALGORITHM"],
        algorithm_version=raw["ALGORITHM_VERSION"],
    )

    if rep is Representation.GAUSSIAN_SE3:
        for k in _GAUSSIAN_REQUIRED:
            if k not in raw:
                raise FormatError(f"missing field for gaussian_se3: {k}")
        h.tangent_convention = _enum(
            TangentConvention, "TANGENT_CONVENTION", raw["TANGENT_CONVENTION"]
        )
        h.tangent_order = _enum(TangentOrder, "TANGENT_ORDER", raw["TANGENT_ORDER"])
        if raw["ROTATION_PARAM"] != "axis_angle":
            raise FormatError(
                f"unsupported ROTATION_PARAM: {raw['ROTATION_PARAM']!r} (expected axis_angle)"
            )
        h.rotation_param = raw["ROTATION_PARAM"]
    elif rep is Representation.ENSEMBLE_SE3:
        for k in _ENSEMBLE_REQUIRED:
            if k not in raw:
                raise FormatError(f"missing field for ensemble_se3: {k}")
        h.weighted = _bool("WEIGHTED", raw["WEIGHTED"])
        h.weight_format = _enum(WeightFormat, "WEIGHT_FORMAT", raw["WEIGHT_FORMAT"])
        h.weights_normalized = _bool("WEIGHTS_NORMALIZED", raw["WEIGHTS_NORMALIZED"])
        if "PARTICLE_COUNT_HINT" in raw:
            try:
                h.particle_count_hint = int(raw["PARTICLE_COUNT_HINT"])
            except ValueError as e:
                raise FormatError(f"PARTICLE_COUNT_HINT not an integer: {raw['PARTICLE_COUNT_HINT']!r}") from e

    return h


def _enum(cls, key: str, value: str):
    try:
        return cls(value)
    except ValueError as e:
        allowed = ", ".join(m.value for m in cls)
        raise FormatError(f"invalid {key}: {value!r} (allowed: {allowed})") from e


def _bool(key: str, value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise FormatError(f"{key} must be 'true' or 'false', got {value!r}")
