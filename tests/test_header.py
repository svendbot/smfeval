import io

import pytest

from src import (
  FORMAT_VERSION,
  FormatError,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
  WeightFormat,
)
from src.io import parse_header, write_header

COMMON = f"""\
#%FORMAT {FORMAT_VERSION}
#%REPRESENTATION {{rep}}
#%POSE_FRAME world
#%BODY_FRAME imu
#%GAUGE {{gauge}}
#%TIMESTAMP_UNIT seconds
#%ALGORITHM testbot
#%ALGORITHM_VERSION 1.0
"""

GAUSSIAN_TAIL = """\
#%TANGENT_CONVENTION right_perturbation
#%TANGENT_ORDER translation_rotation
#%ROTATION_PARAM axis_angle
"""

ENSEMBLE_TAIL = """\
#%WEIGHTED true
#%WEIGHT_FORMAT log
#%WEIGHTS_NORMALIZED false
#%PARTICLE_COUNT_HINT 500
"""


def _parse(text: str) -> SquareHeader:
  return parse_header(io.StringIO(text))


def test_parse_gaussian():
  h = _parse(
    COMMON.format(rep="gaussian_se3", gauge="gravity_yaw") + GAUSSIAN_TAIL
  )
  assert h.format_version == FORMAT_VERSION
  assert h.body_frame == "imu"
  assert h.representation is Representation.GAUSSIAN_SE3
  assert h.gauge is Gauge.GRAVITY_YAW
  assert h.tangent_convention is TangentConvention.RIGHT
  assert h.tangent_order is TangentOrder.TRANS_ROT


def test_parse_ensemble_optional_count():
  h = _parse(COMMON.format(rep="ensemble_se3", gauge="sim3") + ENSEMBLE_TAIL)
  assert h.representation is Representation.ENSEMBLE_SE3
  assert h.weighted is True
  assert h.weight_format is WeightFormat.LOG
  assert h.weights_normalized is False
  assert h.particle_count_hint == 500


def test_parse_deterministic_minimal():
  h = _parse(COMMON.format(rep="deterministic", gauge="fixed"))
  assert h.representation is Representation.DETERMINISTIC
  assert h.gauge is Gauge.FIXED


def test_invalid_gauge_rejected():
  with pytest.raises(FormatError, match="invalid GAUGE"):
    _parse(COMMON.format(rep="deterministic", gauge="bogus"))


def test_missing_common_field():
  text = COMMON.format(rep="deterministic", gauge="fixed")
  text = text.replace("#%GAUGE fixed\n", "")
  with pytest.raises(FormatError, match="missing required header field: GAUGE"):
    _parse(text)


def test_missing_gaussian_field():
  text = COMMON.format(rep="gaussian_se3", gauge="se3") + GAUSSIAN_TAIL
  text = text.replace("#%TANGENT_ORDER translation_rotation\n", "")
  with pytest.raises(
    FormatError, match="missing field for gaussian_se3: TANGENT_ORDER"
  ):
    _parse(text)


def test_unsupported_rotation_param():
  text = COMMON.format(rep="gaussian_se3", gauge="se3") + GAUSSIAN_TAIL
  text = text.replace("axis_angle", "quaternion")
  with pytest.raises(FormatError, match="unsupported ROTATION_PARAM"):
    _parse(text)


def test_duplicate_field_rejected():
  text = COMMON.format(rep="deterministic", gauge="fixed") + "#%GAUGE se3\n"
  with pytest.raises(FormatError, match="duplicate header field"):
    _parse(text)


def test_stops_at_first_data_line():
  text = (
    COMMON.format(rep="deterministic", gauge="fixed")
    + "1.234567 0 0 0 0 0 0 1\n"
  )
  f = io.StringIO(text)
  parse_header(f)
  assert f.readline().startswith("1.234567")


def test_blank_and_plain_comment_lines_skipped():
  text = (
    "# regular comment\n"
    "\n"
    + COMMON.format(rep="deterministic", gauge="fixed")
    + "\n"
    + "1.0 0 0 0 0 0 0 1\n"
  )
  f = io.StringIO(text)
  h = parse_header(f)
  assert h.representation is Representation.DETERMINISTIC
  assert f.readline().startswith("1.0")


@pytest.mark.parametrize(
  "rep,tail,gauge",
  [
    (Representation.GAUSSIAN_SE3, GAUSSIAN_TAIL, "gravity_yaw"),
    (Representation.ENSEMBLE_SE3, ENSEMBLE_TAIL, "sim3"),
    (Representation.DETERMINISTIC, "", "fixed"),
  ],
)
def test_round_trip(rep, tail, gauge):
  src = COMMON.format(rep=rep.value, gauge=gauge) + tail
  h = _parse(src)
  out = io.StringIO()
  write_header(out, h)
  h2 = _parse(out.getvalue())
  assert h == h2
