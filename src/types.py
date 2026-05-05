from dataclasses import dataclass
from enum import Enum


class FormatError(ValueError):
    pass


class Representation(str, Enum):
    GAUSSIAN_SE3 = "gaussian_se3"
    ENSEMBLE_SE3 = "ensemble_se3"
    DETERMINISTIC = "deterministic"


class Gauge(str, Enum):
    FIXED = "fixed"
    SE3 = "se3"
    GRAVITY_YAW = "gravity_yaw"
    SIM3 = "sim3"


class TangentConvention(str, Enum):
    RIGHT = "right_perturbation"
    LEFT = "left_perturbation"


class TangentOrder(str, Enum):
    TRANS_ROT = "translation_rotation"
    ROT_TRANS = "rotation_translation"


class WeightFormat(str, Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass
class Header:
    format_version: str
    representation: Representation
    pose_frame: str
    gauge: Gauge
    timestamp_unit: str
    algorithm: str
    algorithm_version: str

    tangent_convention: TangentConvention | None = None
    tangent_order: TangentOrder | None = None
    rotation_param: str | None = None

    weighted: bool | None = None
    weight_format: WeightFormat | None = None
    weights_normalized: bool | None = None
    particle_count_hint: int | None = None
