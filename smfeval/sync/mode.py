from enum import Enum


class SyncMode(str, Enum):
  NEAREST = "nearest"
  INTERPOLATE_REF = "interpolate_ref"
