from enum import Enum


class SyncMode(str, Enum):
  NEAREST = "nearest"
  INTERPOLATE_GT = "interpolate_gt"
