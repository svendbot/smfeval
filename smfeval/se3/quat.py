import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
  return Rotation.from_quat(q).as_matrix()


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
  return Rotation.from_matrix(R).as_quat()
