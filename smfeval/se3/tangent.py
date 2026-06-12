import numpy as np

from smfeval.format import TangentOrder


def reorder_tangent(
  xi: np.ndarray, src: TangentOrder, dst: TangentOrder
) -> np.ndarray:
  if src is dst:
    return xi
  return np.concatenate([xi[3:], xi[:3]])


def reorder_covariance(
  cov: np.ndarray, src: TangentOrder, dst: TangentOrder
) -> np.ndarray:
  if src is dst:
    return cov
  P = np.block(
    [
      [np.zeros((3, 3)), np.eye(3)],
      [np.eye(3), np.zeros((3, 3))],
    ]
  )
  return P @ cov @ P.T
