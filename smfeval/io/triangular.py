"""Row-major lower-triangle packing of the 6x6 tangent covariance.

A SQUARE gaussian row and a covariance sidecar both store the 21 lower-triangle
entries `c11 c21 c22 c31 ... c66` in this order. Reader, writer, and the sidecar
loader share these two helpers so the packing cannot drift between them.
"""

import numpy as np

# (row, col) of each lower-triangle entry, row-major (21 pairs)
_LOWER_INDICES = [(i, j) for i in range(6) for j in range(i + 1)]


def pack_lower_triangular(cov: np.ndarray) -> list[float]:
  """Row-major lower-triangle entries of a symmetric 6x6 matrix (21 values)."""
  return [float(cov[i, j]) for i, j in _LOWER_INDICES]


def unpack_lower_triangular(entries: list[float]) -> np.ndarray:
  """Expand 21 row-major lower-triangle entries into a symmetric 6x6 matrix."""
  cov = np.zeros((6, 6))
  for k, (i, j) in enumerate(_LOWER_INDICES):
    cov[i, j] = cov[j, i] = entries[k]
  return cov
