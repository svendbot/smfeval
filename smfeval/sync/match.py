from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
  est_indices: np.ndarray
  ref_indices: np.ndarray
  n_total: int
  n_matched: int
  n_dropped: int
  gap_seconds: np.ndarray

  @property
  def gap_quantiles_ms(self) -> dict[str, float]:
    if self.gap_seconds.size == 0:
      return {"median": 0.0, "p95": 0.0, "p99": 0.0}
    ms = self.gap_seconds * 1e3
    return {
      "median": float(np.median(ms)),
      "p95": float(np.quantile(ms, 0.95)),
      "p99": float(np.quantile(ms, 0.99)),
    }


def _matching_time_indices(
  stamps_1: np.ndarray, stamps_2: np.ndarray, max_diff: float, offset_2: float
) -> tuple[list[int], list[int]]:
  matching_1: list[int] = []
  matching_2: list[int] = []
  s2 = stamps_2 + offset_2
  for i, t1 in enumerate(stamps_1):
    diffs = np.abs(s2 - t1)
    j = int(np.argmin(diffs))
    if diffs[j] <= max_diff:
      matching_1.append(i)
      matching_2.append(j)
  return matching_1, matching_2


def match_timestamps(
  est_ts: np.ndarray,
  ref_ts: np.ndarray,
  t_max_diff: float = 0.01,
  t_offset: float = 0.0,
) -> MatchResult:
  """Nearest-neighbor matching with tolerance.

  Iterates over estimate timestamps and selects the nearest reference
  timestamp; pairs above `t_max_diff` are dropped. `t_offset` is added to
  estimate timestamps before matching to correct for known clock skew.
  """
  est_ts = np.asarray(est_ts, dtype=float)
  ref_ts = np.asarray(ref_ts, dtype=float)

  m_est, m_ref = _matching_time_indices(
    est_ts + t_offset, ref_ts, t_max_diff, 0.0
  )
  est_idx = np.array(m_est, dtype=int)
  ref_idx = np.array(m_ref, dtype=int)

  if est_idx.size:
    gaps = np.abs((est_ts[est_idx] + t_offset) - ref_ts[ref_idx])
  else:
    gaps = np.zeros(0)

  return MatchResult(
    est_indices=est_idx,
    ref_indices=ref_idx,
    n_total=len(est_ts),
    n_matched=int(est_idx.size),
    n_dropped=int(len(est_ts) - est_idx.size),
    gap_seconds=gaps,
  )
