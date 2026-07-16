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


def nearest_indices(sorted_vals: np.ndarray, queries: np.ndarray) -> np.ndarray:
  """Index of the nearest entry in ``sorted_vals`` for each query.

  ``sorted_vals`` must be ascending; ties pick the earlier index.
  """
  j = np.clip(np.searchsorted(sorted_vals, queries), 0, sorted_vals.size - 1)
  jl = np.maximum(j - 1, 0)
  take_left = np.abs(sorted_vals[jl] - queries) <= np.abs(
    sorted_vals[j] - queries
  )
  return np.where(take_left, jl, j)


def match_timestamps(
  est_ts: np.ndarray,
  ref_ts: np.ndarray,
  t_max_diff: float = 0.01,
  t_offset: float = 0.0,
) -> MatchResult:
  """Nearest-neighbor matching with tolerance.

  For each estimate timestamp the nearest reference timestamp is
  selected; pairs above `t_max_diff` are dropped. `t_offset` is added to
  estimate timestamps before matching to correct for known clock skew.
  """
  est_ts = np.asarray(est_ts, dtype=float)
  ref_ts = np.asarray(ref_ts, dtype=float)
  shifted = est_ts + t_offset

  order = np.argsort(ref_ts, kind="stable")
  j = order[nearest_indices(ref_ts[order], shifted)]
  all_gaps = np.abs(shifted - ref_ts[j])
  keep = all_gaps <= t_max_diff

  est_idx = np.flatnonzero(keep)
  ref_idx = j[keep]
  gaps = all_gaps[keep] if est_idx.size else np.zeros(0)

  return MatchResult(
    est_indices=est_idx,
    ref_indices=ref_idx,
    n_total=len(est_ts),
    n_matched=int(est_idx.size),
    n_dropped=int(len(est_ts) - est_idx.size),
    gap_seconds=gaps,
  )
