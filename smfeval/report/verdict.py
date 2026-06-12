r"""Compact verdict blocks for the ``nees`` and ``pair`` verbs.

The verdict is the product: three lines a stranger can read without the
paper. The covariance scale gap is

.. math::

   k = \frac{\operatorname{median}\,\mathrm{NEES}}{\chi^2_{d,0.5}}

— the factor by which the published covariance is too tight (k > 1) or
too loose (k < 1), since under a calibrated belief the per-pose NEES is
:math:`\chi^2_d` distributed (median 2.366 for d=3). The per-axis factor
is :math:`\sqrt{k}` (variance vs standard deviation). The qualitative
direction comes from the two-sided ANEES :math:`\chi^2` interval, so a
filter within sampling noise of calibrated prints "consistent" even when
k is not exactly 1.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import chi2

from smfeval.scoring.logscore import AneesResult, anees_consistency

_DIRECTION = {
  "optimistic": "too tight",
  "conservative": "too loose",
  "consistent": "consistent",
  "undefined": "undefined",
}


@dataclass
class NeesVerdict:
  median_nees: float
  calibrated_median: float  # chi2.ppf(0.5, dof)
  k: float  # median_nees / calibrated_median
  per_axis_factor: float  # sqrt(k)
  coverage: float  # fraction inside the nominal ellipsoid
  nominal_coverage: float
  dof: int
  n: int
  direction: str  # too tight | too loose | consistent
  anees: AneesResult

  def to_dict(self) -> dict:
    return asdict(self)


def nees_verdict(
  nees: np.ndarray,
  *,
  dof: int = 3,
  nominal: float = 0.90,
  alpha: float = 0.05,
) -> NeesVerdict:
  """Summarize a per-pose NEES series into the three-line verdict."""
  vals = np.asarray(nees, dtype=float)
  vals = vals[np.isfinite(vals)]
  cal_median = float(chi2.ppf(0.5, df=dof))
  q_nominal = float(chi2.ppf(nominal, df=dof))
  res = anees_consistency(vals, dof=dof, alpha=alpha)
  median = float(np.median(vals)) if vals.size else float("nan")
  k = median / cal_median
  return NeesVerdict(
    median_nees=median,
    calibrated_median=cal_median,
    k=k,
    per_axis_factor=float(np.sqrt(k)) if k >= 0 else float("nan"),
    coverage=float((vals <= q_nominal).mean()) if vals.size else float("nan"),
    nominal_coverage=nominal,
    dof=dof,
    n=int(vals.size),
    direction=_DIRECTION[res.verdict],
    anees=res,
  )


def _fmt(x: float) -> str:
  """Compact magnitude formatting: 2.37, 12.4, 537, 5.63e3."""
  if not np.isfinite(x):
    return "nan"
  if x == 0:
    return "0"
  if abs(x) >= 1e3 or abs(x) < 1e-2:
    return f"{x:.3g}".replace("e+0", "e").replace("e-0", "e-")
  return f"{x:.3g}"


def render_nees_verdict(v: NeesVerdict) -> str:
  pct = int(round(v.nominal_coverage * 100))
  if v.direction == "consistent":
    gap = f"covariance scale consistent (k = {_fmt(v.k)})"
  else:
    gap = (
      f"covariance scale gap k = {_fmt(v.k)}, "
      f"~{_fmt(v.per_axis_factor)}x {v.direction} per axis"
    )
  return "\n".join(
    [
      f"median NEES {_fmt(v.median_nees)}   "
      f"(calibrated: {v.calibrated_median:.2f})",
      gap,
      f"{pct}% coverage: {v.coverage:.3f}  "
      f"(calibrated: {v.nominal_coverage:.3f})",
    ]
  )


def render_pair_verdict(v: NeesVerdict, *, caveat: str) -> str:
  """Pair variant: k only lower-bounds the gap (see the propriety caveat)."""
  if v.direction == "consistent":
    gap = f"pairwise scale gap k >= {_fmt(v.k)}  (lower bound)"
  else:
    gap = (
      f"pairwise scale gap k >= {_fmt(v.k)}, "
      f">={_fmt(v.per_axis_factor)}x {v.direction} per axis  (lower bound)"
    )
  lines = [
    caveat,
    "",
    f"pairwise median NEES {_fmt(v.median_nees)}   "
    f"(calibrated: {v.calibrated_median:.2f})",
    gap,
    f"verdict: {v.anees.verdict}  "
    f"(ANEES {_fmt(v.anees.anees)} vs chi2 interval "
    f"[{_fmt(v.anees.lo)}, {_fmt(v.anees.hi)}])",
  ]
  return "\n".join(lines)
