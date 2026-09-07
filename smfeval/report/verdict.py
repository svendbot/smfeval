r"""Compact verdict blocks for the ``nees`` and ``pair`` verbs.

The verdict is the product: three lines a stranger can read without any
further reading. The covariance scale gap is

.. math::

   k = \frac{\operatorname{median}\,\mathrm{NEES}}{\chi^2_{d,0.5}}

— the factor by which the published covariance is too tight (k > 1) or
too loose (k < 1), since under a calibrated belief the per-pose NEES is
:math:`\chi^2_d` distributed (median 2.366 for d=3).

A published :math:`\Sigma = c\,\Sigma_\mathrm{true}` gives :math:`k = 1/c`,
so per axis the standard deviation is off by :math:`\sqrt{c}`: a factor
:math:`\sqrt{k}` when it is too tight, :math:`1/\sqrt{k}` when it is too
loose. The reported per-axis factor is therefore always :math:`\ge 1` and
``scale_direction`` says which way it points.

Whether a gap is reported at all comes from the two-sided ANEES
:math:`\chi^2` interval, so a filter within sampling noise of calibrated
prints "consistent" even when k is not exactly 1.
"""

import re
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from smfeval.scoring.logscore import AneesResult, anees_consistency
from smfeval.scoring.pairwise import PairResult

_DIRECTION = {
  "optimistic": "too tight",
  "conservative": "too loose",
  "consistent": "consistent",
  "undefined": "undefined",
}


@dataclass
class NeesVerdict:
  """Verdict statistics; the k/direction fields derive from these."""

  coverage: float  # fraction inside the nominal ellipsoid
  nominal_coverage: float
  dof: int
  anees: AneesResult

  @property
  def median_nees(self) -> float:
    return self.anees.median

  @property
  def n(self) -> int:
    return self.anees.n

  @property
  def calibrated_median(self) -> float:
    return float(chi2.ppf(0.5, df=self.dof))

  @property
  def k(self) -> float:
    return self.median_nees / self.calibrated_median

  @property
  def per_axis_factor(self) -> float:
    r"""Per-axis sigma discrepancy implied by k, as a magnitude :math:`\ge 1`.

    :math:`\sqrt{k}` when the covariance is too tight, :math:`1/\sqrt{k}`
    when it is too loose. Reporting :math:`\sqrt{k}` in both directions
    understated an under-confident filter: k = 0.25 means sigma is 2x too
    loose per axis, not 0.5x. Pair with :attr:`scale_direction`.
    """
    if not np.isfinite(self.k) or self.k <= 0.0:
      return float("nan")
    root = float(np.sqrt(self.k))
    return root if self.k >= 1.0 else 1.0 / root

  @property
  def scale_direction(self) -> str:
    """Which way k points, independent of the ANEES consistency verdict."""
    if not np.isfinite(self.k) or self.k <= 0.0:
      return "undefined"
    return "too tight" if self.k >= 1.0 else "too loose"

  @property
  def direction(self) -> str:
    return _DIRECTION[self.anees.verdict]

  def to_dict(self) -> dict:
    return {
      "median_nees": self.median_nees,
      "calibrated_median": self.calibrated_median,
      "k": self.k,
      "per_axis_factor": self.per_axis_factor,
      "scale_direction": self.scale_direction,
      "coverage": self.coverage,
      "nominal_coverage": self.nominal_coverage,
      "dof": self.dof,
      "n": self.n,
      "direction": self.direction,
      "anees": self.anees.to_dict(),
    }


def nees_verdict(
  nees: np.ndarray,
  *,
  dof: int = 3,
  nominal: float = 0.90,
  alpha: float = 0.05,
  anees: AneesResult | None = None,
) -> NeesVerdict:
  """Summarize a per-pose NEES series into the three-line verdict.

  Pass ``anees`` when an :func:`anees_consistency` result for the same
  series is already in hand (e.g. from :class:`PairResult`) to avoid
  recomputing it.
  """
  vals = np.asarray(nees, dtype=float)
  vals = vals[np.isfinite(vals)]
  q_nominal = float(chi2.ppf(nominal, df=dof))
  res = (
    anees
    if anees is not None
    else anees_consistency(vals, dof=dof, alpha=alpha)
  )
  return NeesVerdict(
    coverage=float((vals <= q_nominal).mean()) if vals.size else float("nan"),
    nominal_coverage=nominal,
    dof=dof,
    anees=res,
  )


def _fmt(x: float) -> str:
  """Compact magnitude formatting: 2.37, 12.4, 537, 5.63e3, 4.2e10."""
  if not np.isfinite(x):
    return "nan"
  return re.sub(r"e\+?(-?)0*(\d)", r"e\1\2", f"{x:.3g}")


def render_nees_verdict(v: NeesVerdict) -> str:
  pct = int(round(v.nominal_coverage * 100))
  if v.direction == "consistent":
    gap = f"covariance scale consistent (k = {_fmt(v.k)})"
  else:
    # The gap clause is a statement about k, so its direction word comes
    # from k too; the ANEES verdict only decides whether to print it.
    gap = (
      f"covariance scale gap k = {_fmt(v.k)}, "
      f"~{_fmt(v.per_axis_factor)}x {v.scale_direction} per axis"
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


def pair_verdict_dict(res: PairResult, v: NeesVerdict, caveat: str) -> dict:
  """The pair verb's machine-readable schema (pinned by pair_smoke golden)."""
  return {
    "n_matched": res.n_matched,
    "n_scored": res.n_scored,
    "join_frac": res.join_frac,
    "gap_median_ms": res.gap_median_ms,
    "med_d_norm_m": res.med_d_norm,
    "log_score_mean": res.log_score_mean,
    "k_pair_lower_bound": res.k_pair,
    "k_axis_lower_bound": list(res.k_axis),
    "verdict": v.to_dict(),
    "caveat": caveat,
  }


def render_pair_verdict(res: PairResult, v: NeesVerdict, *, caveat: str) -> str:
  """Pair variant: k only lower-bounds the gap (see the propriety caveat)."""
  if v.direction == "consistent":
    gap = f"pairwise scale gap k >= {_fmt(v.k)}  (lower bound)"
  else:
    gap = (
      f"pairwise scale gap k >= {_fmt(v.k)}, "
      f">={_fmt(v.per_axis_factor)}x {v.scale_direction} per axis  "
      "(lower bound)"
    )
  lines = [
    f"matched {res.n_matched} pose pairs, scored {res.n_scored}  "
    f"(join {res.join_frac:.2f}, median gap {res.gap_median_ms:.1f} ms)",
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
