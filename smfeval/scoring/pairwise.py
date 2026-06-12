r"""No-reference pairwise calibration: score two filters against each other.

Align A to B directly (Umeyama on matched translations — ground truth is
never consulted), form the per-pose tangent difference :math:`d_t`, and
score it under :math:`N(0, \Sigma_A + \Sigma_B)`:

.. math::

   \mathrm{NEES}_\mathrm{pair}
     = d^\top (\Sigma_A + \Sigma_B)^{-1} d
     \sim \chi^2_3 \quad \text{(translation slice, median 2.366)}.

An elevated pairwise NEES certifies overconfidence with no reference
consulted — the structural advantage over GT-only evaluation.

Propriety caveat (applies to every number this module emits): the
pairwise score is strictly proper only if the reference filter's stated
:math:`\Sigma` is its true error covariance and the two filters' errors
are independent. Both violations push *conservative*: an understated
:math:`\Sigma_B` shrinks :math:`\Sigma_\mathrm{eff}`, and common-mode
error cancels in :math:`d`. :math:`\mathrm{NEES}_\mathrm{pair}` is
therefore a LOWER BOUND on miscalibration. Quantitatively, if A
understates its covariance by a factor :math:`k` against an otherwise
calibrated B with equal *true* error covariance, the pairwise NEES
concentrates on :math:`(2k/(k+1))\,\chi^2_3`; with equal *published*
covariances the scale is :math:`(k+1)/2`. Diluted either way, but still
detected with high power at a few hundred poses (see
``tests/test_power.py`` and ``tests/test_pairwise.py``).

The translation slice is the default and the headline: the Umeyama fit
uses translations only, so the rotation slice absorbs no gauge
correction and its pairwise NEES conflates calibration with the
unaligned heading difference.

Ported from the slam_benchmark audit (``scripts/e9_pairwise.py``).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from smfeval.align import fit_alignment, propagate_step
from smfeval.format import Representation, SquareHeader, TangentOrder
from smfeval.scoring.logscore import AneesResult, anees_consistency
from smfeval.se3.lie import pose_matrix, relative, se3_log, trans_slice
from smfeval.steps import GaussianStep, Step
from smfeval.sync import match_timestamps

PROPRIETY_CAVEAT = (
  "propriety caveat: pairwise scores are strictly proper only under a\n"
  "truthful reference sigma and independent errors; both violations push\n"
  "conservative, so NEES_pair lower-bounds miscalibration."
)

_CHI2_MED_3 = float(chi2.ppf(0.5, df=3))
_CHI2_MED_1 = float(chi2.ppf(0.5, df=1))


class PairInputError(ValueError):
  """The two trajectories cannot be scored against each other as-is."""


@dataclass
class PairResult:
  n_matched: int
  join_frac: float
  gap_median_ms: float
  n_scored: int
  med_d_norm: float
  nees: np.ndarray  # (n,) translation NEES_pair (nan where unscorable)
  nees_axis: np.ndarray  # (n, 3) per-axis NEES from diagonal variances
  log_score_mean: float  # mean pairwise Gaussian -log p (translation)
  anees: AneesResult = field(repr=False)
  k_pair: float = float("nan")  # median(nees) / chi2_med(3); lower bound
  k_axis: tuple[float, ...] = (float("nan"),) * 3


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise PairInputError(msg)


def _med(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(np.median(a)) if a.size else float("nan")


def pair_translation_nees(
  header_a: SquareHeader,
  steps_a: list[Step],
  header_b: SquareHeader,
  steps_b: list[Step],
  *,
  t_max_diff: float = 0.01,
  min_matched: int = 10,
  alpha: float = 0.05,
) -> PairResult:
  """Score filter A against filter B with no ground truth.

  A is aligned to B (SE(3) Umeyama on matched translations), then each
  matched pose pair contributes a translation NEES under the summed
  covariance. Raises :class:`PairInputError` when the inputs are not
  two gaussian_se3 trajectories in the same frames/convention with at
  least ``min_matched`` timestamp matches.
  """
  for name, h in (("A", header_a), ("B", header_b)):
    _require(
      h.representation is Representation.GAUSSIAN_SE3,
      f"pair needs gaussian_se3 inputs; {name} is {h.representation.value}",
    )
  _require(
    header_a.tangent_convention == header_b.tangent_convention,
    f"tangent convention mismatch: {header_a.tangent_convention} vs "
    f"{header_b.tangent_convention}; covariances cannot be summed",
  )
  _require(
    header_a.pose_frame == header_b.pose_frame,
    f"pose frames differ: {header_a.pose_frame!r} vs "
    f"{header_b.pose_frame!r}; pre-align into a common pose frame",
  )
  _require(
    header_a.body_frame == header_b.body_frame,
    f"body frames differ: {header_a.body_frame!r} vs "
    f"{header_b.body_frame!r}; pass --body-frame-transform to re-express "
    "A in B's body frame",
  )

  order_a = header_a.tangent_order or TangentOrder.TRANS_ROT
  order_b = header_b.tangent_order or TangentOrder.TRANS_ROT

  ts_a = np.array([s.timestamp for s in steps_a])
  ts_b = np.array([s.timestamp for s in steps_b])
  m = match_timestamps(ts_a, ts_b, t_max_diff=t_max_diff)
  _require(
    m.n_matched >= min_matched,
    f"only {m.n_matched} matched poses (need >= {min_matched}); "
    "check timestamps or loosen --t_max_diff",
  )
  a = [steps_a[i] for i in m.est_indices]
  b = [steps_b[j] for j in m.gt_indices]

  fit = fit_alignment(
    np.array([s.translation for s in a]),
    np.array([s.translation for s in b]),
    mode="se3",
  )
  a = [
    propagate_step(
      s,
      fit.transform,
      scale=fit.scale,
      tangent_convention=header_a.tangent_convention,
      tangent_order=order_a,
    )
    for s in a
  ]

  ti_a, ti_b = trans_slice(order_a), trans_slice(order_b)
  n = len(a)
  d = np.full((n, 3), np.nan)
  cov = np.full((n, 3, 3), np.nan)
  for k, (sa, sb) in enumerate(zip(a, b, strict=True)):
    if not (isinstance(sa, GaussianStep) and isinstance(sb, GaussianStep)):
      continue
    xi = se3_log(
      relative(
        pose_matrix(sa.translation, sa.quat_xyzw),
        pose_matrix(sb.translation, sb.quat_xyzw),
      ),
      order=order_a,
    )
    d[k] = xi[ti_a]
    cov[k] = sa.covariance[ti_a, ti_a] + sb.covariance[ti_b, ti_b]

  nees = np.full(n, np.nan)
  logs = np.full(n, np.nan)
  nees_ax = np.full((n, 3), np.nan)
  for k in range(n):
    if not (np.all(np.isfinite(cov[k])) and np.all(np.isfinite(d[k]))):
      continue
    sign, logdet = np.linalg.slogdet(cov[k])
    if sign <= 0:
      continue
    nees[k] = float(d[k] @ np.linalg.solve(cov[k], d[k]))
    logs[k] = 0.5 * nees[k] + 0.5 * (logdet + 3.0 * np.log(2.0 * np.pi))
    var = np.diag(cov[k])
    if np.all(var > 0):
      nees_ax[k] = d[k] ** 2 / var

  fin = np.isfinite(nees)
  return PairResult(
    n_matched=m.n_matched,
    join_frac=m.n_matched / min(len(ts_a), len(ts_b)),
    gap_median_ms=m.gap_quantiles_ms["median"],
    n_scored=int(fin.sum()),
    med_d_norm=_med(np.linalg.norm(d, axis=1)),
    nees=nees,
    nees_axis=nees_ax,
    log_score_mean=float(np.mean(logs[fin])) if fin.any() else float("nan"),
    anees=anees_consistency(nees, dof=3, alpha=alpha),
    k_pair=_med(nees) / _CHI2_MED_3,
    k_axis=tuple(_med(nees_ax[:, i]) / _CHI2_MED_1 for i in range(3)),
  )
