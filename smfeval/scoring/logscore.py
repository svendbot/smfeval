r"""Closed-form Gaussian log score on the translation marginal.

The logarithmic (or *ignorance*) score :math:`-\log f(y)` was introduced
by Good (1952) and is strictly proper (Gneiting & Raftery, 2007).

We score the 3-D translation block of the SE(3) belief. Orientation is
not scored: a proper score on SO(3) needs a belief density whose
normaliser is intractable for the natural rotation families (paper,
§II.b / §V.d). The translation marginal of the pose Gaussian is the
Gaussian on the translation sub-vector with the matching sub-covariance,
obtained by indexing the residual and covariance with
:func:`smfeval.se3.lie.trans_slice`.

References:
-----------
Good, I. J. (1952). *Rational decisions*. JRSS B 14(1), 107–114.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2

from smfeval.format import TangentOrder
from smfeval.se3.lie import (
  pose_matrix,
  relative,
  se3_log,
  trans_slice,
)
from smfeval.steps import GaussianStep


@dataclass
class GaussianLogScore:
  r"""Translation-marginal negative log density (smaller is better)."""

  translation: float

  def to_dict(self) -> dict[str, float]:
    return asdict(self)


def _gaussian_neg_log_density(xi: np.ndarray, cov: np.ndarray) -> float:
  r"""Negative log density of :math:`\xi` under :math:`N(0, \Sigma)`.

  :math:`\tfrac12(\xi^\top \Sigma^{-1}\xi + \log\det\Sigma + d\log 2\pi)`,
  or ``inf`` when :math:`\Sigma` is not positive definite.
  """
  sign, logdet = np.linalg.slogdet(cov)
  if sign <= 0:
    return float("inf")
  inv = np.linalg.solve(cov, xi)
  quad = float(xi @ inv)
  d = cov.shape[0]
  return 0.5 * (quad + logdet + d * np.log(2.0 * np.pi))


def gaussian_log_score(
  step: GaussianStep,
  ref_translation: np.ndarray,
  ref_quat_xyzw: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> GaussianLogScore:
  r"""Translation-marginal negative log density of the reference under the belief.

  With residual :math:`\xi = \log_{T_\mathrm{mean}}(T_\mathrm{obs})
  \in \mathbb{R}^6`, covariance :math:`\Sigma \in \mathbb{R}^{6\times 6}`,
  and translation selector :math:`I_t` (as configured by ``tangent_order``),

  .. math::

     -\log p(\xi_{I_t}) = \tfrac12\bigl(\xi_{I_t}^\top \Sigma_{I_t I_t}^{-1}
       \xi_{I_t} + \log\det\Sigma_{I_t I_t} + 3\log 2\pi\bigr).

  The marginal density is the joint with the rotation block integrated
  out, which for a Gaussian is the translation sub-vector under the
  matching sub-covariance.
  """
  T_mean = pose_matrix(step.translation, step.quat_xyzw)
  T_obs = pose_matrix(ref_translation, ref_quat_xyzw)
  xi = se3_log(relative(T_mean, T_obs), order=tangent_order)
  t_idx = trans_slice(tangent_order)
  trans = _gaussian_neg_log_density(xi[t_idx], step.covariance[t_idx, t_idx])
  return GaussianLogScore(translation=trans)


@dataclass
class ScoreComponents:
  r"""Log-score and its additive calibration/sharpness split for one slice.

  The Gaussian log score decomposes exactly:

  .. math::
     -\log p = \underbrace{\tfrac12\,\xi^\top\Sigma^{-1}\xi}_{\text{calibration}}
       + \underbrace{\tfrac12\bigl(\log\det\Sigma + d\log 2\pi\bigr)}_{\text{sharpness}}.

  ``calibration`` is :math:`\tfrac12` the NEES (``nees`` exposed raw, since
  :math:`\mathbb{E}[\text{NEES}]=d` under a calibrated belief gives it a
  Σ-independent reference the raw score lacks); ``sharpness`` is the
  log-volume penalty. ``log_score == calibration + sharpness``. All three are
  ``inf`` when :math:`\Sigma` is not positive definite.
  """

  log_score: float
  calibration: float
  sharpness: float
  nees: float
  dof: int

  def to_dict(self) -> dict[str, float]:
    return asdict(self)


@dataclass
class DecomposedLogScore:
  r"""Translation log score with its calibration/sharpness split."""

  translation: ScoreComponents

  def to_dict(self) -> dict[str, dict[str, float]]:
    return asdict(self)


def _score_components(xi: np.ndarray, cov: np.ndarray) -> ScoreComponents:
  r"""Split :math:`-\log\mathcal N(\xi;0,\Sigma)` into calibration + sharpness."""
  d = cov.shape[0]
  sign, logdet = np.linalg.slogdet(cov)
  if sign <= 0:
    return ScoreComponents(
      log_score=float("inf"),
      calibration=float("inf"),
      sharpness=float("inf"),
      nees=float("inf"),
      dof=d,
    )
  nees = float(xi @ np.linalg.solve(cov, xi))
  calibration = 0.5 * nees
  sharpness = 0.5 * (logdet + d * np.log(2.0 * np.pi))
  return ScoreComponents(
    log_score=calibration + sharpness,
    calibration=calibration,
    sharpness=sharpness,
    nees=nees,
    dof=d,
  )


def gaussian_log_score_components(
  step: GaussianStep,
  ref_translation: np.ndarray,
  ref_quat_xyzw: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> DecomposedLogScore:
  r"""Translation log score split into calibration/sharpness.

  Same residual/covariance as :func:`gaussian_log_score`; this variant exposes
  the two additive components so attribution can read the calibration term
  (graded, χ²-referenced) while the raw score remains the proper headline.
  """
  T_mean = pose_matrix(step.translation, step.quat_xyzw)
  T_obs = pose_matrix(ref_translation, ref_quat_xyzw)
  xi = se3_log(relative(T_mean, T_obs), order=tangent_order)
  t_idx = trans_slice(tangent_order)
  return DecomposedLogScore(
    translation=_score_components(xi[t_idx], step.covariance[t_idx, t_idx]),
  )


def student_t_neg_log_density(
  xi: np.ndarray, cov: np.ndarray, nu: float
) -> float:
  r"""Negative log density under a covariance-matched multivariate Student-t.

  Scores :math:`\xi` under a Student-t whose *covariance equals* ``cov``
  (a covariance-matched heavy-tailed belief).

  The t scale matrix is :math:`S=\Sigma\,(\nu-2)/\nu` (needs :math:`\nu>2`), so
  the belief keeps the published second moment but has heavier tails; the
  Gaussian is the :math:`\nu\to\infty` limit. The Mahalanobis term enters as
  :math:`\tfrac{\nu+d}{2}\log(1+m/\nu)` — logarithmic in :math:`m`, so a tail
  outlier contributes :math:`\sim\log m` instead of :math:`m` (bounded
  influence). This is the score-side analogue of a robust (Student-t) likelihood
  — `do(robust likelihood)` evaluated on the published belief, no filter re-run.
  Returns ``inf`` for non-PD ``cov``.
  """
  d = cov.shape[0]
  scale = cov * ((nu - 2.0) / nu)
  sign, logdet = np.linalg.slogdet(scale)
  if sign <= 0:
    return float("inf")
  m = float(xi @ np.linalg.solve(scale, xi))
  return float(
    -gammaln((nu + d) / 2.0)
    + gammaln(nu / 2.0)
    + 0.5 * d * np.log(nu * np.pi)
    + 0.5 * logdet
    + 0.5 * (nu + d) * np.log1p(m / nu)
  )


def student_t_logscore_sweep(
  steps: list,
  ref_translations: np.ndarray,
  ref_quats: np.ndarray,
  nus: list[float],
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> tuple[list[float], dict[float, list[float]]]:
  """Per-step Gaussian and covariance-matched Student-t negative log densities.

  Returns ``(gaussian_nll, {nu: student_t_nll})`` over the GaussianStep entries,
  skipping non-Gaussian steps. The Gaussian column is the nu->inf limit.
  """
  gauss: list[float] = []
  tcols: dict[float, list[float]] = {nu: [] for nu in nus}
  for step, ref_t, ref_q in zip(
    steps, ref_translations, ref_quats, strict=True
  ):
    if not isinstance(step, GaussianStep):
      continue
    xi = se3_log(
      relative(
        pose_matrix(step.translation, step.quat_xyzw),
        pose_matrix(ref_t, ref_q),
      ),
      order=tangent_order,
    )
    t_idx = trans_slice(tangent_order)
    xi_t = xi[t_idx]
    cov_t = step.covariance[t_idx, t_idx]
    gauss.append(_gaussian_neg_log_density(xi_t, cov_t))
    for nu in nus:
      tcols[nu].append(student_t_neg_log_density(xi_t, cov_t, nu))
  return gauss, tcols


@dataclass
class AneesResult:
  r"""Average-NEES consistency test against the two-sided χ² interval.

  For ``n`` per-step NEES values each :math:`\sim\chi^2_d` under a calibrated
  belief, :math:`n\cdot\text{ANEES}\sim\chi^2_{nd}`, giving the acceptance
  interval :math:`[\chi^2_{nd,\alpha/2}/n,\ \chi^2_{nd,1-\alpha/2}/n]` for ANEES.
  ``anees > hi`` ⇒ ``optimistic`` (over-confident: error exceeds Σ);
  ``anees < lo`` ⇒ ``conservative`` (under-confident: Σ too large);
  otherwise ``consistent``. Non-finite NEES values (non-PD Σ) are dropped.
  """

  anees: float
  dof: int
  n: int
  lo: float
  hi: float
  verdict: str
  # Robust companion to the mean ANEES (outlier-dominated): median ~= dof when
  # the bulk is calibrated, so median << anees flags tail over-confidence.
  median: float = float("nan")

  def to_dict(self) -> dict[str, float | int | str]:
    return asdict(self)


def anees_consistency(
  nees_values: np.ndarray, dof: int, alpha: float = 0.05
) -> AneesResult:
  r"""Two-sided χ² consistency verdict for a sequence of per-step NEES values."""
  vals = np.asarray(nees_values, dtype=float)
  vals = vals[np.isfinite(vals)]
  n = int(vals.size)
  if n == 0:
    nan = float("nan")
    return AneesResult(nan, dof, 0, nan, nan, "undefined", nan)
  anees = float(vals.mean())
  median = float(np.median(vals))
  lo = float(chi2.ppf(alpha / 2.0, df=n * dof) / n)
  hi = float(chi2.ppf(1.0 - alpha / 2.0, df=n * dof) / n)
  verdict = (
    "optimistic"
    if anees > hi
    else "conservative"
    if anees < lo
    else "consistent"
  )
  return AneesResult(
    anees=anees, dof=dof, n=n, lo=lo, hi=hi, verdict=verdict, median=median
  )
