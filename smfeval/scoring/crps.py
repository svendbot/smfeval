r"""Continuous Ranked Probability Score (CRPS) on SE(3).

Translation is scored per-axis; rotation uses the SO(3) geodesic kernel.

CRPS was introduced by Matheson & Winkler (1976); the kernel-score and
energy-form characterisation we use here is from Gneiting & Raftery
(2007). The SO(3) variant follows the kernel-score construction with the
geodesic distance as a (negative-definite) kernel.

References:
-----------
Matheson, J. E. & Winkler, R. L. (1976). *Scoring rules for continuous
probability distributions*. Management Science 22(10), 1087–1096.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

import numpy as np
from scipy.stats import norm

from smfeval.format import TangentOrder
from smfeval.scoring._kernel import crps_estimator
from smfeval.scoring._predictive import rotation_samples
from smfeval.se3.lie import so3_exp, so3_log, trans_slice
from smfeval.se3.quat import quat_xyzw_to_rot
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)

_ROT_CRPS_PILOT = 64
_ROT_CRPS_MAX_SAMPLES = 512
_ROT_CRPS_REL_TARGET = 0.05  # stop when SE < 5% of |score|
_ROT_CRPS_ABS_FLOOR = 1e-4  # ...but never tighten below 0.006°


def _gaussian_crps(
  mu: np.ndarray, sigma: np.ndarray, y: np.ndarray
) -> np.ndarray:
  r"""Closed-form per-axis CRPS for :math:`N(\mu, \sigma^2)` at :math:`y`.

  .. math::

     \mathrm{CRPS}(N(\mu,\sigma^2), y)
     = \sigma\left[\omega\bigl(2\Phi(\omega)-1\bigr)
                   + 2\phi(\omega) - 1/\sqrt{\pi}\right],
     \quad \omega = (y-\mu)/\sigma.

  Gneiting & Raftery (2007), closed form following eq. (20).
  """
  z = (y - mu) / sigma
  return sigma * (
    z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI
  )


def translation_crps(
  pred_step: Step,
  gt_translation: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> float:
  r"""Mean per-axis CRPS over the three translation components.

  Gaussian predictive: closed form (Gneiting & Raftery 2007, after eq. 20).
  Ensemble predictive: per-axis U-statistic estimator over the full
  ensemble (the filter sets the sample count; we do not subsample).
  Deterministic: per-axis absolute error.
  """
  match pred_step:
    case GaussianStep():
      sl = trans_slice(tangent_order)
      sigma = np.sqrt(np.diag(pred_step.covariance)[sl])
      return float(
        _gaussian_crps(pred_step.translation, sigma, gt_translation).mean()
      )
    case EnsembleStep():
      samples = pred_step.particles[:, :3]
      axis_scores = [
        crps_estimator(samples[:, i], float(gt_translation[i]))
        for i in range(3)
      ]
      return float(np.mean(axis_scores))
    case DeterministicStep():
      return float(np.abs(pred_step.translation - gt_translation).mean())


def _geodesic_distances(
  samples: np.ndarray, R_obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  r"""Per-sample geodesic angles to R_obs and the full m×m pairwise matrix.

  Uses :math:`\mathrm{tr}(R_a^\top R_b) = \sum_{ij} R_a[i,j] R_b[i,j]`
  (Frobenius inner product) so the angle reduces to a single einsum +
  arccos — no batched SVD/quat decomposition.
  """
  traces_obs = np.einsum("aij,ij->a", samples, R_obs)
  d_obs = np.arccos(np.clip((traces_obs - 1.0) / 2.0, -1.0, 1.0))
  traces_pair = np.einsum("aij,bij->ab", samples, samples)
  D = np.arccos(np.clip((traces_pair - 1.0) / 2.0, -1.0, 1.0))
  return d_obs, D


def _rotation_crps_from_distances(
  d_obs: np.ndarray, D: np.ndarray
) -> tuple[float, float]:
  r"""Kernel-CRPS U-statistic with jackknife SE.

  Score :math:`\hat S = \bar d_{\mathrm{obs}} - \tfrac12 U`, where
  :math:`U = \tfrac{2}{m(m-1)}\sum_{i<j} D_{ij}`. Leave-one-out updates
  reuse row-sums of :math:`D` so the jackknife is :math:`O(m)` given
  the cached distances.
  """
  m = d_obs.size
  sum_d = float(d_obs.sum())
  sum_pairs = 0.5 * float(D.sum())  # symmetric, zero diagonal
  row_sums = D.sum(axis=1)  # sum over j of D[i, j]
  t1 = sum_d / m
  t2 = 2.0 * sum_pairs / (m * (m - 1))
  score = t1 - 0.5 * t2
  t1_loo = (sum_d - d_obs) / (m - 1)
  t2_loo = 2.0 * (sum_pairs - row_sums) / ((m - 1) * (m - 2))
  s_loo = t1_loo - 0.5 * t2_loo
  se = float(np.sqrt((m - 1) / m * np.sum((s_loo - s_loo.mean()) ** 2)))
  return float(score), se


def rotation_crps(
  pred_step: Step,
  gt_quat_xyzw: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  rng: np.random.Generator | None = None,
) -> float:
  r"""CRPS on SO(3) via the geodesic kernel.

  Kernel :math:`k(R_1, R_2) = d_\mathrm{geo}(R_1, R_2)` gives

  .. math::

     \mathrm{CRPS}_{SO(3)}(F, R_\mathrm{obs})
     = \mathbb{E}\,d_\mathrm{geo}(R, R_\mathrm{obs})
     - \tfrac12\,\mathbb{E}\,d_\mathrm{geo}(R, R').

  Negative-definiteness of :math:`d_\mathrm{geo}` gives strict propriety
  (Sejdinovic et al., 2013).

  For a Gaussian predictive we draw tangent samples adaptively: pilot
  of :data:`_ROT_CRPS_PILOT`, then double until the jackknife SE of the
  U-statistic falls under ``max(rel * |score|, abs_floor)`` or
  :data:`_ROT_CRPS_MAX_SAMPLES` is reached. For an ensemble we score
  every particle. Deterministic predictives reduce to the geodesic
  angle.

  References:
  -----------
  Grimit, Gneiting, Berrocal & Johnson (2006) for the circular kernel
  CRPS; Sejdinovic, Sriperumbudur, Gretton & Fukumizu (2013) for
  propriety of distance-based kernel scores.
  """
  R_obs = quat_xyzw_to_rot(gt_quat_xyzw)

  match pred_step:
    case GaussianStep():
      rng = rng if rng is not None else np.random.default_rng(0)
      m = _ROT_CRPS_PILOT
      while True:
        omegas, R_mean = rotation_samples(pred_step, m, rng, tangent_order)
        samples = R_mean @ so3_exp(omegas)  # (m, 3, 3) via broadcasting
        d_obs, D = _geodesic_distances(samples, R_obs)
        score, se = _rotation_crps_from_distances(d_obs, D)
        target = max(_ROT_CRPS_REL_TARGET * abs(score), _ROT_CRPS_ABS_FLOOR)
        if se <= target or m >= _ROT_CRPS_MAX_SAMPLES:
          return score
        m = min(2 * m, _ROT_CRPS_MAX_SAMPLES)
    case EnsembleStep():
      n = pred_step.particles.shape[0]
      if n == 0:
        return float("nan")
      samples = np.array([quat_xyzw_to_rot(p[3:]) for p in pred_step.particles])
      d_obs, D = _geodesic_distances(samples, R_obs)
      if n == 1:
        return float(d_obs[0])
      score, _ = _rotation_crps_from_distances(d_obs, D)
      return score
    case DeterministicStep():
      R_pred = quat_xyzw_to_rot(pred_step.quat_xyzw)
      return float(np.linalg.norm(so3_log(R_pred.T @ R_obs)))
