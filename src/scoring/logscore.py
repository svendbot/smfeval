r"""Closed-form Gaussian log score in SE(3) tangent space.

Decomposed into the joint score and its translation / rotation marginals.

The logarithmic (or *ignorance*) score :math:`-\log f(y)` was introduced
by Good (1952) and is strictly proper (Gneiting & Raftery, 2007).

For SLAM beliefs the 6×6 SE(3) covariance mixes translation and
rotation, so a single joint scalar hides calibration pathologies that
target only one of the two blocks — e.g. a LiDAR filter that is
well-calibrated in translation but overconfident in yaw when the
geometry degenerates along the motion direction. We therefore also
report the *marginal* log scores on the 3-D translation block and the
3-D rotation block. The marginal of a joint Gaussian is the
Gaussian on the corresponding sub-vector with the matching
sub-covariance, so the marginal scores are obtained by indexing the
6-vector residual and the 6×6 covariance with
:func:`src.se3.lie.trans_slice` / :func:`src.se3.lie.rot_slice`. The
three numbers are not independent (the joint encodes their
cross-covariance) but together they decompose the scalar.

References:
----------
Good, I. J. (1952). *Rational decisions*. JRSS B 14(1), 107–114.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

from dataclasses import asdict, dataclass

import numpy as np

from src.format import TangentOrder
from src.se3.lie import pose_matrix, relative, rot_slice, se3_log, trans_slice
from src.steps import GaussianStep


@dataclass
class GaussianLogScore:
  r"""Joint and block-marginal negative log densities (smaller is better)."""

  joint: float
  translation: float
  rotation: float

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
  gt_translation: np.ndarray,
  gt_quat_xyzw: np.ndarray,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> GaussianLogScore:
  r"""Negative log densities of the GT pose under the predictive Gaussian.

  Returns the joint SE(3) score and its translation / rotation marginals.

  With residual :math:`\xi = \log_{T_\mathrm{mean}}(T_\mathrm{obs})
  \in \mathbb{R}^6`, covariance :math:`\Sigma \in \mathbb{R}^{6\times 6}`,
  and block selectors :math:`I_t, I_r` for translation and rotation
  (3 entries each, as configured by ``tangent_order``),

  .. math::

     -\log p(\xi)       &= \tfrac12\bigl(\xi^\top \Sigma^{-1}\xi
         + \log\det\Sigma + 6\log 2\pi\bigr), \\
       -\log p(\xi_{I_t}) &= \tfrac12\bigl(\xi_{I_t}^\top \Sigma_{I_t I_t}^{-1}
         \xi_{I_t} + \log\det\Sigma_{I_t I_t} + 3\log 2\pi\bigr),

  and analogously for :math:`I_r`. The block-marginal density is
  simply the joint with the other block integrated out, which for a
  Gaussian is the sub-vector under the matching sub-covariance.
  """
  T_mean = pose_matrix(step.translation, step.quat_xyzw)
  T_obs = pose_matrix(gt_translation, gt_quat_xyzw)
  xi = se3_log(relative(T_mean, T_obs), order=tangent_order)
  cov = step.covariance

  t_idx = trans_slice(tangent_order)
  r_idx = rot_slice(tangent_order)

  joint = _gaussian_neg_log_density(xi, cov)
  trans = _gaussian_neg_log_density(xi[t_idx], cov[t_idx, t_idx])
  rot = _gaussian_neg_log_density(xi[r_idx], cov[r_idx, r_idx])
  return GaussianLogScore(joint=joint, translation=trans, rotation=rot)
