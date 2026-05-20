r"""Energy score on SE(3) tangent space (joint translation + rotation).

The energy score generalises CRPS to multivariate forecasts. It was
introduced by Székely (2003) and characterised as a strictly proper
multivariate kernel score by Gneiting & Raftery (2007).

References
----------
Székely, G. J. (2003). *E-statistics: The energy of statistical
samples*. Bowling Green State University, Tech. Rep. 03-05.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

import numpy as np

from src.scoring._kernel import energy_score_estimator, sample_gaussian_tangent
from src.se3.lie import pose_matrix, relative, se3_log
from src.steps import EnsembleStep, GaussianStep, Step
from src.types import TangentOrder


def energy_score(
    pred_step: Step,
    gt_translation: np.ndarray,
    gt_quat_xyzw: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
    n_samples: int = 128,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Joint energy score in SE(3) tangent space at the predictive mean.

    For samples :math:`X, X' \stackrel{iid}{\sim} F` and observation
    :math:`y` projected through :math:`\log_{T_\mathrm{ref}}` into
    :math:`\mathbb{R}^6`,

    .. math::

       \mathrm{ES}(F, y) = \mathbb{E}\,\lVert X - y\rVert
           - \tfrac12\,\mathbb{E}\,\lVert X - X'\rVert.

    Strictly proper for any negative-definite norm (Gneiting & Raftery,
    2007, §4.2).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    T_obs = pose_matrix(gt_translation, gt_quat_xyzw)

    if isinstance(pred_step, GaussianStep):
        T_mean = pose_matrix(pred_step.translation, pred_step.quat_xyzw)
        samples = sample_gaussian_tangent(np.zeros(6), pred_step.covariance, n_samples, rng)
        obs_tangent = se3_log(relative(T_mean, T_obs), order=tangent_order)
        return energy_score_estimator(samples, obs_tangent)

    if isinstance(pred_step, EnsembleStep):
        n = pred_step.particles.shape[0]
        if n == 0:
            return float("nan")
        # Energy score is invariant to the linearization point as long as
        # samples and observation are projected through the same map. Particle
        # 0 is convenient. Weights are not yet honored here (v0.2 limitation).
        T_ref = pose_matrix(pred_step.particles[0, :3], pred_step.particles[0, 3:])
        sample_tangents = np.stack(
            [
                se3_log(relative(T_ref, pose_matrix(p[:3], p[3:])), order=tangent_order)
                for p in pred_step.particles
            ]
        )
        obs_tangent = se3_log(relative(T_ref, T_obs), order=tangent_order)
        return energy_score_estimator(sample_tangents, obs_tangent)

    T_mean = pose_matrix(pred_step.translation, pred_step.quat_xyzw)
    return float(np.linalg.norm(se3_log(relative(T_mean, T_obs), order=tangent_order)))
