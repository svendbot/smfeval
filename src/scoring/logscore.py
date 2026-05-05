"""Closed-form Gaussian log score in SE(3) tangent space.

Skipped for ensembles in v0.1 (kernel density estimation with manifold-aware
bandwidths is out of scope here).
"""

import numpy as np

from src.se3.lie import pose_matrix, relative, se3_log
from src.steps import GaussianStep
from src.types import TangentOrder


def gaussian_log_score(
    step: GaussianStep,
    gt_translation: np.ndarray,
    gt_quat_xyzw: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> float:
    """Negative log density (smaller is better) of the GT pose under the Gaussian belief."""
    T_mean = pose_matrix(step.translation, step.quat_xyzw)
    T_obs = pose_matrix(gt_translation, gt_quat_xyzw)
    xi = se3_log(relative(T_mean, T_obs), order=tangent_order)
    cov = step.covariance
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    inv = np.linalg.solve(cov, xi)
    quad = float(xi @ inv)
    d = cov.shape[0]
    return 0.5 * (quad + logdet + d * np.log(2.0 * np.pi))
