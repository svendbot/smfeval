"""CRPS for translation (per-axis) and rotation (SO(3) geodesic kernel)."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.scoring._kernel import crps_estimator
from src.scoring._predictive import rotation_samples, translation_samples
from src.se3.lie import so3_exp
from src.se3.quat import quat_xyzw_to_rot
from src.steps import Step
from src.types import TangentOrder

_DEFAULT_N_SAMPLES = 256


def translation_crps(
    pred_step: Step,
    gt_translation: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
    n_samples: int = _DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean per-axis CRPS over the three translation components."""
    rng = rng if rng is not None else np.random.default_rng(0)
    samples, _ = translation_samples(pred_step, n_samples, rng, tangent_order)
    axis_scores = [crps_estimator(samples[:, i], float(gt_translation[i])) for i in range(3)]
    return float(np.mean(axis_scores))


def rotation_crps(
    pred_step: Step,
    gt_quat_xyzw: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
    n_samples: int = _DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> float:
    """CRPS via the SO(3) geodesic kernel k(R₁,R₂) = d_geo(R₁,R₂)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    R_obs = quat_xyzw_to_rot(gt_quat_xyzw)
    omegas, R_mean = rotation_samples(pred_step, n_samples, rng, tangent_order)
    m = omegas.shape[0]
    if m == 0:
        return float("nan")

    samples = np.array([R_mean @ so3_exp(w) for w in omegas])  # (m, 3, 3)
    rel_obs = np.einsum("mij,jk->mik", samples.transpose(0, 2, 1), R_obs)
    obs_angles = np.linalg.norm(Rotation.from_matrix(rel_obs).as_rotvec(), axis=1)
    term1 = float(obs_angles.mean())
    if m == 1:
        return term1

    pair_rels = np.einsum("aij,bjk->abik", samples.transpose(0, 2, 1), samples)
    flat = pair_rels.reshape(-1, 3, 3)
    pair_angles = np.linalg.norm(
        Rotation.from_matrix(flat).as_rotvec(), axis=1
    ).reshape(m, m)
    iu = np.triu_indices(m, k=1)
    term2 = 2.0 * pair_angles[iu].sum() / (m * (m - 1))
    return term1 - 0.5 * term2
