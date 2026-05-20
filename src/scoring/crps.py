r"""Continuous Ranked Probability Score (CRPS) for translation
(per-axis) and rotation (SO(3) geodesic kernel).

CRPS was introduced by Matheson & Winkler (1976); the kernel-score and
energy-form characterisation we use here is from Gneiting & Raftery
(2007). The SO(3) variant follows the kernel-score construction with the
geodesic distance as a (negative-definite) kernel.

References
----------
Matheson, J. E. & Winkler, R. L. (1976). *Scoring rules for continuous
probability distributions*. Management Science 22(10), 1087–1096.

Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules,
prediction, and estimation*. JASA 102(477), 359–378.
"""

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
    r"""Mean per-axis CRPS over the three translation components.

    For each axis :math:`i \in \{x, y, z\}` we evaluate the univariate
    CRPS via the energy-form identity

    .. math::

       \mathrm{CRPS}(F, y) = \mathbb{E}\,|X - y|
           - \tfrac12\,\mathbb{E}\,|X - X'|,

    with :math:`X, X' \stackrel{iid}{\sim} F`, then average over axes.

    References
    ----------
    Gneiting & Raftery (2007), eq. (20).
    """
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
    r"""CRPS on SO(3) via the geodesic kernel
    :math:`k(R_1, R_2) = d_\mathrm{geo}(R_1, R_2)`.

    Using the kernel-score / energy-form identity on the manifold,

    .. math::

       \mathrm{CRPS}_{SO(3)}(F, R_\mathrm{obs})
       = \mathbb{E}\,d_\mathrm{geo}(R, R_\mathrm{obs})
       - \tfrac12\,\mathbb{E}\,d_\mathrm{geo}(R, R'),

    where :math:`d_\mathrm{geo}(R_1, R_2) = \lVert\log(R_1^\top R_2)\rVert`
    is the geodesic angle on SO(3) and :math:`R, R' \stackrel{iid}{\sim} F`.
    Negative-definiteness of :math:`d_\mathrm{geo}` ensures the rule is
    strictly proper (Sejdinovic et al., 2013).

    References
    ----------
    Gneiting & Raftery (2007); Sejdinovic, D., Sriperumbudur, B.,
    Gretton, A. & Fukumizu, K. (2013). *Equivalence of distance-based and
    RKHS-based statistics in hypothesis testing*. Annals of Statistics
    41(5), 2263–2291.
    """
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
