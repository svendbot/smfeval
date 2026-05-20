"""Propagate a fitted alignment transform to means, covariances, and particles.

Conventions (from spec):
- right-perturbation (T_world = T_mean · Exp(ξ)): under T·T_mean, Σ unchanged.
- left-perturbation (T_world = Exp(ξ) · T_mean): Σ ← Ad_T · Σ · Ad_T^⊤.
- ensembles: each particle ← T·p_i, weights invariant.

For Sim(3), the scale `s` enters the translation block of the Adjoint and
multiplies translations.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.se3.lie import adjoint, trans_slice
from src.steps import DeterministicStep, EnsembleStep, GaussianStep, Step
from src.types import TangentConvention, TangentOrder


def _scaled_adjoint(T: np.ndarray, scale: float, order: TangentOrder) -> np.ndarray:
    Ad = adjoint(T, order)
    if scale != 1.0:
        ti = trans_slice(order)
        Ad[ti, :] *= scale
    return Ad


def _apply_to_pose(
    T: np.ndarray, scale: float, translation: np.ndarray, quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3]
    new_t = scale * (R @ translation) + T[:3, 3]
    new_R = R @ Rotation.from_quat(quat_xyzw).as_matrix()
    return new_t, Rotation.from_matrix(new_R).as_quat()


def propagate_step(
    step: Step,
    transform: np.ndarray,
    scale: float = 1.0,
    tangent_convention: TangentConvention | None = None,
    tangent_order: TangentOrder | None = None,
) -> Step:
    """Apply T (and scale, for Sim(3)) to a step, propagating uncertainty."""
    if isinstance(step, DeterministicStep):
        new_t, new_q = _apply_to_pose(
            transform, scale, step.translation, step.quat_xyzw
        )
        return DeterministicStep(timestamp=step.timestamp, translation=new_t, quat_xyzw=new_q)

    if isinstance(step, GaussianStep):
        if tangent_convention is None or tangent_order is None:
            raise ValueError("gaussian propagation needs tangent convention and order")
        new_t, new_q = _apply_to_pose(
            transform, scale, step.translation, step.quat_xyzw
        )
        if tangent_convention is TangentConvention.RIGHT:
            new_cov = step.covariance.copy()
            if scale != 1.0:
                ti = trans_slice(tangent_order)
                new_cov[ti, :] *= scale
                new_cov[:, ti] *= scale
        else:
            Ad = _scaled_adjoint(transform, scale, tangent_order)
            new_cov = Ad @ step.covariance @ Ad.T
        return GaussianStep(
            timestamp=step.timestamp,
            translation=new_t,
            quat_xyzw=new_q,
            covariance=new_cov,
        )

    if isinstance(step, EnsembleStep):
        R = transform[:3, :3]
        d = transform[:3, 3]
        new = step.particles.copy()
        new[:, :3] = scale * (step.particles[:, :3] @ R.T) + d
        rots = Rotation.from_quat(step.particles[:, 3:]).as_matrix()
        new_rots = R @ rots  # broadcast (3,3) × (n,3,3) → (n,3,3)
        new[:, 3:] = Rotation.from_matrix(new_rots).as_quat()
        return EnsembleStep(
            timestamp=step.timestamp,
            particles=new,
            weights=None if step.weights is None else step.weights.copy(),
        )
