"""Propagate a fitted alignment transform to means, covariances, and particles.

Conventions (from spec):
- right-perturbation (T_world = T_mean · Exp(ξ)): under T·T_mean, Σ unchanged.
- left-perturbation (T_world = Exp(ξ) · T_mean): Σ ← Ad_T · Σ · Ad_T^⊤.
- ensembles: each particle ← T·p_i, weights invariant.

For Sim(3), the scale `s` enters the translation block of the Adjoint and
multiplies translations.
"""

from typing import TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from smfeval.format import TangentConvention, TangentOrder
from smfeval.se3.lie import adjoint, invert, trans_slice
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep

StepT = TypeVar("StepT", GaussianStep, EnsembleStep, DeterministicStep)


def _scaled_adjoint(
  T: np.ndarray, scale: float, order: TangentOrder
) -> np.ndarray:
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
  step: StepT,
  transform: np.ndarray,
  scale: float = 1.0,
  tangent_convention: TangentConvention | None = None,
  tangent_order: TangentOrder | None = None,
) -> StepT:
  """Apply T (and scale, for Sim(3)) to a step, propagating uncertainty."""
  match step:
    case DeterministicStep():
      new_t, new_q = _apply_to_pose(
        transform, scale, step.translation, step.quat_xyzw
      )
      return DeterministicStep(
        timestamp=step.timestamp, translation=new_t, quat_xyzw=new_q
      )

    case GaussianStep():
      if tangent_convention is None or tangent_order is None:
        raise ValueError(
          "gaussian propagation needs tangent convention and order"
        )
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

    case EnsembleStep():
      R = transform[:3, :3]
      d = transform[:3, 3]
      new = step.particles.copy()
      new[:, :3] = scale * (step.particles[:, :3] @ R.T) + d
      rots = Rotation.from_quat(step.particles[:, 3:]).as_matrix()
      new_rots = R @ rots  # broadcast (3,3) x (n,3,3) -> (n,3,3)
      new[:, 3:] = Rotation.from_matrix(new_rots).as_quat()
      return EnsembleStep(
        timestamp=step.timestamp,
        particles=new,
        weights=None if step.weights is None else step.weights.copy(),
      )
    case _:
      raise TypeError(f"unsupported step type {type(step).__name__}")


def apply_body_transform(
  step: StepT,
  T_off: np.ndarray,
  tangent_convention: TangentConvention | None = None,
  tangent_order: TangentOrder | None = None,
) -> StepT:
  """Right-multiply each pose by ``T_off`` to re-express it in a new body frame.

  Given ``T_world_body_old`` and ``T_off = T_body_old__body_new`` (the pose of
  the new body frame as seen from the old), the corrected pose is
  ``T_world_body_new = T_world_body_old · T_off``. This is the dual of
  alignment, which left-multiplies; the tangent transforms differently:

  - right-perturbation:  ``Σ ← Ad_{T_off^{-1}} · Σ · Ad_{T_off^{-1}}^⊤``
  - left-perturbation:   ``Σ`` unchanged

  The math: with right-perturbation ``T = T_mean · Exp(ξ)``, right-multiplying
  by ``T_off`` gives ``T · T_off = T_mean · T_off · Exp(Ad_{T_off^{-1}} · ξ)``.
  With left-perturbation ``T = Exp(ξ) · T_mean``, only ``T_mean`` shifts.
  """
  R_off = T_off[:3, :3]
  t_off = T_off[:3, 3]

  def _new_mean(
    t: np.ndarray, q_xyzw: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    R_mean = Rotation.from_quat(q_xyzw).as_matrix()
    new_t = R_mean @ t_off + t
    new_R = R_mean @ R_off
    return new_t, Rotation.from_matrix(new_R).as_quat()

  match step:
    case DeterministicStep():
      new_t, new_q = _new_mean(step.translation, step.quat_xyzw)
      return DeterministicStep(
        timestamp=step.timestamp, translation=new_t, quat_xyzw=new_q
      )

    case GaussianStep():
      if tangent_convention is None or tangent_order is None:
        raise ValueError(
          "gaussian body-frame transform needs tangent convention and order"
        )
      new_t, new_q = _new_mean(step.translation, step.quat_xyzw)
      if tangent_convention is TangentConvention.RIGHT:
        Ad_inv = adjoint(invert(T_off), tangent_order)
        new_cov = Ad_inv @ step.covariance @ Ad_inv.T
      else:
        new_cov = step.covariance.copy()
      return GaussianStep(
        timestamp=step.timestamp,
        translation=new_t,
        quat_xyzw=new_q,
        covariance=new_cov,
      )

    case EnsembleStep():
      new = step.particles.copy()
      rots = Rotation.from_quat(step.particles[:, 3:]).as_matrix()
      new[:, :3] = step.particles[:, :3] + np.einsum("nij,j->ni", rots, t_off)
      new_rots = np.einsum("nij,jk->nik", rots, R_off)
      new[:, 3:] = Rotation.from_matrix(new_rots).as_quat()
      return EnsembleStep(
        timestamp=step.timestamp,
        particles=new,
        weights=None if step.weights is None else step.weights.copy(),
      )
    case _:
      raise TypeError(f"unsupported step type {type(step).__name__}")
