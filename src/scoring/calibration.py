"""Calibration: PIT/KS, interval coverage, standardized residuals.

Orthogonal to scoring rules — an algorithm with good CRPS can still be
miscalibrated, and vice versa.

PIT here uses two univariate scalarisations of the SE(3) residual:
- translation magnitude ‖t - μ_t‖
- rotation angle d_geo(R, R_μ)

The PIT for each pair is the fraction of predictive samples whose scalar lies
≤ the observed scalar; under a well-calibrated belief this is U(0,1).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest

from src.scoring._predictive import rotation_samples, translation_samples
from src.se3.lie import pose_matrix, relative, se3_log, so3_log, trans_slice
from src.se3.quat import quat_xyzw_to_rot
from src.steps import GaussianStep, Step
from src.types import TangentOrder

_DEFAULT_N_SAMPLES = 256


@dataclass
class CalibrationResult:
    pit_translation: np.ndarray
    pit_rotation: np.ndarray
    ks_p_translation: float
    ks_p_rotation: float
    coverage: float
    nominal_coverage: float
    z_translation_mean: float
    z_translation_std: float


def _z_translation(step: Step, gt_t: np.ndarray, gt_q: np.ndarray, order: TangentOrder) -> float:
    """Mahalanobis-normalised translation residual: Gaussian only, NaN otherwise."""
    if not isinstance(step, GaussianStep):
        return float("nan")
    T_mean = pose_matrix(step.translation, step.quat_xyzw)
    T_obs = pose_matrix(gt_t, gt_q)
    xi = se3_log(relative(T_mean, T_obs), order=order)
    ti = trans_slice(order)
    rho = xi[ti]
    cov_t = step.covariance[ti, ti]
    try:
        L = np.linalg.cholesky(cov_t + 1e-12 * np.eye(3))
    except np.linalg.LinAlgError:
        return float("nan")
    z = np.linalg.solve(L, rho)
    return float(np.linalg.norm(z) / np.sqrt(3))


def _pit(samples: np.ndarray, observation: float) -> float:
    if samples.size == 0:
        return float("nan")
    return float((samples <= observation).mean())


def calibrate(
    pred_steps: list[Step],
    gt_translations: np.ndarray,
    gt_quats: np.ndarray,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
    alpha: float = 0.1,
    n_samples: int = _DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> CalibrationResult:
    rng = rng if rng is not None else np.random.default_rng(0)
    pit_t: list[float] = []
    pit_r: list[float] = []
    z_t: list[float] = []
    inside: list[bool] = []
    for step, gt_t, gt_q in zip(pred_steps, gt_translations, gt_quats):
        t_samples, mu_t = translation_samples(step, n_samples, rng, tangent_order)
        sm = np.linalg.norm(t_samples - mu_t, axis=1)
        om = float(np.linalg.norm(gt_t - mu_t))
        pit_t.append(_pit(sm, om))

        omegas, R_mean = rotation_samples(step, n_samples, rng, tangent_order)
        sa = np.linalg.norm(omegas, axis=1)
        oa = float(np.linalg.norm(so3_log(R_mean.T @ quat_xyzw_to_rot(gt_q))))
        pit_r.append(_pit(sa, oa))

        z_t.append(_z_translation(step, gt_t, gt_q, tangent_order))
        if sm.size:
            inside.append(om <= float(np.quantile(sm, 1.0 - alpha)))

    pit_t_arr = np.array(pit_t)
    pit_r_arr = np.array(pit_r)
    z_t_arr = np.array([z for z in z_t if not np.isnan(z)])

    ks_t = float(kstest(pit_t_arr, "uniform").pvalue) if pit_t_arr.size else float("nan")
    ks_r = float(kstest(pit_r_arr, "uniform").pvalue) if pit_r_arr.size else float("nan")
    cov = float(np.mean(inside)) if inside else float("nan")
    z_mean = float(z_t_arr.mean()) if z_t_arr.size else float("nan")
    z_std = float(z_t_arr.std(ddof=1)) if z_t_arr.size > 1 else float("nan")

    return CalibrationResult(
        pit_translation=pit_t_arr,
        pit_rotation=pit_r_arr,
        ks_p_translation=ks_t,
        ks_p_rotation=ks_r,
        coverage=cov,
        nominal_coverage=1.0 - alpha,
        z_translation_mean=z_mean,
        z_translation_std=z_std,
    )
