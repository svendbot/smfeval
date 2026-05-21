"""Continuous-time GT interpolation via piecewise Gaussian Process on SE(3).

Implements the construction of Zhang & Scaramuzza (2019), §IV.B
(arXiv:1906.03996): for each query timestamp, take a local window of GT
samples bracketing the query, choose the middle sample as ``T_ref``, express
the surrounding poses as ``ξ_i = log(T_ref⁻¹ · T_i) ∈ se(3)``, and fit
independent squared-exponential GPs on each of the six components of ``ξ``
as a function of time. The predictive ``μ_ξ*`` at the query time is mapped
back to ``T* = T_ref · Exp(μ_ξ*)``; the predictive variance ``v*`` is shared
across all six components (the kernel does not depend on the data), giving
``Σ_ξ* = v* · I_6`` on the right-perturbation tangent at ``T*``.

The piecewise / windowed scheme follows the paper's practical choice
(§IV.B): "we select the segments so that the adjacent segments overlap and
use the same hyperparameters for all segments". Defaults pick a window of
10 GT samples around each query and use a squared-exponential kernel with
length scale 0.1 s and unit signal variance — small enough to track local
curvature, large enough to smooth GT noise.

Query times outside the GT range are flagged in the returned ``keep`` mask
rather than extrapolated; the cross-check use case in smfeval has no
business extrapolating into regions where the GP is reverting to its prior.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.se3.lie import invert, pose_matrix, se3_exp, se3_log
from src.types import TangentOrder


def interpolate_gt_at(
    query_times: np.ndarray,
    gt_times: np.ndarray,
    gt_translations: np.ndarray,
    gt_quats: np.ndarray,
    window: int = 10,
    length_scale_s: float = 0.1,
    signal_variance: float = 1.0,
    noise_variance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-GP interpolation of an SE(3) trajectory at query times.

    Parameters
    ----------
    query_times : (Q,) array of timestamps at which to interpolate.
    gt_times : (N,) GT sample times, must be sorted.
    gt_translations : (N, 3) GT translations.
    gt_quats : (N, 4) GT quaternions in xyzw.
    window : number of nearest GT samples to use per query (paper recommends
        ~50% overlap between segments, equivalent to a symmetric local window).
    length_scale_s : SE-kernel length scale in seconds.
    signal_variance : SE-kernel signal variance.
    noise_variance : observation noise on the GT samples; small but non-zero
        for numerical stability of K_zz inversion.

    Returns
    -------
    translations : (Q, 3) interpolated translations (zeros where ``keep`` is False).
    quats : (Q, 4) interpolated quaternions xyzw.
    covariances : (Q, 6, 6) tangent-space predictive covariance in
        ``translation_rotation`` order. Same scalar variance on all six diagonal
        entries (kernel is shared across components, per the paper).
    keep : (Q,) bool — False where the query fell outside ``[gt_times[0],
        gt_times[-1]]``.
    """
    query_times = np.asarray(query_times, dtype=float)
    gt_times = np.asarray(gt_times, dtype=float)
    gt_translations = np.asarray(gt_translations, dtype=float)
    gt_quats = np.asarray(gt_quats, dtype=float)
    n_q = len(query_times)
    n_gt = len(gt_times)

    if n_gt < 2:
        raise ValueError("need at least 2 GT samples to interpolate")
    window = min(window, n_gt)

    out_t = np.zeros((n_q, 3))
    out_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_q, 1))
    out_cov = np.zeros((n_q, 6, 6))
    keep = (query_times >= gt_times[0]) & (query_times <= gt_times[-1])

    for i, qt in enumerate(query_times):
        if not keep[i]:
            continue

        # Local window: `window` GT samples centered on the query insertion point.
        center = int(np.searchsorted(gt_times, qt))
        lo = max(0, center - window // 2)
        hi = min(n_gt, lo + window)
        lo = max(0, hi - window)
        idx = slice(lo, hi)
        t_win = gt_times[idx]

        # Reference pose: middle of the window. Local tangent expansion ξ_i.
        ref_local = (hi - lo) // 2
        T_ref = pose_matrix(gt_translations[lo + ref_local], gt_quats[lo + ref_local])
        T_ref_inv = invert(T_ref)
        xis = np.zeros((hi - lo, 6))
        for j in range(hi - lo):
            T_j = pose_matrix(gt_translations[lo + j], gt_quats[lo + j])
            xis[j] = se3_log(T_ref_inv @ T_j, TangentOrder.TRANS_ROT)

        # Squared-exponential kernel — eq. (32). With shared kernel across the
        # 6 components, the predictive variance is a scalar, the same for all
        # components; the predictive mean is component-wise GP regression.
        dt = t_win - qt
        dt_pair = t_win[:, None] - t_win[None, :]
        K_zz = signal_variance * np.exp(-0.5 * (dt_pair / length_scale_s) ** 2)
        K_zz += noise_variance * np.eye(hi - lo)
        K_qz = signal_variance * np.exp(-0.5 * (dt / length_scale_s) ** 2)

        try:
            alpha = np.linalg.solve(K_zz, xis)
            v_kk = np.linalg.solve(K_zz, K_qz)
        except np.linalg.LinAlgError:
            keep[i] = False
            continue
        mu_xi = K_qz @ alpha
        var_xi = max(float(signal_variance - K_qz @ v_kk), 0.0)

        T_interp = T_ref @ se3_exp(mu_xi, TangentOrder.TRANS_ROT)
        out_t[i] = T_interp[:3, 3]
        out_q[i] = Rotation.from_matrix(T_interp[:3, :3]).as_quat()
        out_cov[i] = var_xi * np.eye(6)

    return out_t, out_q, out_cov, keep
