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
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

from smfeval.format import TangentOrder
from smfeval.se3.lie import invert, pose_matrix, se3_exp, se3_log

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


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

  Returns:
  --------
  translations : (Q, 3) interpolated translations (zeros where ``keep`` is False).
  quats : (Q, 4) interpolated quaternions xyzw.
  covariances : (Q, 6, 6) tangent-space predictive covariance in
      ``translation_rotation`` order. Same scalar variance on all six diagonal
      entries (kernel is shared across components, per the paper).
  keep : (Q,) bool — False where the query fell outside the GT time span
      ``[gt_times[0], gt_times[-1]]``.
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
  out_q = np.tile(_IDENTITY_QUAT_XYZW, (n_q, 1))
  out_cov = np.zeros((n_q, 6, 6))
  keep = np.zeros(n_q, dtype=bool)
  in_range = (query_times >= gt_times[0]) & (query_times <= gt_times[-1])

  T_all = np.stack(
    [pose_matrix(gt_translations[k], gt_quats[k]) for k in range(n_gt)]
  )

  i = 0
  while i < n_q:
    if not in_range[i]:
      i += 1
      continue
    qt = query_times[i]

    center = int(np.searchsorted(gt_times, qt))
    lo = max(0, center - window // 2)
    hi = min(n_gt, lo + window)
    lo = max(0, hi - window)
    t_win = gt_times[lo:hi]

    ref_local = (hi - lo) // 2
    T_ref = T_all[lo + ref_local]
    T_ref_inv = invert(T_ref)
    xis = np.array(
      [
        se3_log(T_ref_inv @ T_all[lo + j], TangentOrder.TRANS_ROT)
        for j in range(hi - lo)
      ]
    )

    dt = t_win - qt
    dt_pair = t_win[:, None] - t_win[None, :]
    K_zz = signal_variance * np.exp(-0.5 * (dt_pair / length_scale_s) ** 2)
    K_zz += noise_variance * np.eye(hi - lo)
    K_qz = signal_variance * np.exp(-0.5 * (dt / length_scale_s) ** 2)

    # SE kernel + positive noise_variance jitter is mathematically SPD;
    # cho_factor is the cheap path and we trust it to succeed.
    c_and_lower = cho_factor(K_zz)
    alpha = cho_solve(c_and_lower, xis)
    v_kk = cho_solve(c_and_lower, K_qz)
    mu_xi = K_qz @ alpha
    var_xi = max(float(signal_variance - K_qz @ v_kk), 0.0)

    T_interp = T_ref @ se3_exp(mu_xi, TangentOrder.TRANS_ROT)
    out_t[i] = T_interp[:3, 3]
    out_q[i] = Rotation.from_matrix(T_interp[:3, :3]).as_quat()
    out_cov[i] = var_xi * np.eye(6)
    keep[i] = True
    i += 1

  return out_t, out_q, out_cov, keep
