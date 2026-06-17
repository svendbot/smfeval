r"""Relative (short-window) CRPS on SE(3) translation increments.

Absolute-pose CRPS is structurally insensitive to :math:`\sigma`
calibration in the overconfident regime that every LIO filter we have
measured operates in: for :math:`N(\mu, \sigma^2)` against the reference
:math:`y`, ``CRPS`` :math:`\to |y-\mu|` once :math:`|z| = |y-\mu|/\sigma
\gg 1`, so any change in :math:`\sigma` that keeps :math:`|z| \gg 1` is
invisible (Gneiting & Raftery 2007, closed form). With per-scan
:math:`\sigma \approx 1.4` mm and APE :math:`\approx 286` mm we get
:math:`z \approx 200` — deep in the linear regime.

That is the wrong question for an *odometry* filter. ``\sigma_post`` is a
**local** precision quantity, bounded by the steady-state Kalman gain;
it is never expected to track integrated APE. The metric that matches it
is the *relative pose error* (RPE) over a short window :math:`\Delta t`:

.. math::

   \Delta\mu_\mathrm{est} = p_\mathrm{est}(t+\Delta t) - p_\mathrm{est}(t)
   \qquad
   \Delta\mu_\mathrm{ref}  = p_\mathrm{ref} (t+\Delta t) - p_\mathrm{ref} (t)

   \Sigma_\mathrm{rel} = \Sigma_\mathrm{pos}(t) + \Sigma_\mathrm{pos}(t+\Delta t)

   \mathrm{CRPS_{rel}}(\Delta t)
     = \frac1{|P|}\sum_{(i,j)\in P}
       \overline{\mathrm{CRPS}\bigl(N(\Delta\mu_\mathrm{est},
                                     \Sigma_\mathrm{rel}),
                                    \Delta\mu_\mathrm{ref}\bigr)}

(per-axis mean, mirroring :func:`translation_crps`). The increment is
formed on the **SE(3)-aligned** estimate, so the position covariance has
already been rotated into the reference frame by the alignment stage; RPE is
otherwise frame-invariant so no further alignment is needed.

:math:`\Sigma_\mathrm{rel} = \Sigma_i + \Sigma_j` is the
independent-errors *upper* bound — consecutive EKF posteriors are
positively correlated through the shared map and IMU prior, but
FAST-LIO-class filters do not publish the cross-covariance. The bound
makes the test conservative: if relative CRPS / ``mean(z²)`` still
indicates overconfidence under this bound, the filter is genuinely
locally overconfident.

At short :math:`\Delta t` the realised RPE is also local, so
:math:`z = |\mathrm{RPE}|/\sigma_\mathrm{rel}` can be :math:`O(1)` —
putting CRPS back in the *nonlinear* regime where it is sensitive to
:math:`\sigma` calibration. Reporting several windows (e.g. 0.1, 1, 10 s)
exposes the local-precision / drift profile as a function of horizon.

See ``plans/0.5-relative-crps.md``.
"""

from dataclasses import dataclass

import numpy as np

from smfeval.format import TangentOrder
from smfeval.scoring.crps import _gaussian_crps
from smfeval.scoring.logscore import AneesResult, anees_consistency
from smfeval.scoring.summary import ScoreSummary, summarize
from smfeval.se3.lie import trans_slice
from smfeval.steps import GaussianStep, Step


@dataclass
class RelativeCrpsResult:
  r"""Relative translation CRPS at one window :math:`\Delta t`."""

  window_s: float
  n_pairs: int
  crps: ScoreSummary
  mean_z2: float
  sigma_rel_median_m: float
  rpe_rmse_m: float


def _default_tolerance(ts: np.ndarray, tolerance_s: float | None) -> float:
  """Window-pair tolerance: caller's value, else half the median period."""
  if tolerance_s is not None:
    return tolerance_s
  dt_med = float(np.median(np.diff(np.sort(ts)))) if ts.size > 1 else 0.0
  return 0.5 * dt_med


def _window_pairs(
  timestamps: np.ndarray, window_s: float, tol_s: float
) -> tuple[np.ndarray, np.ndarray]:
  r"""Index pairs ``(i, j)`` with ``t_j - t_i`` closest to ``window_s``.

  For each ``i`` the nearest later timestamp to ``t_i + window_s`` is
  taken, accepted only when ``j > i`` and the residual is within
  ``tol_s``. Timestamp-based (not fixed-stride) so it is robust to
  non-uniform sampling and dropped scans.
  """
  ts = np.asarray(timestamps, dtype=float)
  target = ts + window_s
  j = np.searchsorted(ts, target)
  j = np.clip(j, 0, ts.size - 1)
  jm1 = np.maximum(j - 1, 0)
  pick_left = np.abs(ts[jm1] - target) < np.abs(ts[j] - target)
  j = np.where(pick_left, jm1, j)
  i = np.arange(ts.size)
  valid = (j > i) & (np.abs(ts[j] - target) <= tol_s)
  return i[valid], j[valid]


@dataclass
class RelativeCalibrationResult:
  r"""Windowed translation calibration term (NEES, dof 3) at one :math:`\Delta t`.

  The horizon slice of the log-score calibration term. For each window-pair the
  relative-translation residual :math:`r = \Delta p_\mathrm{ref} -
  \Delta p_\mathrm{est}` is scored under :math:`\Sigma_\mathrm{rel} =
  \Sigma_{tt}(i) + \Sigma_{tt}(j)`, giving a 3-dof NEES whose ANEES is tested
  against the two-sided χ² interval.

  **One-sided by construction.** :math:`\Sigma_\mathrm{rel}` is the *iid upper
  bound* — consecutive posteriors are positively correlated through the shared
  map/IMU prior, but FAST-LIO-class filters do not publish :math:`\Sigma_{ij}`.
  Ignoring that (positive) cross-covariance inflates :math:`\Sigma_\mathrm{rel}`,
  which *deflates* the NEES. So an ``optimistic`` verdict here is a genuine
  lower bound on over-confidence (the true NEES is larger); a ``consistent`` or
  ``conservative`` verdict cannot certify short-window calibration.
  """

  window_s: float
  anees: AneesResult
  calibration_median: float
  sharpness_median: float
  sigma_rel_median_m: float


def relative_calibration(
  steps: list[Step],
  ref_translations: np.ndarray,
  *,
  windows_s: list[float],
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  tolerance_s: float | None = None,
  alpha: float = 0.05,
) -> list[RelativeCalibrationResult]:
  r"""Windowed translation calibration term (dof-3 NEES + ANEES χ² verdict).

  The calibration-component analogue of :func:`relative_translation_crps`:
  where the latter saturates to :math:`|\mathrm{RPE}|` in the overconfident
  regime, the NEES stays graded. One result per window (skipping windows with
  no valid pairs). See :class:`RelativeCalibrationResult` for the one-sided
  caveat from the iid :math:`\Sigma_\mathrm{rel}` bound.
  """
  if not all(isinstance(s, GaussianStep) for s in steps):
    raise TypeError(
      "relative_calibration requires GaussianStep inputs "
      "(a published position covariance); got a non-Gaussian step"
    )
  ts = np.array([s.timestamp for s in steps], dtype=float)
  mu = np.array([s.translation for s in steps], dtype=float)
  sl = trans_slice(tangent_order)
  cov_tt = np.array([s.covariance[sl, sl] for s in steps], dtype=float)
  ref = np.asarray(ref_translations, dtype=float)
  tol = _default_tolerance(ts, tolerance_s)

  results: list[RelativeCalibrationResult] = []
  for w in windows_s:
    i, j = _window_pairs(ts, w, tol)
    if i.size == 0:
      continue
    r = (ref[j] - ref[i]) - (mu[j] - mu[i])  # relative residual (M, 3)
    cov_rel = cov_tt[i] + cov_tt[j]  # iid upper bound, (M, 3, 3)
    # batched _score_components: nan/inf where Sigma_rel is not PD
    nees = np.full(i.size, np.inf)
    sign, logdet = np.linalg.slogdet(cov_rel)
    pd = sign > 0
    if pd.any():
      sol = np.linalg.solve(cov_rel[pd], r[pd, :, None])[:, :, 0]
      nees[pd] = np.einsum("ij,ij->i", r[pd], sol)
    calib = 0.5 * nees
    sharp = np.where(pd, 0.5 * (logdet + 3.0 * np.log(2.0 * np.pi)), np.inf)
    sig = np.sqrt(cov_rel[:, [0, 1, 2], [0, 1, 2]].sum(axis=1))
    finite = np.isfinite(calib)
    results.append(
      RelativeCalibrationResult(
        window_s=float(w),
        anees=anees_consistency(nees, dof=3, alpha=alpha),
        calibration_median=(
          float(np.median(calib[finite])) if finite.any() else float("nan")
        ),
        sharpness_median=(
          float(np.median(sharp[finite])) if finite.any() else float("nan")
        ),
        sigma_rel_median_m=float(np.median(sig)),
      )
    )
  return results


def relative_translation_crps(
  steps: list[Step],
  ref_translations: np.ndarray,
  *,
  windows_s: list[float],
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  tolerance_s: float | None = None,
  rng: np.random.Generator | None = None,
) -> list[RelativeCrpsResult]:
  r"""Short-window relative translation CRPS at each window in ``windows_s``.

  Args:
    steps: SE(3)-aligned estimate steps. Must be :class:`GaussianStep` —
      the relative metric needs a published position covariance.
    ref_translations: Matched reference translations, element-aligned
      with ``steps``.
    windows_s: Relative-pose windows :math:`\Delta t` in seconds.
    tangent_order: Tangent block order of the step covariances.
    rng: Generator for the bootstrap CI on the per-pair CRPS series.
    tolerance_s: Max allowed deviation of a realised pair gap from the
      requested window. Defaults to half the median sampling period.

  Returns:
    One :class:`RelativeCrpsResult` per window (skipping windows with no
    valid pairs).
  """
  if not all(isinstance(s, GaussianStep) for s in steps):
    raise TypeError(
      "relative_translation_crps requires GaussianStep inputs "
      "(a published position covariance); got a non-Gaussian step"
    )
  ts = np.array([s.timestamp for s in steps], dtype=float)
  mu = np.array([s.translation for s in steps], dtype=float)
  sl = trans_slice(tangent_order)
  var = np.array([np.diag(s.covariance)[sl] for s in steps], dtype=float)
  ref = np.asarray(ref_translations, dtype=float)

  tol = _default_tolerance(ts, tolerance_s)

  results: list[RelativeCrpsResult] = []
  for w in windows_s:
    i, j = _window_pairs(ts, w, tol)
    if i.size == 0:
      continue
    de = mu[j] - mu[i]  # predicted relative translation (aligned frame)
    dg = ref[j] - ref[i]  # realised relative translation
    sigma = np.sqrt(var[i] + var[j])  # (M, 3) per-axis iid bound
    crps_axis = _gaussian_crps(de, sigma, dg)  # (M, 3)
    per_pair = crps_axis.mean(axis=1)  # mean over the 3 axes
    z2 = float((((dg - de) / sigma) ** 2).mean())
    sigma_rel = np.sqrt((sigma**2).sum(axis=1))  # L2 norm of sigma per pair
    rpe = np.linalg.norm(dg - de, axis=1)
    results.append(
      RelativeCrpsResult(
        window_s=float(w),
        n_pairs=int(i.size),
        crps=summarize(per_pair, rng=rng),
        mean_z2=z2,
        sigma_rel_median_m=float(np.median(sigma_rel)),
        rpe_rmse_m=float(np.sqrt((rpe**2).mean())),
      )
    )
  return results
