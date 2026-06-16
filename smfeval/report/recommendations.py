"""Plain-language recommendations cross-referencing report sections."""

import math

from smfeval.report.builder import Report
from smfeval.sync.mode import SyncMode


def recommendations(rep: Report) -> list[str]:
  out: list[str] = []
  sync = rep.sync
  alignment = rep.alignment
  ensemble = rep.ensemble
  cal = rep.calibration

  risk_excess = sync.get("risk_excess_count", 0) or 0
  n_matched = sync.get("n_matched", 0) or 0
  if (
    n_matched
    and risk_excess / n_matched > 0.01
    and sync.get("mode", SyncMode.NEAREST) is SyncMode.NEAREST
  ):
    frac = 100.0 * risk_excess / n_matched
    out.append(
      f"{frac:.1f}% of pairs have sync risk > {sync['risk_threshold']:.1f}; "
      "consider cross-checking with --sync=interpolate_gt to confirm "
      "calibration findings."
    )

  traj_len = alignment.get("trajectory_length_m")
  dof = alignment.get("dof_removed", 0)
  if traj_len and dof and dof > 0 and traj_len < 50.0 * dof:
    out.append(
      f"{dof} DoF removed over {traj_len:.0f} m of trajectory; "
      "post-alignment residuals are biased low. Consider --n_to_align "
      "to fit on a prefix and score on the remainder."
    )

  if ensemble and ensemble.get("degeneracy_fraction", 0.0) > 0.01:
    frac = 100.0 * ensemble["degeneracy_fraction"]
    out.append(
      f"Ensemble degeneracy: {frac:.1f}% of timesteps have N_eff < N/10. "
      "Filter is approaching collapse — investigate resampling cadence."
    )

  if cal:
    ks = cal.get("ks_p_translation", float("nan"))
    cov = cal.get("coverage", float("nan"))
    nominal = cal.get("nominal_coverage", float("nan"))
    if not (math.isnan(ks) or math.isnan(cov) or math.isnan(nominal)):
      if ks < 0.05 and cov < nominal - 0.05:
        out.append(
          "Coverage below nominal combined with KS p < 0.05 — the "
          "filter is over-confident (claimed Σ too tight, reference "
          "falls outside the predicted intervals); widen process "
          "noise. Miscalibration is unlikely to be explained by "
          "sync error alone."
        )
      elif ks < 0.05 and cov > nominal + 0.05:
        out.append(
          "Coverage exceeds nominal and KS p < 0.05 — the filter "
          "is under-confident (claimed Σ too loose); tighten "
          "process noise."
        )

  return out
