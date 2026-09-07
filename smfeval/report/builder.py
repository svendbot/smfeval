"""Assemble the report dict from sync, alignment, scoring, and calibration outputs."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smfeval.align.fit import AlignmentFit
from smfeval.format import Gauge
from smfeval.scoring.calibration import CalibrationResult
from smfeval.scoring.ensemble_diag import EnsembleDiagnostic
from smfeval.scoring.summary import ScoreSummary
from smfeval.sync.match import MatchResult
from smfeval.sync.mode import SyncMode
from smfeval.sync.risk import DEFAULT_SYNC_RISK_THRESHOLD

# Version of the JSON report contract (docs/report.schema.json). Bump when the
# report structure changes. Separate from the package and SQUARE format versions.
# 2.0: scores are translation-only; rotation/joint scores and the SO(3)
# Gaussian-validity section were removed.
# 3.0: the PIT/KS check was removed — calibration now carries "coverage_p"
# (exact binomial p on the ellipsoidal hit rate) and "n_coverage" in place of
# "ks_p_translation"; sync carries "risk_n", the pairs with a defined risk.
REPORT_SCHEMA_VERSION = "3.0"


@dataclass
class Report:
  schema_version: str = REPORT_SCHEMA_VERSION
  sync: dict[str, Any] = field(default_factory=dict)
  alignment: dict[str, Any] = field(default_factory=dict)
  ensemble: dict[str, Any] | None = None
  scores: dict[str, Any] = field(default_factory=dict)
  calibration: dict[str, Any] = field(default_factory=dict)
  # Populated under --calibration: {"absolute": {joint,translation,rotation},
  # "windowed": [...]} of the log-score calibration/sharpness split.
  calibration_split: dict[str, Any] | None = None
  # Populated under --calibration: per-window track-frame bias/variance
  # (bias_fraction + dominant axis), read by diagnose() for systematic bias.
  bias_variance: list[dict[str, Any]] | None = None
  recommendations: list[str] = field(default_factory=list)
  # Structured, actionable diagnoses; list of Diagnosis dataclasses.
  diagnoses: list[Any] = field(default_factory=list)


def _quantiles(arr: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
  if arr.size == 0:
    return {f"p{int(q * 100):02d}": float("nan") for q in qs}
  return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def build_report(
  match: MatchResult,
  fit: AlignmentFit,
  declared_gauge: Gauge,
  sync_risks: np.ndarray | None,
  ensemble: EnsembleDiagnostic | None,
  scores: dict[str, ScoreSummary],
  calibration: CalibrationResult | None,
  trajectory_length_m: float | None,
  sync_mode: SyncMode = SyncMode.NEAREST,
  sync_risk_threshold: float = DEFAULT_SYNC_RISK_THRESHOLD,
) -> Report:
  rep = Report()

  # Pairs whose sync risk is defined (nan where the step has no sigma, e.g. a
  # deterministic estimate): both the excess count and the quantiles are over
  # these, and risk_n is their count so consumers divide by the right total.
  defined_risks = (
    sync_risks[np.isfinite(sync_risks)]
    if sync_risks is not None
    else np.zeros(0)
  )
  risk_excess = int((defined_risks > sync_risk_threshold).sum())
  rep.sync = {
    "mode": sync_mode,
    "n_matched": match.n_matched,
    "n_total": match.n_total,
    "n_dropped": match.n_dropped,
    "gap_quantiles_ms": match.gap_quantiles_ms,
    "risk_threshold": sync_risk_threshold,
    "risk_n": int(defined_risks.size),
    "risk_excess_count": risk_excess,
    "risk_quantiles": (
      {
        "median": float(np.median(defined_risks)),
        "p95": float(np.quantile(defined_risks, 0.95)),
        "p99": float(np.quantile(defined_risks, 0.99)),
      }
      if defined_risks.size
      else None
    ),
  }

  rep.alignment = {
    "declared_gauge": declared_gauge.value,
    "mode": fit.mode,
    "dof_removed": fit.dof_removed,
    "scale": fit.scale,
    "translation": fit.fitted_translation.tolist(),
    "rotation_matrix": fit.fitted_rotation.tolist(),
    "residual_quantiles": _quantiles(fit.residuals, (0.5, 0.95)),
    "trajectory_length_m": trajectory_length_m,
  }

  if ensemble is not None:
    rep.ensemble = {
      "n_nominal": ensemble.n_nominal,
      "n_eff_quantiles": _quantiles(ensemble.n_eff, (0.5, 0.05, 0.01)),
      "n_unique_quantiles": _quantiles(ensemble.n_unique, (0.5, 0.05, 0.01)),
      "degeneracy_fraction": ensemble.degeneracy_fraction,
    }

  rep.scores = {k: v.to_dict() for k, v in scores.items()}

  if calibration is not None:
    rep.calibration = {
      "coverage": calibration.coverage,
      "nominal_coverage": calibration.nominal_coverage,
      "n_coverage": calibration.n_coverage,
      "coverage_p": calibration.coverage_p,
      "z_translation_mean": calibration.z_translation_mean,
      "z_translation_std": calibration.z_translation_std,
    }

  return rep
