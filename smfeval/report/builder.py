"""Assemble the report dict from sync, alignment, scoring, and calibration outputs."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smfeval.align.fit import AlignmentFit
from smfeval.format import Gauge
from smfeval.scoring.calibration import CalibrationResult
from smfeval.scoring.ensemble_diag import EnsembleDiagnostic
from smfeval.scoring.gaussian_diag import GaussianValidityDiagnostic
from smfeval.scoring.summary import ScoreSummary
from smfeval.sync.match import MatchResult
from smfeval.sync.mode import SyncMode

# Version of the JSON report contract (docs/report.schema.json). Bump when the
# report structure changes. Separate from the package and SQUARE format versions.
REPORT_SCHEMA_VERSION = "1.0"


@dataclass
class Report:
  schema_version: str = REPORT_SCHEMA_VERSION
  sync: dict[str, Any] = field(default_factory=dict)
  alignment: dict[str, Any] = field(default_factory=dict)
  ensemble: dict[str, Any] | None = None
  gaussian_validity: dict[str, Any] | None = None
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
  sync_risk_threshold: float,
  ensemble: EnsembleDiagnostic | None,
  gaussian_validity: GaussianValidityDiagnostic | None,
  scores: dict[str, ScoreSummary],
  calibration: CalibrationResult | None,
  trajectory_length_m: float | None,
  sync_mode: SyncMode = SyncMode.NEAREST,
) -> Report:
  rep = Report()

  risk_excess = (
    int((sync_risks > sync_risk_threshold).sum())
    if sync_risks is not None and sync_risks.size
    else 0
  )
  rep.sync = {
    "mode": sync_mode,
    "n_matched": match.n_matched,
    "n_total": match.n_total,
    "n_dropped": match.n_dropped,
    "gap_quantiles_ms": match.gap_quantiles_ms,
    "risk_threshold": sync_risk_threshold,
    "risk_excess_count": risk_excess,
    "risk_quantiles": (
      {
        "median": float(np.median(sync_risks)),
        "p95": float(np.quantile(sync_risks, 0.95)),
        "p99": float(np.quantile(sync_risks, 0.99)),
      }
      if sync_risks is not None and sync_risks.size
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

  if gaussian_validity is not None:
    rep.gaussian_validity = {
      "n_total": gaussian_validity.n_total,
      "max_sigma_r": gaussian_validity.max_sigma_r,
      "soft_limit_rad": gaussian_validity.soft_limit_rad,
      "hard_limit_rad": gaussian_validity.hard_limit_rad,
      "n_exceeding_soft": gaussian_validity.n_exceeding_soft,
      "n_exceeding_hard": gaussian_validity.n_exceeding_hard,
    }

  rep.scores = {k: v.to_dict() for k, v in scores.items()}

  if calibration is not None:
    rep.calibration = {
      "ks_p_translation": calibration.ks_p_translation,
      "ks_p_rotation": calibration.ks_p_rotation,
      "coverage": calibration.coverage,
      "nominal_coverage": calibration.nominal_coverage,
      "z_translation_mean": calibration.z_translation_mean,
      "z_translation_std": calibration.z_translation_std,
    }

  return rep
