"""A3: each FailureMode fires on a crafted Report, and not otherwise."""

from smfeval.report.builder import Report
from smfeval.report.diagnostics import FailureMode, Severity, diagnose


def _slice(
  verdict: str,
  anees: float,
  *,
  dof: int = 3,
  lo: float = 2.9,
  hi: float = 3.1,
  nees_median: float | None = None,
) -> dict:
  return {
    "anees": anees,
    "dof": dof,
    "n": 100,
    "lo": lo,
    "hi": hi,
    "verdict": verdict,
    "calibration_median": anees / 2.0,
    "sharpness_median": -5.0,
    # default median = mean ⇒ "bulk"; pass a small value for a "tail" regime.
    "nees_median": anees if nees_median is None else nees_median,
  }


def _actions(rep: Report, mode) -> str:
  d = next(d for d in diagnose(rep) if d.mode == mode)
  return " ".join(d.recommended_actions)


def test_regime_tail_routes_to_student_t():
  # optimistic mean but median ≈ dof (bulk calibrated) ⇒ heavy-dynamics tail.
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("optimistic", 5e4, dof=6, hi=6.1, nees_median=5.0),
        "translation": _slice("optimistic", 6e4, nees_median=2.8),
        "rotation": _slice("consistent", 3.0),
      }
    }
  )
  acts = _actions(rep, FailureMode.TRANSLATION_OVERCONFIDENT)
  assert "tail" in acts.lower() and "student-t" in acts.lower()


def test_regime_bulk_routes_to_ess():
  # mean AND median both ≫ dof ⇒ bulk over-confidence.
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("optimistic", 5e4, dof=6, hi=6.1, nees_median=4e4),
        "translation": _slice("optimistic", 6e4, nees_median=5e4),
        "rotation": _slice("consistent", 3.0),
      }
    }
  )
  acts = _actions(rep, FailureMode.TRANSLATION_OVERCONFIDENT)
  assert "bulk" in acts.lower() and "ess" in acts.lower()


def _modes(rep: Report) -> set:
  return {d.mode for d in diagnose(rep)}


def test_no_diagnoses_when_all_consistent():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("consistent", 6.0, dof=6),
        "translation": _slice("consistent", 3.0),
        "rotation": _slice("consistent", 3.0),
      }
    }
  )
  assert diagnose(rep) == []


def test_translation_overconfident_isolated():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("optimistic", 5e4, dof=6),
        "translation": _slice("optimistic", 6e4),
        "rotation": _slice("consistent", 3.0),
      }
    }
  )
  modes = _modes(rep)
  assert FailureMode.TRANSLATION_OVERCONFIDENT in modes
  assert FailureMode.ROTATION_OVERCONFIDENT not in modes
  assert FailureMode.OVERCONFIDENT not in modes


def test_rotation_overconfident_isolated():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("optimistic", 5e4, dof=6),
        "translation": _slice("consistent", 3.0),
        "rotation": _slice("optimistic", 6e4),
      }
    }
  )
  modes = _modes(rep)
  assert FailureMode.ROTATION_OVERCONFIDENT in modes
  assert FailureMode.TRANSLATION_OVERCONFIDENT not in modes


def test_joint_overconfident_when_both_channels_hot():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("optimistic", 5e4, dof=6),
        "translation": _slice("optimistic", 6e4),
        "rotation": _slice("optimistic", 4e4),
      }
    }
  )
  # both channels hot → falls through to the joint OVERCONFIDENT diagnosis.
  assert FailureMode.OVERCONFIDENT in _modes(rep)


def test_underconfident():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("conservative", 1.0, dof=6),
        "translation": _slice("conservative", 0.5),
        "rotation": _slice("conservative", 0.5),
      }
    }
  )
  modes = _modes(rep)
  assert FailureMode.UNDERCONFIDENT in modes
  # severity is INFO for underconfidence.
  d = next(d for d in diagnose(rep) if d.mode == FailureMode.UNDERCONFIDENT)
  assert d.severity == Severity.INFO


def test_local_overconfidence_from_windowed():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("consistent", 6.0, dof=6),
        "translation": _slice("consistent", 3.0),
        "rotation": _slice("consistent", 3.0),
      },
      "windowed": [
        {
          "window_s": 0.1,
          "anees": 1900.0,
          "dof": 3,
          "n": 2800,
          "lo": 2.9,
          "hi": 3.1,
          "verdict": "optimistic",
          "calibration_median": 390.0,
          "sharpness_median": -14.0,
          "sigma_rel_m": 0.009,
        },
      ],
    }
  )
  assert FailureMode.LOCAL_OVERCONFIDENCE in _modes(rep)


def test_horizon_accumulation_when_anees_grows():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("consistent", 6.0, dof=6),
        "translation": _slice("consistent", 3.0),
        "rotation": _slice("consistent", 3.0),
      },
      "windowed": [
        {
          "window_s": 0.1,
          "anees": 1900.0,
          "dof": 3,
          "n": 2800,
          "lo": 2.9,
          "hi": 3.1,
          "verdict": "optimistic",
          "calibration_median": 390.0,
          "sharpness_median": -14.0,
          "sigma_rel_m": 0.009,
        },
        {
          "window_s": 10.0,
          "anees": 13000.0,
          "dof": 3,
          "n": 2700,
          "lo": 2.9,
          "hi": 3.1,
          "verdict": "optimistic",
          "calibration_median": 3500.0,
          "sharpness_median": -25.0,
          "sigma_rel_m": 0.009,
        },
      ],
    }
  )
  modes = _modes(rep)
  assert FailureMode.HORIZON_ACCUMULATION in modes
  assert FailureMode.LOCAL_OVERCONFIDENCE in modes  # both fire


def test_no_accumulation_when_flat():
  rep = Report(
    calibration_split={
      "absolute": {
        "joint": _slice("consistent", 6.0, dof=6),
        "translation": _slice("consistent", 3.0),
        "rotation": _slice("consistent", 3.0),
      },
      "windowed": [
        {
          "window_s": 0.1,
          "anees": 1900.0,
          "dof": 3,
          "n": 2800,
          "lo": 2.9,
          "hi": 3.1,
          "verdict": "optimistic",
          "calibration_median": 390.0,
          "sharpness_median": -14.0,
          "sigma_rel_m": 0.009,
        },
        {
          "window_s": 10.0,
          "anees": 2100.0,
          "dof": 3,
          "n": 2700,
          "lo": 2.9,
          "hi": 3.1,
          "verdict": "optimistic",
          "calibration_median": 420.0,
          "sharpness_median": -14.0,
          "sigma_rel_m": 0.009,
        },
      ],
    }
  )
  assert FailureMode.HORIZON_ACCUMULATION not in _modes(rep)


def test_gaussian_tangent_invalid_critical():
  rep = Report(
    gaussian_validity={
      "n_total": 1000,
      "n_exceeding_soft": 200,
      "n_exceeding_hard": 200,
    }
  )
  d = next(
    d for d in diagnose(rep) if d.mode == FailureMode.GAUSSIAN_TANGENT_INVALID
  )
  assert d.severity == Severity.CRITICAL


def test_sync_risk_fires_only_for_nearest():
  rep = Report(
    sync={
      "n_matched": 1000,
      "risk_excess_count": 500,
      "risk_threshold": 0.3,
      "mode": "nearest",
    }
  )
  assert FailureMode.SYNC_RISK in _modes(rep)
  rep_interp = Report(
    sync={
      "n_matched": 1000,
      "risk_excess_count": 500,
      "risk_threshold": 0.3,
      "mode": "interpolate_gt",
    }
  )
  assert FailureMode.SYNC_RISK not in _modes(rep_interp)


def test_ensemble_degeneracy():
  rep = Report(ensemble={"degeneracy_fraction": 0.2})
  assert FailureMode.ENSEMBLE_DEGENERACY in _modes(rep)


def _bv(bias_fraction: float, axis: str) -> dict:
  return {
    "window_s": 0.1,
    "n_pairs": 100,
    "mse": 1.0,
    "bias_fraction": bias_fraction,
    "bias": [0.1, 0.0, 0.5],
    "std": [0.1, 0.1, 0.1],
    "dominant_axis": axis,
  }


def test_systematic_bias_vertical_routes_to_gravity():
  rep = Report(bias_variance=[_bv(0.75, "vertical")])
  d = next(d for d in diagnose(rep) if d.mode == FailureMode.SYSTEMATIC_BIAS)
  assert "gravity" in " ".join(d.recommended_actions).lower()
  assert d.severity == Severity.CRITICAL


def test_systematic_bias_along_routes_to_scale():
  rep = Report(bias_variance=[_bv(0.4, "along")])
  d = next(d for d in diagnose(rep) if d.mode == FailureMode.SYSTEMATIC_BIAS)
  assert "scale" in " ".join(d.recommended_actions).lower()
  assert d.severity == Severity.WARNING


def test_no_systematic_bias_when_variance_dominated():
  rep = Report(bias_variance=[_bv(0.02, "along")])  # near-zero bias_fraction
  assert FailureMode.SYSTEMATIC_BIAS not in {d.mode for d in diagnose(rep)}


def test_empty_report_no_diagnoses():
  assert diagnose(Report()) == []
