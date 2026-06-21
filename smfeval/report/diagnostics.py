r"""A3: actionable diagnosis layer — map calibration slices to a fault + fix.

The scoring core (calibration term, windowed/horizon term, gaussian validity,
sync risk) tells you *whether* something is wrong; this layer reads those
slices and emits structured :class:`Diagnosis` objects naming a
:class:`FailureMode`, the signals that triggered it, and the recommended
action. It is the machine-readable form of the attribution protocol:

  slice read                         → fault isolated → action
  ----------------------------------   --------------   -------------------------
  translation calibration optimistic   accel / scale    which sensor channel
  windowed: hot at smallest Δt         local            measurement model / sync
  windowed: ANEES grows with Δt        accumulation     tune filter vs loop close
  sync risk excess                     competing cfdr   --sync=interpolate_ref

``diagnose(rep)`` reads only what is present on the :class:`Report`: the
translation calibration split (populated under ``--calibration``), sync, and
ensemble sections. Prior/posterior attribution is left as a hook for when a
prior-dump SQUARE is scored alongside (B2).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum

from smfeval.report.builder import Report


class FailureMode(str, Enum):
  OVERCONFIDENT = "overconfident"
  UNDERCONFIDENT = "underconfident"
  SYSTEMATIC_BIAS = "systematic_bias"
  TRANSLATION_OVERCONFIDENT = "translation_overconfident"
  LOCAL_OVERCONFIDENCE = "local_overconfidence"
  HORIZON_ACCUMULATION = "horizon_accumulation"
  SYNC_RISK = "sync_risk"
  ENSEMBLE_DEGENERACY = "ensemble_degeneracy"


class Severity(str, Enum):
  INFO = "info"
  WARNING = "warning"
  CRITICAL = "critical"


@dataclass
class Diagnosis:
  mode: FailureMode
  severity: Severity
  signals_triggered: list[str]
  explanation: str
  recommended_actions: list[str] = field(default_factory=list)

  def to_dict(self) -> dict:
    return asdict(self)


# ANEES that exceeds the chi2 upper bound by this factor is flagged CRITICAL
# rather than WARNING (an order of magnitude past the consistency interval).
_CRITICAL_ANEES_FACTOR = 10.0
# windowed ANEES must grow by at least this factor from the shortest to the
# longest horizon to read as accumulation rather than flat local overconfidence.
_ACCUMULATION_GROWTH_FACTOR = 2.0


def _sev_for_anees(anees: float, hi: float) -> Severity:
  if hi > 0 and anees > _CRITICAL_ANEES_FACTOR * hi:
    return Severity.CRITICAL
  return Severity.WARNING


def _signal(slice_name: str, s: dict) -> str:
  med = s.get("nees_median", float("nan"))
  return (
    f"{slice_name} ANEES {s['anees']:.3g} (median {med:.3g}) vs χ² interval "
    f"[{s['lo']:.3g}, {s['hi']:.3g}] (dof {s['dof']}) → {s['verdict']}"
  )


def _regime(s: dict) -> str:
  """Classify the over-confidence regime.

  Either a heavy-dynamics TAIL (median NEES ≈ dof, bulk calibrated, mean
  inflated by outliers) or the BULK (median also hot).
  """
  med = s.get("nees_median", float("nan"))
  hi = s.get("hi", float("nan"))
  if med != med or hi != hi:
    return "unknown"
  return "tail" if med <= hi else "bulk"


def _regime_action(s: dict) -> str:
  """Route the fix by regime — the productized B2 mean-vs-median lesson."""
  r = _regime(s)
  if r == "tail":
    return (
      "Median NEES ≈ dof but the mean is inflated → the over-confidence is a "
      "heavy-dynamics TAIL, not the bulk. A robust (Student-t) likelihood "
      "targets it; a global Σ inflation would make the (calibrated) bulk "
      "under-confident."
    )
  if r == "bulk":
    return (
      "Median NEES also exceeds the interval → BULK over-confidence (not just a "
      "tail). Widen the covariance — e.g. the ESS c∝n_eff inflation — and/or "
      "the dominant noise term."
    )
  return "Compare mean vs median NEES to tell a heavy-dynamics tail from bulk."


def _diagnose_absolute(absolute: dict) -> list[Diagnosis]:
  """Channel attribution from the translation calibration slice."""
  out: list[Diagnosis] = []
  trans = absolute.get("translation")
  if not trans:
    return out
  tv = trans["verdict"]

  if tv == "optimistic":
    out.append(
      Diagnosis(
        mode=FailureMode.TRANSLATION_OVERCONFIDENT,
        severity=_sev_for_anees(trans["anees"], trans["hi"]),
        signals_triggered=[_signal("translation", trans)],
        explanation=(
          "Translation covariance is too tight — the reference falls far outside "
          "the predicted ellipsoid (the calibration term stays graded where "
          "CRPS/coverage saturate)."
        ),
        recommended_actions=[
          "Suspect the accelerometer noise / scale / time-offset (the "
          "translation channel).",
          "Check the windowed slice: hot only at long Δt → drift accumulation; "
          "hot at 0.1 s → local measurement-model overconfidence.",
          _regime_action(trans),
        ],
      )
    )
  elif tv == "conservative":
    out.append(
      Diagnosis(
        mode=FailureMode.UNDERCONFIDENT,
        severity=Severity.INFO,
        signals_triggered=[_signal("translation", trans)],
        explanation=(
          "Translation covariance is looser than the realised error needs."
        ),
        recommended_actions=["Tighten the dominant process-noise term."],
      )
    )
  return out


def _diagnose_windowed(windowed: list[dict]) -> list[Diagnosis]:
  """Horizon attribution from the windowed (relative) calibration slices."""
  out: list[Diagnosis] = []
  rows = sorted(
    (w for w in windowed if w.get("n", 1)), key=lambda w: w["window_s"]
  )
  if not rows:
    return out
  shortest, longest = rows[0], rows[-1]

  if shortest["verdict"] == "optimistic":
    out.append(
      Diagnosis(
        mode=FailureMode.LOCAL_OVERCONFIDENCE,
        severity=_sev_for_anees(shortest["anees"], shortest["hi"]),
        signals_triggered=[
          f"Δt={shortest['window_s']:g}s " + _signal("windowed", shortest)
        ],
        explanation=(
          "Over-confident even at the shortest horizon — a *local* precision "
          "problem (one-sided under the iid Σ_rel bound, so this is a lower "
          "bound on the over-confidence)."
        ),
        recommended_actions=[
          "Local over-confidence points at the measurement model (per-scan "
          "likelihood / ESS), not long-horizon drift.",
          "Rule out the sync confounder first: re-score with "
          "--sync=interpolate_ref (if ANEES drops, sync; if not, it is real).",
        ],
      )
    )

  grows = (
    longest["anees"] > _ACCUMULATION_GROWTH_FACTOR * shortest["anees"]
    and longest["window_s"] > shortest["window_s"]
  )
  if grows:
    out.append(
      Diagnosis(
        mode=FailureMode.HORIZON_ACCUMULATION,
        severity=Severity.WARNING,
        signals_triggered=[
          f"windowed ANEES grows {shortest['anees']:.3g} "
          f"(Δt={shortest['window_s']:g}s) → {longest['anees']:.3g} "
          f"(Δt={longest['window_s']:g}s)"
        ],
        explanation=(
          "The calibration excess grows with horizon — error accumulates "
          "faster than the (flat) local σ admits."
        ),
        recommended_actions=[
          "Accumulation that only appears at long Δt argues for loop closure "
          "/ global correction over local filter retuning.",
          "Cross-check the bias/variance decomposition (B1): a growing "
          "along-track bias points at scale/time-offset.",
        ],
      )
    )
  return out


_BIAS_FRACTION_THRESHOLD = 0.3
_AXIS_CHANNEL = {
  "vertical": "gravity alignment / initial attitude (a vertical-dominated "
  "systematic offset is the gravity-leak / init signature)",
  "along": "scale or time-offset (an along-track systematic offset grows with "
  "speed)",
  "cross": "a lateral extrinsic / heading error",
}


def _diagnose_bias_variance(bv: list[dict]) -> list[Diagnosis]:
  """B1: a high bias_fraction ⇒ systematic error.

  The dominant axis localises the channel (vertical→gravity/init,
  along→scale/time-offset, cross→extrinsic). Distinguishes 'recalibrate
  the rig/init' from 'tune the noise model'.
  """
  rows = [w for w in bv if w.get("bias_fraction") == w.get("bias_fraction")]
  if not rows:
    return []
  worst = max(rows, key=lambda w: w["bias_fraction"])
  if worst["bias_fraction"] < _BIAS_FRACTION_THRESHOLD:
    return []
  axis = worst["dominant_axis"]
  sev = Severity.CRITICAL if worst["bias_fraction"] > 0.6 else Severity.WARNING
  return [
    Diagnosis(
      mode=FailureMode.SYSTEMATIC_BIAS,
      severity=sev,
      signals_triggered=[
        f"bias_fraction {worst['bias_fraction']:.2f} at Δt={worst['window_s']:g}s "
        f"(dominant axis: {axis}); ‖bias‖ vs spread → systematic"
      ],
      explanation=(
        f"The windowed increment error is dominated by a SYSTEMATIC offset "
        f"(bias_fraction {worst['bias_fraction']:.2f}), not random spread — and "
        f"it is largest in the {axis} channel."
      ),
      recommended_actions=[
        f"Recalibrate / fix {_AXIS_CHANNEL.get(axis, axis)} — not the noise "
        "model (this is bias, not variance).",
        "Contrast: a near-zero bias_fraction would point at random noise "
        "(measurement-model / ESS), not the rig or init.",
      ],
    )
  ]


def diagnose(rep: Report) -> list[Diagnosis]:
  """Map the report's slices to structured, actionable diagnoses."""
  out: list[Diagnosis] = []

  split = rep.calibration_split or {}
  if split.get("absolute"):
    out.extend(_diagnose_absolute(split["absolute"]))
  if split.get("windowed"):
    out.extend(_diagnose_windowed(split["windowed"]))
  if rep.bias_variance:
    out.extend(_diagnose_bias_variance(rep.bias_variance))

  sync = rep.sync or {}
  n_matched = sync.get("n_matched", 0) or 0
  risk_excess = sync.get("risk_excess_count", 0) or 0
  mode = sync.get("mode")
  mode_val = getattr(mode, "value", mode)
  if n_matched and risk_excess / n_matched > 0.01 and mode_val == "nearest":
    out.append(
      Diagnosis(
        mode=FailureMode.SYNC_RISK,
        severity=Severity.WARNING,
        signals_triggered=[
          f"{100.0 * risk_excess / n_matched:.1f}% of pairs exceed sync risk "
          f"{sync.get('risk_threshold', 0.3):.1f}"
        ],
        explanation=(
          "A competing confounder: timestamp-matching error shrinks short-"
          "window Σ_rel the same way local over-confidence does."
        ),
        recommended_actions=[
          "Re-score with --sync=interpolate_ref to separate sync from a genuine "
          "calibration fault before trusting short-horizon verdicts.",
        ],
      )
    )

  ens = rep.ensemble or {}
  if ens.get("degeneracy_fraction", 0.0) > 0.01:
    frac = 100.0 * ens["degeneracy_fraction"]
    out.append(
      Diagnosis(
        mode=FailureMode.ENSEMBLE_DEGENERACY,
        severity=Severity.WARNING,
        signals_triggered=[f"{frac:.1f}% of timesteps have N_eff < N/10"],
        explanation="The particle filter is approaching weight collapse.",
        recommended_actions=["Investigate resampling cadence / proposal."],
      )
    )

  return out
