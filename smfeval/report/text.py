"""Render the structured report into the plain-text layout from the spec."""

import math

from smfeval.report.builder import Report
from smfeval.report.diagnostics import sync_risk_excess_fraction
from smfeval.scoring import SCORE_LABELS
from smfeval.sync.mode import SyncMode
from smfeval.sync.risk import DEFAULT_SYNC_RISK_THRESHOLD


def _fmt_int(n: int | None) -> str:
  return f"{n:,}" if n is not None else "—"


def _fmt_pct(x: float) -> str:
  return f"{100.0 * x:.1f}%" if not math.isnan(x) else "n/a"


def _fmt_float(x: float, prec: int = 4) -> str:
  return f"{x:.{prec}f}" if not math.isnan(x) else "n/a"


def render_report(rep: Report) -> str:
  out: list[str] = []
  out.append("=== smfeval scoring report ===")
  out.append("")
  out.append(_render_sync(rep))
  out.append(_render_alignment(rep))
  if rep.ensemble is not None:
    out.append(_render_ensemble(rep))
  out.append(_render_scores(rep))
  out.append(_render_calibration(rep))
  if rep.calibration_split:
    out.append(_render_calibration_split(rep))
  if rep.diagnoses:
    out.append(_render_diagnoses(rep))
  out.append(_render_recommendations(rep))
  return "\n".join(out)


def _render_sync(rep: Report) -> str:
  s = rep.sync
  gaps = s.get("gap_quantiles_ms", {})
  risk_q = s.get("risk_quantiles")
  mode: SyncMode = s.get("mode", SyncMode.NEAREST)
  lines = [
    "Synchronization",
    f"  Mode:                   {mode.value}",
    f"  Pairs matched:          {_fmt_int(s.get('n_matched'))} / {_fmt_int(s.get('n_total'))}",
    f"  Dropped:                {_fmt_int(s.get('n_dropped'))}",
  ]
  match mode:
    case SyncMode.NEAREST:
      lines.append(
        f"  Timestamp gap (ms):     median {_fmt_float(gaps.get('median', float('nan')), 2)},"
        f" p95 {_fmt_float(gaps.get('p95', float('nan')), 2)},"
        f" p99 {_fmt_float(gaps.get('p99', float('nan')), 2)}"
      )
      if risk_q is not None:
        lines.append(
          f"  Sync risk (v·Δt / σ):   median {_fmt_float(risk_q['median'], 4)},"
          f" p95 {_fmt_float(risk_q['p95'], 4)},"
          f" p99 {_fmt_float(risk_q['p99'], 4)}"
        )
        excess = s.get("risk_excess_count", 0)
        if excess:
          frac = 100.0 * sync_risk_excess_fraction(s)
          threshold = s.get("risk_threshold", DEFAULT_SYNC_RISK_THRESHOLD)
          lines.append(
            f"                          [warning] {excess} pairs ({frac:.1f}%) "
            f"exceed risk {threshold:.1f}"
          )
    case SyncMode.INTERPOLATE_REF:
      if risk_q is not None:
        lines.append(
          f"  GP σ (m):   median {_fmt_float(risk_q['median'], 4)},"
          f" p95 {_fmt_float(risk_q['p95'], 4)},"
          f" p99 {_fmt_float(risk_q['p99'], 4)}"
        )
  lines.append("")
  return "\n".join(lines)


def _render_alignment(rep: Report) -> str:
  a = rep.alignment
  t = a.get("translation", [0.0, 0.0, 0.0])
  res = a.get("residual_quantiles", {})
  traj_len = a.get("trajectory_length_m")
  lines = [
    "Alignment",
    f"  Gauge (declared):       {a.get('declared_gauge')}",
    f"  Mode applied:           {a.get('mode')}   ({a.get('dof_removed')} DoF)",
    f"  Fitted Δxyz:            ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m",
  ]
  if a.get("scale", 1.0) != 1.0:
    lines.append(f"  Fitted scale:           {a['scale']:.6f}")
  lines.append(
    f"  Fit residual (m):       median {_fmt_float(res.get('p50', float('nan')))},"
    f" p95 {_fmt_float(res.get('p95', float('nan')))}"
  )
  if traj_len is not None:
    lines.append(
      f"                          {a.get('dof_removed')} DoF removed "
      f"over {traj_len:.0f} m of trajectory"
    )
  lines.append("")
  return "\n".join(lines)


def _render_ensemble(rep: Report) -> str:
  e = rep.ensemble or {}
  n_eff = e.get("n_eff_quantiles", {})
  n_uniq = e.get("n_unique_quantiles", {})
  lines = [
    "Ensemble diagnostics",
    f"  Nominal N:              {e.get('n_nominal')}",
    f"  N_eff from weights:     median {_fmt_float(n_eff.get('p50', float('nan')), 0)},"
    f" p05 {_fmt_float(n_eff.get('p05', float('nan')), 0)}"
    f" p01 {_fmt_float(n_eff.get('p01', float('nan')), 0)}",
    f"  Unique particles:       median {_fmt_float(n_uniq.get('p50', float('nan')), 0)},"
    f" p05 {_fmt_float(n_uniq.get('p05', float('nan')), 0)}"
    f" p01 {_fmt_float(n_uniq.get('p01', float('nan')), 0)}",
  ]
  deg = e.get("degeneracy_fraction", 0.0)
  if deg > 0.0:
    lines.append(
      f"                          [warning] {_fmt_pct(deg)} of timesteps show "
      "degeneracy (N_eff < N/10)"
    )
  lines.append("")
  return "\n".join(lines)


_SCORE_PRECISION = 3


def _fmt_summary_line(label: str, s: dict, unit: str) -> list[str]:
  """Three-line TUM-style summary block for a single prequential score.

  Bottom line reports the Politis–White mean block length used by the
  stationary bootstrap; values much greater than 1 indicate strong
  temporal dependence in the score series.
  """
  p = _SCORE_PRECISION
  n = int(s.get("n", 0))
  ci_lvl = float(s.get("ci_level", 0.95))
  pct = int(round(ci_lvl * 100))
  unit_str = f" {unit}" if unit else ""
  head = (
    f"  {label + ':':<26}  mean {_fmt_float(s.get('mean', float('nan')), p)}{unit_str}"
    f"   [{pct}% CI {_fmt_float(s.get('ci_low', float('nan')), p)},"
    f" {_fmt_float(s.get('ci_high', float('nan')), p)}]   (n={n})"
  )
  body = (
    f"                              median {_fmt_float(s.get('median', float('nan')), p)},"
    f" std {_fmt_float(s.get('std', float('nan')), p)},"
    f" min {_fmt_float(s.get('min', float('nan')), p)},"
    f" max {_fmt_float(s.get('max', float('nan')), p)}"
  )
  ell = s.get("block_length", float("nan"))
  diag = (
    f"                              block length (Politis–White):"
    f" {_fmt_float(ell, 1)}"
  )
  return [head, body, diag]


def _render_scores(rep: Report) -> str:
  lines = ["Scores"]
  for key, s in rep.scores.items():
    label, unit = SCORE_LABELS.get(key, (key, ""))
    lines.extend(_fmt_summary_line(label, s, unit))
  lines.append("")
  return "\n".join(lines)


def _render_calibration(rep: Report) -> str:
  c = rep.calibration
  if not c:
    return "Calibration\n  (skipped)\n"
  lines = ["Calibration"]
  p_cov = c.get("coverage_p", float("nan"))
  suffix = (
    "  [warning] possible miscalibration"
    if not math.isnan(p_cov) and p_cov < 0.05
    else ""
  )
  lines.append(
    f"  {int(c.get('nominal_coverage', 0.9) * 100)}% Mahalanobis coverage:"
    f"  {_fmt_pct(c.get('coverage', float('nan')))}     "
    f"(nominal {_fmt_pct(c.get('nominal_coverage', float('nan')))})"
  )
  lines.append(
    f"  Coverage test (binomial): p = {_fmt_float(p_cov, 3)}"
    f"  (n={_fmt_int(c.get('n_coverage'))}){suffix}"
  )
  z_mean = c.get("z_translation_mean", float("nan"))
  z_std = c.get("z_translation_std", float("nan"))
  if not math.isnan(z_mean):
    comment = ""
    if not math.isnan(z_std):
      if z_std > 1.1:
        comment = "   (over-confident)"
      elif z_std < 0.9:
        comment = "   (under-confident)"
    lines.append(
      f"  Translation z-score:    mean {_fmt_float(z_mean, 2)},"
      f" std {_fmt_float(z_std, 2)}{comment}"
    )
  lines.append("")
  return "\n".join(lines)


def _render_calibration_split(rep: Report) -> str:
  split = rep.calibration_split or {}
  lines = ["Log-score calibration / sharpness split (ANEES χ² verdict)"]
  absolute = split.get("absolute") or {}
  for name in ("translation",):
    s = absolute.get(name)
    if not s:
      continue
    lines.append(
      f"  {name:<12} dof {s['dof']}  ANEES {_fmt_float(s['anees'], 3)}"
      f" (median {_fmt_float(s.get('nees_median', float('nan')), 3)})"
      f"  χ²[{_fmt_float(s['lo'], 2)}, {_fmt_float(s['hi'], 2)}]"
      f"  → {s['verdict']}"
    )
  windowed = split.get("windowed") or []
  if windowed:
    lines.append("  windowed (horizon, dof-3, one-sided):")
    for w in windowed:
      lines.append(
        f"    Δt {w['window_s']:>6g}s  ANEES {_fmt_float(w['anees'], 3)}"
        f"  → {w['verdict']}"
      )
  lines.append("")
  return "\n".join(lines)


def _render_diagnoses(rep: Report) -> str:
  lines = ["Diagnoses (attribution → action)"]
  for d in rep.diagnoses:
    mode = getattr(d.mode, "value", d.mode)
    sev = getattr(d.severity, "value", d.severity)
    lines.append(f"  [{sev}] {mode}")
    lines.append(f"      {d.explanation}")
    for sig in d.signals_triggered:
      lines.append(f"      · {sig}")
    for act in d.recommended_actions:
      lines.append(f"      → {act}")
  lines.append("")
  return "\n".join(lines)


def _render_recommendations(rep: Report) -> str:
  if not rep.recommendations:
    return "Recommendations\n  (none)\n"
  lines = ["Recommendations"]
  for r in rep.recommendations:
    lines.append(f"  - {r}")
  lines.append("")
  return "\n".join(lines)
