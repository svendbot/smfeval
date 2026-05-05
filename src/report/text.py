"""Render the structured report into the plain-text layout from the spec."""

import math

from src.report.builder import Report


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
    out.append(_render_recommendations(rep))
    return "\n".join(out)


def _render_sync(rep: Report) -> str:
    s = rep.sync
    gaps = s.get("gap_quantiles_ms", {})
    risk_q = s.get("risk_quantiles")
    lines = [
        "Synchronization",
        f"  Pairs matched:          {_fmt_int(s.get('n_matched'))} / {_fmt_int(s.get('n_total'))}",
        f"  Dropped:                {_fmt_int(s.get('n_dropped'))}",
        f"  Timestamp gap (ms):     median {_fmt_float(gaps.get('median', float('nan')), 2)},"
        f" p95 {_fmt_float(gaps.get('p95', float('nan')), 2)},"
        f" p99 {_fmt_float(gaps.get('p99', float('nan')), 2)}",
    ]
    if risk_q is not None:
        lines.append(
            f"  Sync risk (v·Δt / σ):   median {_fmt_float(risk_q['median'], 2)},"
            f" p95 {_fmt_float(risk_q['p95'], 2)},"
            f" p99 {_fmt_float(risk_q['p99'], 2)}"
        )
        excess = s.get("risk_excess_count", 0)
        n = s.get("n_matched", 0) or 0
        if excess:
            frac = 100.0 * excess / n if n else 0.0
            lines.append(
                f"                          ⚠ {excess} pairs ({frac:.1f}%) "
                f"exceed risk {s.get('risk_threshold', 0.3):.1f}"
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
            f"                          ⚠ {_fmt_pct(deg)} of timesteps show "
            "degeneracy (N_eff < N/10)"
        )
    lines.append("")
    return "\n".join(lines)


def _render_scores(rep: Report) -> str:
    sc = rep.scores
    lines = ["Scores"]
    if "translation_crps" in sc:
        lines.append(f"  Translation CRPS:       {_fmt_float(sc['translation_crps'])} m")
    if "rotation_crps" in sc:
        lines.append(f"  Rotation CRPS:          {_fmt_float(sc['rotation_crps'])} rad")
    if "energy_score" in sc:
        lines.append(f"  Energy score (SE(3)):   {_fmt_float(sc['energy_score'])}")
    if "log_score" in sc:
        lines.append(f"  Log score:              {_fmt_float(sc['log_score'], 2)}")
    if "interval_score" in sc:
        lines.append(f"  Interval score:         {_fmt_float(sc['interval_score'])}")
    lines.append("")
    return "\n".join(lines)


def _render_calibration(rep: Report) -> str:
    c = rep.calibration
    if not c:
        return "Calibration\n  (skipped)\n"
    lines = ["Calibration"]
    ks_t = c.get("ks_p_translation", float("nan"))
    suffix = "  ⚠ possible miscalibration" if not math.isnan(ks_t) and ks_t < 0.05 else ""
    lines.append(f"  PIT uniformity (KS):    p = {_fmt_float(ks_t, 3)}{suffix}")
    lines.append(
        f"  {int(c.get('nominal_coverage', 0.9) * 100)}% interval coverage:"
        f"  {_fmt_pct(c.get('coverage', float('nan')))}     "
        f"(nominal {_fmt_pct(c.get('nominal_coverage', float('nan')))})"
    )
    z_mean = c.get("z_translation_mean", float("nan"))
    z_std = c.get("z_translation_std", float("nan"))
    if not math.isnan(z_mean):
        comment = ""
        if not math.isnan(z_std):
            if z_std > 1.1:
                comment = "   (slightly under-confident)"
            elif z_std < 0.9:
                comment = "   (slightly over-confident)"
        lines.append(
            f"  Translation z-score:    mean {_fmt_float(z_mean, 2)},"
            f" std {_fmt_float(z_std, 2)}{comment}"
        )
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
