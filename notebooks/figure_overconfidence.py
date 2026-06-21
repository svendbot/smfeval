#!/usr/bin/env python3
r"""README hero figure: a calibrated mean, an overconfident belief.

Self-contained reproduction from the shipped ``notebooks/data`` (FAST-LIO2 on
Oxford Spires ``christ-church-03``). One macro panel + two nested zooms:

  macro : the estimate track coloured by per-pose translation NEES (the whole
          run sits far above the calibrated 2.37, so it is not a local glitch),
          with the observer track underneath.
  zoom A: the estimate and observer pose at one representative step; the gap
          between them is z sigma.
  zoom B: the reported 90% region blown up until its shape is visible -- it is
          millimetres across, and the truth is z sigma away, off-frame.

The point: ATE/RPE see only the small mean gap (zoom A's "cm"); smfeval sees
that the belief calls that gap z sigma -- the covariance is the lie.

Run: ``python notebooks/figure_overconfidence.py`` (numpy, scipy, matplotlib,
smfeval). Sourced from public data.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from smfeval.align import (
  align_mode_for_gauge,
  apply_body_transform,
  fit_alignment,
  propagate_step,
)
from smfeval.io import load_square, load_tum
from smfeval.scoring import gaussian_log_score_components
from smfeval.se3 import quat_xyzw_to_rot, trans_slice
from smfeval.se3.lie import homogeneous
from smfeval.sync import match_timestamps

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "docs" / "img" / "overconfidence"

C_REF = "#111111"
C_EST = "#1E66F5"
C_ERR = "#E5322D"
CALIBRATED = float(
  chi2.median(df=3)
)  # 2.366: per-pose NEES under a true belief


def _gunzip(src: Path, dst: Path) -> None:
  with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)


def load_run(tmp: Path):
  """Aligned estimate steps + matched reference, scored per pose (NEES, world error)."""
  _gunzip(DATA / "christ-church-03_fast_lio2.SQUARE.gz", tmp / "est.SQUARE")
  _gunzip(DATA / "christ-church-03_ref.tum.gz", tmp / "ref.tum")
  header, est = load_square(tmp / "est.SQUARE")
  _, ref = load_tum(tmp / "ref.tum", pose_frame="world", body_frame="lidar")

  tf = json.loads((DATA / "imu_to_lidar.json").read_text())
  T_off = homogeneous(np.array(tf["R"]).reshape(3, 3), np.array(tf["t"]))
  order = header.tangent_order
  est = [
    apply_body_transform(
      s,
      T_off,
      tangent_convention=header.tangent_convention,
      tangent_order=order,
    )
    for s in est
  ]

  m = match_timestamps(
    np.array([s.timestamp for s in est]),
    np.array([s.timestamp for s in ref]),
    t_max_diff=0.01,
  )
  matched = [est[i] for i in m.est_indices]
  ref_t = np.array([ref[j].translation for j in m.ref_indices])
  ref_q = np.array([ref[j].quat_xyzw for j in m.ref_indices])

  # align under the file's declared gauge (se3 here): full Umeyama removes every
  # rigid freedom, so the residual is as small as a mean-only metric can make it.
  # Whatever NEES survives that is the belief's fault, not misalignment.
  fit = fit_alignment(
    np.array([s.translation for s in matched]),
    ref_t,
    mode=align_mode_for_gauge(header.gauge),
  )
  aligned = [
    propagate_step(
      s,
      fit.transform,
      scale=fit.scale,
      tangent_convention=header.tangent_convention,
      tangent_order=order,
    )
    for s in matched
  ]

  t_idx = trans_slice(order)
  est_xyz = np.array([s.translation for s in aligned])
  nees = np.array(
    [
      gaussian_log_score_components(s, t, q, order).translation.nees
      for s, t, q in zip(aligned, ref_t, ref_q, strict=True)
    ]
  )
  cov_t = np.array([s.covariance[t_idx, t_idx] for s in aligned])  # body block
  rot = np.array([quat_xyzw_to_rot(s.quat_xyzw) for s in aligned])
  return est_xyz, ref_t, nees, cov_t, rot


def planar_ellipse(cov2: np.ndarray, conf: float):
  """(width, height, angle_deg) of the conf-region ellipse for a 2x2 cov."""
  w, V = np.linalg.eigh(cov2)
  w = np.clip(w, 0.0, None)
  s = chi2.ppf(conf, df=2)
  order = np.argsort(w)[::-1]
  w, V = w[order], V[:, order]
  ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
  return 2 * np.sqrt(s * w[0]), 2 * np.sqrt(s * w[1]), ang


def scale_bar(ax, half: float):
  """Round-number scale bar in the lower-left of a zoom axes (m / cm / mm)."""
  for L in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
    if L < 1.4 * half:
      break
  x0, x1 = ax.get_xlim()
  y0, y1 = ax.get_ylim()
  x = x0 + 0.08 * (x1 - x0)
  y = y0 + 0.10 * (y1 - y0)
  ax.plot([x, x + L], [y, y], color="0.2", lw=2.5, solid_capstyle="butt")
  lab = (
    f"{L:.0f} m"
    if L >= 1
    else f"{L * 100:.0f} cm"
    if L >= 0.01
    else f"{L * 1000:.0f} mm"
  )
  ax.text(
    x + L / 2,
    y + 0.03 * (y1 - y0),
    lab,
    ha="center",
    va="bottom",
    fontsize=9.5,
    color="0.2",
  )


def pick_pose(est_xyz, ref_t, nees, conf=0.90, min_err=0.02, mid=(0.1, 0.9)):
  """Pick a representative overconfident pose.

  Mid-run, truth outside the conf region, planar error >= min_err, NEES
  nearest the run median (not cherry-picked).
  """
  n = len(nees)
  floor = chi2.ppf(conf, df=3)
  err = np.linalg.norm((ref_t - est_xyz)[:, :2], axis=1)
  keep = np.isfinite(nees)
  keep[: int(mid[0] * n)] = False
  keep[int(mid[1] * n) :] = False
  keep &= (err >= min_err) & (nees > floor)
  idx = np.where(keep)[0]
  return int(idx[np.argmin(np.abs(nees[idx] - np.median(nees[keep])))])


def main() -> int:
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    est_xyz, ref_t, nees, cov_t, rot = load_run(Path(td))

  k = pick_pose(est_xyz, ref_t, nees)
  p_est, p_ref = est_xyz[k, :2], ref_t[k, :2]
  e_w = p_ref - p_est
  err_m = float(np.linalg.norm(e_w))
  z = float(np.sqrt(nees[k]))
  # plain-language gloss of z: how far past the 99% credible radius the truth
  # sits (the 99% radius for 3 dof is sqrt(chi2.ppf(0.99, 3)) ~= 3.37)
  mult99 = z / float(np.sqrt(chi2.ppf(0.99, df=3)))
  cov_w = rot[k] @ cov_t[k] @ rot[k].T
  wmaj, wmin, ang = planar_ellipse(cov_w[:2, :2], 0.90)
  rmse = float(np.sqrt(np.mean(np.sum((ref_t - est_xyz) ** 2, axis=1))))
  print(
    f"n={len(nees)} median NEES={np.median(nees):.1f} (z={np.sqrt(np.median(nees)):.0f})  "
    f"APE RMSE={rmse * 100:.1f} cm"
  )
  print(
    f"pose idx={k} NEES={nees[k]:.0f} z={z:.0f} err={err_m * 100:.1f} cm 90%major={wmaj * 1000:.1f} mm"
  )

  plt.rcParams.update(
    {
      "font.size": 12,
      "axes.labelsize": 12,
      "xtick.labelsize": 11,
      "ytick.labelsize": 11,
    }
  )
  fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")

  # macro: observer track + estimate track coloured by per-pose NEES
  ax.plot(ref_t[:, 0], ref_t[:, 1], "-", color=C_REF, lw=1.2, zorder=1)
  pts = est_xyz[:, :2]
  segs = np.stack([pts[:-1], pts[1:]], axis=1)
  cval = 0.5 * (nees[:-1] + nees[1:])
  vmax = float(np.percentile(cval, 98))
  lc = LineCollection(
    segs, cmap="plasma", norm=LogNorm(vmin=CALIBRATED, vmax=vmax), zorder=2
  )
  lc.set_array(cval)
  lc.set_linewidth(2.0)
  ax.add_collection(lc)
  cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02, extend="max")
  cb.set_label("per-pose translation NEES", fontsize=11)
  ticks = [t for t in (CALIBRATED, 1e2, 1e3, 1e4, 1e5) if t <= vmax]
  cb.set_ticks(ticks)
  cb.ax.set_yticklabels(
    [
      "2.37" if t == CALIBRATED else f"$10^{{{int(np.log10(t))}}}$"
      for t in ticks
    ]
  )
  ax.set_aspect("equal")
  ax.autoscale_view()
  ax.set_xlabel("x [m]")
  ax.set_ylabel("y [m]")
  ax.legend(
    handles=[
      Line2D([0], [0], color=C_REF, lw=1.2, label="observer trajectory"),
      Line2D([0], [0], color=C_EST, lw=2.0, label="FAST-LIO2 estimator"),
    ],
    loc="upper left",
    frameon=False,
    fontsize=11,
  )
  # one stacked box (the track is tall and narrow, so left+right boxes collide):
  # what a mean-only metric reports, then what a calibrated belief should read.
  ax.text(
    0.02,
    0.02,
    f"trajectory APE RMSE = {rmse * 100:.1f} cm\n"
    r"calibrated $\Rightarrow$ NEES = 2.37",
    transform=ax.transAxes,
    fontsize=11,
    va="bottom",
    linespacing=1.6,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
  )

  W = 30
  sl = slice(max(0, k - W), min(len(est_xyz), k + W + 1))
  segE, segG = est_xyz[sl], ref_t[sl]

  # zoom A: the two tracks + the gap
  c = 0.5 * (p_est + p_ref)
  halfA = max(err_m, wmaj) * 1.5
  insA = ax.inset_axes([0.17, 0.28, 0.34, 0.34])
  insA.plot(segG[:, 0], segG[:, 1], "-", color=C_REF, lw=2.2, zorder=2)
  insA.plot(segE[:, 0], segE[:, 1], "-", color=C_EST, lw=2.2, zorder=2)
  insA.plot(*p_est, "o", color=C_EST, ms=6, mec="white", mew=0.8, zorder=4)
  insA.plot(*p_ref, "s", color=C_REF, ms=7, mec="white", mew=0.8, zorder=4)
  insA.annotate(
    "",
    xy=p_ref,
    xytext=p_est,
    arrowprops=dict(arrowstyle="-|>", color=C_ERR, lw=2.0),
    zorder=5,
  )
  insA.text(
    0.04,
    0.96,
    f"{err_m * 100:.0f} cm = {z:.0f}$\\sigma$\n({mult99:.0f}x outside 99%)",
    transform=insA.transAxes,
    color=C_ERR,
    fontsize=11,
    fontweight="bold",
    va="top",
    ha="left",
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
  )
  insA.set_xlim(c[0] - halfA, c[0] + halfA)
  insA.set_ylim(c[1] - halfA, c[1] + halfA)
  insA.set_aspect("equal")
  insA.set_xticks([])
  insA.set_yticks([])
  scale_bar(insA, halfA)
  insA.set_title("estimate vs true track", fontsize=10)
  ax.indicate_inset_zoom(insA, edgecolor="0.4", lw=1.0, alpha=0.9)

  # zoom B: the reported 90% region, to scale
  halfB = wmaj * 1.8
  insB = ax.inset_axes([0.60, 0.58, 0.28, 0.28])
  insB.plot(segE[:, 0], segE[:, 1], "-", color=C_EST, lw=2.0, zorder=2)
  insB.add_patch(
    Ellipse(
      p_est,
      wmaj,
      wmin,
      angle=ang,
      fill=True,
      alpha=0.45,
      facecolor=C_EST,
      edgecolor=C_EST,
      lw=1.8,
      zorder=3,
    )
  )
  insB.plot(*p_est, "o", color=C_EST, ms=6, mec="white", mew=0.8, zorder=4)
  u = e_w / (err_m + 1e-12)
  insB.annotate(
    f"true pose\n{z:.0f}$\\sigma$ this way",
    xy=p_est + u * halfB * 0.95,
    xytext=p_est + u * halfB * 0.30,
    color=C_ERR,
    fontsize=9.5,
    fontweight="bold",
    ha="center",
    va="center",
    arrowprops=dict(arrowstyle="-|>", color=C_ERR, lw=1.8),
    zorder=5,
  )
  insB.set_xlim(p_est[0] - halfB, p_est[0] + halfB)
  insB.set_ylim(p_est[1] - halfB, p_est[1] + halfB)
  insB.set_aspect("equal")
  insB.set_xticks([])
  insB.set_yticks([])
  scale_bar(insB, halfB)
  insB.set_title("reported 90% region", fontsize=10)
  insA.indicate_inset(
    [p_est[0] - halfB, p_est[1] - halfB, 2 * halfB, 2 * halfB],
    insB,
    edgecolor="0.4",
    lw=1.0,
    alpha=0.9,
  )

  OUT.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(str(OUT) + ".png", dpi=200, bbox_inches="tight")
  print(f"wrote {OUT}.png")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
