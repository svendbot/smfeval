# %% [markdown]
# # smfeval — score the belief, not just the mean
#
# This notebook reproduces the headline verdict on one Oxford Spires
# sequence (`2024-03-18-christ-church-03`): FAST-LIO2's per-scan
# covariance vs its realised error — median NEES, the covariance scale
# gap *k*, and 90% coverage.
#
# Runtime: a few seconds on a free Colab instance. The only smfeval
# dependencies are numpy and scipy.

# %%
# %pip install -q "smfeval>=0.4"

# %%
import gzip
import shutil
import urllib.request
from pathlib import Path

BASE = (
  "https://raw.githubusercontent.com/svendbot/smfeval/main/notebooks/data"
)
FILES = {
  "est.SQUARE": "christ-church-03_fast_lio2.SQUARE.gz",
  "gt.tum": "christ-church-03_gt.tum.gz",
  "imu_to_lidar.json": "imu_to_lidar.json",
}

for local, remote in FILES.items():
  if Path(local).exists():
    continue
  raw, _ = urllib.request.urlretrieve(f"{BASE}/{remote}")
  if remote.endswith(".gz"):
    with gzip.open(raw, "rb") as src, open(local, "wb") as dst:
      shutil.copyfileobj(src, dst)
  else:
    shutil.copy(raw, local)
print("data ready:", [f for f in FILES])

# %% [markdown]
# ## The thirty-second verdict
#
# One command. The estimate declares `BODY_FRAME imu`; the Spires GT is in
# the LiDAR frame, so the filter's own extrinsic is passed along.

# %%
# !smfeval nees est.SQUARE gt.tum --gt-body-frame lidar --body-frame-transform imu_to_lidar.json

# %% [markdown]
# Median NEES in the millions against a calibrated reference of 2.37: the
# published covariance is several orders of magnitude too tight, and the
# 90% credible ellipsoid never contains the truth. The belief is wrong
# even where the trajectory is accurate — this is what mean-based metrics
# (ATE/RPE) cannot see.
#
# ## The same numbers through the library API

# %%
import json

import numpy as np

from smfeval.align import (
  align_mode_for_gauge,
  apply_body_transform,
  fit_alignment,
  propagate_step,
)
from smfeval.io import load_square, load_tum
from smfeval.report.verdict import nees_verdict, render_nees_verdict
from smfeval.scoring import gaussian_log_score_components
from smfeval.se3.lie import homogeneous
from smfeval.sync import match_timestamps

est_header, est_steps = load_square(Path("est.SQUARE"))
_, gt_steps = load_tum(Path("gt.tum"), pose_frame="world", body_frame="lidar")

# re-express the estimate in the GT's (LiDAR) body frame
tf = json.loads(Path("imu_to_lidar.json").read_text())
T_off = homogeneous(np.array(tf["R"]).reshape(3, 3), np.array(tf["t"]))
order = est_header.tangent_order
est_steps = [
  apply_body_transform(
    s,
    T_off,
    tangent_convention=est_header.tangent_convention,
    tangent_order=order,
  )
  for s in est_steps
]

# match timestamps, fit the gauge-implied alignment, score
est_ts = np.array([s.timestamp for s in est_steps])
gt_ts = np.array([s.timestamp for s in gt_steps])
m = match_timestamps(est_ts, gt_ts, t_max_diff=0.01)
matched = [est_steps[i] for i in m.est_indices]
gt_t = np.array([gt_steps[j].translation for j in m.gt_indices])
gt_q = np.array([gt_steps[j].quat_xyzw for j in m.gt_indices])

fit = fit_alignment(
  np.array([s.translation for s in matched]),
  gt_t,
  mode=align_mode_for_gauge(est_header.gauge),
)
aligned = [
  propagate_step(
    s,
    fit.transform,
    scale=fit.scale,
    tangent_convention=est_header.tangent_convention,
    tangent_order=order,
  )
  for s in matched
]

nees = np.array(
  [
    gaussian_log_score_components(s, t, q, order).translation.nees
    for s, t, q in zip(aligned, gt_t, gt_q)
  ]
)
verdict = nees_verdict(nees, dof=3)
print(render_nees_verdict(verdict))

# %% [markdown]
# ## Where the NEES mass sits vs where it should
#
# Under a calibrated belief the per-pose translation NEES is
# $\chi^2_3$-distributed. Plotting the realised distribution against that
# reference shows the gap is not a tail effect — the entire bulk is
# displaced by orders of magnitude.

# %%
import matplotlib.pyplot as plt
from scipy.stats import chi2

finite = nees[np.isfinite(nees) & (nees > 0)]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.logspace(-2, np.log10(finite.max()), 80)
ax.hist(finite, bins=bins, density=True, alpha=0.6, label="realised NEES")
x = np.logspace(-2, 2, 400)
ax.plot(x, chi2.pdf(x, df=3) * x * np.log(10), "k--", label=r"$\chi^2_3$ (calibrated)")
ax.axvline(verdict.median_nees, color="C3", label=f"median = {verdict.median_nees:.3g}")
ax.axvline(2.366, color="k", lw=0.8, label="calibrated median = 2.37")
ax.set_xscale("log")
ax.set_xlabel("translation NEES (log scale)")
ax.set_ylabel("density")
ax.set_title(
  f"FAST-LIO2 on christ-church-03: k = {verdict.k:.3g} "
  f"(~{verdict.per_axis_factor:.0f}x too tight per axis)"
)
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## No ground truth? Score two filters against each other
#
# `smfeval pair a.SQUARE b.SQUARE` aligns filter A to filter B (the
# reference is never consulted) and scores the difference under the summed
# covariances — an elevated pairwise NEES certifies overconfidence with no
# ground truth at all, and it is a *lower bound* on the miscalibration.
# See the README for details.
