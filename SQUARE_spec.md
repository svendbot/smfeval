# SQUARE v0.3 — Format and Scoring Report Summary

## File Format

A text-based format for storing probabilistic SLAM output, designed as a principled extension of TUM. Each file stores an algorithm's native belief representation at each timestep; ground truth, scoring choices, and derived quantities stay out of the format.

### Header

Comment lines prefixed with `#%` declare metadata. Required fields depend on representation.

**Common to all representations:**

```
#%FORMAT SQUARE/0.3
#%REPRESENTATION <gaussian_se3 | ensemble_se3 | deterministic>
#%POSE_FRAME world
#%BODY_FRAME <name>
#%GAUGE <fixed | se3 | gravity_yaw | sim3>
#%TIMESTAMP_UNIT seconds
#%ALGORITHM <name>
#%ALGORITHM_VERSION <version>
```

`BODY_FRAME` names the rigid body whose pose is reported (e.g. `imu`, `lidar`, `base_link`). It is a property of the *publisher*, not the dataset: FAST-LIO2's `state_ikfom` is in the IMU frame, so its SQUARE files declare `BODY_FRAME imu`; the Oxford Spires GT lives in the LiDAR frame and declares `BODY_FRAME lidar`. Names are free-form strings; matching is by exact string equality.

When estimate and ground-truth body frames differ, the scoring tool requires `--body-frame-transform PATH` (JSON: `{"R": [9 floats row-major], "t": [3 floats]}`). The transform is `T_est_body__gt_body` in standard ROS `target_T_source` convention: `R` maps a vector in GT-body coords to estimate-body coords, and `t` is the GT-body origin expressed in estimate-body coords. Equivalently, it is the pose of the GT body frame as seen from the estimate body frame. Applied to the estimate by right-multiplication: `T_world_gt_body = T_world_est_body · T_off`. For `right_perturbation` Gaussian covariances, `Σ ← Ad_{T_off^{-1}} · Σ · Ad_{T_off^{-1}}^⊤`; for `left_perturbation`, `Σ` is unchanged.

**Convention trap (FAST-LIO).** FAST-LIO's `extrinsic_R`, `extrinsic_T` are consumed in `IMU_Processing.hpp` as `Lidar_R_wrt_IMU` and `Lidar_T_wrt_IMU` — these are `T_lidar_imu` (the rotation maps IMU-frame vectors to LiDAR-frame). For an IMU-publishing FAST-LIO file scored against LiDAR-frame GT, the `--body-frame-transform` is the **inverse**: `R = extrinsic_R^⊤`, `t = -extrinsic_R^⊤ · extrinsic_T`.

**Gaussian-specific:**

```
#%TANGENT_CONVENTION <right_perturbation | left_perturbation>
#%TANGENT_ORDER <translation_rotation | rotation_translation>
#%ROTATION_PARAM axis_angle
```

**Ensemble-specific:**

```
#%WEIGHTED <true | false>
#%WEIGHT_FORMAT <linear | log>
#%WEIGHTS_NORMALIZED <true | false>
#%PARTICLE_COUNT_HINT <N>   (optional)
```

### Frame and Gauge

`POSE_FRAME world` is a label, not a guarantee — two algorithms that both write `world` rarely live in the same frame. Rather than try to express absolute frames, smfeval declares the algorithm's **gauge**: which degrees of freedom were pinned by initialization vs. left free for the data to determine. This is an intrinsic property of the algorithm, knowable without reference to ground truth.

| `GAUGE` | Pinned by initialization | Free DoF | Default alignment |
|---|---|---|---|
| `fixed` | absolute world frame (GPS, prior map, AR anchor) | 0 | none |
| `se3` | first pose at identity, no IMU | 6 | SE(3) Umeyama |
| `gravity_yaw` | gravity (pitch, roll) via IMU | 4 (xyz + yaw) | yaw + translation |
| `sim3` | first pose at identity, scale unknown | 7 (6 + scale) | Sim(3) Umeyama |

A stereo-inertial system writes `gravity_yaw`; monocular ORB-SLAM writes `sim3`; a lidar system localizing in a prior map writes `fixed`. Whether ground truth happens to live in the matching frame is a property of the dataset, not the algorithm. The scoring tool reads `GAUGE` to pick a sensible default `--align` mode and applies the resulting transform to both means and uncertainty (`Σ ↦ Ad_T Σ Ad_T^⊤` for Gaussian, particle-wise for ensembles).

### Row Formats

**`gaussian_se3`** — one row per timestep:

```
timestamp tx ty tz qx qy qz qw L00 L10 L11 L20 L21 L22 L30 L31 L32 L33 L40 L41 L42 L43 L44 L50 L51 L52 L53 L54 L55
```

29 columns: timestamp, mean pose (TUM-compatible first 8 columns), then 21 lower-triangular entries of the 6×6 tangent-space covariance in row-major order.

**`ensemble_se3`** — one row per particle, grouped by timestep:

```
timestamp particle_id [weight] tx ty tz qx qy qz qw
```

Weight column present iff `WEIGHTED true`. Particles are contiguous within a timestep; timestamps are monotonically non-decreasing. `particle_id` is a within-timestep row index with no cross-timestep meaning.

**`deterministic`** — identical to TUM format, so existing ground-truth files are valid input.

### Structural Rules

- Timestamps printed with ≥6 decimal places; parsers group ensemble rows by string equality.
- Writers flush at timestep boundaries; readers tolerate a truncated final timestep (append-safe).
- Ensemble size may vary across timesteps (supports KLD-sampling, dynamic resampling).
- Weights stored in the filter's native form (linear or log, normalized or not); normalization is a scoring-time derivation.
- Gaussian covariances are the 6×6 pose marginal in a declared tangent-space convention; writers convert from algorithm-native conventions at write time.

### Design Principle

Store raw belief; derive everything else. Scoring rules, $N_\text{eff}$, unique-particle counts, PIT values, aggregate statistics — all reconstructible from the file plus ground truth. New scoring rules or diagnostics added later will work on existing files without reprocessing.

## Scoring Report

Produced by the scoring tool given a SQUARE file and a ground-truth file. Structured in three sections: data quality, scores, and recommendations.

### Synchronization

Two strategies are available via `--sync`:

- **`nearest`** (default): for each estimate timestamp $t_i^\text{est}$, the GT pose at the nearest $t_j^\text{gt}$ is selected; pairs with $|t_i^\text{est} - t_j^\text{gt}| > $ `--t_max_diff` are dropped. An optional `--t_offset` is applied to estimates before matching to correct for known clock skew. Default tolerance matches evo (0.01 s). This is the same strategy APE scoring uses — timestamp pairing is interoperable with evo.

- **`interpolate_gt`**: piecewise Gaussian Process on SE(3) following Zhang & Scaramuzza (2019, §IV.B). For each estimate timestamp, a local window of GT samples is taken; the middle pose is chosen as $T_\text{ref}$; the surrounding GT poses are expressed as $\xi_i = \log(T_\text{ref}^{-1} T_i) \in \mathfrak{se}(3)$; six independent squared-exponential GPs (length scale `--sync_length_scale`, default 0.1 s; window `--sync_window`, default 10 samples) are fit on the components of $\xi$ as functions of time; the predictive $\mu_{\xi^*}$ at the query is mapped back to $T^* = T_\text{ref} \cdot \mathrm{Exp}(\mu_{\xi^*})$. The kernel is shared across the six components, so the predictive covariance is a scalar $v^*$ shared across all diagonal entries, reported per-pair as "GP σ" in the report. Query times outside the GT range are skipped, not extrapolated. Use this as a cross-check when the nearest-neighbor sync risk fires — it removes timestamp slop from the residual.

Sync risk (nearest mode): $r = \lVert v_\text{gt} \rVert \cdot |\Delta t| / \sigma_\text{trans}$ — when $r > 0.3$ for a significant fraction of pairs, the calibration findings could be partly explained by sync error and the recommendation section flags this with a pointer to `--sync=interpolate_gt`.

### Alignment

After synchronization, the estimate is brought into the ground-truth frame using the alignment mode implied by the file's `GAUGE` (overridable via `--align {none, origin, se3, gravity_yaw, sim3}`). A least-squares transform `T` is fit on the matched mean poses and propagated to the full belief:

- `gaussian_se3`: mean ← `T·μ`, covariance ← `Ad_T · Σ · Ad_T^⊤`. For Sim(3), the translation block of the Adjoint carries the scale.
- `ensemble_se3`: each particle ← `T·pᵢ`; weights are invariant under rigid/Sim(3) transforms of the support.
- `deterministic`: pose ← `T·p`.

The fitted transform and the DoF removed are reported in the data-quality section. Alignment uses the same data it scores against, so post-alignment residuals are biased low — strongest on short trajectories and high-DoF gauges. The recommendations section flags this when alignment removes a non-trivial fraction of the trajectory's effective DoF; `--n_to_align` (fit on a prefix, score on the remainder) mitigates the bias for users who care.

### Report Structure

```
=== smfeval scoring report ===

Synchronization
  Pairs matched:          9,847 / 10,000
  Dropped (gap > 10 ms):  153
  Timestamp gap (ms):     median 0.4, p95 1.8, p99 3.2  // Intervals from HDI or CI
  Sync risk (v·Δt / σ):   median 0.02, p95 0.18, p99 0.41
                          ⚠ 127 pairs (1.3%) exceed risk 0.3

Alignment
  Gauge (declared):       gravity_yaw   (pitch/roll pinned by IMU)
  Mode applied:           gravity_yaw   (4 DoF: xyz + yaw)
  Fitted Δyaw:            1.84°
  Fitted Δxyz:            (0.012, -0.003, 0.041) m
  Fit residual (m):       median 0.018, p95 0.072
                          4 DoF removed over 312 m of trajectory

Ensemble diagnostics   (ensemble_se3 only)
  Nominal N:              500
  N_eff from weights:     median 347, p05 14 p01 12
  Unique particles:       median 423, p05 38 p01 30
                          ⚠ 3.2% of timesteps show degeneracy (N_eff < N/10)

Scores
  Translation CRPS:       0.0234 m
  Rotation CRPS:          0.0089 rad
  Energy score (SE(3)):   0.0312
  Log score:              2.41      (Gaussian only; skipped for ensembles)

Calibration
  PIT uniformity (KS):    p = 0.04  ⚠ possible miscalibration
  90% interval coverage:  84.2%     (nominal 90%)
  Translation z-score:    mean 0.02, std 1.18   (slightly under-confident)

Recommendations
  - 1.3% of pairs have sync risk > 0.3; consider cross-checking
    with --sync=interpolate_gt to confirm calibration findings.  (Requires
  - Coverage below nominal combined with KS p < 0.05 suggests the
    filter is mildly under-confident. Miscalibration is unlikely to
    be explained by sync error alone.
```

### Diagnostic Categories

- **Data quality**: sync pairing statistics, dropped pairs, sync risk distribution, alignment mode and fitted transform, ensemble degeneracy. These tell you whether the scores below are trustworthy and on what frame they were computed.
- **Scores**: proper scoring rules — translation CRPS, rotation CRPS (via SO(3) geodesic kernel), joint energy score, log score (Gaussian only), interval score. Translation and rotation reported separately, matching evo's APE/RPE convention.
- **Calibration**: PIT histogram summary, interval coverage vs. nominal, standardized residual statistics. Orthogonal to score magnitude — an algorithm can have good CRPS but poor calibration.
- **Recommendations**: plain-language interpretation. Cross-references between diagnostics (e.g., sync risk × PIT deviation) to distinguish artifacts from real findings.

### What the Report Deliberately Does Not Do

- No automatic "pass/fail" verdict; thresholds are recommendations, not judgments.
- No cross-algorithm comparison in a single report; comparison is a separate tool that consumes multiple reports.
- No modification of the input files; the SQUARE file is the immutable record of the algorithm's output.

## Appendix: escape hatch — TUM poses + covariance without a SQUARE header

SQUARE adoption is not a precondition for getting a verdict. A filter that
prints its covariance anywhere can be scored from header-less files in two
ways; both reuse the SQUARE covariance packing (row-major lower triangle of
the 6×6 tangent covariance: `c11 c21 c22 c31 … c66`, 21 entries).

**(a) Wide TUM (29 columns).** Standard TUM pose columns followed by the 21
covariance entries:

```
timestamp x y z qx qy qz qw c11 c21 c22 c31 ... c66
```

A wide-TUM file is byte-identical to a SQUARE `gaussian_se3` body without
the header.

**(b) Sidecar covariance file (`--cov`).** Plain 8-column TUM poses plus a
separate file with rows

```
timestamp c11 c21 c22 c31 ... c66
```

(22 fields). `#` comments are allowed. The sidecar must match the pose file
1:1 by timestamp (tolerance 1e-6 s) — covariance attachment is
identity-critical, so there is no nearest-neighbour fallback. Optional
header lines `#%TANGENT_CONVENTION <v>` / `#%TANGENT_ORDER <v>` in the
sidecar override the CLI defaults.

Because the header is missing, the metadata it would carry comes from
flags: `--est-body-frame` (required), `--est-pose-frame` (default `world`),
`--tangent-convention` (default `right_perturbation`), `--tangent-order`
(default `translation_rotation`), `--gauge` (default `se3`). The assumed
values are echoed on stderr so a wrong guess is visible:

```
$ smfeval nees est.tum gt.tum --cov est.cov --est-body-frame imu --gt-body-frame imu
note: bare-TUM estimate read as gaussian_se3 (body_frame='imu', ...)
median NEES 2.31   (calibrated: 2.37)
...
```

The `pair` verb requires SQUARE inputs (it needs both filters' declared
conventions to sum covariances safely).

## Summary

The format stores the SLAM algorithm's native probabilistic output — Gaussian mean + covariance for filters, weighted particles for ensemble methods — with headers declaring the conventions and **gauge** needed to interpret them. The gauge tells the scoring tool which DoF the algorithm pinned and which it left free, so the right alignment mode (none, SE(3), gravity-yaw, or Sim(3)) is the default, not a guess. Everything else derived — scores, $N_\text{eff}$, PIT, calibration, the alignment transform itself — is computed at scoring time. Synchronization uses nearest-neighbor matching with a tolerance, matching TUM and evo conventions for interoperability. The scoring report surfaces both the numbers and the caveats needed to trust them, including the bias introduced by alignment, and reports multiple proper scoring rules side by side because no single rule captures the full shape of a predictive distribution.
