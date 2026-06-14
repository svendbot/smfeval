# SQUARE v0.3 — Format Specification

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
