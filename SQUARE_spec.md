# SQUARE-Format_Specification

**Version:** SQUARE/0.3

The spec version is the SQUARE format version, single-sourced from
`smfeval.format.FORMAT_VERSION` and checked in CI. It changes only when the
byte format changes, not with the smfeval release version.

## File Format

A text format for probabilistic SLAM output, extending TUM. Each file stores an
algorithm's belief at each timestep. Reference, scoring choices, and derived
quantities are not stored.

### Header

Lines prefixed with `#%` declare metadata. Required fields depend on
representation.

Common to all representations:

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

`BODY_FRAME` names the rigid body whose pose is reported (`imu`, `lidar`,
`base_link`). It is a property of the publisher, not the dataset. FAST-LIO2's
`state_ikfom` is in the IMU frame and declares `BODY_FRAME imu`; Oxford Spires
GT is in the LiDAR frame and declares `BODY_FRAME lidar`. Names are free-form
strings matched by exact equality.

When estimate and GT body frames differ, the scoring tool requires
`--body-frame-transform PATH`, a JSON object `{"R": [9 floats row-major], "t":
[3 floats]}`. `T_off` is the pose of the GT body frame in the estimate body
frame (ROS `target_T_source` convention). `R` maps a GT-body vector to
estimate-body coordinates and `t` is the GT-body origin in estimate-body
coordinates. It is applied by right-multiplication, `T_world_gt_body =
T_world_est_body · T_off`. For `right_perturbation` covariances,
`Σ ← Ad_{T_off^{-1}} · Σ · Ad_{T_off^{-1}}^⊤`; for `left_perturbation`, `Σ` is
unchanged.

Gaussian-specific:

```
#%TANGENT_CONVENTION <right_perturbation | left_perturbation>
#%TANGENT_ORDER <translation_rotation | rotation_translation>
#%ROTATION_PARAM axis_angle
```

Ensemble-specific:

```
#%WEIGHTED <true | false>
#%WEIGHT_FORMAT <linear | log>
#%WEIGHTS_NORMALIZED <true | false>
#%PARTICLE_COUNT_HINT <N>   (optional)
```

### Frame and Gauge

`POSE_FRAME world` is a label, not a frame guarantee. `GAUGE` declares which
degrees of freedom the algorithm pinned at initialization and which it left
free. It is intrinsic to the algorithm and independent of reference. The
scoring tool reads `GAUGE` to pick the default `--align` mode and applies the
transform to means and covariances (`Σ ↦ Ad_T Σ Ad_T^⊤` for Gaussian,
particle-wise for ensembles).

| `GAUGE` | Pinned at initialization | Free DoF | Default alignment |
|---|---|---|---|
| `fixed` | absolute world frame (GPS, prior map, AR anchor) | 0 | none |
| `se3` | first pose at identity, no IMU | 6 | SE(3) Umeyama |
| `gravity_yaw` | gravity (pitch, roll) via IMU | 4 (xyz + yaw) | yaw + translation |
| `sim3` | first pose at identity, scale unknown | 7 (6 + scale) | Sim(3) Umeyama |

### Row Formats

`gaussian_se3`, one row per timestep:

```
timestamp tx ty tz qx qy qz qw L00 L10 L11 L20 L21 L22 L30 L31 L32 L33 L40 L41 L42 L43 L44 L50 L51 L52 L53 L54 L55
```

29 columns. Timestamp, the TUM-compatible mean pose (first 8 columns), then the
21 lower-triangular entries of the 6×6 tangent covariance in row-major order.

`ensemble_se3`, one row per particle, grouped by timestep:

```
timestamp particle_id [weight] tx ty tz qx qy qz qw
```

The weight column is present iff `WEIGHTED true`. Particles are contiguous
within a timestep and timestamps are monotonically non-decreasing.
`particle_id` is a within-timestep row index with no cross-timestep meaning.

`deterministic` is identical to TUM, so existing reference files are valid
input.

### Structural Rules

- Timestamps have at least 6 decimal places. Parsers group ensemble rows by
  string equality.
- Writers flush at timestep boundaries. Readers tolerate a truncated final
  timestep (append-safe).
- Ensemble size may vary across timesteps (KLD-sampling, dynamic resampling).
- Weights are stored in the filter's native form (linear or log, normalized or
  not). Normalization is a scoring-time derivation.
- Gaussian covariances are the 6×6 pose marginal in the declared tangent
  convention. Writers convert from algorithm-native conventions at write time.

### Design Principle

Store raw belief, derive everything else. Scoring rules, $N_\text{eff}$,
unique-particle counts, PIT values, and aggregate statistics are reconstructible
from the file plus reference. New scoring rules work on existing files
without reprocessing.

## Escape hatch (TUM poses with covariance, no header)

A header-less file can be scored two ways. Both reuse the SQUARE covariance
packing, the row-major lower triangle of the 6×6 tangent covariance
(`c11 c21 c22 c31 ... c66`, 21 entries).

(a) Wide TUM, 29 columns. TUM pose columns followed by the 21 covariance
entries:

```
timestamp x y z qx qy qz qw c11 c21 c22 c31 ... c66
```

This is byte-identical to a SQUARE `gaussian_se3` body without the header.

(b) Sidecar file (`--cov`). Plain 8-column TUM poses plus a file with rows:

```
timestamp c11 c21 c22 c31 ... c66
```

22 fields. `#` comments are allowed. The sidecar must match the pose file 1:1 by
timestamp (tolerance 1e-6 s). There is no nearest-neighbour fallback, because
covariance attachment is identity-critical. Optional `#%TANGENT_CONVENTION` and
`#%TANGENT_ORDER` lines in the sidecar override the CLI defaults.

Missing header metadata comes from flags: `--est-body-frame` (required),
`--est-pose-frame` (default `world`), `--tangent-convention` (default
`right_perturbation`), `--tangent-order` (default `translation_rotation`),
`--gauge` (default `se3`). The assumed values are echoed on stderr.

```
$ smfeval nees est.tum gt.tum --cov est.cov --est-body-frame imu --ref-body-frame imu
note: bare-TUM estimate read as gaussian_se3 (body_frame='imu', ...)
median NEES 2.31   (calibrated: 2.37)
...
```

The `pair` verb requires SQUARE inputs, since it needs both filters' declared
conventions to sum covariances.
