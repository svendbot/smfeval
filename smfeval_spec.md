# smfeval v0.2 — Format and Scoring Report Summary

## File Format

A text-based format for storing probabilistic SLAM output, designed as a principled extension of TUM. Each file stores an algorithm's native belief representation at each timestep; ground truth, scoring choices, and derived quantities stay out of the format.

### Header

Comment lines prefixed with `#%` declare metadata. Required fields depend on representation.

**Common to all representations:**

```
#%FORMAT smfeval/0.2
#%REPRESENTATION <gaussian_se3 | ensemble_se3 | deterministic>
#%POSE_FRAME world
#%GAUGE <fixed | se3 | gravity_yaw | sim3>
#%TIMESTAMP_UNIT seconds
#%ALGORITHM <name>
#%ALGORITHM_VERSION <version>
```

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
>> How can the algorithm (Fast-LIO, Point_LIO, ...) know which Gauge to use? This can only be known when the experiement is setup.
>> Is this really never documented. Relying on Kabesh seems crazy.

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
>> TODO(ola): These symbols not to be clearly defined (even if it is obvious!)

```
timestamp tx ty tz qx qy qz qw L00 L10 L11 L20 L21 L22 L30 L31 L32 L33 L40 L41 L42 L43 L44 L50 L51 L52 L53 L54 L55
```

29 columns: timestamp, mean pose (TUM-compatible first 8 columns), then 21 lower-triangular entries of the 6×6 tangent-space covariance in row-major order.

**`ensemble_se3`** — one row per particle, grouped by timestep:

```
timestamp particle_id [weight] tx ty tz qx qy qz qw
```

Weight column present iff `WEIGHTED true`. Particles are contiguous within a timestep; timestamps are monotonically non-decreasing. `particle_id` is a within-timestep row index with no cross-timestep meaning.

>> TODO(ola): is this really necessary. If we know tha the algorithm is not probabilistic then just use existing TUM format? No need to reproduce or handle it in this library.
>> Just give a clear error message with reference to existing infrastructure.
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

Produced by the scoring tool given a smfeval file and a ground-truth file. Structured in three sections: data quality, scores, and recommendations.

### Synchronization

Ground truth is matched to estimates using **nearest-neighbor matching with a tolerance**, following the convention from TUM (Sturm et al.) inherited by evo. For each estimate timestamp $t_i^\text{est}$, the ground truth pose at the nearest $t_j^\text{gt}$ is selected; pairs with $|t_i^\text{est} - t_j^\text{gt}| > $ `--t_max_diff` are dropped. An optional `--t_offset` is applied to estimates before matching to correct for known clock skew. Default tolerance matches evo (0.01 s).

This is the same strategy APE scoring uses, so timestamp pairing is interoperable with evo — a given estimate/ground-truth pair will match identically in both tools.

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
- No modification of the input files; the smfeval file is the immutable record of the algorithm's output.

## Related Work

No existing tool combines proper scoring rules, SE(3)-aware geometry, a standardized probabilistic trajectory format, and evo-style practitioner tooling. The ingredients live in separate literatures.

**Closest methodological precedent.** Zhang and Scaramuzza (2019), *Rethinking Trajectory Evaluation for SLAM: a Probabilistic, Continuous-Time Approach* (arXiv:1906.03996), formulate trajectory evaluation probabilistically and model ground truth as a piecewise Gaussian Process in continuous time, generalizing absolute and relative error to likelihoods. Their framing places uncertainty on the ground-truth side; smfeval places it on the estimate side. The two views are complementary, and their continuous-time ground-truth interpolation is a principled alternative to nearest-neighbor synchronization that smfeval could support.

**Deterministic predecessors.** Zhang and Scaramuzza (2018), *A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry* (IROS 2018), produced the `rpg_trajectory_evaluation` toolbox, which implements ATE and RE for deterministic trajectories. evo (Grupp, github.com/MichaelGrupp/evo) is the de facto community tool in the same space. Both are deterministic; smfeval extends their workflow to probabilistic output.

**Position paper motivating this work.** Sansoni and Tosetti (2026), The SLAM Confidence Trap, argue that the SLAM community has prioritized benchmark scores over principled uncertainty estimation, producing systems that are geometrically accurate but probabilistically inconsistent, and call for a paradigm shift in which real-time uncertainty becomes a primary metric of success. smfeval is a concrete answer to that call.

**Theoretical foundations.** Gneiting and Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation* (JASA 102:359–378), is the canonical reference for the scoring rules used here (CRPS, energy score, log score, interval score). Extensions to manifolds relevant for SE(3) scoring are developed in Mardia et al. (2016) for general Riemannian manifolds and in Takasu et al. (2018) for the sphere. Bonnier and Oberhauser (2024) extend proper scoring rules to trajectories as first-class objects, a direction worth tracking for future versions.

>> I know Mardia through Directional Stats and Thomas Hamelryck. Could be good to speak with.

**Adjacent SLAM work.** PALoc (arXiv:2401.17826) derives factor-graph covariances for ground-truth trajectory generation — the inverse direction of smfeval's concern. NEES (Normalized Estimation Error Squared) has long been used in the filtering community as a consistency check but is rarely reported in modern optimization-based SLAM papers and is subsumed by the calibration section of the smfeval report.

## Why Multiple Scores

Every proper scoring rule captures something real about the predictive distribution but none captures everything. CRPS is sensitive to distributional shape in the bulk; log score is sensitive to tails and punishes overconfidence; interval score targets a specified coverage level; energy score handles joint structure across dimensions; calibration diagnostics like PIT and coverage are orthogonal to all of the above. Like the parable of the blind monks and the elephant, each rule touches a different part of the animal — one reports a snake, another a tree trunk, another a fan — and only by combining them do you get a faithful picture of what the algorithm is actually doing. The scoring report reports all of them together, alongside the data-quality diagnostics that tell you whether to trust any of them, precisely because no single number is sufficient.

## Summary

The format stores the SLAM algorithm's native probabilistic output — Gaussian mean + covariance for filters, weighted particles for ensemble methods — with headers declaring the conventions and **gauge** needed to interpret them. The gauge tells the scoring tool which DoF the algorithm pinned and which it left free, so the right alignment mode (none, SE(3), gravity-yaw, or Sim(3)) is the default, not a guess. Everything else derived — scores, $N_\text{eff}$, PIT, calibration, the alignment transform itself — is computed at scoring time. Synchronization uses nearest-neighbor matching with a tolerance, matching TUM and evo conventions for interoperability. The scoring report surfaces both the numbers and the caveats needed to trust them, including the bias introduced by alignment, and reports multiple proper scoring rules side by side because no single rule captures the full shape of a predictive distribution.

## References
 - Kanti V. Mardia, John T. Kent, and Arnab K. Laha. Score matching estimators for directional
   distributions. Preprint, 2016. URL https://arxiv.org/abs/1604.08470. arXiv:1604.08470.

 - Waghmare, Kartik, and Johanna Ziegel. "Proper scoring rules for estimation and forecast evaluation."
  Annual Review of Statistics and Its Application 13 (2025).

 - Yuya Takasu, Keisuke Yano, and Fumiyasu Komaki. Scoring rules for statistical models on spheres.
   Stat. Probab. Lett., 138:111–115, 2018. URL https://doi.org/10.1016/j.spl.2018.02.054.

 - Bonnier, Patric, and Harald Oberhauser. "Proper scoring rules, gradients, divergences, and entropies
   for paths and time series." Bayesian Analysis 20.4 (2025): 1615-1646.
