# Validation: FAST-LIO2 belief exporter

- **Sequence:** Oxford Spires `2024-03-18-christ-church-03` (public)
- **Run:** `fast_lio2_belief_spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0_20260608_175631` (slam_benchmark)
- **Status:** verified (audited against source and empirical error scatter, Sec. V.d standard)

Command (the estimate declares `BODY_FRAME imu`; Spires reference is in the LiDAR
frame, so the extrinsic from the filter's own config is passed):

```
smfeval nees traj.SQUARE ref-tum.txt \
  --ref-body-frame lidar \
  --body-frame-transform spires_imu_to_lidar.json
```

Output (verbatim):

```
median NEES 1.04e3   (calibrated: 2.37)
covariance scale gap k = 441, ~21x too tight per axis
90% coverage: 0.000  (calibrated: 0.900)
```

The exporter publishes the raw per-scan IKFoM covariance; the verdict shows
the well-documented overconfidence of that covariance, which is the point —
the exporter is judged on whether it faithfully exports the filter's belief,
not on whether that belief is calibrated.

## Deriving the body-frame transform (convention trap)

FAST-LIO's `extrinsic_R`, `extrinsic_T` are consumed in `IMU_Processing.hpp`
as `Lidar_R_wrt_IMU` / `Lidar_T_wrt_IMU` — these are `T_lidar_imu` (the
rotation maps IMU-frame vectors to LiDAR-frame). For an IMU-publishing file
scored against LiDAR-frame reference, the `--body-frame-transform` is the
**inverse**: `R = extrinsic_R^T`, `t = -extrinsic_R^T · extrinsic_T`. That
inverse is what `spires_imu_to_lidar.json` stores.
