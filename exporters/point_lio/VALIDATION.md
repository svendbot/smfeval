# Validation: Point-LIO belief exporter

- **Sequence:** Oxford Spires `2024-03-18-christ-church-03` (public)
- **Run:** `point_lio_belief_spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0_20260608_175111` (slam_benchmark)
- **Status:** verified (audited against source and empirical error scatter, Sec. V.d standard)

Command:

```
smfeval nees traj.SQUARE gt-tum.txt \
  --ref-body-frame lidar \
  --body-frame-transform spires_imu_to_lidar.json
```

Output (verbatim):

```
median NEES 39.7   (calibrated: 2.37)
covariance scale gap k = 16.8, ~4.09x too tight per axis
90% coverage: 0.016  (calibrated: 0.900)
```

The verdict reflects the filter's raw per-scan covariance.
