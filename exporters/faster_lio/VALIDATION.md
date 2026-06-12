# Validation: Faster-LIO belief exporter

- **Sequence:** Oxford Spires `2024-03-18-christ-church-03` (public)
- **Run:** `faster_lio_belief_spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0_20260608_164534` (slam_benchmark)
- **Status:** verified (audited against source and empirical error scatter, Sec. V.d standard)

Command:

```
smfeval nees traj.SQUARE gt-tum.txt \
  --gt-body-frame lidar \
  --body-frame-transform spires_imu_to_lidar.json
```

Output (verbatim):

```
median NEES 5.09e8   (calibrated: 2.37)
covariance scale gap k = 2.15e8, ~1.47e4x too tight per axis
90% coverage: 0.000  (calibrated: 0.900)
```

Note: upstream Faster-LIO contained a rot/pos block swap and a dead write in
its odometry-covariance publication; `belief-publisher.patch` fixes both (see
the patch header). The verdict reflects the filter's raw per-scan covariance.
