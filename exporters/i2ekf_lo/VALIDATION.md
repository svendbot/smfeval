# Validation: I2EKF-LO belief exporter

- **Sequence:** Oxford Spires `2024-03-18-christ-church-03` (public)
- **Run:** `i2ekf_lo_belief_spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0_20260608_163912` (slam_benchmark)
- **Status:** verified (audited against source and empirical error scatter, Sec. V.d standard)

Command (I2EKF-LO already estimates in the LiDAR body frame, so no
body-frame transform is needed against the Spires GT):

```
smfeval nees traj.SQUARE gt-tum.txt --gt-body-frame lidar
```

Output (verbatim):

```
median NEES 4.18e10   (calibrated: 2.37)
covariance scale gap k = 1.77e10, ~1.33e5x too tight per axis
90% coverage: 0.000  (calibrated: 0.900)
```

The verdict reflects the filter's raw per-scan covariance.
