# Validation: FAST-LIO2 belief exporter

- **Sequence:** Oxford Spires `2024-03-18-christ-church-03` (public)
- **Run:** `fast_lio2_belief_spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0_20260608_175631` (slam_benchmark)
- **Status:** verified (audited against source and empirical error scatter, Sec. V.d standard)

Command (the estimate declares `BODY_FRAME imu`; Spires GT is in the LiDAR
frame, so the extrinsic from the filter's own config is passed):

```
smfeval nees traj.SQUARE gt-tum.txt \
  --gt-body-frame lidar \
  --body-frame-transform spires_imu_to_lidar.json
```

Output (verbatim):

```
median NEES 1.06e7   (calibrated: 2.37)
covariance scale gap k = 4.48e6, ~2.12e3x too tight per axis
90% coverage: 0.000  (calibrated: 0.900)
```

The exporter publishes the raw per-scan IKFoM covariance; the verdict shows
the well-documented overconfidence of that covariance, which is the point —
the exporter is judged on whether it faithfully exports the filter's belief,
not on whether that belief is calibrated.
