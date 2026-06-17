# Data license for the `real_*` and `pair_smoke` fixtures

The trajectory excerpts in the `real_fast_lio2/`, `real_faster_lio/`,
`real_point_lio/`, `real_i2ekf_lo/`, and `pair_smoke/` directories are
derived from the **Oxford Spires Dataset** (sequence
`2024-03-18-christ-church-03`):

- `ref.tum` files are truncated excerpts of the dataset's reference
  trajectory.
- `est.smfeval` / `a.smfeval` / `b.smfeval` files are SLAM filter outputs
  (FAST-LIO2, Faster-LIO, Point-LIO, I2EKF-LO with belief exporters)
  computed on the dataset's sensor recordings, i.e. derivative works.

These files are redistributed under the dataset's license, **Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** — *not* the Apache-2.0 license that covers the
smfeval source code.

Attribution:

> Tao, Y., Muñoz-Bañón, M. Á., Zhang, L., Wang, J., Fu, L. F. T., &
> Fallon, M. (2025). The Oxford Spires Dataset: Benchmarking large-scale
> LiDAR-visual localisation, reconstruction and radiance field methods.
> *The International Journal of Robotics Research*.
> https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/

Each fixture directory's `PROVENANCE` file records the exact source run
and truncation rule (`scripts/make_regression_fixtures.py`).
