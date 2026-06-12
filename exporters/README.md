# Exporters — make your filter write its belief

Each directory holds everything needed to make one SLAM filter publish its
posterior covariance and convert it to a SQUARE file smfeval can score:

- `belief-publisher.patch` — the diff against a pinned upstream commit that
  makes the filter publish its belief (often only a few lines);
- `UPSTREAM` — the upstream repository and commit the patch applies to,
  as `URL @ sha`;
- `bag_to_square.py` — standalone script (needs `rosbag`; **not** part of
  the smfeval package) converting the recorded topic to SQUARE;
- `VALIDATION.md` — a validation run: the exact `smfeval nees` command and
  its verbatim verdict on one named public sequence;
- `spires_imu_to_lidar.json` — where present, the body-frame extrinsic used
  by the validation command (derived from the filter's own config).

## Status

| Filter    | Upstream                              | Commit         | Status   | Validation sequence              | k (scale gap, raw Σ) |
|-----------|---------------------------------------|----------------|----------|----------------------------------|----------------------|
| FAST-LIO2 | github.com/hku-mars/FAST_LIO          | `7cc4175de6f8` | verified | Spires christ-church-03          | 4.48e6               |
| Faster-LIO| github.com/gaoxiang12/faster-lio      | `ea0e0910a4cf` | verified | Spires christ-church-03          | 2.15e8               |
| Point-LIO | github.com/hku-mars/Point-LIO         | `97b0042e397e` | verified | Spires christ-church-03          | 2.47e3               |
| I2EKF-LO  | github.com/YWL0720/I2EKF-LO           | `8d2158cda30e` | verified | Spires christ-church-03          | 1.77e10              |

**verified** — audited against the filter source *and* the empirical error
scatter (the paper's Sec. V.d standard): the exported covariance is the
filter's actual posterior, in the declared frame and convention.
**contributed** — community-submitted with the required artifacts but not
yet source-audited.

A wrong export produces a wrong verdict that gets attributed to the tool, so
promotion from `contributed` to `verified` requires the audit. The k column
is the filter's raw per-scan covariance scale gap on the validation
sequence — exporters are judged on whether they faithfully export the
belief, not on whether that belief is calibrated.

## Contributing an exporter

Open a PR adding a directory with the four artifacts above (see
`.github/PULL_REQUEST_TEMPLATE.md`):

1. the diff and the upstream `URL @ sha` it applies to;
2. `smfeval validate --strict your.SQUARE` output (per-row covariance SPD,
   plausible magnitudes, not degenerate-zero);
3. the validation run on a named public sequence with the verbatim verdict.

`scripts/check_exporters.py` (run in CI) enforces the directory layout
mechanically. New exporters land as `contributed`.
