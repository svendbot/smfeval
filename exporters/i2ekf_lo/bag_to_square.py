#!/usr/bin/env python3
"""Convert /Odometry's pose.covariance to SQUARE gaussian_se3.

I2EKF-LO is patched (belief-publisher.patch) to populate pose.covariance
with the state.cov top-left 6x6 permuted from native (rot[0..2], pos[3..5])
into REP-103 (pos, rot) order.

BODY_FRAME=lidar: I2EKF-LO is a LiDAR-only filter; state.pos_end is the
LiDAR-in-world pose used by reference, so no body-frame transform is needed for
Spires (unlike IKFoM filters which publish IMU-frame poses).

Caveat: I2EKF-LO has no IMU propagation step. The covariance evolution is
driven by scan-to-map residuals + process noise only, so magnitudes are
not directly comparable to IKFoM-based filters.

Usage: pose_cov_to_unc.py <bag> <out.SQUARE> [topic=/Odometry]
"""

import sys

import rosbag

_TS = "{:.9f}"
_V = "{:.17g}"

ALGORITHM = "i2ekf_lo_belief"
ALGORITHM_VERSION = "0"
BODY_FRAME = "lidar"
GAUGE = "se3"


def main():
  bag_path, out_path, *rest = sys.argv[1:]
  topic = rest[0] if rest else "/Odometry"

  with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as f:
    wrote_header = False
    for _, msg, _ in bag.read_messages(topics=[topic]):
      if not wrote_header:
        f.write("#%FORMAT SQUARE/0.3\n")
        f.write("#%REPRESENTATION gaussian_se3\n")
        f.write("#%POSE_FRAME world\n")
        f.write(f"#%BODY_FRAME {BODY_FRAME}\n")
        f.write(f"#%GAUGE {GAUGE}\n")
        f.write("#%TIMESTAMP_UNIT seconds\n")
        f.write(f"#%ALGORITHM {ALGORITHM}\n")
        f.write(f"#%ALGORITHM_VERSION {ALGORITHM_VERSION}\n")
        f.write("#%TANGENT_CONVENTION right_perturbation\n")
        f.write("#%TANGENT_ORDER translation_rotation\n")
        f.write("#%ROTATION_PARAM axis_angle\n")
        wrote_header = True

      ts = msg.header.stamp.to_sec()
      p = msg.pose.pose.position
      q = msg.pose.pose.orientation
      cov = msg.pose.covariance
      # i2ekf_lo's iEKF runs at flg_EKF_inited=true from the first scan
      # and uses the simple (I-KH) form (laserMapping.cpp:1346) which
      # loses PSD numerically until the filter stabilises. Drop rows
      # with non-positive diagonal or non-finite values; smfeval rejects
      # the whole file on the first bad row otherwise.
      diag = [cov[k * 6 + k] for k in range(6)]
      if any((not (d > 0.0)) for d in diag):  # NaN, ±inf, ≤0 all false
        continue
      tri = [cov[i * 6 + j] for i in range(6) for j in range(i + 1)]
      cols = [
        _TS.format(ts),
        _V.format(p.x),
        _V.format(p.y),
        _V.format(p.z),
        _V.format(q.x),
        _V.format(q.y),
        _V.format(q.z),
        _V.format(q.w),
        *(_V.format(v) for v in tri),
      ]
      f.write(" ".join(cols) + "\n")


if __name__ == "__main__":
  main()
