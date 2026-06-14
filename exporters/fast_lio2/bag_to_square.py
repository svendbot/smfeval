#!/usr/bin/env python3
"""Convert /Belief (fast_lio/Belief) messages in a rosbag to SQUARE gaussian_se3.

Conventions are pinned by msg/Belief.msg:
  TANGENT_CONVENTION right_perturbation
  TANGENT_ORDER      translation_rotation   (pos at [0..2], rot at [3..5])
  ROTATION_PARAM     axis_angle
The 6x6 covariance is stored row-major in msg.covariance; SQUARE wants the
21 lower-triangular entries in row-major order.

BODY_FRAME defaults to ``imu`` for FAST-LIO2: the state pose is published in
the IMU frame (see use-ikfom.hpp). Scoring against a GT in a different body
frame (e.g. Spires GT is in the LiDAR frame) requires smfeval's
--body-frame-transform extrinsic.

Usage: pose_cov_to_unc.py <bag> <out.SQUARE> [topic=/Belief]
"""
import sys
import rosbag

_TS = "{:.9f}"
_V = "{:.17g}"

ALGORITHM = "fast_lio2_belief"
ALGORITHM_VERSION = "0"
BODY_FRAME = "imu"


def main():
    bag_path, out_path, *rest = sys.argv[1:]
    topic = rest[0] if rest else "/Belief"

    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as f:
        wrote_header = False
        for _, msg, _ in bag.read_messages(topics=[topic]):
            if not wrote_header:
                gauge = (getattr(msg, "gauge", "") or "se3").strip()
                f.write("#%FORMAT SQUARE/0.3\n")
                f.write("#%REPRESENTATION gaussian_se3\n")
                f.write("#%POSE_FRAME world\n")
                f.write(f"#%BODY_FRAME {BODY_FRAME}\n")
                f.write(f"#%GAUGE {gauge}\n")
                f.write("#%TIMESTAMP_UNIT seconds\n")
                f.write(f"#%ALGORITHM {ALGORITHM}\n")
                f.write(f"#%ALGORITHM_VERSION {ALGORITHM_VERSION}\n")
                f.write("#%TANGENT_CONVENTION right_perturbation\n")
                f.write("#%TANGENT_ORDER translation_rotation\n")
                f.write("#%ROTATION_PARAM axis_angle\n")
                wrote_header = True

            ts = msg.header.stamp.to_sec()
            p = msg.pose.position
            q = msg.pose.orientation
            cov = msg.covariance  # 36 floats, row-major 6x6
            tri = [cov[i * 6 + j] for i in range(6) for j in range(i + 1)]
            cols = [
                _TS.format(ts),
                _V.format(p.x), _V.format(p.y), _V.format(p.z),
                _V.format(q.x), _V.format(q.y), _V.format(q.z), _V.format(q.w),
                *(_V.format(v) for v in tri),
            ]
            f.write(" ".join(cols) + "\n")


if __name__ == "__main__":
    main()
