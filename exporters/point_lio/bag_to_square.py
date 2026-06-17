#!/usr/bin/env python3
"""Convert /Odometry's pose.covariance to SQUARE gaussian_se3.

Point-LIO is patched (belief-publisher.patch) to populate
nav_msgs/Odometry.pose.covariance with the active IKFoM filter's top-left
6x6 in (pos, rot) tangent order. We just lower-triangulate it into SQUARE.

Conventions are pinned by the patch:
  TANGENT_CONVENTION right_perturbation
  TANGENT_ORDER      translation_rotation   (pos[0..2], rot[3..5])
  ROTATION_PARAM     axis_angle

BODY_FRAME=imu: Point-LIO publishes the IMU-frame pose (`child_frame_id =
"body"`, same as FAST-LIO2). Scoring against Spires LiDAR-frame reference needs
smfeval's --body-frame-transform — see evaluate.py.

Usage: pose_cov_to_unc.py <bag> <out.SQUARE> [topic=/Odometry]
"""
import sys

import rosbag

_TS = "{:.9f}"
_V = "{:.17g}"

ALGORITHM = "point_lio_belief"
ALGORITHM_VERSION = "0"
BODY_FRAME = "imu"
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
            cov = msg.pose.covariance  # 36 floats, row-major 6x6
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
