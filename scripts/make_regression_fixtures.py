#!/usr/bin/env python3
"""Cut real-data regression fixtures from slam_benchmark Oxford Spires runs.

Selection: one run per audited filter on Spires christ-church-03 (all four
filters have runs on this sequence), rows ROW_LO..ROW_HI of traj.SQUARE
(skipping the stationary start), with the reference TUM truncated to the matching
timestamp span. Mechanical gates: gaussian_se3 representation, all rows
finite, translation covariance SPD throughout, >= 30 s of span.

The committed excerpts are derived from the Oxford Spires dataset and are
redistributed under CC BY-NC-SA 4.0 with attribution — see
tests/fixtures/regression/DATA_LICENSE.md.

Point SLAM_BENCHMARK at a local slam_benchmark checkout, then:
  SLAM_BENCHMARK=/path/to/slam_benchmark \
    uv run python scripts/make_regression_fixtures.py
then regenerate goldens:
  UPDATE_FIXTURES=1 uv run pytest tests/test_regression.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from smfeval.format import Representation
from smfeval.io import load_square, write_header, write_steps
from smfeval.se3.lie import homogeneous, invert

_bench_root = os.environ.get("SLAM_BENCHMARK")
if not _bench_root:
  sys.exit("set SLAM_BENCHMARK to a local slam_benchmark checkout")
BENCH = Path(_bench_root)
EVAL = BENCH / "evaluation"
REF_TUM = (
  BENCH
  / "datasets/data/oxford_spires/sequences/2024-03-18-christ-church-03"
  / "processed/trajectory/ref-tum.txt"
)
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/regression"

SEQ = "spires_2024-03-18-christ-church-03_1710755015_2024-03-18-09-43-36_0"
ROW_LO, ROW_HI = 50, 360
REF_MARGIN_S = 0.5

# filter -> (algo dir prefix, regression cmd, needs imu->lidar transform)
FILTERS = {
  "fast_lio2": ("fast_lio2_belief", "score", True),
  "faster_lio": ("faster_lio_belief", "nees", True),
  "point_lio": ("point_lio_belief", "nees", True),
  "i2ekf_lo": ("i2ekf_lo_belief", "nees", False),
}


def newest_run(algo: str) -> Path:
  runs = sorted(EVAL.glob(f"{algo}_{SEQ}_*"))
  if not runs:
    sys.exit(f"no runs for {algo}_{SEQ}")
  return runs[-1]


def imu_to_lidar(algo: str) -> dict:
  """Invert the R_lidar__imu / t_lidar__imu extrinsics from the algo config."""
  cfg = yaml.safe_load(
    (BENCH / "algorithms" / algo / "configs" / "spires.yaml").read_text()
  )
  m = cfg.get("mapping") or cfg.get("lio") or {}
  T = homogeneous(
    np.array(m["extrinsic_R"], dtype=float).reshape(3, 3),
    np.array(m["extrinsic_T"], dtype=float),
  )
  T_inv = invert(T)
  return {"R": T_inv[:3, :3].flatten().tolist(), "t": T_inv[:3, 3].tolist()}


def check_steps(steps: list, name: str) -> None:
  if len(steps) != ROW_HI - ROW_LO:
    sys.exit(f"{name}: only {len(steps)} rows in [{ROW_LO}, {ROW_HI})")
  span = steps[-1].timestamp - steps[0].timestamp
  if span < 30.0:
    sys.exit(f"{name}: span {span:.1f} s < 30 s")
  for k, s in enumerate(steps):
    if not (
      np.all(np.isfinite(s.translation))
      and np.all(np.isfinite(s.quat_xyzw))
      and np.all(np.isfinite(s.covariance))
    ):
      sys.exit(f"{name}: non-finite row {k}")
    np.linalg.cholesky(s.covariance[:3, :3])  # raises if not SPD


def cut_ref(t0: float, t1: float) -> list[str]:
  rows = []
  for line in REF_TUM.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
      continue
    t = float(line.split()[0])
    if t0 - REF_MARGIN_S <= t <= t1 + REF_MARGIN_S:
      rows.append(line)
  return rows


def write_fixture(name: str, run: Path, est_name: str = "est.smfeval") -> tuple:
  header, steps = load_square(run / "traj.SQUARE")
  if header.representation is not Representation.GAUSSIAN_SE3:
    sys.exit(f"{name}: not gaussian_se3")
  cut = steps[ROW_LO:ROW_HI]
  check_steps(cut, name)

  scen = OUT / name
  scen.mkdir(parents=True, exist_ok=True)
  with (scen / est_name).open("w") as f:
    write_header(f, header)
    write_steps(f, cut, header)
  return cut, scen


def provenance(scen: Path, runs: list[Path]) -> None:
  lines = [
    "Source: slam_benchmark evaluation runs on Oxford Spires",
    "Sequence: 2024-03-18-christ-church-03",
    *[f"Run: {r.name}" for r in runs],
    f"Rows: traj.SQUARE data rows [{ROW_LO}, {ROW_HI}) "
    f"(stationary start skipped); reference span +-{REF_MARGIN_S} s",
    "Cut: scripts/make_regression_fixtures.py, 2026-06-12",
    "License: see ../DATA_LICENSE.md (CC BY-NC-SA 4.0, Oxford Spires)",
  ]
  (scen / "PROVENANCE").write_text("\n".join(lines) + "\n")


def main() -> None:
  for name, (algo, cmd, needs_tf) in FILTERS.items():
    run = newest_run(algo)
    cut, scen = write_fixture(f"real_{name}", run)
    ref_rows = cut_ref(cut[0].timestamp, cut[-1].timestamp)
    (scen / "ref.tum").write_text("\n".join(ref_rows) + "\n")
    args: dict = {"cmd": cmd, "ref_body_frame": "lidar", "seed": 0, "extra": []}
    if cmd == "score":
      args["n_samples"] = 32
    if needs_tf:
      (scen / "imu_to_lidar.json").write_text(json.dumps(imu_to_lidar(algo)))
      args["body_frame_transform"] = "imu_to_lidar.json"
    (scen / "args.json").write_text(json.dumps(args, indent=2) + "\n")
    provenance(scen, [run])
    print(
      f"real_{name}: {len(cut)} est rows, {len(ref_rows)} ref rows ({run.name})"
    )

  # pair fixture: two filters, same sequence and span, no reference involved.
  run_a = newest_run("fast_lio2_belief")
  run_b = newest_run("point_lio_belief")
  cut_a, scen = write_fixture("pair_smoke", run_a, "a.smfeval")
  hb, steps_b = load_square(run_b / "traj.SQUARE")
  cut_b = steps_b[ROW_LO:ROW_HI]
  check_steps(cut_b, "pair_smoke/b")
  with (scen / "b.smfeval").open("w") as f:
    write_header(f, hb)
    write_steps(f, cut_b, hb)
  (scen / "args.json").write_text(
    json.dumps({"cmd": "pair", "extra": []}, indent=2) + "\n"
  )
  provenance(scen, [run_a, run_b])
  print(f"pair_smoke: {len(cut_a)}/{len(cut_b)} rows")


if __name__ == "__main__":
  main()
