"""Guard against drift between the notebook's library pipeline and the CLI.

notebooks/figure1_verdict.py shows the nees verdict twice: once through
the CLI and once hand-composed from the library API. The two must print
the same verdict — if _prepare's defaults change (sync, alignment fit,
t_max_diff), this test fails before the notebook silently diverges.
"""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from smfeval.align import (
  align_mode_for_gauge,
  apply_body_transform,
  fit_alignment,
  propagate_step,
)
from smfeval.cli.main import main
from smfeval.io import load_square, load_tum
from smfeval.report.verdict import nees_verdict, render_nees_verdict
from smfeval.scoring import gaussian_log_score_components
from smfeval.se3.lie import homogeneous
from smfeval.sync import match_timestamps

_DATA = Path(__file__).parent.parent / "notebooks" / "data"


@pytest.fixture
def notebook_files(tmp_path: Path) -> Path:
  for local, remote in {
    "est.SQUARE": "christ-church-03_fast_lio2.SQUARE.gz",
    "ref.tum": "christ-church-03_ref.tum.gz",
  }.items():
    with (
      gzip.open(_DATA / remote, "rb") as src,
      (tmp_path / local).open("wb") as dst,
    ):
      shutil.copyfileobj(src, dst)
  shutil.copy(_DATA / "imu_to_lidar.json", tmp_path / "imu_to_lidar.json")
  return tmp_path


def _library_verdict(d: Path) -> str:
  """The notebook's library-API cell, verbatim logic."""
  est_header, est_steps = load_square(d / "est.SQUARE")
  _, ref_steps = load_tum(d / "ref.tum", pose_frame="world", body_frame="lidar")

  tf = json.loads((d / "imu_to_lidar.json").read_text())
  T_off = homogeneous(np.array(tf["R"]).reshape(3, 3), np.array(tf["t"]))
  order = est_header.tangent_order
  est_steps = [
    apply_body_transform(
      s,
      T_off,
      tangent_convention=est_header.tangent_convention,
      tangent_order=order,
    )
    for s in est_steps
  ]

  est_ts = np.array([s.timestamp for s in est_steps])
  ref_ts = np.array([s.timestamp for s in ref_steps])
  m = match_timestamps(est_ts, ref_ts, t_max_diff=0.01)
  matched = [est_steps[i] for i in m.est_indices]
  ref_t = np.array([ref_steps[j].translation for j in m.ref_indices])
  ref_q = np.array([ref_steps[j].quat_xyzw for j in m.ref_indices])

  fit = fit_alignment(
    np.array([s.translation for s in matched]),
    ref_t,
    mode=align_mode_for_gauge(est_header.gauge),
  )
  aligned = [
    propagate_step(
      s,
      fit.transform,
      scale=fit.scale,
      tangent_convention=est_header.tangent_convention,
      tangent_order=order,
    )
    for s in matched
  ]
  nees = np.array(
    [
      gaussian_log_score_components(s, t, q, order).translation.nees
      for s, t, q in zip(aligned, ref_t, ref_q, strict=True)
    ]
  )
  return render_nees_verdict(nees_verdict(nees, dof=3))


def test_notebook_library_path_matches_cli(
  notebook_files: Path, capsys: pytest.CaptureFixture
):
  rc = main(
    [
      "nees",
      str(notebook_files / "est.SQUARE"),
      str(notebook_files / "ref.tum"),
      "--ref-body-frame",
      "lidar",
      "--body-frame-transform",
      str(notebook_files / "imu_to_lidar.json"),
    ]
  )
  cli_out = capsys.readouterr().out.strip()
  assert rc == 0
  assert _library_verdict(notebook_files) == cli_out
