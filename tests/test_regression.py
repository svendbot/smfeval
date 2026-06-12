"""Golden-report regression tests.

Each scenario under ``tests/fixtures/regression/<name>/`` holds:
- ``gt.tum`` (or ``gt.smfeval``) — ground-truth trajectory (score/nees)
- ``est.smfeval``               — estimate trajectory (score/nees)
- ``a.smfeval`` / ``b.smfeval`` — the two trajectories (pair)
- ``args.json``                 — CLI args spec (see _build_argv); the
  optional ``cmd`` key selects the verb (``score`` default, ``nees``,
  ``pair``); ``body_frame_transform`` names a JSON file in the scenario dir
- ``expected_report.json``      — committed golden report

The test runs the CLI and compares the JSON output against the golden
file with numeric tolerance. Set ``UPDATE_FIXTURES=1`` to rewrite the
golden file in place (use after intentional changes).
"""

import json
import math
import os
from pathlib import Path

import pytest

from smfeval.cli.main import main

FIXTURES = Path(__file__).parent / "fixtures" / "regression"
RTOL = 1e-9
ATOL = 1e-12


def _scenarios() -> list[Path]:
  if not FIXTURES.exists():
    return []
  return sorted(p for p in FIXTURES.iterdir() if p.is_dir())


def _build_argv(scenario: Path, json_out: Path) -> tuple[list[str], bool]:
  """Build the CLI argv for a scenario; second element: JSON on stdout."""
  spec = json.loads((scenario / "args.json").read_text())
  cmd = spec.get("cmd", "score")

  if cmd == "pair":
    argv = ["pair", str(scenario / "a.smfeval"), str(scenario / "b.smfeval")]
    if spec.get("body_frame_transform"):
      argv += [
        "--body-frame-transform",
        str(scenario / spec["body_frame_transform"]),
      ]
    argv += list(spec.get("extra", []))
    argv += ["--json"]
    return argv, True

  gt = scenario / ("gt.tum" if (scenario / "gt.tum").exists() else "gt.smfeval")
  est = scenario / "est.smfeval"
  argv = [cmd, str(est), str(gt), "--seed", str(spec.get("seed", 0))]
  if cmd == "score" and "n_samples" in spec:
    argv += ["--n_samples", str(spec["n_samples"])]
  if spec.get("gt_body_frame"):
    argv += ["--gt-body-frame", spec["gt_body_frame"]]
  if spec.get("gt_pose_frame"):
    argv += ["--gt-pose-frame", spec["gt_pose_frame"]]
  if spec.get("body_frame_transform"):
    argv += [
      "--body-frame-transform",
      str(scenario / spec["body_frame_transform"]),
    ]
  argv += list(spec.get("extra", []))
  if cmd == "nees":
    argv += ["--json"]
    return argv, True
  argv += ["--json-out", str(json_out)]
  return argv, False


def _compare(actual: object, expected: object, path: str = "") -> list[str]:
  """Return a list of human-readable diff messages."""
  if isinstance(expected, dict):
    if not isinstance(actual, dict):
      return [f"{path}: expected dict, got {type(actual).__name__}"]
    diffs: list[str] = []
    for k in set(actual) | set(expected):
      sub = f"{path}.{k}" if path else k
      if k not in actual:
        diffs.append(f"{sub}: missing in actual")
      elif k not in expected:
        diffs.append(f"{sub}: extra in actual")
      else:
        diffs.extend(_compare(actual[k], expected[k], sub))
    return diffs
  if isinstance(expected, list):
    if not isinstance(actual, list):
      return [f"{path}: expected list, got {type(actual).__name__}"]
    if len(actual) != len(expected):
      return [f"{path}: length {len(actual)} vs {len(expected)}"]
    return [
      d
      for i, (a, e) in enumerate(zip(actual, expected, strict=False))
      for d in _compare(a, e, f"{path}[{i}]")
    ]
  if isinstance(expected, float):
    if not isinstance(actual, (int, float)):
      return [f"{path}: expected number, got {type(actual).__name__}"]
    if math.isnan(expected) and math.isnan(actual):
      return []
    if math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
      return []
    return [f"{path}: {actual!r} vs {expected!r}"]
  if actual != expected:
    return [f"{path}: {actual!r} vs {expected!r}"]
  return []


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda p: p.name)
def test_regression(
  scenario: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
  if not _scenarios():
    pytest.skip("no regression scenarios in tests/fixtures/regression/")

  out = tmp_path / "report.json"
  argv, json_on_stdout = _build_argv(scenario, out)
  rc = main(argv)
  captured = capsys.readouterr()
  assert rc == 0, f"CLI exited with {rc}: {captured.err}"

  actual = json.loads(captured.out if json_on_stdout else out.read_text())

  golden = scenario / "expected_report.json"
  if os.environ.get("UPDATE_FIXTURES"):
    golden.write_text(json.dumps(actual, indent=2) + "\n")
    pytest.skip(f"updated {golden}")

  if not golden.exists():
    pytest.fail(
      f"no golden report at {golden}. Run with UPDATE_FIXTURES=1 to create it."
    )

  expected = json.loads(golden.read_text())
  diffs = _compare(actual, expected)
  assert not diffs, "report mismatch:\n  " + "\n  ".join(diffs)
