"""Golden-report regression tests.

Each scenario under ``tests/fixtures/regression/<name>/`` holds:
- ``gt.tum`` (or ``gt.smfeval``) — ground-truth trajectory
- ``est.smfeval``               — estimate trajectory
- ``args.json``                 — CLI args spec (see _build_argv)
- ``expected_report.json``      — committed golden report

The test runs the CLI ``score`` command and compares the JSON report against
the golden file with numeric tolerance. Set ``UPDATE_FIXTURES=1`` to rewrite
the golden file in place (use after intentional changes).
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


def _build_argv(scenario: Path, json_out: Path) -> list[str]:
  spec = json.loads((scenario / "args.json").read_text())
  gt = scenario / ("gt.tum" if (scenario / "gt.tum").exists() else "gt.smfeval")
  est = scenario / "est.smfeval"
  argv = ["score", str(est), str(gt), "--seed", str(spec.get("seed", 0))]
  if "n_samples" in spec:
    argv += ["--n_samples", str(spec["n_samples"])]
  if spec.get("gt_body_frame"):
    argv += ["--gt-body-frame", spec["gt_body_frame"]]
  if spec.get("gt_pose_frame"):
    argv += ["--gt-pose-frame", spec["gt_pose_frame"]]
  argv += list(spec.get("extra", []))
  argv += ["--json-out", str(json_out)]
  return argv


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
def test_regression(scenario: Path, tmp_path: Path) -> None:
  if not _scenarios():
    pytest.skip("no regression scenarios in tests/fixtures/regression/")

  out = tmp_path / "report.json"
  rc = main(_build_argv(scenario, out))
  assert rc == 0, f"CLI exited with {rc}"

  actual = json.loads(out.read_text())

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
