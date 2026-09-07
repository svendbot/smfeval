"""Golden-report regression tests.

Each scenario under ``tests/fixtures/regression/<name>/`` holds:
- ``ref.tum`` (or ``ref.smfeval``) — reference trajectory (score/nees)
- ``est.smfeval``               — estimate trajectory (score/nees)
- ``a.smfeval`` / ``b.smfeval`` — the two trajectories (pair)
- ``args.json``                 — CLI args spec (see _build_argv); the
  optional ``cmd`` key selects the verb (``score`` default, ``nees``,
  ``pair``); ``body_frame_transform`` names a JSON file in the scenario dir
- ``expected_report.json``      — committed golden report

The test runs the CLI and compares the JSON output against the golden
file with numeric tolerance. Set ``UPDATE_FIXTURES=1`` to rewrite the
golden file in place (use after intentional changes).

Goldens are written with floats rounded to ``GOLDEN_SIG_DIGITS``, so
regenerating on a machine whose BLAS sums in a different order is a no-op
instead of a diff in the last digit or two.
"""

import json
import math
import os
from pathlib import Path

import pytest

from smfeval.cli.main import main

FIXTURES = Path(__file__).parent / "fixtures" / "regression"
# Cross-platform tolerance. Sampled scores and bootstrap block lengths vary in
# the last few digits across BLAS and Python builds, so an exact-match
# tolerance is not portable. 1e-6 still catches real regressions, which move
# values by far more than that.
RTOL = 1e-6
ATOL = 1e-9
# Precision goldens are stored at. Digits past this are never read -- the
# comparison above stops at 1e-6 -- but writing them made UPDATE_FIXTURES
# produce a diff on any machine but the one that last ran it. 12 leaves six
# orders of margin under RTOL and four over the ~1e-16 relative spread we
# actually observe between builds.
GOLDEN_SIG_DIGITS = 12


def _round_sig(x: float, sig: int) -> float:
  """Round to ``sig`` significant digits; non-finite and zero pass through."""
  if not math.isfinite(x) or x == 0.0:
    return x
  return round(x, sig - 1 - math.floor(math.log10(abs(x))))


def _rounded(obj: object, sig: int = GOLDEN_SIG_DIGITS) -> object:
  """Recursively round every float in a decoded JSON structure.

  Ints (including bools) pass through untouched so the golden keeps their
  JSON type.
  """
  if isinstance(obj, dict):
    return {k: _rounded(v, sig) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_rounded(v, sig) for v in obj]
  if isinstance(obj, float):
    return _round_sig(obj, sig)
  return obj


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

  ref = scenario / (
    "ref.tum" if (scenario / "ref.tum").exists() else "ref.smfeval"
  )
  est = scenario / "est.smfeval"
  argv = [cmd, str(est), str(ref)]
  if cmd == "score":
    argv += ["--seed", str(spec.get("seed", 0))]
    if "n_samples" in spec:
      argv += ["--n_samples", str(spec["n_samples"])]
  if spec.get("ref_body_frame"):
    argv += ["--ref-body-frame", spec["ref_body_frame"]]
  if spec.get("ref_pose_frame"):
    argv += ["--ref-pose-frame", spec["ref_pose_frame"]]
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
    golden.write_text(json.dumps(_rounded(actual), indent=2) + "\n")
    pytest.skip(f"updated {golden}")

  if not golden.exists():
    pytest.fail(
      f"no golden report at {golden}. Run with UPDATE_FIXTURES=1 to create it."
    )

  expected = json.loads(golden.read_text())
  diffs = _compare(actual, expected)
  assert not diffs, "report mismatch:\n  " + "\n  ".join(diffs)


def test_round_sig_keeps_significant_digits_across_magnitudes():
  assert _round_sig(0.0006262220495590752, 12) == 0.000626222049559
  assert _round_sig(184.46873624905618, 12) == 184.468736249
  assert _round_sig(5856719983.084327, 12) == 5856719983.08


def test_round_sig_passes_through_zero_and_non_finite():
  assert _round_sig(0.0, 12) == 0.0
  assert math.isnan(_round_sig(float("nan"), 12))
  assert _round_sig(float("inf"), 12) == float("inf")


def test_rounded_preserves_int_and_bool_types():
  out = _rounded({"n": 309, "flag": True, "x": 1.23456789012345}, sig=12)
  assert isinstance(out["n"], int) and out["n"] == 309
  assert out["flag"] is True
  assert out["x"] == 1.23456789012


def test_rounded_recurses_into_lists_and_nested_dicts():
  out = _rounded({"a": [{"b": 1.23456789012345}]}, sig=6)
  assert out == {"a": [{"b": 1.23457}]}


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda p: p.name)
def test_golden_is_already_rounded(scenario: Path) -> None:
  """Rounding a committed golden is a fixed point.

  This is what stops UPDATE_FIXTURES from re-dirtying a golden on a machine
  whose BLAS sums in a different order: the stored value carries no digits
  past GOLDEN_SIG_DIGITS for that noise to land in.
  """
  golden = scenario / "expected_report.json"
  if not golden.exists():
    pytest.skip(f"no golden at {golden}")
  raw = golden.read_text()
  assert json.dumps(_rounded(json.loads(raw)), indent=2) + "\n" == raw
