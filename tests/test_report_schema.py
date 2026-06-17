"""The `score --json` report validates against docs/report.schema.json."""

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parent.parent
SCHEMA = REPO / "docs" / "report.schema.json"
DET = REPO / "tests" / "fixtures" / "regression" / "det_smoke"


def _schema() -> dict:
  return json.loads(SCHEMA.read_text())


def test_schema_is_valid_jsonschema():
  jsonschema.Draft202012Validator.check_schema(_schema())


def test_score_json_matches_schema(capsys):
  from smfeval.cli.main import main

  rc = main(
    [
      "score",
      str(DET / "est.smfeval"),
      str(DET / "gt.tum"),
      "--ref-body-frame",
      "imu",
      "--seed",
      "0",
      "--n_samples",
      "8",
      "--json",
    ]
  )
  assert rc == 0
  report = json.loads(capsys.readouterr().out)
  jsonschema.validate(report, _schema())


def test_committed_score_golden_matches_schema():
  # det_smoke's golden is a full score report; it must satisfy the schema too.
  golden = json.loads((DET / "expected_report.json").read_text())
  jsonschema.validate(golden, _schema())
