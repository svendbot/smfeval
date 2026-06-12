from collections.abc import Iterable
from typing import TextIO

from smfeval.format import Representation, SquareHeader
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step

_TS_FMT = "{:.9f}"
_VAL_FMT = "{:.17g}"


def write_step(f: TextIO, step: Step, header: SquareHeader) -> None:
  if isinstance(step, GaussianStep):
    if header.representation is not Representation.GAUSSIAN_SE3:
      raise ValueError("GaussianStep requires gaussian_se3 header")
    _write_gaussian(f, step)
  elif isinstance(step, EnsembleStep):
    if header.representation is not Representation.ENSEMBLE_SE3:
      raise ValueError("EnsembleStep requires ensemble_se3 header")
    _write_ensemble(f, step, header)
  elif isinstance(step, DeterministicStep):
    if header.representation is not Representation.DETERMINISTIC:
      raise ValueError("DeterministicStep requires deterministic header")
    _write_deterministic(f, step)
  else:
    raise ValueError(f"unknown step type: {type(step).__name__}")


def write_steps(f: TextIO, steps: Iterable[Step], header: SquareHeader) -> None:
  for step in steps:
    write_step(f, step, header)


def _write_gaussian(f: TextIO, step: GaussianStep) -> None:
  parts = [_TS_FMT.format(step.timestamp)]
  parts += [_VAL_FMT.format(x) for x in step.translation]
  parts += [_VAL_FMT.format(x) for x in step.quat_xyzw]
  parts += [
    _VAL_FMT.format(step.covariance[i, j])
    for i in range(6)
    for j in range(i + 1)
  ]
  f.write(" ".join(parts) + "\n")


def _write_ensemble(
  f: TextIO, step: EnsembleStep, header: SquareHeader
) -> None:
  weighted = bool(header.weighted)
  if weighted and step.weights is None:
    raise ValueError("WEIGHTED true but step has no weights")
  if not weighted and step.weights is not None:
    raise ValueError("WEIGHTED false but step provides weights")
  n = step.particles.shape[0]
  if weighted and step.weights is not None and step.weights.shape[0] != n:
    raise ValueError("weights length mismatch with particles")
  ts = _TS_FMT.format(step.timestamp)
  for pid in range(n):
    parts = [ts, str(pid)]
    if weighted:
      assert step.weights is not None
      parts.append(_VAL_FMT.format(step.weights[pid]))
    parts += [_VAL_FMT.format(x) for x in step.particles[pid]]
    f.write(" ".join(parts) + "\n")


def _write_deterministic(f: TextIO, step: DeterministicStep) -> None:
  parts = [_TS_FMT.format(step.timestamp)]
  parts += [_VAL_FMT.format(x) for x in step.translation]
  parts += [_VAL_FMT.format(x) for x in step.quat_xyzw]
  f.write(" ".join(parts) + "\n")
