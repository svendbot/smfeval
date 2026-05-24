from collections.abc import Iterator
from typing import TextIO

import numpy as np

from src.format import FormatError, Representation, SquareHeader
from src.steps import DeterministicStep, EnsembleStep, GaussianStep, Step


def iter_steps(f: TextIO, header: SquareHeader) -> Iterator[Step]:
  if header.representation is Representation.GAUSSIAN_SE3:
    yield from _iter_gaussian(f)
  elif header.representation is Representation.ENSEMBLE_SE3:
    yield from _iter_ensemble(f, header)
  else:
    yield from _iter_deterministic(f)


def _data_lines(f: TextIO) -> Iterator[tuple[str, bool]]:
  for raw in f:
    s = raw.strip()
    if s and not s.startswith("#"):
      yield s, raw.endswith("\n")


def _iter_with_truncation(
  items: Iterator[tuple[str, bool]], expected_fields: int
) -> Iterator[str]:
  buffered: tuple[str, bool] | None = None
  for text, terminated in items:
    if buffered is not None:
      yield buffered[0]
    buffered = (text, terminated)
  if buffered is None:
    return
  text, terminated = buffered
  if terminated or len(text.split()) == expected_fields:
    yield text


def _iter_gaussian(f: TextIO) -> Iterator[GaussianStep]:
  for line in _iter_with_truncation(_data_lines(f), 29):
    parts = line.split()
    if len(parts) != 29:
      raise FormatError(
        f"gaussian_se3 row has {len(parts)} fields, expected 29"
      )
    try:
      vals = [float(p) for p in parts]
    except ValueError as e:
      raise FormatError(f"non-numeric value in row: {line!r}") from e
    yield GaussianStep(
      timestamp=vals[0],
      translation=np.array(vals[1:4]),
      quat_xyzw=np.array(vals[4:8]),
      covariance=_expand_lower_triangular(vals[8:]),
    )


def _expand_lower_triangular(entries: list[float]) -> np.ndarray:
  cov = np.zeros((6, 6))
  k = 0
  for i in range(6):
    for j in range(i + 1):
      cov[i, j] = entries[k]
      cov[j, i] = entries[k]
      k += 1
  return cov


def _iter_deterministic(f: TextIO) -> Iterator[DeterministicStep]:
  for line in _iter_with_truncation(_data_lines(f), 8):
    parts = line.split()
    if len(parts) != 8:
      raise FormatError(
        f"deterministic row has {len(parts)} fields, expected 8"
      )
    try:
      vals = [float(p) for p in parts]
    except ValueError as e:
      raise FormatError(f"non-numeric value in row: {line!r}") from e
    yield DeterministicStep(
      timestamp=vals[0],
      translation=np.array(vals[1:4]),
      quat_xyzw=np.array(vals[4:8]),
    )


def _iter_ensemble(f: TextIO, header: SquareHeader) -> Iterator[EnsembleStep]:
  weighted = bool(header.weighted)
  expected = 10 if weighted else 9

  current_ts_str: str | None = None
  particles: list[list[float]] = []
  weights: list[float] = []

  def _flush() -> EnsembleStep | None:
    if not particles:
      return None
    return EnsembleStep(
      timestamp=float(current_ts_str) if current_ts_str else 0.0,
      particles=np.array(particles),
      weights=np.array(weights) if weighted else None,
    )

  for line in _iter_with_truncation(_data_lines(f), expected):
    parts = line.split()
    if len(parts) != expected:
      raise FormatError(
        f"ensemble_se3 row has {len(parts)} fields, expected {expected}"
      )
    ts_str = parts[0]
    try:
      int(parts[1])
      if weighted:
        weight = float(parts[2])
        pose = [float(p) for p in parts[3:]]
      else:
        weight = 0.0
        pose = [float(p) for p in parts[2:]]
    except ValueError as e:
      raise FormatError(f"non-numeric value in row: {line!r}") from e

    if current_ts_str is not None and ts_str != current_ts_str:
      step = _flush()
      if step is not None:
        yield step
      particles = []
      weights = []

    current_ts_str = ts_str
    particles.append(pose)
    if weighted:
      weights.append(weight)

  step = _flush()
  if step is not None:
    yield step
