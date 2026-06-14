from collections.abc import Iterator
from typing import TextIO

import numpy as np

from smfeval.format import FormatError, Representation, SquareHeader
from smfeval.io.triangular import unpack_lower_triangular
from smfeval.steps import DeterministicStep, EnsembleStep, GaussianStep, Step


def iter_steps(f: TextIO, header: SquareHeader) -> Iterator[Step]:
  if header.representation is Representation.GAUSSIAN_SE3:
    yield from _iter_gaussian(f)
  elif header.representation is Representation.ENSEMBLE_SE3:
    yield from _iter_ensemble(f, header)
  else:
    yield from _iter_deterministic(f)


def _data_lines(f: TextIO) -> Iterator[tuple[int, str, bool]]:
  """Yield (1-based line number, stripped text, newline-terminated) data rows."""
  for lineno, raw in enumerate(f, start=1):
    s = raw.strip()
    if s and not s.startswith("#"):
      yield lineno, s, raw.endswith("\n")


def _iter_with_truncation(
  items: Iterator[tuple[int, str, bool]], expected_fields: int
) -> Iterator[tuple[int, str]]:
  """Yield (lineno, text), tolerating a single truncated final row.

  A non-terminated last line is dropped only when its field count already
  differs from expected (a genuine partial write); otherwise it is yielded so
  the row validators can report it.
  """
  buffered: tuple[int, str, bool] | None = None
  for lineno, text, terminated in items:
    if buffered is not None:
      yield buffered[0], buffered[1]
    buffered = (lineno, text, terminated)
  if buffered is None:
    return
  lineno, text, terminated = buffered
  if terminated or len(text.split()) == expected_fields:
    yield lineno, text


def _iter_gaussian(f: TextIO) -> Iterator[GaussianStep]:
  for lineno, line in _iter_with_truncation(_data_lines(f), 29):
    parts = line.split()
    if len(parts) != 29:
      raise FormatError(
        f"row {lineno}: gaussian_se3 row has {len(parts)} fields, expected 29"
      )
    try:
      vals = [float(p) for p in parts]
    except ValueError as e:
      raise FormatError(f"row {lineno}: non-numeric value: {line!r}") from e
    yield GaussianStep(
      timestamp=vals[0],
      translation=np.array(vals[1:4]),
      quat_xyzw=np.array(vals[4:8]),
      covariance=unpack_lower_triangular(vals[8:]),
    )


def _iter_deterministic(f: TextIO) -> Iterator[DeterministicStep]:
  for lineno, line in _iter_with_truncation(_data_lines(f), 8):
    parts = line.split()
    if len(parts) != 8:
      raise FormatError(
        f"row {lineno}: deterministic row has {len(parts)} fields, expected 8"
      )
    try:
      vals = [float(p) for p in parts]
    except ValueError as e:
      raise FormatError(f"row {lineno}: non-numeric value: {line!r}") from e
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
  next_pid = 0
  seen_ts: set[str] = set()

  def _flush() -> EnsembleStep | None:
    if not particles:
      return None
    return EnsembleStep(
      timestamp=float(current_ts_str),  # type: ignore[arg-type]
      particles=np.array(particles),
      weights=np.array(weights) if weighted else None,
    )

  for lineno, line in _iter_with_truncation(_data_lines(f), expected):
    parts = line.split()
    if len(parts) != expected:
      raise FormatError(
        f"row {lineno}: ensemble_se3 row has {len(parts)} fields, "
        f"expected {expected}"
      )
    ts_str = parts[0]
    try:
      pid = int(parts[1])
      if weighted:
        weight = float(parts[2])
        pose = [float(p) for p in parts[3:]]
      else:
        weight = 0.0
        pose = [float(p) for p in parts[2:]]
    except ValueError as e:
      raise FormatError(f"row {lineno}: non-numeric value: {line!r}") from e

    if current_ts_str is not None and ts_str != current_ts_str:
      if ts_str in seen_ts:
        raise FormatError(
          f"row {lineno}: timestamp {ts_str} recurs after another timestamp; "
          "all ensemble rows for a timestep must be contiguous"
        )
      step = _flush()
      if step is not None:
        yield step
      seen_ts.add(current_ts_str)
      particles = []
      weights = []
      next_pid = 0

    if pid != next_pid:
      raise FormatError(
        f"row {lineno}: particle_id {pid}, expected {next_pid} "
        "(ids must run 0..n-1 within a timestep)"
      )
    current_ts_str = ts_str
    particles.append(pose)
    next_pid += 1
    if weighted:
      weights.append(weight)

  step = _flush()
  if step is not None:
    yield step
