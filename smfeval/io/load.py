from pathlib import Path

from smfeval.format import SquareHeader, TumHeader
from smfeval.io.header import parse_header
from smfeval.io.reader import _iter_deterministic, iter_steps
from smfeval.steps import DeterministicStep, Step


def looks_like_tum(path: Path) -> bool:
  """Detect bare-TUM trajectory files (no ``#%FORMAT`` header).

  Matches the CLI help: GT may be smfeval or TUM.
  """
  with path.open() as f:
    for line in f:
      s = line.strip()
      if not s:
        continue
      if s.startswith("#%FORMAT"):
        return False
      if s.startswith("#"):
        continue
      return True
  return True


def load_tum(
  path: Path, pose_frame: str, body_frame: str
) -> tuple[TumHeader, list[DeterministicStep]]:
  header = TumHeader(pose_frame=pose_frame, body_frame=body_frame)
  with path.open() as f:
    steps = list(_iter_deterministic(f))
  return header, steps


def load_square(path: Path) -> tuple[SquareHeader, list[Step]]:
  with path.open() as f:
    header = parse_header(f)
    steps = list(iter_steps(f, header))
  return header, steps
