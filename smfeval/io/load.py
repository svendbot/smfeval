from pathlib import Path

import numpy as np

from smfeval.format import (
  FORMAT_VERSION,
  FormatError,
  Gauge,
  Representation,
  SquareHeader,
  TangentConvention,
  TangentOrder,
  TumHeader,
)
from smfeval.io.header import parse_header
from smfeval.io.reader import (
  _iter_deterministic,
  _iter_gaussian,
  iter_steps,
)
from smfeval.io.triangular import unpack_lower_triangular
from smfeval.steps import DeterministicStep, GaussianStep, Step

# sidecar covariance rows: timestamp + 21 lower-triangle entries
_SIDECAR_FIELDS = 22
_SIDECAR_ATOL_S = 1e-6


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


def sniff_tum_columns(path: Path) -> int:
  """Field count of the first data row of a header-less trajectory file."""
  with path.open() as f:
    for line in f:
      s = line.strip()
      if s and not s.startswith("#"):
        return len(s.split())
  return 0


def load_tum(
  path: Path, pose_frame: str, body_frame: str
) -> tuple[TumHeader, list[DeterministicStep]]:
  header = TumHeader(pose_frame=pose_frame, body_frame=body_frame)
  with path.open() as f:
    steps = list(_iter_deterministic(f))
  return header, steps


def _escape_hatch_header(
  pose_frame: str,
  body_frame: str,
  tangent_convention: TangentConvention,
  tangent_order: TangentOrder,
  gauge: Gauge,
  algorithm: str,
) -> SquareHeader:
  return SquareHeader(
    format_version=FORMAT_VERSION,
    representation=Representation.GAUSSIAN_SE3,
    pose_frame=pose_frame,
    body_frame=body_frame,
    gauge=gauge,
    timestamp_unit="seconds",
    algorithm=algorithm,
    algorithm_version="0",
    tangent_convention=tangent_convention,
    tangent_order=tangent_order,
    rotation_param="axis_angle",
  )


def load_tum_gaussian(
  path: Path,
  *,
  pose_frame: str,
  body_frame: str,
  tangent_convention: TangentConvention = TangentConvention.RIGHT,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  gauge: Gauge = Gauge.SE3,
) -> tuple[SquareHeader, list[GaussianStep]]:
  """Load a wide-TUM file: header-less rows of ``t x y z qx qy qz qw`` plus
  the 21 row-major lower-triangle entries of the 6x6 tangent covariance.

  Byte-compatible with a SQUARE gaussian_se3 body; the frame and tangent
  metadata that the missing header would carry comes from the keyword
  arguments instead.
  """  # noqa: D205
  header = _escape_hatch_header(
    pose_frame, body_frame, tangent_convention, tangent_order, gauge, "tum+cov"
  )
  with path.open() as f:
    steps = list(_iter_gaussian(f))
  return header, steps


def load_cov_sidecar(
  path: Path,
) -> tuple[
  np.ndarray, np.ndarray, TangentConvention | None, TangentOrder | None
]:
  """Read a sidecar covariance file for a bare-TUM estimate.

  Rows are ``timestamp c11 c21 c22 c31 ... c66`` (22 fields): the row-major
  lower triangle of the 6x6 tangent covariance, identical packing to SQUARE
  gaussian rows. ``#`` comments are allowed; optional header lines
  ``#%TANGENT_CONVENTION <v>`` / ``#%TANGENT_ORDER <v>`` override the CLI
  defaults. Returns (timestamps (n,), covariances (n, 6, 6), convention
  override, order override).
  """
  ts: list[float] = []
  covs: list[np.ndarray] = []
  convention: TangentConvention | None = None
  order: TangentOrder | None = None
  for raw in path.read_text().splitlines():
    s = raw.strip()
    if not s:
      continue
    if s.startswith("#%TANGENT_CONVENTION"):
      convention = TangentConvention(s.split(maxsplit=1)[1].strip())
      continue
    if s.startswith("#%TANGENT_ORDER"):
      order = TangentOrder(s.split(maxsplit=1)[1].strip())
      continue
    if s.startswith("#"):
      continue
    parts = s.split()
    if len(parts) != _SIDECAR_FIELDS:
      raise FormatError(
        f"covariance sidecar row has {len(parts)} fields, expected "
        f"{_SIDECAR_FIELDS} (timestamp + 21 lower-triangle entries)"
      )
    try:
      vals = [float(p) for p in parts]
    except ValueError as e:
      raise FormatError(f"non-numeric value in sidecar row: {s!r}") from e
    ts.append(vals[0])
    covs.append(unpack_lower_triangular(vals[1:]))
  if not ts:
    raise FormatError(f"covariance sidecar {path} has no data rows")
  return np.asarray(ts), np.stack(covs), convention, order


def attach_covariances(
  steps: list[DeterministicStep],
  ts: np.ndarray,
  covs: np.ndarray,
  *,
  atol: float = _SIDECAR_ATOL_S,
) -> list[GaussianStep]:
  """Attach sidecar covariances to TUM poses, matched 1:1 by timestamp.

  The sidecar must cover every pose exactly (within ``atol`` seconds) —
  covariance attachment is identity-critical, so there is no silent
  nearest-neighbour fallback.
  """
  if len(steps) != ts.size:
    raise FormatError(
      f"covariance sidecar has {ts.size} rows for {len(steps)} poses; "
      "they must match 1:1"
    )
  out: list[GaussianStep] = []
  for s, t, cov in zip(steps, ts, covs, strict=True):
    if abs(s.timestamp - t) > atol:
      raise FormatError(
        f"sidecar timestamp {t!r} does not match pose timestamp "
        f"{s.timestamp!r} (tolerance {atol} s); rows must align 1:1"
      )
    out.append(
      GaussianStep(
        timestamp=s.timestamp,
        translation=s.translation,
        quat_xyzw=s.quat_xyzw,
        covariance=cov,
      )
    )
  return out


def load_tum_with_sidecar(
  path: Path,
  cov_path: Path,
  *,
  pose_frame: str,
  body_frame: str,
  tangent_convention: TangentConvention = TangentConvention.RIGHT,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  gauge: Gauge = Gauge.SE3,
) -> tuple[SquareHeader, list[GaussianStep]]:
  """Bare-TUM poses plus a sidecar covariance file -> gaussian_se3 steps.

  ``#%TANGENT_*`` header lines in the sidecar override the keyword
  defaults (the file knows its own convention better than the caller).
  """
  _, det_steps = load_tum(path, pose_frame=pose_frame, body_frame=body_frame)
  ts, covs, conv_override, order_override = load_cov_sidecar(cov_path)
  header = _escape_hatch_header(
    pose_frame,
    body_frame,
    conv_override or tangent_convention,
    order_override or tangent_order,
    gauge,
    "tum+cov",
  )
  return header, attach_covariances(det_steps, ts, covs)


def load_square(path: Path) -> tuple[SquareHeader, list[Step]]:
  with path.open() as f:
    header = parse_header(f)
    steps = list(iter_steps(f, header))
  return header, steps


def load_estimate(
  path: Path,
  *,
  cov: Path | None = None,
  pose_frame: str = "world",
  body_frame: str | None = None,
  tangent_convention: TangentConvention = TangentConvention.RIGHT,
  tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
  gauge: Gauge = Gauge.SE3,
) -> tuple[SquareHeader, list[Step]]:
  """Load an estimate: SQUARE natively, or the bare-TUM escape hatches.

  A header-less file may be wide TUM (29 columns: pose + the 21
  lower-triangle covariance entries), plain TUM with a ``cov`` sidecar,
  or plain TUM (deterministic). Either way ``body_frame`` must declare
  what the missing header would have. Raises FormatError on
  undeclared or unrecognized input.
  """
  if not looks_like_tum(path):
    return load_square(path)

  if body_frame is None:
    raise FormatError(
      "estimate file is header-less (bare TUM) so its body frame is "
      "unknown; pass --est-body-frame <name>"
    )
  ncols = sniff_tum_columns(path)
  if ncols == 29:
    return load_tum_gaussian(
      path,
      pose_frame=pose_frame,
      body_frame=body_frame,
      tangent_convention=tangent_convention,
      tangent_order=tangent_order,
      gauge=gauge,
    )
  if ncols == 8 and cov is not None:
    return load_tum_with_sidecar(
      path,
      cov,
      pose_frame=pose_frame,
      body_frame=body_frame,
      tangent_convention=tangent_convention,
      tangent_order=tangent_order,
      gauge=gauge,
    )
  if ncols == 8:
    tum, steps = load_tum(path, pose_frame=pose_frame, body_frame=body_frame)
    header = tum.to_square()
    header.gauge = gauge
    return header, steps
  raise FormatError(
    f"header-less estimate has {ncols} columns; expected 8 (TUM, "
    "optionally with --cov sidecar) or 29 (TUM + 21 lower-triangle "
    "covariance entries)"
  )
