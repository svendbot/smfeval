#!/usr/bin/env python3
"""Mechanical checks on the exporters/ directory layout (run in CI).

Every exporter directory must carry: a *.patch, an UPSTREAM file with a
parsable ``URL @ sha`` line, and a VALIDATION.md whose verdict block shows
a positive median NEES (a zero or missing NEES means the validation run is
degenerate or absent). This is also what makes the directory countable:
the script prints the exporter count (the adoption metric).
"""

import re
import sys
from pathlib import Path

EXPORTERS = Path(__file__).resolve().parent.parent / "exporters"
UPSTREAM_RE = re.compile(r"^https?://\S+ @ [0-9a-f]{7,40}$")
NEES_RE = re.compile(r"median NEES ([0-9.eE+-]+)")


def check(d: Path) -> list[str]:
  errs: list[str] = []
  if not list(d.glob("*.patch")):
    errs.append("no *.patch file")
  upstream = d / "UPSTREAM"
  if not upstream.is_file():
    errs.append("no UPSTREAM file")
  elif not any(
    UPSTREAM_RE.match(s.strip())
    for s in upstream.read_text().splitlines()
    if s.strip()
  ):
    errs.append("UPSTREAM has no parsable 'URL @ sha' line")
  validation = d / "VALIDATION.md"
  if not validation.is_file():
    errs.append("no VALIDATION.md")
  else:
    m = NEES_RE.search(validation.read_text())
    if m is None:
      errs.append("VALIDATION.md has no 'median NEES' verdict line")
    elif not float(m.group(1)) > 0:
      errs.append(f"VALIDATION.md median NEES is degenerate ({m.group(1)})")
  return errs


def main() -> int:
  dirs = sorted(
    p for p in EXPORTERS.iterdir() if p.is_dir() and not p.name.startswith(".")
  )
  failed = False
  for d in dirs:
    errs = check(d)
    if errs:
      failed = True
      for e in errs:
        print(f"FAIL {d.name}: {e}", file=sys.stderr)
    else:
      print(f"ok   {d.name}")
  print(f"{len(dirs)} exporter(s)")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
