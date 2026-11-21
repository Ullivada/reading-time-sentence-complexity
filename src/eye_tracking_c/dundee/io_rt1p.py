"""Parse Dundee RT_dataset1P (time-ordered) fixation files."""

from __future__ import annotations

import re
from pathlib import Path

# Filename: sa01ma1p.dat -> subject 'a', text_id 1
RT1P_NAME = re.compile(r"^s([a-j])(\d{2})ma1p\.dat$", re.IGNORECASE)


def parse_rt1p_filename(name: str) -> tuple[str, int] | None:
    m = RT1P_NAME.match(name)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def load_rt1p_events(path: Path) -> list[tuple[int, int]]:
    """
    Return list of (WNUM, FDUR) in file order (time order for one subject×text).

    Skips *Blink, non-positive WNUM, non-positive FDUR.
    """
    events: list[tuple[int, int]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        header = True
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if header:
                header = False
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            word = parts[0]
            if word.startswith("*"):
                continue
            try:
                wnum = int(parts[6])
                fdur = int(parts[7])
            except ValueError:
                continue
            if wnum <= 0 or fdur <= 0:
                continue
            events.append((wnum, fdur))
    return events
