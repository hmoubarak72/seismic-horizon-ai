"""Interpreter seed picks: records, validation and CSV import/export.

Only picks with ``status == "approved"`` constrain the horizon engine. AI-proposed
picks enter as ``status="proposed"`` with ``source="ai:<model>"`` and must be approved by
the interpreter first (interpreter authority).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import getpass
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

PICK_FIELDS = [
    "pick_id", "inline", "crossline", "time_ms", "event", "phase", "confidence",
    "status", "source", "interpreter", "timestamp_utc", "label", "x", "y",
]


@dataclasses.dataclass
class SeedPick:
    inline: float
    crossline: float
    time_ms: float
    event: str = "target"
    phase: str = "peak"  # peak | trough | zero_crossing_up | zero_crossing_down
    confidence: float = 1.0
    status: str = "approved"  # approved | proposed | rejected
    source: str = "interpreter"
    interpreter: str = dataclasses.field(default_factory=lambda: _user())
    timestamp_utc: str = dataclasses.field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    label: str = ""
    x: float | None = None
    y: float | None = None
    pick_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover
        return "unknown"


class PickConflictError(ValueError):
    pass


@dataclasses.dataclass
class PickSet:
    picks: list[SeedPick] = dataclasses.field(default_factory=list)

    def add(self, pick: SeedPick) -> None:
        self.picks.append(pick)

    def extend(self, picks: Iterable[SeedPick]) -> None:
        self.picks.extend(picks)

    def approved(self, event: str | None = None) -> list[SeedPick]:
        return [p for p in self.picks if p.status == "approved" and (event is None or p.event == event)]

    def to_grid_constraints(self, geometry, event: str | None = None, conflict_tolerance_samples: float = 0.5):
        """Map approved picks to (il_idx, xl_idx, t_sample) arrays.

        Several picks in one bin are averaged if they agree within the tolerance,
        otherwise :class:`PickConflictError` lists the conflicting bins: the engine never
        decides silently between two contradicting interpreter picks.
        """
        picks = self.approved(event)
        if not picks:
            raise ValueError("No approved seed picks for this event.")
        rows: dict[tuple[int, int], list[float]] = {}
        for p in picks:
            i = int(round(geometry.il_index(p.inline)))
            j = int(round(geometry.xl_index(p.crossline)))
            if not (0 <= i < geometry.ilines.size and 0 <= j < geometry.xlines.size):
                raise ValueError(f"Pick {p.pick_id} at IL {p.inline} XL {p.crossline} lies outside the survey/ROI.")
            rows.setdefault((i, j), []).append(float(geometry.sample_index(p.time_ms)))
        conflicts = {k: v for k, v in rows.items() if np.ptp(v) > conflict_tolerance_samples}
        if conflicts:
            raise PickConflictError(f"Conflicting picks in {len(conflicts)} bins (index: samples): {dict(list(conflicts.items())[:5])}")
        keys = np.array(list(rows.keys()), dtype=np.int64)
        times = np.array([np.mean(v) for v in rows.values()])
        return keys[:, 0], keys[:, 1], times

    # ------------------------------------------------------------------- I/O
    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=PICK_FIELDS)
            w.writeheader()
            for p in self.picks:
                w.writerow(dataclasses.asdict(p))
        return path

    @classmethod
    def from_csv(cls, path: str | Path) -> "PickSet":
        out = cls()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                kw = {k: v for k, v in row.items() if k in PICK_FIELDS and v not in (None, "")}
                for k in ("inline", "crossline", "time_ms", "confidence", "x", "y"):
                    if k in kw:
                        kw[k] = float(kw[k])
                out.add(SeedPick(**kw))
        return out

    @classmethod
    def from_arrays(cls, inlines, crosslines, times_ms, **common) -> "PickSet":
        return cls([SeedPick(float(a), float(b), float(c), **common) for a, b, c in zip(inlines, crosslines, times_ms)])
