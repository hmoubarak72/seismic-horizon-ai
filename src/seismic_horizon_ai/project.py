"""Interpretation projects: one file that holds a whole piece of interpretation work.

A project (``*.shproj``, JSON) records which seismic volume is interpreted, every seed pick
with its full provenance, the fault sticks and reference horizon in use, the area and
display settings, and (optionally) the last extracted horizon. Bulky arrays (extracted
horizon, confidence, reference grid, fault sticks, fault cut-offs) go into a sidecar
``<name>.shproj.npz`` next to the project file.

Design rules:
* Measured data is never copied into the project: the project only *points* to the HDF5
  volume (and the SEG-Y it came from). Paths are stored both absolute and relative to the
  project file, so a project folder can be moved together with its data.
* Saving is atomic (write to a temporary file, then replace), so a crash while saving never
  leaves a half-written project.
* Loading refuses a project written by a newer, incompatible format version.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .horizon_engine.seed_picks import PICK_FIELDS, PickSet, SeedPick
from .seismic_io.interpretation_io import FaultSticks

PROJECT_SUFFIX = ".shproj"
FORMAT = "seismic_horizon_ai.project"
FORMAT_VERSION = 1


class ProjectError(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclasses.dataclass
class FaultEntry:
    """A fault loaded from a sticks file; ``sticks`` are in full-survey grid coordinates."""

    name: str
    source_path: str = ""
    time_shift_ms: float = 0.0
    sticks: list[np.ndarray] = dataclasses.field(default_factory=list)

    def to_fault_sticks(self) -> FaultSticks:
        return FaultSticks([np.asarray(s, float) for s in self.sticks], self.name)


@dataclasses.dataclass
class HorizonState:
    """Last extracted horizon on the full survey grid (time in ms, NaN = not interpreted)."""

    time_ms: np.ndarray
    confidence: np.ndarray | None = None
    cuts: list[np.ndarray] = dataclasses.field(default_factory=list)  # (m, 2) full-grid (il_idx, xl_idx)
    summary: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Project:
    name: str = "Untitled"
    path: Path | None = None
    volume_h5: str = ""
    source_segy: str = ""
    picks: PickSet = dataclasses.field(default_factory=PickSet)
    faults: list[FaultEntry] = dataclasses.field(default_factory=list)
    reference_path: str = ""
    reference_shift_ms: float = 0.0
    reference: np.ndarray | None = None  # fractional sample index grid
    horizon: HorizonState | None = None
    settings: dict = dataclasses.field(default_factory=dict)
    notes: str = ""
    created_utc: str = dataclasses.field(default_factory=_now)
    modified_utc: str = ""
    app_version: str = __version__

    # ------------------------------------------------------------------ paths
    @property
    def sidecar(self) -> Path | None:
        return None if self.path is None else self.path.with_name(self.path.name + ".npz")

    def _rel(self, p: str) -> str:
        if not p or self.path is None:
            return ""
        try:
            return os.path.relpath(Path(p).resolve(), self.path.parent.resolve())
        except ValueError:  # different drive on Windows
            return ""

    def resolve(self, absolute: str, relative: str = "") -> str:
        """Best existing location of a referenced file: relative to the project first, then absolute."""
        if relative and self.path is not None:
            cand = (self.path.parent / relative).resolve()
            if cand.exists():
                return str(cand)
        return absolute

    # ------------------------------------------------------------------ save
    def save(self, path: str | Path | None = None) -> Path:
        if path is not None:
            path = Path(path)
            if path.suffix.lower() != PROJECT_SUFFIX:
                path = path.with_name(path.name + PROJECT_SUFFIX)
            self.path = path
        if self.path is None:
            raise ProjectError("No project file name given.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.modified_utc = _now()
        self.app_version = __version__

        arrays: dict[str, np.ndarray] = {}
        faults_meta = []
        for k, f in enumerate(self.faults):
            for s_i, s in enumerate(f.sticks):
                arrays[f"fault{k}_stick{s_i}"] = np.asarray(s, float)
            faults_meta.append({"name": f.name, "source_path": f.source_path, "source_rel": self._rel(f.source_path),
                                "time_shift_ms": f.time_shift_ms, "n_sticks": len(f.sticks)})
        if self.reference is not None:
            arrays["reference"] = np.asarray(self.reference, np.float32)
        horizon_meta = None
        if self.horizon is not None:
            arrays["horizon_time_ms"] = np.asarray(self.horizon.time_ms, np.float32)
            if self.horizon.confidence is not None:
                arrays["horizon_confidence"] = np.asarray(self.horizon.confidence, np.float32)
            for c_i, c in enumerate(self.horizon.cuts):
                arrays[f"cut{c_i}"] = np.asarray(c, float)
            horizon_meta = {"n_cuts": len(self.horizon.cuts), "has_confidence": self.horizon.confidence is not None,
                            "summary": self.horizon.summary}

        doc: dict[str, Any] = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "app_version": self.app_version,
            "name": self.name, "created_utc": self.created_utc, "modified_utc": self.modified_utc,
            "notes": self.notes,
            "volume": {"h5": self.volume_h5, "h5_rel": self._rel(self.volume_h5),
                       "source_segy": self.source_segy, "source_segy_rel": self._rel(self.source_segy)},
            "picks": [dataclasses.asdict(p) for p in self.picks.picks],
            "faults": faults_meta,
            "reference": None if self.reference is None and not self.reference_path else
            {"path": self.reference_path, "rel": self._rel(self.reference_path), "time_shift_ms": self.reference_shift_ms,
             "stored": self.reference is not None},
            "horizon": horizon_meta,
            "settings": self.settings,
            "sidecar": self.sidecar.name if arrays else None,
        }
        _atomic_write_text(self.path, json.dumps(doc, indent=2, default=_json_default))
        side = self.sidecar
        if arrays:
            tmp = side.with_name(side.name + ".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, side)
        elif side.exists():
            side.unlink()
        return self.path

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Could not read project {path}: {exc}") from exc
        if doc.get("format") != FORMAT:
            raise ProjectError(f"{path.name} is not a Seismic Horizon Interpreter project.")
        if int(doc.get("format_version", 0)) > FORMAT_VERSION:
            raise ProjectError(f"{path.name} was written by a newer version of the app (format {doc['format_version']}); please update.")
        p = cls(name=doc.get("name", path.stem), path=path, notes=doc.get("notes", ""),
                created_utc=doc.get("created_utc", ""), modified_utc=doc.get("modified_utc", ""),
                app_version=doc.get("app_version", ""), settings=doc.get("settings") or {})
        vol = doc.get("volume") or {}
        p.volume_h5 = p.resolve(vol.get("h5", ""), vol.get("h5_rel", ""))
        p.source_segy = p.resolve(vol.get("source_segy", ""), vol.get("source_segy_rel", ""))
        for row in doc.get("picks", []):
            p.picks.add(SeedPick(**{k: v for k, v in row.items() if k in PICK_FIELDS}))

        arrays: dict[str, np.ndarray] = {}
        side = p.sidecar
        if doc.get("sidecar"):
            if not side.exists():
                raise ProjectError(f"The project's data file {side.name} is missing next to {path.name}.")
            with np.load(side, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        for k, f in enumerate(doc.get("faults", [])):
            sticks = [arrays[f"fault{k}_stick{s}"] for s in range(int(f.get("n_sticks", 0))) if f"fault{k}_stick{s}" in arrays]
            p.faults.append(FaultEntry(f.get("name", f"fault{k}"), p.resolve(f.get("source_path", ""), f.get("source_rel", "")),
                                       float(f.get("time_shift_ms", 0.0)), sticks))
        ref = doc.get("reference")
        if ref:
            p.reference_path = p.resolve(ref.get("path", ""), ref.get("rel", ""))
            p.reference_shift_ms = float(ref.get("time_shift_ms", 0.0))
            if "reference" in arrays:
                p.reference = arrays["reference"].astype(float)
        hz = doc.get("horizon")
        if hz and "horizon_time_ms" in arrays:
            p.horizon = HorizonState(
                arrays["horizon_time_ms"].astype(float),
                arrays["horizon_confidence"].astype(float) if "horizon_confidence" in arrays else None,
                [arrays[f"cut{c}"] for c in range(int(hz.get("n_cuts", 0))) if f"cut{c}" in arrays],
                hz.get("summary") or {})
        return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def export_grid_csv(path: str | Path, time_ms: np.ndarray, geometry, confidence: np.ndarray | None = None) -> Path:
    """Export a full-grid horizon (e.g. one reloaded from a project) as CSV; NaN cells are skipped."""
    import csv

    path = Path(path)
    ii, jj = np.nonzero(np.isfinite(time_ms))
    have_xy = geometry.affine is not None
    if have_xy:
        x, y = geometry.world_xy(ii, jj)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["inline", "crossline", "x", "y", "time_ms", "confidence"])
        for k, (i, j) in enumerate(zip(ii, jj)):
            w.writerow([int(geometry.ilines[i]), int(geometry.xlines[j]),
                        f"{x[k]:.2f}" if have_xy else "", f"{y[k]:.2f}" if have_xy else "",
                        f"{time_ms[i, j]:.3f}",
                        "" if confidence is None or not np.isfinite(confidence[i, j]) else f"{confidence[i, j]:.3f}"])
    return path
