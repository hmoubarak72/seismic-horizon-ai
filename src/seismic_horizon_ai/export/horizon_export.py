"""Horizon export (CSV / XYZ) with QC flags and a provenance sidecar.

Uninterpreted cells (NaN) are never written as values. Every exported row carries
its confidence and flags so downstream users cannot mistake extrapolated or
low-confidence areas for interpreted surface.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..horizon_engine.horizon_3d import HorizonResult
from ..seismic_io.survey_geometry import SurveyGeometry


def export_horizon_csv(
    path: str | Path,
    result: HorizonResult,
    geometry: SurveyGeometry,
    confidence: np.ndarray | None = None,
    extra_flags: dict[str, np.ndarray] | None = None,
) -> Path:
    path = Path(path)
    t_ms = geometry.time_ms(result.t)
    have_xy = geometry.affine is not None
    ii, jj = np.nonzero(np.isfinite(result.t))
    if have_xy:
        x, y = geometry.world_xy(ii, jj)
    flags = {"extrapolated": result.extrapolated, "clipped": result.clipped, **(extra_flags or {})}
    fields = ["inline", "crossline", "x", "y", "time_ms", "confidence", "domain"] + list(flags)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for k, (i, j) in enumerate(zip(ii, jj)):
            row = [
                int(geometry.ilines[i]), int(geometry.xlines[j]),
                f"{x[k]:.2f}" if have_xy else "", f"{y[k]:.2f}" if have_xy else "",
                f"{t_ms[i, j]:.3f}",
                "" if confidence is None else f"{confidence[i, j]:.3f}",
                int(result.labels[i, j]),
            ] + [int(bool(flags[f][i, j])) for f in flags]
            w.writerow(row)
    sidecar = result.provenance.to_dict()
    sidecar["export"] = {"format": "csv", "crs": geometry.crs, "units": {"time_ms": "ms TWT", "x": "CRS units", "y": "CRS units"},
                         "n_rows": int(ii.size), "n_uninterpreted_live_cells": int(result.unconstrained.sum())}
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return path


def export_horizon_xyz(path: str | Path, result: HorizonResult, geometry: SurveyGeometry) -> Path:
    """Plain ``x y time_ms`` text (requires georeferencing)."""
    if geometry.affine is None:
        raise ValueError("XYZ export requires world coordinates; the survey has no valid CDP coordinates.")
    path = Path(path)
    ii, jj = np.nonzero(np.isfinite(result.t))
    x, y = geometry.world_xy(ii, jj)
    z = geometry.time_ms(result.t[ii, jj])
    np.savetxt(path, np.c_[x, y, z], fmt="%.2f %.2f %.3f", header=f"x y time_ms  crs={geometry.crs}")
    return path
