"""Import of existing interpretations: XYZ horizons and fault sticks (e.g. OpendTect exports).

Imported interpretations are *reference* data (expert opinion), kept separate from
measured seismic and from the engine's own output.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .survey_geometry import SurveyGeometry


def world_to_index(geometry: SurveyGeometry, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the bin-to-world affine: world (x, y) -> fractional (il_idx, xl_idx)."""
    if geometry.affine is None:
        raise ValueError("Geometry has no bin-to-world transform.")
    A = np.vstack([geometry.affine, [0.0, 0.0, 1.0]])
    r = np.linalg.inv(A) @ np.vstack([np.asarray(x, float), np.asarray(y, float), np.ones(np.size(x))])
    return r[0], r[1]


def read_xyz_horizon(path: str | Path, geometry: SurveyGeometry, time_shift_ms: float = 0.0, max_offgrid: float = 0.25) -> np.ndarray:
    """Grid an ``x y time_ms`` horizon file onto the survey; returns fractional sample indices (NaN = absent).

    ``time_shift_ms`` is added to the file times (e.g. to reconcile a different time datum).
    Points further than ``max_offgrid`` bins from a bin centre are ignored.
    """
    d = np.loadtxt(path, usecols=(0, 1, 2))
    i, j = world_to_index(geometry, d[:, 0], d[:, 1])
    ii, jj = np.rint(i).astype(int), np.rint(j).astype(int)
    ok = (np.hypot(i - ii, j - jj) <= max_offgrid) & (ii >= 0) & (jj >= 0) & (ii < geometry.ilines.size) & (jj < geometry.xlines.size)
    out = np.full((geometry.ilines.size, geometry.xlines.size), np.nan)
    out[ii[ok], jj[ok]] = geometry.sample_index(d[ok, 2] + time_shift_ms)
    return out


@dataclasses.dataclass
class FaultSticks:
    """Fault sticks in grid coordinates: list of (n, 3) arrays (il_idx, xl_idx, sample_index)."""

    sticks: list[np.ndarray]
    name: str = "fault"

    def cutoff_polyline(self, horizon: np.ndarray, search: int = 3) -> np.ndarray:
        """Map-view fault trace where each stick crosses a horizon.

        For every stick, the horizon time is looked up near each stick node (nearest
        finite value within ``search`` bins, since horizons are usually blank at the
        fault itself) and the stick is intersected where ``t_stick - t_horizon`` changes sign.
        Returns (m, 2) fractional (il_idx, xl_idx) points ordered by stick.
        """
        n_il, n_xl = horizon.shape
        pts = []
        for s in self.sticks:
            th = []
            for i, j, _ in s:
                i0, j0 = int(round(i)), int(round(j))
                win = horizon[max(0, i0 - search): i0 + search + 1, max(0, j0 - search): j0 + search + 1]
                th.append(np.nanmedian(win) if np.isfinite(win).any() else np.nan)
            d = s[:, 2] - np.asarray(th)
            for k in range(len(s) - 1):
                if np.isfinite(d[k]) and np.isfinite(d[k + 1]) and d[k] * d[k + 1] <= 0 and d[k] != d[k + 1]:
                    w = d[k] / (d[k] - d[k + 1])
                    pts.append(s[k, :2] + w * (s[k + 1, :2] - s[k, :2]))
                    break
        return np.asarray(pts).reshape(-1, 2)


def read_fault_sticks(path: str | Path, geometry: SurveyGeometry, time_shift_ms: float = 0.0, name: str | None = None) -> FaultSticks:
    """Read ``x y time_ms stick_index node_index`` fault sticks (OpendTect export layout)."""
    d = np.loadtxt(path)
    i, j = world_to_index(geometry, d[:, 0], d[:, 1])
    t = geometry.sample_index(d[:, 2] + time_shift_ms)
    sticks = []
    for sid in np.unique(d[:, 3]):
        m = d[:, 3] == sid
        order = np.argsort(d[m, 4])
        sticks.append(np.c_[i[m][order], j[m][order], t[m][order]])
    return FaultSticks(sticks, name or Path(path).stem)
