"""Fault evidence and fault barriers for the horizon engine.

Two sources of fault information are kept strictly separate:

* **Interpreted fault cut-offs** (map-view polylines approved by the interpreter) become
  *hard barriers*: edges of the horizon grid that no slope, smoothing or correlation
  equation may cross.
* **Fault probability** volumes (here a simple planarity-based discontinuity proxy;
  later a validated 3D CNN) only *down-weight* equations, except where the probability
  reaches the configured hard threshold.

Barriers are stored as two boolean edge arrays:
``block_il[i, j]`` blocks the edge (i, j)-(i+1, j) and ``block_xl[i, j]`` blocks (i, j)-(i, j+1).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage


@dataclasses.dataclass
class FaultBarriers:
    block_il: np.ndarray  # (n_il-1, n_xl)
    block_xl: np.ndarray  # (n_il, n_xl-1)
    source: str = "interpreter"

    @classmethod
    def empty(cls, n_il: int, n_xl: int) -> "FaultBarriers":
        return cls(np.zeros((max(n_il - 1, 0), n_xl), bool), np.zeros((n_il, max(n_xl - 1, 0)), bool), "none")

    def union(self, other: "FaultBarriers") -> "FaultBarriers":
        return FaultBarriers(self.block_il | other.block_il, self.block_xl | other.block_xl, f"{self.source}+{other.source}")

    @property
    def n_blocked(self) -> int:
        return int(self.block_il.sum() + self.block_xl.sum())

    def cell_mask(self) -> np.ndarray:
        """(n_il, n_xl) mask of cells touching a blocked edge (for display and QC)."""
        n_il, n_xl = self.block_xl.shape[0], self.block_il.shape[1]
        m = np.zeros((n_il, n_xl), bool)
        m[:-1] |= self.block_il
        m[1:] |= self.block_il
        m[:, :-1] |= self.block_xl
        m[:, 1:] |= self.block_xl
        return m


def _segments_intersect(p1, p2, q1, q2) -> np.ndarray:
    """Vectorised proper/touching intersection test of segment p1p2 with segments q1q2 (arrays)."""
    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    d1 = orient(q1, q2, p1)
    d2 = orient(q1, q2, p2)
    d3 = orient(p1, p2, q1)
    d4 = orient(p1, p2, q2)
    return (d1 * d2 <= 0) & (d3 * d4 <= 0) & ~((d1 == 0) & (d2 == 0))


def barriers_from_polylines(n_il: int, n_xl: int, polylines: list[np.ndarray]) -> FaultBarriers:
    """Rasterise interpreted map-view fault traces (in fractional (il_idx, xl_idx)) into blocked edges.

    An edge between two neighbouring cell centres is blocked when the straight segment
    joining them intersects any fault polyline segment.
    """
    fb = FaultBarriers.empty(n_il, n_xl)
    fb.source = "interpreter"
    ii_il, jj_il = np.meshgrid(np.arange(n_il - 1), np.arange(n_xl), indexing="ij")
    ii_xl, jj_xl = np.meshgrid(np.arange(n_il), np.arange(n_xl - 1), indexing="ij")
    for line in polylines:
        line = np.asarray(line, dtype=float)
        for a, b in zip(line[:-1], line[1:]):
            lo = np.floor(np.minimum(a, b)) - 1
            hi = np.ceil(np.maximum(a, b)) + 1
            for (I, J, di, dj, arr) in ((ii_il, jj_il, 1, 0, fb.block_il), (ii_xl, jj_xl, 0, 1, fb.block_xl)):
                sel = (I >= lo[0]) & (I <= hi[0]) & (J >= lo[1]) & (J <= hi[1])
                if not sel.any():
                    continue
                q1 = np.stack([I[sel], J[sel]], -1).astype(float)
                q2 = q1 + np.array([di, dj], float)
                hit = _segments_intersect(np.asarray(a), np.asarray(b), q1, q2)
                arr[sel] |= hit
    return fb


def barriers_from_fault_mask(fault_cells: np.ndarray) -> FaultBarriers:
    """Block every edge touching a cell flagged as fault (map-view boolean mask)."""
    f = np.asarray(fault_cells, bool)
    return FaultBarriers(f[:-1] | f[1:], f[:, :-1] | f[:, 1:], "mask")


def discontinuity_proxy(planarity: np.ndarray, sigma_t: float = 4.0, low: float = 0.2, high: float = 0.7) -> np.ndarray:
    """Heuristic fault-probability proxy from structure-tensor planarity.

    ``1 - planarity`` is smoothed along time (faults are vertically extended, noise is
    not) and mapped linearly from [low, high] to [0, 1]. This is a *derived attribute*,
    not a validated fault model: it also responds to noise, dim zones and chaotic
    facies, so by default it only down-weights equations (see HorizonConfig).
    """
    d = 1.0 - np.asarray(planarity, dtype=np.float32)
    d = ndimage.gaussian_filter1d(d, sigma_t, axis=-1, mode="nearest")
    return np.clip((d - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)
