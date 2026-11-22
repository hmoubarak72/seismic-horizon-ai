"""Survey geometry: inline/crossline axes, trace lookup and the bin-to-world transform."""

from __future__ import annotations

import dataclasses
import json
from functools import reduce
from math import gcd
from typing import Any

import numpy as np


class GeometryError(ValueError):
    """Raised when survey geometry cannot be established safely."""


def _regular_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Return a regular axis covering ``values`` with the greatest common step.

    Gaps (missing lines) are kept as part of the axis so they show up as missing
    traces instead of being silently compressed out.
    """
    uniq = np.unique(values)
    if uniq.size == 0:
        raise GeometryError(f"No {name} numbers found in trace headers.")
    if uniq.size == 1:
        return uniq.astype(np.int64)
    diffs = np.diff(uniq).astype(np.int64)
    step = int(reduce(gcd, diffs.tolist()))
    if step <= 0:
        raise GeometryError(f"Could not determine a positive {name} increment.")
    return np.arange(uniq[0], uniq[-1] + step, step, dtype=np.int64)


@dataclasses.dataclass
class SurveyGeometry:
    """Regular 3D post-stack survey geometry.

    Attributes
    ----------
    ilines, xlines : 1D int arrays of line numbers (regular, gaps included).
    samples_ms : 1D float array of sample times (or depths) in ms (or m).
    trace_index : (n_il, n_xl) int array; index of the SEG-Y trace for each bin, ``-1`` if missing.
    cdp_x, cdp_y : (n_il, n_xl) float arrays of scaled CDP coordinates (NaN where missing).
    affine : 2x3 matrix mapping ``[il_idx, xl_idx, 1]`` to ``[x, y]`` (least-squares fit), or None.
    affine_rms : RMS misfit of the affine fit in coordinate units.
    crs : coordinate reference system string if found (e.g. ``"EPSG:23031"``), else None.
    """

    ilines: np.ndarray
    xlines: np.ndarray
    samples_ms: np.ndarray
    trace_index: np.ndarray
    cdp_x: np.ndarray
    cdp_y: np.ndarray
    affine: np.ndarray | None = None
    affine_rms: float | None = None
    crs: str | None = None
    domain: str = "time"  # "time" or "depth"; SEG-Y does not encode it reliably
    measurement_system: str | None = None

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ilines.size, self.xlines.size, self.samples_ms.size)

    @property
    def dt_ms(self) -> float:
        return float(self.samples_ms[1] - self.samples_ms[0]) if self.samples_ms.size > 1 else float("nan")

    @property
    def live_mask(self) -> np.ndarray:
        return self.trace_index >= 0

    @property
    def il_step(self) -> int:
        return int(self.ilines[1] - self.ilines[0]) if self.ilines.size > 1 else 1

    @property
    def xl_step(self) -> int:
        return int(self.xlines[1] - self.xlines[0]) if self.xlines.size > 1 else 1

    def il_index(self, iline: float) -> float:
        return (iline - self.ilines[0]) / self.il_step

    def xl_index(self, xline: float) -> float:
        return (xline - self.xlines[0]) / self.xl_step

    def sample_index(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        return (np.asarray(time_ms, dtype=float) - self.samples_ms[0]) / self.dt_ms

    def time_ms(self, sample_index: float | np.ndarray) -> float | np.ndarray:
        return self.samples_ms[0] + np.asarray(sample_index, dtype=float) * self.dt_ms

    def world_xy(self, il_idx: np.ndarray | float, xl_idx: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Map (fractional) bin indices to world coordinates using the affine fit."""
        if self.affine is None:
            raise GeometryError("No valid bin-to-world transform; coordinates are missing or degenerate.")
        il_idx = np.asarray(il_idx, dtype=float)
        xl_idx = np.asarray(xl_idx, dtype=float)
        x = self.affine[0, 0] * il_idx + self.affine[0, 1] * xl_idx + self.affine[0, 2]
        y = self.affine[1, 0] * il_idx + self.affine[1, 1] * xl_idx + self.affine[1, 2]
        return x, y

    def bin_spacing(self) -> tuple[float, float] | None:
        """(inline-direction, crossline-direction) bin size in coordinate units."""
        if self.affine is None:
            return None
        return float(np.hypot(self.affine[0, 0], self.affine[1, 0])), float(np.hypot(self.affine[0, 1], self.affine[1, 1]))

    # ------------------------------------------------------------ serialisation
    def to_json(self) -> str:
        d: dict[str, Any] = {
            "ilines": self.ilines.tolist(),
            "xlines": self.xlines.tolist(),
            "samples_ms": self.samples_ms.tolist(),
            "affine": None if self.affine is None else self.affine.tolist(),
            "affine_rms": self.affine_rms,
            "crs": self.crs,
            "domain": self.domain,
            "measurement_system": self.measurement_system,
        }
        return json.dumps(d)

    # ----------------------------------------------------------- construction
    @classmethod
    def regular(cls, n_il: int, n_xl: int, n_t: int, dt_ms: float = 4.0, il0: int = 1, xl0: int = 1,
                bin_size: tuple[float, float] = (25.0, 25.0), origin: tuple[float, float] = (0.0, 0.0),
                crs: str | None = None) -> "SurveyGeometry":
        """A complete regular geometry (synthetic data, tests)."""
        ti = np.arange(n_il * n_xl, dtype=np.int64).reshape(n_il, n_xl)
        ii, jj = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing="ij")
        A = np.array([[bin_size[0], 0.0, origin[0]], [0.0, bin_size[1], origin[1]]])
        return cls(
            ilines=np.arange(il0, il0 + n_il), xlines=np.arange(xl0, xl0 + n_xl),
            samples_ms=np.arange(n_t) * dt_ms, trace_index=ti,
            cdp_x=origin[0] + ii * bin_size[0], cdp_y=origin[1] + jj * bin_size[1],
            affine=A, affine_rms=0.0, crs=crs,
        )

    @classmethod
    def from_headers(
        cls,
        il: np.ndarray,
        xl: np.ndarray,
        samples_ms: np.ndarray,
        cdp_x: np.ndarray | None = None,
        cdp_y: np.ndarray | None = None,
        crs: str | None = None,
        measurement_system: str | None = None,
    ) -> tuple["SurveyGeometry", np.ndarray]:
        """Build geometry from per-trace header arrays.

        Returns the geometry and an int array with the trace numbers of duplicate
        traces (traces whose (il, xl) bin was already occupied; the first one wins).
        """
        il = np.asarray(il, dtype=np.int64)
        xl = np.asarray(xl, dtype=np.int64)
        if il.shape != xl.shape or il.ndim != 1:
            raise GeometryError("Inline and crossline header arrays must be 1D and equal length.")
        if il.size == 0:
            raise GeometryError("SEG-Y file contains no traces.")
        if np.unique(il).size == 1 and np.unique(xl).size == 1 and il.size > 1:
            raise GeometryError(
                "All traces share one inline/crossline pair: the configured header bytes are probably wrong."
            )
        ilines = _regular_axis(il, "inline")
        xlines = _regular_axis(xl, "crossline")
        if ilines.size * xlines.size > 50 * il.size:
            raise GeometryError(
                f"Regular grid would have {ilines.size}x{xlines.size} bins for only {il.size} traces; "
                "header bytes are probably wrong or the survey is irregular (2D / rotated binning)."
            )
        il_step = int(ilines[1] - ilines[0]) if ilines.size > 1 else 1
        xl_step = int(xlines[1] - xlines[0]) if xlines.size > 1 else 1
        ii = (il - ilines[0]) // il_step
        jj = (xl - xlines[0]) // xl_step

        trace_index = np.full((ilines.size, xlines.size), -1, dtype=np.int64)
        flat = ii * xlines.size + jj
        # first occurrence wins; later ones are duplicates
        order = np.arange(il.size)
        _, first = np.unique(flat, return_index=True)
        is_first = np.zeros(il.size, dtype=bool)
        is_first[first] = True
        duplicates = order[~is_first]
        trace_index.ravel()[flat[first]] = first

        cx = np.full(trace_index.shape, np.nan)
        cy = np.full(trace_index.shape, np.nan)
        affine = None
        rms = None
        if cdp_x is not None and cdp_y is not None:
            cdp_x = np.asarray(cdp_x, dtype=float)
            cdp_y = np.asarray(cdp_y, dtype=float)
            cx.ravel()[flat[first]] = cdp_x[first]
            cy.ravel()[flat[first]] = cdp_y[first]
            affine, rms = _fit_affine(ii[first], jj[first], cdp_x[first], cdp_y[first])

        geom = cls(
            ilines=ilines,
            xlines=xlines,
            samples_ms=np.asarray(samples_ms, dtype=float),
            trace_index=trace_index,
            cdp_x=cx,
            cdp_y=cy,
            affine=affine,
            affine_rms=rms,
            crs=crs,
            measurement_system=measurement_system,
        )
        return geom, duplicates


def _fit_affine(ii: np.ndarray, jj: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Least-squares affine map from bin indices to world coordinates.

    Returns ``(None, None)`` when coordinates are absent (all zero) or degenerate.
    """
    if x.size < 3 or (np.allclose(x, 0) and np.allclose(y, 0)):
        return None, None
    G = np.column_stack([ii, jj, np.ones_like(ii)]).astype(float)
    if np.linalg.matrix_rank(G) < 3:
        return None, None
    cx, *_ = np.linalg.lstsq(G, x, rcond=None)
    cy, *_ = np.linalg.lstsq(G, y, rcond=None)
    A = np.vstack([cx, cy])
    if abs(np.linalg.det(A[:, :2])) < 1e-12:
        return None, None
    res = np.hypot(G @ cx - x, G @ cy - y)
    return A, float(np.sqrt(np.mean(res**2)))
