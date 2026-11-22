"""2D slope-based horizon tracking along inlines/crosslines and line-intersection QC.

The 2D tracker is the same least-squares engine applied to a single section
(an (n_trace, 1, n_t) "volume"), so 2D and 3D results are numerically consistent.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..attributes.fault_probability import FaultBarriers
from ..attributes.structure_tensor import SlopeVolume, compute_slopes_2d
from ..config import HorizonConfig
from .horizon_3d import HorizonExtractor3D, HorizonResult


@dataclasses.dataclass
class LineHorizon:
    axis: str  # "inline" (a fixed inline index, varying crossline) or "crossline"
    index: int  # fixed line index
    t: np.ndarray  # (n_trace,) fractional samples, NaN = not tracked
    result: HorizonResult


def track_line(
    section: np.ndarray,
    seed_pos: np.ndarray,
    seed_t: np.ndarray,
    axis: str,
    index: int,
    cfg: HorizonConfig | None = None,
    fault_edges: np.ndarray | None = None,
    live: np.ndarray | None = None,
    sigma_smooth: tuple[float, float] = (3.0, 2.0),
) -> LineHorizon:
    """Track a horizon along one section.

    Parameters
    ----------
    section : (n_trace, n_t) amplitudes along the line.
    seed_pos, seed_t : trace positions along the line and seed times (samples).
    fault_edges : optional trace indices k such that the edge k -> k+1 crosses a fault.
    """
    section = np.asarray(section, dtype=np.float32)
    n, n_t = section.shape
    p, plan = compute_slopes_2d(section, sigma_smooth=sigma_smooth)
    slopes = SlopeVolume(p=p[:, None, :], q=np.zeros((n, 1, n_t), np.float32), planarity=plan[:, None, :],
                         params={"mode": "2d", "sigma_smooth": sigma_smooth})
    bar = FaultBarriers.empty(n, 1)
    if fault_edges is not None and len(fault_edges):
        bar.block_il[np.asarray(fault_edges, int), 0] = True
        bar.source = "interpreter"
    ex = HorizonExtractor3D(section[:, None, :], slopes, cfg=cfg, barriers=bar,
                            live_mask=None if live is None else np.asarray(live, bool)[:, None])
    res = ex.extract(np.asarray(seed_pos, int), np.zeros(len(seed_pos), int), np.asarray(seed_t, float))
    return LineHorizon(axis=axis, index=index, t=res.t[:, 0], result=res)


def line_misties(lines: list[LineHorizon], tolerance: float = 1.0) -> list[dict]:
    """Compare tracked times where inline and crossline interpretations intersect.

    Returns one record per intersection with ``|difference| > tolerance`` samples.
    """
    out = []
    ils = [l for l in lines if l.axis == "inline"]
    xls = [l for l in lines if l.axis == "crossline"]
    for a in ils:
        for b in xls:
            # inline a (fixed il index a.index, varies xl) meets crossline b (fixed xl b.index) at (a.index, b.index)
            if b.index >= a.t.size or a.index >= b.t.size:
                continue
            ta, tb = a.t[b.index], b.t[a.index]
            if np.isfinite(ta) and np.isfinite(tb) and abs(ta - tb) > tolerance:
                out.append({"inline_index": a.index, "crossline_index": b.index, "t_inline": float(ta), "t_crossline": float(tb), "difference": float(ta - tb)})
    return out
