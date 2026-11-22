"""Interpretation confidence with explicit drivers (never a bare number).

Each driver is a 0..1 score per cell (1 = supports confidence):

* ``slope_quality``      mean incident edge weight (planarity, fault down-weighting)
* ``correlation``        local density of accepted multigrid correlation pairs (saturating)
* ``pick_density``       distance to the nearest seed in the same fault block
* ``fault_proximity``    distance to the nearest barrier edge
* ``phase``              1 where the horizon is in phase with the event, else 0.5
* ``not_extrapolated``   0.5 outside the pick hull of the block, else 1

Combined confidence is the geometric mean of the drivers; the *limiting driver*
(the lowest score) is reported per cell so the interpreter knows why a zone is weak.
This is a heuristic indicator to steer review, not a calibrated probability.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

from ..horizon_engine.horizon_3d import HorizonResult


@dataclasses.dataclass
class ConfidenceMap:
    confidence: np.ndarray
    drivers: dict[str, np.ndarray]
    limiting_driver: np.ndarray  # index into driver_names, -1 where uninterpreted
    driver_names: list[str]

    def summary(self) -> dict:
        ok = np.isfinite(self.confidence)
        out = {
            "mean_confidence": float(np.nanmean(self.confidence)) if ok.any() else float("nan"),
            "fraction_reliable_ge_0.7": float(np.mean(self.confidence[ok] >= 0.7)) if ok.any() else 0.0,
            "limiting_driver_share": {},
        }
        for k, name in enumerate(self.driver_names):
            out["limiting_driver_share"][name] = float(np.mean(self.limiting_driver[ok] == k)) if ok.any() else 0.0
        return out


def confidence_map(
    result: HorizonResult,
    pick_scale: float = 30.0,
    fault_scale: float = 3.0,
    corr_sigma: float = 4.0,
    off_phase: np.ndarray | None = None,
) -> ConfidenceMap:
    t = result.t
    ok = np.isfinite(t)
    drivers: dict[str, np.ndarray] = {}
    drivers["slope_quality"] = np.clip(result.slope_support, 0, 1)
    # correlation pairs live on coarse grids: spread their support spatially before scoring
    dens = ndimage.gaussian_filter(result.correlation_support, corr_sigma, mode="nearest")
    ref = np.percentile(dens[ok], 75) if ok.any() and np.any(dens[ok] > 0) else 1.0
    drivers["correlation"] = 0.5 + 0.5 * np.clip(dens / max(ref, 1e-12), 0, 1)
    d = np.where(np.isfinite(result.seed_distance), result.seed_distance, np.inf)
    drivers["pick_density"] = np.exp(-d / pick_scale)
    bc = result.barriers.cell_mask()
    if bc.any():
        fd = ndimage.distance_transform_edt(~bc)
        drivers["fault_proximity"] = 1 - 0.7 * np.exp(-fd / fault_scale)
    else:
        drivers["fault_proximity"] = np.ones_like(t)
    drivers["phase"] = np.where(off_phase, 0.5, 1.0) if off_phase is not None else np.ones_like(t)
    drivers["not_extrapolated"] = np.where(result.extrapolated, 0.5, 1.0)
    names = list(drivers)
    stack = np.stack([np.clip(drivers[n], 1e-6, 1) for n in names])
    conf = np.exp(np.mean(np.log(stack), axis=0))
    conf = np.where(ok, conf, np.nan)
    lim = np.where(ok, np.argmin(stack, axis=0), -1)
    return ConfidenceMap(confidence=conf, drivers=drivers, limiting_driver=lim, driver_names=names)
