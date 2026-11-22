"""Horizon QC: slope misfit, cycle-skip candidates, spikes, seed honouring and reference metrics."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from scipy import ndimage

from ..horizon_engine.horizon_3d import HorizonResult, sample_along


def slope_misfit(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge misfit (samples) between surface differences and slopes sampled on the surface."""
    P, Q = sample_along(p, t), sample_along(q, t)
    r_il = (t[1:] - t[:-1]) - 0.5 * (P[1:] + P[:-1])
    r_xl = (t[:, 1:] - t[:, :-1]) - 0.5 * (Q[:, 1:] + Q[:, :-1])
    return r_il, r_xl


def cycle_skip_candidates(result: HorizonResult, p: np.ndarray, q: np.ndarray, period_samples: float, fraction: float = 0.4) -> np.ndarray:
    """Cells where the surface jumps by more than ``fraction`` of a period relative to the local slope.

    Barrier edges are excluded: a jump across an interpreted fault is fault throw, not a cycle skip.
    """
    r_il, r_xl = slope_misfit(result.t, p, q)
    thr = fraction * period_samples
    bad_il = (np.abs(np.nan_to_num(r_il)) > thr) & ~result.barriers.block_il
    bad_xl = (np.abs(np.nan_to_num(r_xl)) > thr) & ~result.barriers.block_xl
    m = np.zeros(result.t.shape, bool)
    m[:-1] |= bad_il; m[1:] |= bad_il; m[:, :-1] |= bad_xl; m[:, 1:] |= bad_xl
    return m


def spikes(t: np.ndarray, threshold: float = 2.0, size: int = 3, barrier_cells: np.ndarray | None = None) -> np.ndarray:
    """Cells deviating more than ``threshold`` samples from their 3x3 median (excluding fault-adjacent cells)."""
    filled = np.where(np.isfinite(t), t, np.nanmedian(t) if np.isfinite(t).any() else 0.0)
    med = ndimage.median_filter(filled, size=size, mode="nearest")
    m = np.isfinite(t) & (np.abs(t - med) > threshold)
    if barrier_cells is not None:
        m &= ~barrier_cells
    return m


@dataclasses.dataclass
class ReferenceMetrics:
    """Agreement with a reference interpretation (an expert pick, not ground truth)."""

    n_compared: int
    coverage: float  # fraction of reference cells that have an extracted value
    mae_ms: float
    rms_ms: float
    bias_ms: float
    within_10ms: float
    within_20ms: float
    within_40ms: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_to_reference(t_ms: np.ndarray, ref_ms: np.ndarray) -> ReferenceMetrics:
    ref_ok = np.isfinite(ref_ms)
    both = ref_ok & np.isfinite(t_ms)
    d = (t_ms - ref_ms)[both]
    n = int(both.sum())
    if n == 0:
        nan = float("nan")
        return ReferenceMetrics(0, 0.0, nan, nan, nan, nan, nan, nan)
    ad = np.abs(d)
    return ReferenceMetrics(
        n_compared=n,
        coverage=n / max(int(ref_ok.sum()), 1),
        mae_ms=float(ad.mean()),
        rms_ms=float(np.sqrt(np.mean(d**2))),
        bias_ms=float(d.mean()),
        within_10ms=float(np.mean(ad <= 10)),
        within_20ms=float(np.mean(ad <= 20)),
        within_40ms=float(np.mean(ad <= 40)),
    )


def seed_misfit(result: HorizonResult) -> np.ndarray:
    si, sj, st = result.seeds
    return result.t[si, sj] - st
