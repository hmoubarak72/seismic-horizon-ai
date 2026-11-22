"""Reflection-phase consistency of a horizon (is it on the same event everywhere?)."""

from __future__ import annotations

import numpy as np

from ..attributes.instantaneous import instantaneous_phase
from ..horizon_engine.horizon_3d import sample_along


def phase_along_horizon(seismic: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Instantaneous phase (radians) sampled on the horizon (unwrapped-safe via complex interpolation)."""
    ph = instantaneous_phase(seismic)
    c = sample_along(np.cos(ph), t) + 1j * sample_along(np.sin(ph), t)
    return np.angle(c)


def phase_consistency(seismic: np.ndarray, t: np.ndarray, reference_phase: float | None = None, tolerance_deg: float = 45.0) -> dict:
    """Circular statistics of horizon phase.

    ``reference_phase`` defaults to the circular mean (e.g. ~0 rad for a peak pick).
    Returns the circular mean, circular standard deviation and the fraction of cells
    within ``tolerance_deg`` of the reference, plus a per-cell off-phase mask.
    """
    phi = phase_along_horizon(seismic, t)
    ok = np.isfinite(phi)
    z = np.exp(1j * phi[ok])
    R = np.abs(z.mean()) if z.size else 0.0
    mean = float(np.angle(z.mean())) if z.size else float("nan")
    ref = mean if reference_phase is None else reference_phase
    dev = np.degrees(np.abs(np.angle(np.exp(1j * (phi - ref)))))
    off = ok & (dev > tolerance_deg)
    return {
        "circular_mean_deg": float(np.degrees(mean)),
        "circular_std_deg": float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12))))) if z.size else float("nan"),
        "fraction_in_phase": float(1 - off.sum() / max(ok.sum(), 1)),
        "off_phase_mask": off,
    }
