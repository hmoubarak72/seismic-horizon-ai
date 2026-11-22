"""Complex-trace attributes: envelope (reflection strength) and instantaneous phase."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def analytic_trace(seismic: np.ndarray) -> np.ndarray:
    return hilbert(np.asarray(seismic, dtype=np.float64), axis=-1)


def envelope(seismic: np.ndarray) -> np.ndarray:
    return np.abs(analytic_trace(seismic)).astype(np.float32)


def instantaneous_phase(seismic: np.ndarray) -> np.ndarray:
    """Instantaneous phase in radians (-pi..pi). A symmetric peak has phase ~0, a trough ~pi."""
    return np.angle(analytic_trace(seismic)).astype(np.float32)


def dominant_period_samples(seismic: np.ndarray, max_traces: int = 2000, seed: int = 0) -> float:
    """Dominant period (samples) from the mean amplitude spectrum of a random trace subset."""
    data = np.asarray(seismic).reshape(-1, seismic.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], size=min(max_traces, data.shape[0]), replace=False)
    sub = data[np.sort(idx)].astype(np.float64)
    sub = sub[np.ptp(sub, axis=1) > 0]
    if sub.size == 0:
        return float("nan")
    spec = np.abs(np.fft.rfft(sub, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(seismic.shape[-1])
    spec[0] = 0
    f = freqs[int(np.argmax(spec))]
    return float(1.0 / f) if f > 0 else float("nan")
