"""Conventional amplitude peak tracking: the baseline the least-squares engine is compared to.

Starting from one seed, the tracker visits traces in breadth-first order and, on each
trace, snaps to the extremum of the chosen polarity within ``search`` samples of the
mean time of already-tracked neighbours. Errors propagate trace to trace, which is
exactly the weakness the global least-squares method avoids.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _snap(trace: np.ndarray, t_guess: float, search: int, polarity: int) -> float | None:
    n = trace.size
    k0 = int(round(t_guess))
    lo, hi = max(1, k0 - search), min(n - 2, k0 + search)
    if lo > hi:
        return None
    seg = polarity * trace[lo : hi + 1]
    k = lo + int(np.argmax(seg))
    if k <= 0 or k >= n - 1:
        return None
    a, b, c = polarity * trace[k - 1], polarity * trace[k], polarity * trace[k + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den < 0 else 0.0
    return k + float(np.clip(off, -0.5, 0.5))


def peak_track(seismic: np.ndarray, seed: tuple[int, int, float], search: int = 3, polarity: int = 1,
               live: np.ndarray | None = None, min_amplitude: float = 0.0) -> np.ndarray:
    """Return (n_il, n_xl) tracked times (NaN where tracking stopped)."""
    n_il, n_xl, _ = seismic.shape
    live = np.ones((n_il, n_xl), bool) if live is None else live
    t = np.full((n_il, n_xl), np.nan)
    i0, j0, t0 = seed
    s = _snap(seismic[i0, j0], t0, search, polarity)
    if s is None:
        return t
    t[i0, j0] = s
    q = deque([(i0, j0)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n_il and 0 <= b < n_xl and live[a, b] and np.isnan(t[a, b]):
                nb = [t[a + u, b + v] for u, v in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= a + u < n_il and 0 <= b + v < n_xl and np.isfinite(t[a + u, b + v])]
                guess = float(np.mean(nb))
                s = _snap(seismic[a, b], guess, search, polarity)
                if s is not None and polarity * seismic[a, b, int(round(s))] >= min_amplitude:
                    t[a, b] = s
                    q.append((a, b))
    return t
