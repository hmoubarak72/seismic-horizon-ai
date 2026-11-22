"""Multigrid trace correlations via smooth dynamic time warping (DTW).

Local slopes alone accumulate small errors over long distances and can be fooled by
noise, so, following Wu & Fomel (2018), traces on coarse grids (strides 4, 8, 16, ...)
are correlated directly to add long-range constraints ``t[b] - t[a] = shift``.

Here each correlation is computed in a window centred on the current horizon
estimate in both traces. The DTW alignment (strain limited to one sample per sample)
is evaluated with forward + backward accumulation so the optimal lag at the window
centre is found without backtracking, then refined to sub-sample precision with a
parabola. A normalised cross-correlation after alignment gives the pair confidence.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class CorrelationPairs:
    a: np.ndarray  # flat cell index of the first trace
    b: np.ndarray  # flat cell index of the second trace
    shift: np.ndarray  # t[b] - t[a], samples
    weight: np.ndarray
    ncc: np.ndarray
    stride: np.ndarray
    n_candidates: int = 0
    n_rejected_low_corr: int = 0
    n_rejected_at_lag_limit: int = 0
    n_rejected_barrier: int = 0

    @classmethod
    def empty(cls) -> "CorrelationPairs":
        z = np.zeros(0)
        return cls(z.astype(np.int64), z.astype(np.int64), z, z, z, z.astype(np.int64))

    def as_tuple(self):
        return self.a, self.b, self.shift, self.weight


def _accumulate(e: np.ndarray, reverse: bool) -> np.ndarray:
    """DTW accumulation along axis 1 of e[P, n, L] with lag steps of at most 1."""
    P, n, L = e.shape
    D = np.empty_like(e)
    order = range(n - 1, -1, -1) if reverse else range(n)
    prev = None
    for i in order:
        if prev is None:
            D[:, i] = e[:, i]
        else:
            left = np.concatenate([np.full((P, 1), np.inf), prev[:, :-1]], axis=1)
            right = np.concatenate([prev[:, 1:], np.full((P, 1), np.inf)], axis=1)
            D[:, i] = e[:, i] + np.minimum(np.minimum(left, prev), right)
        prev = D[:, i]
    return D


def dtw_center_lag(f: np.ndarray, g: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched smooth-DTW lag at the window centre.

    Parameters
    ----------
    f : (P, n) reference windows.
    g : (P, n + 2*max_lag) windows of the other traces, padded by ``max_lag`` each side,
        so ``f[:, i]`` may align with ``g[:, i + max_lag + l]`` for ``l`` in [-max_lag, max_lag].

    Returns ``(lag, ncc, at_limit)``: sub-sample lag at the centre, normalised
    cross-correlation after integer alignment, and whether the lag hit the limit.
    """
    P, n = f.shape
    L = 2 * max_lag + 1

    def norm(x):
        x = x - x.mean(axis=1, keepdims=True)
        s = np.sqrt((x**2).mean(axis=1, keepdims=True))
        return x / np.where(s > 0, s, 1.0)

    fn = norm(f)
    gn = norm(g)
    lags = np.arange(-max_lag, max_lag + 1)
    gi = np.arange(n)[:, None] + max_lag + lags[None, :]  # (n, L)
    e = (fn[:, :, None] - gn[:, gi]) ** 2  # (P, n, L)
    Df = _accumulate(e, False)
    Db = _accumulate(e, True)
    c = n // 2
    cost = Df[:, c] + Db[:, c] - e[:, c]  # (P, L)
    k = np.argmin(cost, axis=1)
    at_limit = (k == 0) | (k == L - 1)
    # sub-sample refinement: maximise a centre-tapered NCC around the DTW lag
    taper = np.exp(-0.5 * ((np.arange(n) - c) / max(n / 4.0, 1.0)) ** 2)
    pos_base = np.arange(n)[None, :] + max_lag
    rows = np.arange(P)[:, None]

    def tapered_ncc(shift: np.ndarray) -> np.ndarray:
        pos = np.clip(pos_base + shift[:, None], 0, g.shape[1] - 1)
        k0 = np.floor(pos).astype(int)
        k1 = np.minimum(k0 + 1, g.shape[1] - 1)
        w = pos - k0
        gs = (1 - w) * gn[rows, k0] + w * gn[rows, k1]
        a_ = fn * taper
        b_ = gs * taper
        a_ = a_ - a_.mean(axis=1, keepdims=True)
        b_ = b_ - b_.mean(axis=1, keepdims=True)
        den = np.sqrt((a_**2).sum(axis=1) * (b_**2).sum(axis=1))
        return (a_ * b_).sum(axis=1) / np.where(den > 0, den, 1.0)

    lag = lags[k].astype(float)
    for h in (1.0, 0.25):
        cm, c0, cp = tapered_ncc(lag - h), tapered_ncc(lag), tapered_ncc(lag + h)
        denom = cm - 2 * c0 + cp
        ok = denom < -1e-12
        frac = np.where(ok, 0.5 * (cm - cp) / np.where(ok, denom, -1.0), 0.0)
        lag = lag + h * np.clip(frac, -1.0, 1.0)
    lag = np.clip(lag, -max_lag, max_lag)
    ncc = tapered_ncc(lag)
    return lag, ncc, at_limit


def _windows(traces: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    """Extract windows centred on fractional ``centers`` (linear interpolation, edge-clamped).

    Fractional centring keeps the correlation constraints continuous in the horizon
    estimate, which avoids limit cycles when the outer iteration moves a horizon by
    less than one sample.
    """
    n_t = traces.shape[1]
    pos = np.clip(centers[:, None] + np.arange(-half, half + 1)[None, :], 0, n_t - 1)
    k0 = np.floor(pos).astype(int)
    k1 = np.minimum(k0 + 1, n_t - 1)
    w = pos - k0
    rows = np.arange(traces.shape[0])[:, None]
    return (1 - w) * traces[rows, k0] + w * traces[rows, k1]


def multigrid_correlations(
    seismic: np.ndarray,
    horizon: np.ndarray,
    labels: np.ndarray,
    block_il: np.ndarray,
    block_xl: np.ndarray,
    strides: tuple[int, ...] = (4, 8, 16),
    half_window: int = 20,
    max_lag: int = 6,
    min_correlation: float = 0.6,
    weight: float = 1.0,
) -> CorrelationPairs:
    """Build coarse-grid correlation constraints around the current horizon.

    Pairs are axis-aligned neighbours on each coarse grid. A pair is discarded if
    either cell is unsolved, the cells lie in different horizon domains, the straight
    path between them crosses a barrier edge, the DTW lag hits the limit (possible
    cycle skip) or the aligned correlation is below ``min_correlation``.
    """
    n_il, n_xl, n_t = seismic.shape
    out_a, out_b, out_s, out_w, out_c, out_st = [], [], [], [], [], []
    n_cand = n_low = n_lim = n_bar = 0
    # cumulative barrier counts for fast "path crosses barrier" tests
    cum_il = np.concatenate([np.zeros((1, n_xl), int), np.cumsum(block_il, axis=0)], axis=0)  # (n_il, n_xl)
    cum_xl = np.concatenate([np.zeros((n_il, 1), int), np.cumsum(block_xl, axis=1)], axis=1)
    for s in strides:
        if s <= 0:
            continue
        ii, jj = np.meshgrid(np.arange(0, n_il, s), np.arange(0, n_xl, s), indexing="ij")
        for di, dj in ((s, 0), (0, s)):
            ia, ja = ii.ravel(), jj.ravel()
            ib, jb = ia + di, ja + dj
            ok = (ib < n_il) & (jb < n_xl)
            ia, ja, ib, jb = ia[ok], ja[ok], ib[ok], jb[ok]
            ta, tb = horizon[ia, ja], horizon[ib, jb]
            ok = np.isfinite(ta) & np.isfinite(tb) & (labels[ia, ja] >= 0) & (labels[ia, ja] == labels[ib, jb])
            n_cand += int(ok.sum())
            if di:
                crosses = (cum_il[ib, ja] - cum_il[ia, ja]) > 0
            else:
                crosses = (cum_xl[ia, jb] - cum_xl[ia, ja]) > 0
            n_bar += int((ok & crosses).sum())
            ok &= ~crosses
            if not ok.any():
                continue
            ia, ja, ib, jb, ta, tb = ia[ok], ja[ok], ib[ok], jb[ok], ta[ok], tb[ok]
            f = _windows(np.asarray(seismic[ia, ja], dtype=np.float64), ta, half_window)
            g = _windows(np.asarray(seismic[ib, jb], dtype=np.float64), tb, half_window + max_lag)
            lag, ncc, at_lim = dtw_center_lag(f, g, max_lag)
            good = ~at_lim & (ncc >= min_correlation)
            n_lim += int(at_lim.sum())
            n_low += int((~at_lim & (ncc < min_correlation)).sum())
            shift = (tb + lag) - ta
            out_a.append(ia[good] * n_xl + ja[good])
            out_b.append(ib[good] * n_xl + jb[good])
            out_s.append(shift[good])
            out_w.append(weight * ncc[good] ** 2)
            out_c.append(ncc[good])
            out_st.append(np.full(int(good.sum()), s))
    if not out_a:
        cp = CorrelationPairs.empty()
    else:
        cp = CorrelationPairs(
            np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_s),
            np.concatenate(out_w), np.concatenate(out_c), np.concatenate(out_st),
        )
    cp.n_candidates, cp.n_rejected_low_corr, cp.n_rejected_at_lag_limit, cp.n_rejected_barrier = n_cand, n_low, n_lim, n_bar
    return cp
