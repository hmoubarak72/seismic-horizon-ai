"""Initial horizon surface from seed picks (fault-block aware).

Inside the convex hull of the picks of a fault block we use Clough-Tocher C1 cubic
Bezier patches on a Delaunay triangulation (scipy ``CloughTocher2DInterpolator``),
the triangle-patch equivalent of the Akima-derivative Bezier scheme in the Pyrus
workflow. Outside the hull a smoothed thin-plate radial basis function extrapolates,
and those cells are flagged ``extrapolated`` so they can never be presented as
interpreted. Each horizon domain (fault block) is interpolated only from its own
picks; values are never carried across a barrier.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import cKDTree


@dataclasses.dataclass
class InitialSurface:
    t: np.ndarray  # (n_il, n_xl) samples, NaN where the domain has no picks
    extrapolated: np.ndarray  # bool, outside the pick hull of its domain
    seed_distance: np.ndarray  # distance (in bins) to the nearest pick in the same domain
    method: np.ndarray  # 0 none, 1 clough-tocher, 2 rbf, 3 constant


def _interp_points(pts: np.ndarray, vals: np.ndarray, query: np.ndarray, rbf_smoothing: float):
    """Interpolate one domain; returns values and method codes."""
    out = np.full(query.shape[0], np.nan)
    method = np.zeros(query.shape[0], np.int8)
    upts, inv = np.unique(pts, axis=0, return_inverse=True)
    uvals = np.bincount(inv.ravel(), weights=vals) / np.bincount(inv.ravel())
    if upts.shape[0] < 3 or np.linalg.matrix_rank(np.c_[upts - upts.mean(0)]) < 2:
        if upts.shape[0] >= 2:
            # collinear picks: linear trend along the line, constant across it
            d = upts[-1] - upts[0]
            d = d / (np.linalg.norm(d) or 1.0)
            s = (upts - upts[0]) @ d
            sq = (query - upts[0]) @ d
            order = np.argsort(s)
            out[:] = np.interp(sq, s[order], uvals[order])
        else:
            out[:] = uvals[0]
        method[:] = 3
        return out, method
    ct = CloughTocher2DInterpolator(upts, uvals)
    v = ct(query)
    inside = np.isfinite(v)
    out[inside] = v[inside]
    method[inside] = 1
    if (~inside).any():
        rbf = RBFInterpolator(upts, uvals, kernel="thin_plate_spline", smoothing=rbf_smoothing, degree=1,
                              neighbors=min(64, upts.shape[0]) if upts.shape[0] > 64 else None)
        out[~inside] = rbf(query[~inside])
        method[~inside] = 2
    return out, method


def initial_surface(
    shape: tuple[int, int],
    seed_i: np.ndarray,
    seed_j: np.ndarray,
    seed_t: np.ndarray,
    labels: np.ndarray | None = None,
    rbf_smoothing: float = 1.0,
) -> InitialSurface:
    """Interpolate seeds to a full grid, independently per horizon domain (``labels``)."""
    n_il, n_xl = shape
    if labels is None:
        labels = np.zeros(shape, int)
    t = np.full(shape, np.nan)
    method = np.zeros(shape, np.int8)
    dist = np.full(shape, np.inf)
    seed_lab = labels[seed_i, seed_j]
    for lab in np.unique(seed_lab[seed_lab >= 0]):
        sel = seed_lab == lab
        cells = np.argwhere(labels == lab)
        pts = np.c_[seed_i[sel], seed_j[sel]].astype(float)
        vals, meth = _interp_points(pts, seed_t[sel].astype(float), cells.astype(float), rbf_smoothing)
        t[cells[:, 0], cells[:, 1]] = vals
        method[cells[:, 0], cells[:, 1]] = meth
        d, _ = cKDTree(pts).query(cells.astype(float))
        dist[cells[:, 0], cells[:, 1]] = d
    return InitialSurface(t=t, extrapolated=(method == 2) | (method == 3), seed_distance=dist, method=method)
