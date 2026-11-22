"""Weighted sparse least-squares solve for a horizon surface t(il, xl).

Equations (one row each, all multiplied by sqrt(weight)):

* slope, inline edges     ``t[i+1, j] - t[i, j] = p_edge[i, j]``
* slope, crossline edges  ``t[i, j+1] - t[i, j] = q_edge[i, j]``
* correlation pairs       ``t[b] - t[a] = shift_ab`` (multigrid, long range)
* soft seeds              ``t[k] = t_seed``

Hard seeds are eliminated from the unknowns, so interpreter picks are honoured
exactly. A weight of exactly zero is a *barrier*: the connected components of the
remaining edge graph define horizon domains. A domain that contains no seed has no
information tying it to the target event; it is returned as NaN and reported
(the solver never bridges a barrier to fill it).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla


@dataclasses.dataclass
class LinearSystemInfo:
    n_unknowns: int
    n_equations: int
    n_components: int
    n_seeded_components: int
    n_unconstrained_cells: int
    solver: str
    solve_seconds: float
    cg_iterations: int | None = None
    residual_rms: dict[str, float] = dataclasses.field(default_factory=dict)


def edge_components(n_il: int, n_xl: int, w_il: np.ndarray, w_xl: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Connected-component labels (n_il, n_xl) over non-zero-weight edges; -1 for inactive cells."""
    N = n_il * n_xl
    idx = np.arange(N).reshape(n_il, n_xl)
    rows, cols = [], []
    m = (w_il > 0) & active[:-1] & active[1:]
    rows.append(idx[:-1][m]); cols.append(idx[1:][m])
    m = (w_xl > 0) & active[:, :-1] & active[:, 1:]
    rows.append(idx[:, :-1][m]); cols.append(idx[:, 1:][m])
    r = np.concatenate(rows); c = np.concatenate(cols)
    G = sparse.coo_matrix((np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    _, labels = csgraph.connected_components(G, directed=False)
    labels = labels.reshape(n_il, n_xl)
    labels[~active] = -1
    return labels


def solve_horizon_lsq(
    p_edge: np.ndarray,
    w_il: np.ndarray,
    q_edge: np.ndarray,
    w_xl: np.ndarray,
    seed_i: np.ndarray,
    seed_j: np.ndarray,
    seed_t: np.ndarray,
    active: np.ndarray,
    corr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    seed_mode: str = "hard",
    soft_seed_weight: float = 100.0,
    solver: str = "auto",
    x0: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, LinearSystemInfo]:
    """Solve for t. Returns ``(t, labels, info)``; ``t`` is NaN in unconstrained domains.

    ``corr`` is ``(a_flat, b_flat, shift, weight)`` with flat cell indices.
    """
    n_il, n_xl = active.shape
    N = n_il * n_xl
    idx = np.arange(N).reshape(n_il, n_xl)
    w_il = np.where(active[:-1] & active[1:], w_il, 0.0)
    w_xl = np.where(active[:, :-1] & active[:, 1:], w_xl, 0.0)
    if labels is None:
        labels = edge_components(n_il, n_xl, w_il, w_xl, active)

    seed_flat = idx[seed_i, seed_j]
    seeded_labels = np.unique(labels.ravel()[seed_flat])
    seeded_labels = seeded_labels[seeded_labels >= 0]
    solve_mask = np.isin(labels, seeded_labels) & active

    # --- assemble rows
    rows_a, rows_b, rhs, wts, kinds = [], [], [], [], []

    def add(a, b, r, w, kind):
        keep = w > 0
        rows_a.append(a[keep]); rows_b.append(b[keep]); rhs.append(r[keep]); wts.append(w[keep]); kinds.append(np.full(int(keep.sum()), kind))

    add(idx[:-1].ravel(), idx[1:].ravel(), p_edge.ravel(), w_il.ravel(), 0)
    add(idx[:, :-1].ravel(), idx[:, 1:].ravel(), q_edge.ravel(), w_xl.ravel(), 1)
    if corr is not None and len(corr[0]):
        a, b, s, w = corr
        same = labels.ravel()[a] == labels.ravel()[b]
        add(a[same], b[same], s[same], w[same], 2)
    a_all = np.concatenate(rows_a); b_all = np.concatenate(rows_b)
    r_all = np.concatenate(rhs); w_all = np.concatenate(wts); k_all = np.concatenate(kinds)
    # keep only equations inside solved domains
    keep = solve_mask.ravel()[a_all] & solve_mask.ravel()[b_all]
    a_all, b_all, r_all, w_all, k_all = a_all[keep], b_all[keep], r_all[keep], w_all[keep], k_all[keep]
    sw = np.sqrt(w_all)
    m = a_all.size
    eq = np.arange(m)
    A = sparse.csr_matrix(
        (np.concatenate([-sw, sw]), (np.concatenate([eq, eq]), np.concatenate([a_all, b_all]))), shape=(m, N)
    )
    bvec = sw * r_all

    t_full = np.full(N, np.nan)
    fixed = np.zeros(N, bool)
    if seed_mode == "hard":
        fixed[seed_flat] = True
        t_full[seed_flat] = seed_t
    elif seed_mode == "soft":
        ws = np.sqrt(soft_seed_weight)
        S = sparse.csr_matrix((np.full(seed_flat.size, ws), (np.arange(seed_flat.size), seed_flat)), shape=(seed_flat.size, N))
        A = sparse.vstack([A, S]).tocsr()
        bvec = np.concatenate([bvec, ws * seed_t])
        k_all = np.concatenate([k_all, np.full(seed_flat.size, 3)])
    else:
        raise ValueError(f"seed_mode must be 'hard' or 'soft', got {seed_mode!r}")

    free = solve_mask.ravel() & ~fixed
    free_idx = np.nonzero(free)[0]
    A_f = A[:, free_idx]
    rhs_vec = bvec.copy()
    if fixed.any():
        rhs_vec -= A[:, fixed] @ t_full[fixed]
    M = (A_f.T @ A_f).tocsc()
    g = A_f.T @ rhs_vec

    t0 = time.perf_counter()
    cg_it = None
    used = solver
    if solver == "auto":
        used = "direct" if free_idx.size <= 300_000 else "cg"
    if free_idx.size == 0:
        x = np.zeros(0)
    elif used == "direct":
        x = spla.spsolve(M, g)
    elif used == "cg":
        d = M.diagonal()
        Minv = spla.LinearOperator(M.shape, matvec=lambda v: v / np.where(d > 0, d, 1.0))
        guess = None
        if x0 is not None:
            guess = np.nan_to_num(x0.ravel()[free_idx], nan=float(np.nanmean(seed_t)))
        it = [0]
        x, status = spla.cg(M, g, x0=guess, rtol=1e-10, maxiter=5000, M=Minv, callback=lambda _: it.__setitem__(0, it[0] + 1))
        cg_it = it[0]
        if status != 0:
            raise RuntimeError(f"CG did not converge (status {status})")
    else:
        raise ValueError(f"Unknown solver {solver!r}")
    dt = time.perf_counter() - t0
    t_full[free_idx] = x
    t_full[~solve_mask.ravel()] = np.nan

    res = A @ np.nan_to_num(t_full) - bvec
    rms = {}
    for k, name in ((0, "slope_inline"), (1, "slope_crossline"), (2, "correlation"), (3, "soft_seed")):
        sel = k_all == k
        if sel.any():
            # residual in samples (undo sqrt-weight scaling)
            wk = (np.concatenate([w_all, np.full(A.shape[0] - w_all.size, soft_seed_weight)]))[sel]
            rms[name] = float(np.sqrt(np.sum(res[sel] ** 2) / max(np.sum(wk), 1e-12)))
    n_comp = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    info = LinearSystemInfo(
        n_unknowns=int(free_idx.size),
        n_equations=int(A.shape[0]),
        n_components=n_comp,
        n_seeded_components=int(seeded_labels.size),
        n_unconstrained_cells=int((active & ~solve_mask).sum()),
        solver=used,
        solve_seconds=dt,
        cg_iterations=cg_it,
        residual_rms=rms,
    )
    return t_full.reshape(n_il, n_xl), labels, info
