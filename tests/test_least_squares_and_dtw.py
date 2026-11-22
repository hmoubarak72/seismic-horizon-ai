import numpy as np
import pytest

from conftest import TARGET_TAU
from seismic_horizon_ai.horizon_engine.least_squares import solve_horizon_lsq
from seismic_horizon_ai.horizon_engine.multigrid_dtw import _windows, dtw_center_lag, multigrid_correlations


def plane(n_il=20, n_xl=15):
    i, j = np.meshgrid(np.arange(n_il, dtype=float), np.arange(n_xl, dtype=float), indexing="ij")
    return 5 + 0.3 * i - 0.2 * j + 0.01 * i * j


def edges(t):
    return t[1:] - t[:-1], t[:, 1:] - t[:, :-1]


def test_exact_recovery_from_consistent_slopes():
    t = plane()
    p, q = edges(t)
    act = np.ones(t.shape, bool)
    sol, labels, info = solve_horizon_lsq(p, np.ones_like(p), q, np.ones_like(q), np.array([3]), np.array([4]), np.array([t[3, 4]]), act)
    np.testing.assert_allclose(sol, t, atol=1e-9)
    assert info.n_seeded_components == 1 and info.n_unconstrained_cells == 0


def test_hard_seeds_are_honoured_exactly():
    t = plane()
    p, q = edges(t)
    p = p + 0.05  # biased slopes
    act = np.ones(t.shape, bool)
    si, sj = np.array([0, 10, 19]), np.array([0, 7, 14])
    st = t[si, sj]
    sol, _, _ = solve_horizon_lsq(p, np.ones_like(p), q, np.ones_like(q), si, sj, st, act)
    np.testing.assert_allclose(sol[si, sj], st, atol=1e-12)


def test_barrier_domain_without_seed_is_left_uninterpreted():
    t = plane()
    p, q = edges(t)
    w_il = np.ones_like(p)
    w_il[9, :] = 0.0  # barrier between inline 9 and 10 across the whole survey
    act = np.ones(t.shape, bool)
    sol, labels, info = solve_horizon_lsq(p, w_il, q, np.ones_like(q), np.array([2]), np.array([2]), np.array([t[2, 2]]), act)
    assert np.isnan(sol[10:]).all()
    np.testing.assert_allclose(sol[:10], t[:10], atol=1e-9)
    assert info.n_components == 2 and info.n_unconstrained_cells == 10 * 15


def test_cg_matches_direct_and_soft_close_to_hard():
    rng = np.random.default_rng(0)
    t = plane(30, 25)
    p, q = edges(t)
    p = p + rng.normal(0, 0.02, p.shape); q = q + rng.normal(0, 0.02, q.shape)
    act = np.ones(t.shape, bool)
    si, sj = np.array([5, 20]), np.array([5, 20]); st = t[si, sj]
    d, _, _ = solve_horizon_lsq(p, np.ones_like(p), q, np.ones_like(q), si, sj, st, act, solver="direct")
    c, _, info = solve_horizon_lsq(p, np.ones_like(p), q, np.ones_like(q), si, sj, st, act, solver="cg")
    np.testing.assert_allclose(c, d, atol=1e-6)
    assert info.cg_iterations > 0
    s, _, _ = solve_horizon_lsq(p, np.ones_like(p), q, np.ones_like(q), si, sj, st, act, seed_mode="soft", soft_seed_weight=1e4)
    assert np.max(np.abs(s - d)) < 0.05


def test_correlation_constraints_are_used():
    t = plane()
    p, q = edges(t)
    act = np.ones(t.shape, bool)
    # zero-weight slopes in a band, bridged only by a correlation pair -> must follow the pair shift
    w_il = np.full_like(p, 1.0)
    a = np.array([0 * 15 + 0]); b = np.array([19 * 15 + 0])
    sol, _, info = solve_horizon_lsq(p, w_il, q, np.ones_like(q), np.array([0]), np.array([0]), np.array([t[0, 0]]), act,
                                     corr=(a, b, np.array([t[19, 0] - t[0, 0]]), np.array([1.0])))
    assert "correlation" in info.residual_rms and info.residual_rms["correlation"] < 1e-8


@pytest.mark.parametrize("shift", [0.0, 0.3, -1.4, 2.0, 4.7])
def test_dtw_recovers_known_shift(shift):
    x = np.arange(200.0)
    tr = np.sin(2 * np.pi * x / 11) * np.exp(-(((x - 100) / 30) ** 2))
    trb = np.interp(x - shift, x, tr)
    f = _windows(tr[None], np.array([100.0]), 20)
    g = _windows(trb[None], np.array([100.0]), 26)
    lag, ncc, lim = dtw_center_lag(f, g, 6)
    assert abs(lag[0] - shift) < 0.05 and ncc[0] > 0.99 and not lim[0]


def test_dtw_flags_lag_limit():
    x = np.arange(200.0)
    tr = np.exp(-(((x - 100) / 3) ** 2))  # single isolated pulse
    trb = np.interp(x - 9, x, tr)
    f = _windows(tr[None], np.array([100.0]), 20)
    g = _windows(trb[None], np.array([100.0]), 23)
    _, _, lim = dtw_center_lag(f, g, 3)
    assert lim[0]


def test_multigrid_pairs_accuracy_and_barriers(fault_case):
    sv, _ = fault_case
    h = sv.horizon(TARGET_TAU)
    n_il, n_xl = h.shape
    labels = np.zeros(h.shape, int)
    block_il = sv.fault_side[:-1] != sv.fault_side[1:]
    block_xl = sv.fault_side[:, :-1] != sv.fault_side[:, 1:]
    cp = multigrid_correlations(sv.seismic, h, labels, block_il, block_xl, strides=(4, 8))
    assert cp.n_rejected_barrier > 0
    true = h.ravel()[cp.b] - h.ravel()[cp.a]
    assert np.median(np.abs(cp.shift - true)) < 0.1
    side = sv.fault_side.ravel()
    assert (side[cp.a] == side[cp.b]).all()  # no accepted pair straddles the fault
