"""Integration tests: synthetic volumes with known horizons and faults."""

import numpy as np
import pytest

from conftest import TARGET_TAU, fault_seeds, random_seeds
from seismic_horizon_ai.attributes.fault_probability import FaultBarriers, barriers_from_polylines
from seismic_horizon_ai.attributes.structure_tensor import compute_slopes
from seismic_horizon_ai.config import HorizonConfig
from seismic_horizon_ai.horizon_engine.horizon_2d import line_misties, track_line
from seismic_horizon_ai.horizon_engine.horizon_3d import HorizonExtractor3D
from seismic_horizon_ai.horizon_engine.peak_tracking import peak_track
from seismic_horizon_ai.synthetic import make_synthetic


def polyline_barriers(sv):
    fl = sv.fault_line
    return barriers_from_polylines(*sv.fault_side.shape, [np.c_[fl[0], fl[1]]])


def test_unfaulted_horizon_recovered(flat_case):
    sv, sl = flat_case
    h = sv.horizon(TARGET_TAU)
    si, sj, st = random_seeds(sv)
    r = HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)
    err = r.t - h
    assert r.converged
    assert np.nanmean(np.abs(err)) < 0.15  # samples (0.6 ms at 4 ms)
    assert np.nanmax(np.abs(err)) < 1.0
    np.testing.assert_allclose(r.t[si, sj], st, atol=1e-9)  # interpreter picks honoured exactly
    assert not r.unconstrained.any()
    assert r.provenance.inputs["seismic"] and r.provenance.parameters["n_seeds"] == 12


def test_multigrid_correlations_improve_accuracy(flat_case):
    sv, sl = flat_case
    h = sv.horizon(TARGET_TAU)
    si, sj, st = random_seeds(sv, n=6, seed=3)
    cfg = HorizonConfig(); cfg.correlation.enabled = False
    no_corr = HorizonExtractor3D(sv.seismic, sl, cfg=cfg).extract(si, sj, st)
    with_corr = HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)
    assert np.nanmean(np.abs(with_corr.t - h)) < np.nanmean(np.abs(no_corr.t - h))
    assert with_corr.iterations[-1]["n_correlation_pairs"] > 100


def test_fault_barrier_prevents_smoothing_across_fault(fault_case):
    sv, sl = fault_case
    h = sv.horizon(TARGET_TAU)
    si, sj, st = fault_seeds(sv)
    bar = polyline_barriers(sv)
    near = bar.cell_mask()
    with_bar = HorizonExtractor3D(sv.seismic, sl, barriers=bar).extract(si, sj, st)
    without = HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)
    e_bar = np.abs(with_bar.t - h)
    e_no = np.abs(without.t - h)
    assert np.nanmean(e_bar) < 0.3
    assert np.nanmean(e_bar[near]) < 0.5 * np.nanmean(e_no[near])
    # the fault throw (7 samples) is preserved across the barrier
    crossing = bar.block_il
    jump = np.abs(with_bar.t[1:] - with_bar.t[:-1])[crossing]
    assert np.median(jump) > 5.0


def test_seedless_fault_block_is_not_bridged(fault_case):
    sv, sl = fault_case
    h = sv.horizon(TARGET_TAU)
    si, sj, st = random_seeds(sv, n=8, seed=5, mask=~sv.fault_side)
    r = HorizonExtractor3D(sv.seismic, sl, barriers=polyline_barriers(sv)).extract(si, sj, st)
    assert np.isnan(r.t[sv.fault_side]).all()
    assert r.unconstrained[sv.fault_side].all()
    assert np.nanmean(np.abs(r.t - h)) < 0.3
    assert any("without seed picks" in n for n in r.provenance.notes)


def test_fault_probability_hard_threshold_creates_barrier(fault_case):
    sv, sl = fault_case
    side = sv.fault_side
    edge_cells = FaultBarriers(side[:-1] != side[1:], side[:, :-1] != side[:, 1:]).cell_mask()
    fp = np.repeat(edge_cells[..., None].astype(np.float32), sv.seismic.shape[-1], axis=-1)
    si, sj, st = fault_seeds(sv, seed=2)
    r = HorizonExtractor3D(sv.seismic, sl, fault_probability=fp).extract(si, sj, st)
    assert r.iterations[-1]["n_probability_barrier_edges"] > 0
    # cells on the fault are isolated by the probability barrier and stay uninterpreted
    assert np.isnan(r.t[edge_cells]).mean() > 0.9
    assert np.nanmean(np.abs(r.t - sv.horizon(TARGET_TAU))) < 0.3


def test_deterministic(flat_case):
    sv, sl = flat_case
    si, sj, st = random_seeds(sv)
    a = HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)
    b = HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)
    np.testing.assert_array_equal(a.t, b.t)


def test_seed_validation(flat_case):
    sv, sl = flat_case
    live = np.ones(sv.seismic.shape[:2], bool); live[0, 0] = False
    with pytest.raises(ValueError):
        HorizonExtractor3D(sv.seismic, sl, live_mask=live).extract(np.array([0]), np.array([0]), np.array([50.0]))
    with pytest.raises(ValueError):
        HorizonExtractor3D(sv.seismic, sl).extract(np.array([1]), np.array([1]), np.array([1e6]))


def test_beats_peak_tracking_baseline_on_noisy_faulted_data():
    sv = make_synthetic(throw=7.0, noise=0.6, seed=21)
    sl = compute_slopes(sv.seismic)
    h = sv.horizon(TARGET_TAU)
    si, sj, st = fault_seeds(sv, seed=4)
    ls = HorizonExtractor3D(sv.seismic, sl, barriers=polyline_barriers(sv)).extract(si, sj, st)
    pk = peak_track(sv.seismic, (int(si[0]), int(sj[0]), float(st[0])), search=3)
    e_ls = np.abs(ls.t - h)
    e_pk = np.abs(np.where(np.isfinite(pk), pk, np.nan) - h)
    # peak tracking either stops early or jumps cycles; the global solution does neither
    assert np.nanmean(e_ls) < 0.5
    assert np.nanmean(e_ls) < np.nanmean(e_pk) or np.isfinite(pk).mean() < np.isfinite(ls.t).mean()


def test_2d_tracking_and_misties(fault_case):
    sv, _ = fault_case
    h = sv.horizon(TARGET_TAU)
    j = 25
    sec = sv.seismic[:, j, :]
    side = sv.fault_side[:, j]
    k = int(np.nonzero(side[:-1] != side[1:])[0][0])
    pos = np.array([5, sec.shape[0] - 6])
    line = track_line(sec, pos, h[pos, j], axis="crossline", index=j, fault_edges=[k])
    assert np.nanmean(np.abs(line.t - h[:, j])) < 0.3
    il_line = track_line(sv.seismic[10], np.array([3, 40]), h[10, [3, 40]], axis="inline", index=10)
    assert line_misties([il_line, line], tolerance=0.5) == []
    il_line.t[j] += 3.0
    assert len(line_misties([il_line, line], tolerance=0.5)) == 1
