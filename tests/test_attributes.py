import numpy as np

from conftest import TARGET_TAU
from seismic_horizon_ai.attributes.fault_probability import (
    FaultBarriers,
    barriers_from_fault_mask,
    barriers_from_polylines,
    discontinuity_proxy,
)
from seismic_horizon_ai.attributes.instantaneous import dominant_period_samples
from seismic_horizon_ai.attributes.structure_tensor import compute_slopes, compute_slopes_2d
from seismic_horizon_ai.config import StructureTensorConfig
from seismic_horizon_ai.horizon_engine.horizon_3d import sample_along


def plane_wave(p, q, shape=(30, 25, 80), period=10.0):
    i, j, t = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    return np.cos(2 * np.pi * (t - p * i - q * j) / period).astype(np.float32)


def test_plane_wave_slopes_exact():
    sl = compute_slopes(plane_wave(0.4, -0.25))
    core = (slice(8, -8), slice(8, -8), slice(10, -10))
    assert np.median(np.abs(sl.p[core] - 0.4)) < 0.01
    assert np.median(np.abs(sl.q[core] + 0.25)) < 0.01
    assert np.median(sl.planarity[core]) > 0.95
    dip, az = sl.dip_azimuth()
    assert abs(np.median(dip[core]) - np.hypot(0.4, 0.25)) < 0.01


def test_synthetic_slopes_along_horizon(flat_case):
    sv, sl = flat_case
    h = sv.horizon(TARGET_TAU)
    pt, qt = sv.true_slopes(TARGET_TAU)
    assert np.median(np.abs(sample_along(sl.p, h) - pt)) < 0.05
    assert np.median(np.abs(sample_along(sl.q, h) - qt)) < 0.05


def test_chunked_equals_unchunked(flat_case):
    sv, sl = flat_case
    small = compute_slopes(sv.seismic, StructureTensorConfig(chunk_inlines=7))
    np.testing.assert_allclose(small.p, sl.p, atol=1e-4)
    np.testing.assert_allclose(small.planarity, sl.planarity, atol=1e-4)


def test_2d_slopes_match_section():
    sec = plane_wave(0.6, 0.0)[:, 5, :]
    p, plan = compute_slopes_2d(sec)
    assert np.median(np.abs(p[8:-8, 10:-10] - 0.6)) < 0.01
    assert np.median(plan[8:-8, 10:-10]) > 0.95


def test_dead_traces_have_zero_confidence(flat_case):
    sv, _ = flat_case
    live = np.ones(sv.seismic.shape[:2], bool)
    live[5, 5] = False
    sl = compute_slopes(sv.seismic[:20], live_mask=live[:20])
    assert sl.planarity[5, 5].max() == 0


def test_discontinuity_proxy_highlights_fault(fault_case):
    sv, sl = fault_case
    fp = discontinuity_proxy(sl.planarity)
    h = sv.horizon(TARGET_TAU)
    side = sv.fault_side
    on_fault = FaultBarriers(side[:-1] != side[1:], side[:, :-1] != side[:, 1:]).cell_mask()
    f_h = sample_along(fp, h)
    assert np.nanmean(f_h[on_fault]) > 2 * np.nanmean(f_h[~on_fault]) + 1e-3


def test_polyline_barrier_separates_blocks(fault_case):
    sv, _ = fault_case
    fl = sv.fault_line
    b = barriers_from_polylines(*sv.fault_side.shape, [np.c_[fl[0], fl[1]]])
    # every inline edge whose ends lie on different sides of the fault must be blocked
    crossing = sv.fault_side[:-1] != sv.fault_side[1:]
    assert b.block_il[crossing].all()
    assert b.n_blocked < 3 * crossing.sum() + 3 * sv.fault_side.shape[1]


def test_dominant_period(flat_case):
    sv, _ = flat_case
    per = dominant_period_samples(sv.seismic)
    assert 8 < per < 16  # Ricker peak 0.09 cycles/sample -> ~11 samples
