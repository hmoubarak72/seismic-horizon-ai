import json

import numpy as np
import pytest

from conftest import TARGET_TAU, random_seeds
from seismic_horizon_ai.attributes.instantaneous import dominant_period_samples
from seismic_horizon_ai.config import load_config
from seismic_horizon_ai.export.horizon_export import export_horizon_csv, export_horizon_xyz
from seismic_horizon_ai.horizon_engine.horizon_3d import HorizonExtractor3D
from seismic_horizon_ai.horizon_engine.seed_picks import PickConflictError, PickSet, SeedPick
from seismic_horizon_ai.quality_control.horizon_residuals import (
    compare_to_reference,
    cycle_skip_candidates,
    seed_misfit,
    spikes,
)
from seismic_horizon_ai.quality_control.phase_consistency import phase_consistency
from seismic_horizon_ai.quality_control.uncertainty import confidence_map
from seismic_horizon_ai.seismic_io.survey_geometry import SurveyGeometry


@pytest.fixture(scope="module")
def flat_result(flat_case):
    sv, sl = flat_case
    si, sj, st = random_seeds(sv)
    return HorizonExtractor3D(sv.seismic, sl).extract(si, sj, st)


def test_reference_metrics():
    ref = np.array([[100.0, 110.0], [np.nan, 130.0]])
    t = np.array([[105.0, 135.0], [120.0, np.nan]])
    m = compare_to_reference(t, ref)
    assert m.n_compared == 2 and m.coverage == pytest.approx(2 / 3)
    assert m.mae_ms == pytest.approx(15.0) and m.rms_ms == pytest.approx(np.sqrt((25 + 625) / 2))
    assert m.within_10ms == 0.5 and m.within_40ms == 1.0


def test_cycle_skip_and_spike_detection(flat_case, flat_result):
    sv, sl = flat_case
    per = dominant_period_samples(sv.seismic)
    clean = cycle_skip_candidates(flat_result, sl.p, sl.q, per)
    assert clean.mean() < 0.01
    import copy

    bad = copy.copy(flat_result)
    bad.t = flat_result.t.copy()
    bad.t[20:30, 20:30] += per  # a patch tracked one cycle too deep
    skips = cycle_skip_candidates(bad, sl.p, sl.q, per)
    assert skips[20, 25] and skips[29, 25] and not skips[25, 25]
    sp_t = flat_result.t.copy(); sp_t[10, 10] += 5
    s = spikes(sp_t)
    assert s[10, 10] and s.sum() == 1


def test_phase_consistency(flat_case, flat_result):
    sv, _ = flat_case
    good = phase_consistency(sv.seismic, flat_result.t)
    # the synthetic target reflector (tau=80) has negative reflectivity: a trough, phase ~ +-180 deg
    assert good["fraction_in_phase"] > 0.95 and abs(abs(good["circular_mean_deg"]) - 180) < 30
    per = dominant_period_samples(sv.seismic)
    shifted = flat_result.t + per / 2  # half a cycle off: wrong event
    bad = phase_consistency(sv.seismic, shifted, reference_phase=np.pi)
    assert bad["fraction_in_phase"] < 0.5


def test_confidence_reports_drivers(flat_result):
    cm = confidence_map(flat_result)
    assert set(cm.driver_names) >= {"slope_quality", "correlation", "pick_density", "fault_proximity"}
    s = cm.summary()
    assert 0 < s["mean_confidence"] <= 1
    assert abs(sum(s["limiting_driver_share"].values()) - 1) < 1e-9
    assert (seed_misfit(flat_result) == 0).all()


def test_export_csv_and_xyz(tmp_path, flat_case, flat_result):
    sv, _ = flat_case
    g = SurveyGeometry.regular(*sv.seismic.shape, dt_ms=4.0, il0=1000, xl0=2000, crs="EPSG:23031")
    cm = confidence_map(flat_result)
    p = export_horizon_csv(tmp_path / "h.csv", flat_result, g, confidence=cm.confidence)
    lines = p.read_text().splitlines()
    assert lines[0].startswith("inline,crossline,x,y,time_ms,confidence")
    assert len(lines) == 1 + np.isfinite(flat_result.t).sum()
    side = json.loads((tmp_path / "h.csv.provenance.json").read_text())
    assert side["export"]["crs"] == "EPSG:23031" and side["product"] == "horizon_3d"
    xyz = np.loadtxt(export_horizon_xyz(tmp_path / "h.xyz", flat_result, g))
    assert xyz.shape[1] == 3
    np.testing.assert_allclose(np.sort(xyz[:, 2]), np.sort(g.time_ms(flat_result.t[np.isfinite(flat_result.t)])), atol=1e-3)


def test_pickset_roundtrip_and_authority(tmp_path):
    g = SurveyGeometry.regular(10, 10, 100, dt_ms=4.0, il0=100, xl0=1)
    ps = PickSet([SeedPick(101, 2, 40.0), SeedPick(105, 7, 60.0, phase="trough")])
    ps.add(SeedPick(103, 3, 44.0, status="proposed", source="ai:waveform_similarity"))
    back = PickSet.from_csv(ps.to_csv(tmp_path / "p.csv"))
    assert len(back.picks) == 3 and back.picks[1].phase == "trough"
    i, j, t = back.to_grid_constraints(g)
    assert len(i) == 2  # the AI proposal is not a constraint until approved
    assert t.tolist() == [10.0, 15.0]
    back.add(SeedPick(101, 2, 52.0))
    with pytest.raises(PickConflictError):
        back.to_grid_constraints(g)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[horizon]\nmax_iterations = 3\n[horizon.correlation]\nstrides = [2, 4]\n")
    cfg = load_config(p)
    assert cfg.horizon.max_iterations == 3 and cfg.horizon.correlation.strides == (2, 4)
    p.write_text("[horizon]\nmax_iteration = 3\n")
    with pytest.raises(ValueError):
        load_config(p)
