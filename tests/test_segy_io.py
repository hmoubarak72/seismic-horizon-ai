import json

import numpy as np
import pytest

from seismic_horizon_ai.preprocessing.roi_manager import ROI
from seismic_horizon_ai.preprocessing.seismic_qc import SeismicQCError, run_segy_qc
from seismic_horizon_ai.provenance import sha256_file
from seismic_horizon_ai.seismic_io.hdf5_manager import SeismicVolume, convert_segy_to_hdf5
from seismic_horizon_ai.seismic_io.segy_reader import read_cube, scan_segy, write_segy


@pytest.fixture()
def cube():
    rng = np.random.default_rng(3)
    c = rng.normal(size=(8, 6, 50)).astype(np.float32)
    c[2, 3] = 0.0  # dead trace
    return c


def test_scan_and_qc_findings(tmp_path, cube):
    p = write_segy(tmp_path / "s.sgy", cube, dt_ms=4.0, il0=100, xl0=20, il_step=2,
                   origin_xy=(450000.0, 6780000.0), bin_size=(12.5, 25.0),
                   skip_bins={(0, 0), (7, 5)}, duplicate_bins={(4, 4)},
                   text_lines={1: "SYNTHETIC TEST", 2: "CRS EPSG:23031 ED50 UTM 31N"})
    scan = scan_segy(p)
    assert scan.n_traces == 8 * 6 - 2 + 1
    assert scan.sample_interval_us == 4000
    assert scan.crs == "EPSG:23031"
    s, g, rep, dead = run_segy_qc(p)
    assert rep.passed
    codes = {f.code for f in rep.findings}
    assert {"geometry.missing_traces", "geometry.duplicate_traces", "amplitude.dead_traces"} <= codes
    assert "geometry.crs_missing" not in codes
    assert g.shape == (8, 6, 50)
    assert g.ilines[0] == 100 and g.il_step == 2
    assert dead[2, 3] and dead.sum() == 1
    assert not g.live_mask[0, 0]
    di, dx = g.bin_spacing()
    assert di == pytest.approx(12.5, abs=0.01) and dx == pytest.approx(25.0, abs=0.01)
    assert "PASSED" in rep.to_markdown()


def test_read_cube_matches(tmp_path, cube):
    p = write_segy(tmp_path / "s.sgy", cube)
    _, g, _, _ = run_segy_qc(p)
    np.testing.assert_allclose(read_cube(p, g), cube)
    np.testing.assert_allclose(read_cube(p, g, slice(1, 3), slice(2, 5), slice(10, 20)), cube[1:3, 2:5, 10:20])


def test_hdf5_roundtrip_and_provenance(tmp_path, cube):
    p = write_segy(tmp_path / "s.sgy", cube, skip_bins={(1, 1)})
    out, rep = convert_segy_to_hdf5(p, tmp_path / "v.h5")
    with SeismicVolume(out) as v:
        assert v.shape == cube.shape
        live = v.live_mask
        assert not live[1, 1] and not live[2, 3]  # missing and dead are masked
        np.testing.assert_allclose(v.load()[live], cube[live])
        assert v.provenance["inputs"]["s.sgy"] == sha256_file(p)
        assert v.qc_report["passed"]
        assert v.data.chunks is not None


def test_roi_conversion(tmp_path, cube):
    p = write_segy(tmp_path / "s.sgy", cube, il0=10, xl0=1)
    out, _ = convert_segy_to_hdf5(p, tmp_path / "roi.h5", roi=ROI((12, 14), (2, 3), (20.0, 60.0)))
    with SeismicVolume(out) as v:
        assert v.shape == (3, 2, 11)
        np.testing.assert_allclose(v.load(), cube[2:5, 1:3, 5:16])
        assert v.geometry.ilines.tolist() == [12, 13, 14]


def test_qc_gate_blocks_bad_file(tmp_path):
    cube = np.zeros((4, 4, 20), np.float32)  # entirely dead
    p = write_segy(tmp_path / "dead.sgy", cube)
    _, _, rep, _ = run_segy_qc(p)
    assert not rep.passed
    with pytest.raises(SeismicQCError):
        convert_segy_to_hdf5(p, tmp_path / "dead.h5")
    data = json.loads(rep.to_json())
    assert data["passed"] is False
