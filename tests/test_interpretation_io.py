import numpy as np

from seismic_horizon_ai.seismic_io.interpretation_io import read_fault_sticks, read_xyz_horizon, world_to_index
from seismic_horizon_ai.seismic_io.survey_geometry import SurveyGeometry


def test_xyz_horizon_and_fault_cutoff(tmp_path):
    g = SurveyGeometry.regular(20, 15, 100, dt_ms=4.0, origin=(1000.0, 2000.0))
    ii, jj = np.meshgrid(np.arange(20), np.arange(15), indexing="ij")
    t_ms = 200.0 + 2.0 * ii
    x, y = g.world_xy(ii.ravel(), jj.ravel())
    np.savetxt(tmp_path / "h.txt", np.c_[x, y, t_ms.ravel()])
    h = read_xyz_horizon(tmp_path / "h.txt", g, time_shift_ms=4.0)
    np.testing.assert_allclose(h, (t_ms + 4.0) / 4.0)
    i, j = world_to_index(g, x[:3], y[:3])
    np.testing.assert_allclose(np.c_[i, j], np.c_[ii.ravel()[:3], jj.ravel()[:3]], atol=1e-9)
    # one vertical-ish fault stick at inline 10.5, crossline 5, from 100 to 400 ms
    sx, sy = g.world_xy(np.full(4, 10.5), np.full(4, 5.0))
    np.savetxt(tmp_path / "f.txt", np.c_[sx, sy, [100.0, 200.0, 300.0, 400.0], np.zeros(4), np.arange(4)])
    fs = read_fault_sticks(tmp_path / "f.txt", g)
    cut = fs.cutoff_polyline(h)
    assert cut.shape == (1, 2)
    np.testing.assert_allclose(cut[0], [10.5, 5.0], atol=1e-9)
