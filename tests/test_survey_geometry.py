import numpy as np
import pytest

from seismic_horizon_ai.preprocessing.roi_manager import ROI
from seismic_horizon_ai.seismic_io.survey_geometry import GeometryError, SurveyGeometry


def _grid(ils, xls):
    il, xl = np.meshgrid(ils, xls, indexing="ij")
    return il.ravel(), xl.ravel()


def test_regular_axes_keep_gaps_as_missing_traces():
    il, xl = _grid([100, 102, 106], [10, 11, 12])  # inline 104 missing entirely
    g, dup = SurveyGeometry.from_headers(il, xl, np.arange(5) * 4.0)
    assert g.ilines.tolist() == [100, 102, 104, 106]
    assert g.il_step == 2
    assert (~g.live_mask).sum() == 3 and not g.live_mask[2].any()
    assert dup.size == 0


def test_duplicates_first_trace_wins():
    il, xl = _grid([1, 2], [1, 2])
    il = np.r_[il, 2]; xl = np.r_[xl, 2]
    g, dup = SurveyGeometry.from_headers(il, xl, np.arange(3) * 2.0)
    assert dup.tolist() == [4]
    assert g.trace_index[1, 1] == 3


def test_degenerate_headers_fail_safely():
    with pytest.raises(GeometryError):
        SurveyGeometry.from_headers(np.zeros(10, int), np.zeros(10, int), np.arange(3.0))


def test_affine_fit_recovers_rotated_bins():
    il, xl = _grid(np.arange(1, 11), np.arange(1, 8))
    ang = np.radians(30)
    i, j = il - 1, xl - 1
    x = 1000 + 12.5 * i * np.cos(ang) - 25 * j * np.sin(ang)
    y = 5000 + 12.5 * i * np.sin(ang) + 25 * j * np.cos(ang)
    g, _ = SurveyGeometry.from_headers(il, xl, np.arange(4.0), x, y)
    assert g.affine_rms < 1e-6
    di, dx = g.bin_spacing()
    assert di == pytest.approx(12.5) and dx == pytest.approx(25.0)
    wx, wy = g.world_xy(3, 4)
    assert wx == pytest.approx(x[3 * 7 + 4]) and wy == pytest.approx(y[3 * 7 + 4])


def test_roi_slices():
    g = SurveyGeometry.regular(20, 10, 100, dt_ms=4.0, il0=100, xl0=1)
    s = ROI(inline_range=(105, 110), crossline_range=(3, 4), time_range_ms=(40, 80)).to_slices(g)
    assert s == (slice(5, 11), slice(2, 4), slice(10, 21))
    with pytest.raises(ValueError):
        ROI(inline_range=(500, 600)).to_slices(g)
