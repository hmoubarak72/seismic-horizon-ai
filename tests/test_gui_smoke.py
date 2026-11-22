"""Headless smoke test of the desktop app: open a volume, pick, extract, draw."""

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_app_end_to_end(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from seismic_horizon_ai.horizon_engine.seed_picks import PickSet, SeedPick
    from seismic_horizon_ai.seismic_io.hdf5_manager import convert_segy_to_hdf5
    from seismic_horizon_ai.seismic_io.segy_reader import write_segy
    from seismic_horizon_ai.synthetic import make_synthetic
    from seismic_horizon_ai.visualization.app import MainWindow

    QSettings("seismic_horizon_ai", "interpreter-test").clear()
    sv = make_synthetic(n_il=30, n_xl=25, n_t=150, throw=0.0, noise=0.1)
    seg = write_segy(tmp_path / "s.sgy", sv.seismic, il0=1, xl0=1)
    h5, _ = convert_segy_to_hdf5(seg, tmp_path / "s.h5")

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    w.open_h5(str(h5))
    h = sv.horizon(80.0)
    w.picks = PickSet([SeedPick(i + 1, j + 1, float(h[i, j] * 4.0)) for i, j in [(5, 5), (20, 18), (10, 20)]])
    w._refresh_all()
    w.run_extraction()
    t0 = time.time()
    while w._thread is not None and time.time() - t0 < 120:
        app.processEvents(); time.sleep(0.05)
    assert w.output is not None
    err = np.nanmean(np.abs(w.horizon_ms - h * 4.0))
    assert err < 2.0  # ms
    for k in range(w.map_kind.count()):
        w.map_kind.setCurrentIndex(k)
    w.draw_timeslice()
    w.export_horizon  # menu action exists
    w.close()


def test_picking_works_after_zoom(tmp_path):
    """Regression: after using the zoom box, clicks must add picks and the zoom must be kept."""
    from matplotlib.backend_bases import MouseEvent
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from seismic_horizon_ai.seismic_io.hdf5_manager import convert_segy_to_hdf5
    from seismic_horizon_ai.seismic_io.segy_reader import write_segy
    from seismic_horizon_ai.synthetic import make_synthetic
    from seismic_horizon_ai.visualization.app import MainWindow

    sv = make_synthetic(n_il=20, n_xl=18, n_t=120, throw=0.0, noise=0.1)
    h5, _ = convert_segy_to_hdf5(write_segy(tmp_path / "z.sgy", sv.seismic), tmp_path / "z.h5")
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    w.open_h5(str(h5))
    w.snap.setCurrentText("none")
    c, ax = w.sec_canvas, w.sec_canvas.ax
    c.draw()

    def mouse(name, x, y, button=1):
        px, py = ax.transData.transform((x, y))
        return MouseEvent(name, c, px, py, button=button)

    # user turns the zoom box on, drags, releases
    w.sec_toolbar.zoom()
    assert "zoom" in str(w.sec_toolbar.mode)
    ax.set_xlim(5, 10); ax.set_ylim(300, 200)  # the zoomed view
    w._section_release(mouse("button_release_event", 7, 250))
    for _ in range(5):
        app.processEvents()
    assert "zoom" not in str(w.sec_toolbar.mode)  # tool switched off automatically

    n0 = len(w.picks.picks)
    w._section_click(mouse("button_press_event", 7.2, 240.0))
    assert len(w.picks.picks) == n0 + 1
    p = w.picks.picks[-1]
    assert p.crossline == 7 and abs(p.time_ms - 240.0) < 1.0
    assert w.sec_canvas.ax.get_xlim() == (5, 10)  # zoom kept after the pick redraw

    # mouse-wheel zoom keeps picking active
    w._section_scroll(MouseEvent("scroll_event", c, *ax.transData.transform((8, 250)), button="up"))
    assert w.sec_canvas.ax.get_xlim()[1] - w.sec_canvas.ax.get_xlim()[0] < 5
    w.reset_section_zoom()
    assert w.sec_canvas.ax.get_xlim()[1] - w.sec_canvas.ax.get_xlim()[0] > 10
    w.close()
