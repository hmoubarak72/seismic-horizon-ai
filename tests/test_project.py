"""Interpretation projects: save/load round trip (core) and New -> Save -> Reopen in the desktop app."""

import json
import os
import time

import numpy as np
import pytest

from seismic_horizon_ai.horizon_engine.seed_picks import PickSet, SeedPick
from seismic_horizon_ai.project import FaultEntry, HorizonState, Project, ProjectError


def test_project_round_trip(tmp_path):
    vol = tmp_path / "data" / "v.h5"
    vol.parent.mkdir(); vol.write_bytes(b"x")
    h = np.full((5, 6), np.nan); h[1:4, 2:5] = 123.5
    p = Project(name="Test", volume_h5=str(vol),
                picks=PickSet([SeedPick(10, 20, 250.0, source="interpreter"), SeedPick(11, 22, 251.5, status="proposed")]),
                faults=[FaultEntry("FaultA", "faultA.txt", 4.0, [np.array([[1.0, 2.0, 30.0], [1.5, 2.5, 60.0]])])],
                reference=np.arange(30, dtype=float).reshape(5, 6), reference_path="ref.txt", reference_shift_ms=4.0,
                horizon=HorizonState(h, np.where(np.isfinite(h), 0.8, np.nan), [np.array([[0.5, 1.0], [2.0, 3.0]])], {"n_picks": 2}),
                settings={"snap": "trough", "roi": [1, 2, 3, 4]})
    out = p.save(tmp_path / "proj" / "my")
    assert out.name == "my.shproj" and out.exists() and (out.parent / "my.shproj.npz").exists()
    doc = json.loads(out.read_text())
    assert doc["format_version"] == 1 and doc["volume"]["h5_rel"]

    q = Project.load(out)
    assert q.name == "Test" and q.settings == p.settings
    assert [(a.inline, a.crossline, a.time_ms, a.status, a.pick_id) for a in q.picks.picks] == \
           [(a.inline, a.crossline, a.time_ms, a.status, a.pick_id) for a in p.picks.picks]
    assert len(q.picks.approved()) == 1  # proposed picks stay proposed
    assert q.faults[0].name == "FaultA" and q.faults[0].time_shift_ms == 4.0
    np.testing.assert_allclose(q.faults[0].sticks[0], p.faults[0].sticks[0])
    np.testing.assert_allclose(q.reference, p.reference)
    np.testing.assert_allclose(q.horizon.time_ms, h)  # NaN stays NaN
    np.testing.assert_allclose(q.horizon.cuts[0], p.horizon.cuts[0])
    assert q.horizon.summary == {"n_picks": 2}

    # moving the project folder together with its data keeps the volume reachable (relative path)
    moved = tmp_path / "moved"
    import shutil
    shutil.copytree(tmp_path / "proj", moved / "proj"); shutil.copytree(tmp_path / "data", moved / "data")
    shutil.rmtree(tmp_path / "data")
    r = Project.load(moved / "proj" / "my.shproj")
    assert os.path.samefile(r.volume_h5, moved / "data" / "v.h5")


def test_project_rejects_bad_files(tmp_path):
    bad = tmp_path / "x.shproj"
    bad.write_text("{not json")
    with pytest.raises(ProjectError):
        Project.load(bad)
    bad.write_text(json.dumps({"format": "something.else"}))
    with pytest.raises(ProjectError):
        Project.load(bad)
    bad.write_text(json.dumps({"format": "seismic_horizon_ai.project", "format_version": 99}))
    with pytest.raises(ProjectError, match="newer"):
        Project.load(bad)


def _wait(app, w, timeout=180):
    t0 = time.time()
    while w._thread is not None and time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.05)
    assert w._thread is None, "background task did not finish"


def test_app_new_save_reopen_project(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHAI_NO_RESTORE", "1")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox

    from seismic_horizon_ai.seismic_io.segy_reader import write_segy
    from seismic_horizon_ai.synthetic import make_synthetic
    import seismic_horizon_ai.visualization.app as appmod

    monkeypatch.setattr(appmod, "DEFAULT_DATA_DIR", tmp_path / "data")
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2] if len(a) > 2 else "") or QMessageBox.Yes)
    QSettings("seismic_horizon_ai", "interpreter-test").clear()

    sv = make_synthetic(n_il=30, n_xl=25, n_t=150, throw=0.0, noise=0.1)
    seg = write_segy(tmp_path / "survey.sgy", sv.seismic, il0=1, xl0=1)
    app = QApplication.instance() or QApplication([])

    w = appmod.MainWindow()
    w.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    proj_file = tmp_path / "projects" / "My F3 work.shproj"
    w.new_project("My F3 work", proj_file, seg)  # SEG-Y: QC + conversion run in the background
    _wait(app, w)
    assert w.vol is not None and proj_file.exists() and not w.dirty
    assert (tmp_path / "data" / "survey" / "survey.h5").exists()
    assert w.project.source_segy.endswith("survey.sgy")

    h = sv.horizon(80.0)
    for i, j in [(5, 5), (20, 18), (10, 20)]:
        w.picks.add(SeedPick(i + 1, j + 1, float(h[i, j] * 4.0)))
    w._mark_dirty(); w._refresh_all()
    assert w.windowTitle().startswith("*")
    w.snap.setCurrentText("trough"); w.margin.setValue(150)
    w.run_extraction(); _wait(app, w)
    assert w.horizon_ms is not None
    hz = w.horizon_ms.copy()
    assert w.save_project()
    assert not w.dirty and not w.windowTitle().startswith("*")
    w.close(); w.vol is None or w.vol.close()

    # a fresh window reopens everything
    w2 = appmod.MainWindow()
    w2.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    assert w2.open_project(str(proj_file))
    assert w2.project.name == "My F3 work"
    assert len(w2.picks.picks) == 3
    assert [p.time_ms for p in w2.picks.picks] == pytest.approx([float(h[i, j] * 4.0) for i, j in [(5, 5), (20, 18), (10, 20)]])
    np.testing.assert_allclose(w2.horizon_ms, hz, atol=1e-3)  # float32 storage
    assert w2.snap.currentText() == "trough" and w2.margin.value() == 150
    assert not w2.dirty
    # reloaded horizon can be drawn and exported without re-running the engine
    w2.draw_section(); w2.draw_map()
    out_csv = tmp_path / "hz.csv"
    w2.export_horizon(str(out_csv))
    assert out_csv.exists() and len(out_csv.read_text().splitlines()) > 100
    # edit -> unsaved -> save again -> reopen sees the edit
    w2.picks.add(SeedPick(15, 12, float(h[14, 11] * 4.0))); w2._mark_dirty()
    assert w2.save_project()
    w2.close()
    w3 = appmod.MainWindow()
    w3.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    assert w3.open_project(str(proj_file)) and len(w3.picks.picks) == 4
    w3.close()
