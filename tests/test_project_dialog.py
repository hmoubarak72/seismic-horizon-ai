"""The File > New project dialog: validation and creation from an HDF5 volume."""

import os

import pytest


def test_new_project_dialog(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHAI_NO_RESTORE", "1")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox

    from seismic_horizon_ai.seismic_io.hdf5_manager import convert_segy_to_hdf5
    from seismic_horizon_ai.seismic_io.segy_reader import write_segy
    from seismic_horizon_ai.synthetic import make_synthetic
    import seismic_horizon_ai.visualization.app as appmod

    sv = make_synthetic(n_il=12, n_xl=10, n_t=80, throw=0.0, noise=0.1)
    h5, _ = convert_segy_to_hdf5(write_segy(tmp_path / "d.sgy", sv.seismic), tmp_path / "d.h5")
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]) or QMessageBox.Ok)
    answers = iter([("", str(tmp_path), str(h5)), ("Line 12 test", str(tmp_path / "p"), str(h5))])

    def fake_exec(dlg):
        vals = next(answers, None)
        if vals is None:
            return QDialog.Rejected
        for edit, v in zip(dlg.findChildren(QLineEdit), vals):
            edit.setText(v)
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    w = appmod.MainWindow()
    w.settings = QSettings("seismic_horizon_ai", "interpreter-test")
    w.new_project_dialog()
    assert warnings and "name" in warnings[0]  # empty name refused, dialog shown again
    target = tmp_path / "p" / "Line 12 test.shproj"
    assert target.exists() and w.project.path == target and w.vol is not None
    assert "Line 12 test" in w.windowTitle()
    w.close()
