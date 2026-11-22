"""Seismic Horizon Interpreter: desktop application (PySide6 + matplotlib + PyVista).

Workflow in the window:
  1. File > Open SEG-Y (QC + conversion) or Open HDF5 volume.
  2. Browse inlines / crosslines / time slices; click on a reflection to add a seed pick
     (right-click removes the nearest pick, Ctrl+Z undoes). Optionally snap to peak/trough.
  3. Optionally load fault sticks and a reference horizon.
  4. Press "Extract horizon". The engine runs in the background and the result appears on
     the sections, the map, the confidence map and in 3D.
  5. Export the horizon (CSV with confidence and flags + provenance) or save the picks.

The interpreter stays in charge: every pick is theirs, nothing is applied silently.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import __version__  # noqa: E402
from ..horizon_engine.seed_picks import PickSet, SeedPick  # noqa: E402
from ..preprocessing.roi_manager import ROI  # noqa: E402
from ..seismic_io.hdf5_manager import SeismicVolume  # noqa: E402
from ..seismic_io.interpretation_io import read_fault_sticks, read_xyz_horizon  # noqa: E402

APP_NAME = "Seismic Horizon Interpreter"
DEFAULT_DATA_DIR = Path("C:/SeismicData") if os.name == "nt" else Path.home() / "SeismicData"


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#1f3b57")); p.setPen(Qt.NoPen); p.drawRoundedRect(2, 2, 60, 60, 12, 12)
    pen = p.pen(); pen.setWidth(4)
    for k, col in enumerate(["#9ad0ec", "#f2c14e", "#9ad0ec"]):
        pen.setColor(QColor(col)); p.setPen(pen)
        y = 18 + 14 * k
        p.drawLine(8, y + 6, 30, y); p.drawLine(34, y + 10, 56, y + 4)
    pen.setColor(QColor("#e4572e")); pen.setWidth(3); p.setPen(pen); p.drawLine(33, 8, 30, 58)
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------------------- workers
class Worker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            out = self.fn(*self.args, progress=self.progress.emit, **self.kwargs)
            self.finished.emit(out)
        except Exception as exc:  # report every failure to the user, never crash the window
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _convert(segy: str, h5: str, force: bool, progress):
    from ..seismic_io.hdf5_manager import convert_segy_to_hdf5

    progress("Checking SEG-Y headers and amplitudes (QC)...")
    out, report = convert_segy_to_hdf5(segy, h5, require_qc_pass=not force)
    return out, report


def _extract(vol, picks, roi, sticks, progress):
    from ..workflow import run_extraction

    return run_extraction(vol, picks, roi, sticks, progress=progress)


# ---------------------------------------------------------------------------- canvas
class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)


# ---------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setWindowIcon(make_icon())
        self.resize(1500, 920)
        self.settings = QSettings("seismic_horizon_ai", "interpreter")

        self.vol: SeismicVolume | None = None
        self.picks = PickSet()
        self.undo_stack: list[tuple[str, SeedPick]] = []
        self.faults = []  # FaultSticks in full grid coords
        self.reference: np.ndarray | None = None  # sample index grid
        self.output = None  # ExtractionOutput
        self.horizon_ms: np.ndarray | None = None
        self.conf_full: np.ndarray | None = None
        self._thread: QThread | None = None
        self._section_cache: dict = {}

        self._build_ui()
        self._build_menus()
        self._refresh_all()
        last = self.settings.value("last_h5", "")
        if last and Path(last).exists():
            self.open_h5(last)  # reopen the last survey so work continues where it stopped

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        # ---- left panel
        left = QWidget(); lv = QVBoxLayout(left); left.setMinimumWidth(330); left.setMaximumWidth(430)
        self.info = QLabel("No survey loaded.\nFile > Open SEG-Y... or Open HDF5 volume...")
        self.info.setWordWrap(True)
        g1 = QGroupBox("Survey"); QVBoxLayout(g1).addWidget(self.info); lv.addWidget(g1)

        g2 = QGroupBox("Seed picks (left-click add, right-click remove, Ctrl+Z undo)")
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        row.addWidget(QLabel("Snap to:"))
        self.snap = QComboBox(); self.snap.addItems(["peak", "trough", "none"])
        row.addWidget(self.snap); row.addStretch(1)
        v2.addLayout(row)
        self.pick_table = QTableWidget(0, 3)
        self.pick_table.setHorizontalHeaderLabels(["Inline", "Crossline", "Time (ms)"])
        self.pick_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pick_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pick_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pick_table.cellDoubleClicked.connect(self._goto_pick)
        v2.addWidget(self.pick_table)
        row = QHBoxLayout()
        b = QPushButton("Delete selected"); b.clicked.connect(self._delete_selected); row.addWidget(b)
        b = QPushButton("Clear all"); b.clicked.connect(self._clear_picks); row.addWidget(b)
        v2.addLayout(row)
        lv.addWidget(g2, 2)

        g3 = QGroupBox("Area to interpret")
        f3 = QFormLayout(g3)
        self.auto_roi = QCheckBox("Automatic: around the picks (+25 lines)"); self.auto_roi.setChecked(True)
        self.auto_roi.toggled.connect(self._roi_toggled)
        f3.addRow(self.auto_roi)
        self.il_min, self.il_max, self.xl_min, self.xl_max = (QSpinBox() for _ in range(4))
        for w in (self.il_min, self.il_max, self.xl_min, self.xl_max):
            w.setRange(0, 10**7); w.setEnabled(False)
        r = QHBoxLayout(); r.addWidget(self.il_min); r.addWidget(QLabel("to")); r.addWidget(self.il_max); f3.addRow("Inlines", r)
        r = QHBoxLayout(); r.addWidget(self.xl_min); r.addWidget(QLabel("to")); r.addWidget(self.xl_max); f3.addRow("Crosslines", r)
        self.margin = QDoubleSpinBox(); self.margin.setRange(20, 1000); self.margin.setValue(120); self.margin.setSuffix(" ms")
        f3.addRow("Time window ±", self.margin)
        lv.addWidget(g3)

        g4 = QGroupBox("Faults and reference")
        v4 = QVBoxLayout(g4)
        self.fault_label = QLabel("No fault sticks loaded."); self.fault_label.setWordWrap(True)
        v4.addWidget(self.fault_label)
        r = QHBoxLayout()
        b = QPushButton("Load fault sticks..."); b.clicked.connect(self.load_faults); r.addWidget(b)
        b = QPushButton("Clear"); b.clicked.connect(self._clear_faults); r.addWidget(b)
        v4.addLayout(r)
        b = QPushButton("Load reference horizon..."); b.clicked.connect(self.load_reference); v4.addWidget(b)
        lv.addWidget(g4)

        self.run_btn = QPushButton("▶  Extract horizon")
        f = QFont(); f.setPointSize(11); f.setBold(True); self.run_btn.setFont(f)
        self.run_btn.setMinimumHeight(42)
        self.run_btn.clicked.connect(self.run_extraction)
        lv.addWidget(self.run_btn)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setVisible(False)
        lv.addWidget(self.progress)
        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMaximumHeight(170)
        self.result_box.setPlaceholderText("Results and QC will appear here.")
        lv.addWidget(self.result_box)
        split.addWidget(left)

        # ---- right: tabs
        self.tabs = QTabWidget()
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)

        # section tab
        sec = QWidget(); sv = QVBoxLayout(sec)
        top = QHBoxLayout()
        self.direction = QComboBox(); self.direction.addItems(["Inline", "Crossline"])
        self.direction.currentIndexChanged.connect(self._direction_changed)
        top.addWidget(QLabel("View:")); top.addWidget(self.direction)
        self.line_slider = QSlider(Qt.Horizontal); self.line_slider.valueChanged.connect(self._line_changed)
        self.line_spin = QSpinBox(); self.line_spin.valueChanged.connect(self._spin_changed)
        top.addWidget(self.line_slider, 1); top.addWidget(self.line_spin)
        self.gain = QSlider(Qt.Horizontal); self.gain.setRange(80, 100); self.gain.setValue(98); self.gain.setMaximumWidth(120)
        self.gain.valueChanged.connect(lambda _v: self.draw_section())
        top.addWidget(QLabel("Contrast")); top.addWidget(self.gain)
        sv.addLayout(top)
        self.sec_canvas = PlotCanvas()
        self.sec_canvas.mpl_connect("button_press_event", self._section_click)
        sv.addWidget(NavigationToolbar2QT(self.sec_canvas, sec))
        sv.addWidget(self.sec_canvas, 1)
        self.tabs.addTab(sec, "Seismic section")

        # time slice tab
        ts = QWidget(); tv = QVBoxLayout(ts); self.ts_tab = ts
        top = QHBoxLayout()
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.sliderReleased.connect(self.draw_timeslice)
        self.t_label = QLabel("")
        top.addWidget(QLabel("Time slice:")); top.addWidget(self.t_slider, 1); top.addWidget(self.t_label)
        tv.addLayout(top)
        self.ts_canvas = PlotCanvas()
        self.ts_canvas.mpl_connect("button_press_event", self._map_click)
        tv.addWidget(NavigationToolbar2QT(self.ts_canvas, ts)); tv.addWidget(self.ts_canvas, 1)
        self.tabs.addTab(ts, "Time slice")

        # map tab
        mp = QWidget(); mv = QVBoxLayout(mp)
        top = QHBoxLayout()
        self.map_kind = QComboBox(); self.map_kind.addItems(["Horizon time", "Confidence", "Difference to reference", "Reference horizon"])
        self.map_kind.currentIndexChanged.connect(lambda _i: self.draw_map())
        top.addWidget(QLabel("Show:")); top.addWidget(self.map_kind); top.addStretch(1)
        top.addWidget(QLabel("Click the map to jump to that inline/crossline"))
        mv.addLayout(top)
        self.map_canvas = PlotCanvas()
        self.map_canvas.mpl_connect("button_press_event", self._map_click)
        mv.addWidget(NavigationToolbar2QT(self.map_canvas, mp)); mv.addWidget(self.map_canvas, 1)
        self.tabs.addTab(mp, "Horizon map")

        # 3D tab (created lazily: VTK start-up is slow)
        self.view3d_holder = QWidget(); QVBoxLayout(self.view3d_holder)
        self.plotter = None
        self.tabs.addTab(self.view3d_holder, "3D view")

        # QC tab
        self.qc_text = QTextEdit(); self.qc_text.setReadOnly(True)
        self.tabs.addTab(self.qc_text, "QC report")
        self.tabs.currentChanged.connect(self._tab_changed)

    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        for text, fn, key in [
            ("Open SEG-Y...", self.open_segy, "Ctrl+O"),
            ("Open HDF5 volume...", self.open_h5, "Ctrl+Shift+O"),
            ("Open recent volume", self.open_recent, None),
            (None, None, None),
            ("Load picks (CSV)...", self.load_picks, "Ctrl+L"),
            ("Save picks (CSV)...", self.save_picks, "Ctrl+S"),
            ("Import picks from horizon file...", self.import_picks_from_horizon, None),
            (None, None, None),
            ("Export horizon (CSV)...", self.export_horizon, "Ctrl+E"),
            ("Save figures...", self.save_figures, None),
            (None, None, None),
            ("Quit", self.close, "Ctrl+Q"),
        ]:
            if text is None:
                m.addSeparator(); continue
            a = QAction(text, self); a.triggered.connect(fn)
            if key:
                a.setShortcut(QKeySequence(key))
            m.addAction(a)
        e = self.menuBar().addMenu("&Edit")
        a = QAction("Undo pick", self); a.setShortcut(QKeySequence.Undo); a.triggered.connect(self.undo); e.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("How to use", self); a.triggered.connect(self.show_help); h.addAction(a)
        a = QAction("About", self); a.triggered.connect(lambda: QMessageBox.about(self, APP_NAME, f"{APP_NAME} {__version__}\n\nPhysics-first horizon interpretation: least-squares horizons from local slopes, multigrid correlations and your seed picks, with faults as hard barriers.")); h.addAction(a)

    # ------------------------------------------------------------------ helpers
    @property
    def g(self):
        return self.vol.geometry if self.vol else None

    def _busy(self, on: bool, msg: str = ""):
        self.progress.setVisible(on)
        self.run_btn.setEnabled(not on)
        if msg:
            self.statusBar().showMessage(msg)

    def _start(self, fn, args, on_done, label):
        if self._thread is not None:
            QMessageBox.information(self, APP_NAME, "Please wait: a task is still running.")
            return
        self._busy(True, label)
        th = QThread(self)
        w = Worker(fn, *args)
        w.moveToThread(th)
        th.started.connect(w.run)
        # bound methods of the window => queued into the GUI thread
        w.progress.connect(self._on_progress)
        w.finished.connect(self._on_done)
        w.failed.connect(self._on_failed)
        self._done_cb = on_done
        self._worker = w
        self._thread = th
        th.start()

    def _on_progress(self, msg: str):
        self.statusBar().showMessage(msg)

    def _on_done(self, out):
        self._finish()
        self._done_cb(out)

    def _on_failed(self, err: str):
        self._finish()
        self._error(err)

    def _finish(self):
        if self._thread:
            self._thread.quit(); self._thread.wait()
        self._thread = None
        self._busy(False)

    def _error(self, err: str):
        self.statusBar().showMessage("Failed.")
        box = QMessageBox(self); box.setIcon(QMessageBox.Warning); box.setWindowTitle(APP_NAME)
        box.setText(err.split("\n\n")[0]); box.setDetailedText(err); box.exec()

    # ------------------------------------------------------------------ files
    def open_segy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEG-Y", self.settings.value("last_dir", ""), "SEG-Y (*.sgy *.segy *.SGY *.SEGY);;All files (*)")
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        default = DEFAULT_DATA_DIR / Path(path).stem / (Path(path).stem + ".h5")
        h5, _ = QFileDialog.getSaveFileName(self, "Save checked volume as (HDF5)", str(default), "HDF5 (*.h5)")
        if not h5:
            return
        Path(h5).parent.mkdir(parents=True, exist_ok=True)

        def done(out):
            out_path, report = out
            self.qc_text.setMarkdown(report.to_markdown())
            self.open_h5(str(out_path))
            self.tabs.setCurrentWidget(self.qc_text)

        def failed_qc(force=False):
            self._start(_convert, (path, h5, force), done, "Converting SEG-Y...")

        failed_qc(False)

    def open_recent(self):
        last = self.settings.value("last_h5", "")
        if last and Path(last).exists():
            self.open_h5(last)
        else:
            QMessageBox.information(self, APP_NAME, "No recent volume.")

    def open_h5(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open HDF5 volume", str(DEFAULT_DATA_DIR), "HDF5 (*.h5 *.hdf5)")
            if not path:
                return
        try:
            vol = SeismicVolume(path)
        except Exception as exc:
            self._error(f"Could not open {path}: {exc}"); return
        if self.vol:
            self.vol.close()
        self.vol = vol
        self.settings.setValue("last_h5", path)
        self._section_cache.clear()
        self.output = None; self.horizon_ms = None; self.conf_full = None; self.reference = None; self.faults = []
        g = self.g
        qc = vol.qc_report
        self.qc_text.setMarkdown(_qc_markdown(qc))
        self.info.setText(
            f"<b>{Path(path).name}</b><br>"
            f"Inlines {g.ilines[0]}–{g.ilines[-1]} ({g.ilines.size})<br>"
            f"Crosslines {g.xlines[0]}–{g.xlines[-1]} ({g.xlines.size})<br>"
            f"Time {g.samples_ms[0]:.0f}–{g.samples_ms[-1]:.0f} ms, dt {g.dt_ms:g} ms<br>"
            f"QC: {'passed' if qc.get('passed') else 'FAILED'} · CRS: {g.crs or 'not in file'}"
        )
        for w, lo, hi in ((self.il_min, g.ilines[0], g.ilines[-1]), (self.il_max, g.ilines[0], g.ilines[-1]),
                          (self.xl_min, g.xlines[0], g.xlines[-1]), (self.xl_max, g.xlines[0], g.xlines[-1])):
            w.setRange(int(lo), int(hi))
        self.il_min.setValue(int(g.ilines[0])); self.il_max.setValue(int(g.ilines[-1]))
        self.xl_min.setValue(int(g.xlines[0])); self.xl_max.setValue(int(g.xlines[-1]))
        self.t_slider.setRange(0, g.samples_ms.size - 1); self.t_slider.setValue(g.samples_ms.size // 3)
        self._direction_changed()
        self.statusBar().showMessage(f"Opened {path}")
        self._refresh_all()

    def load_picks(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load picks", self.settings.value("last_dir", ""), "CSV (*.csv)")
        if not path:
            return
        try:
            ps = PickSet.from_csv(path)
        except Exception as exc:
            self._error(f"Could not read picks: {exc}"); return
        self.picks.extend(ps.picks)
        self.statusBar().showMessage(f"Loaded {len(ps.picks)} picks from {path}")
        self._refresh_all()

    def save_picks(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save picks", self.settings.value("last_dir", "picks.csv"), "CSV (*.csv)")
        if path:
            self.picks.to_csv(path)
            self.statusBar().showMessage(f"Saved {len(self.picks.picks)} picks to {path}")

    def import_picks_from_horizon(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Horizon file (x y time_ms)", self.settings.value("last_dir", ""), "Text (*.txt *.xyz *.dat);;All files (*)")
        if not path:
            return
        dlg = QDialog(self); dlg.setWindowTitle("Import picks from horizon")
        f = QFormLayout(dlg)
        sp = QSpinBox(); sp.setRange(5, 500); sp.setValue(50); f.addRow("Pick every Nth line", sp)
        sh = QDoubleSpinBox(); sh.setRange(-1000, 1000); sh.setValue(0); sh.setSuffix(" ms"); f.addRow("Time shift", sh)
        f.addRow(QLabel("(F3 dGB horizon files need +4 ms)"))
        only = QCheckBox("Only inside the area box"); only.setChecked(not self.auto_roi.isChecked()); f.addRow(only)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); f.addRow(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        g = self.g
        h = read_xyz_horizon(path, g, time_shift_ms=sh.value())
        live = self.vol.live_mask
        ri, rj = self._manual_roi_slices() if only.isChecked() else (slice(0, g.ilines.size), slice(0, g.xlines.size))
        n = 0
        for i in range(ri.start + sp.value() // 2, ri.stop, sp.value()):
            for j in range(rj.start + sp.value() // 2, rj.stop, sp.value()):
                if np.isfinite(h[i, j]) and live[i, j]:
                    self.picks.add(SeedPick(int(g.ilines[i]), int(g.xlines[j]), float(g.time_ms(h[i, j])), source=f"import:{Path(path).name}"))
                    n += 1
        self.statusBar().showMessage(f"Imported {n} picks from {Path(path).name}")
        self._refresh_all()

    def load_faults(self):
        if not self._need_volume():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Fault sticks (x y time_ms stick node)", self.settings.value("last_dir", ""), "Text (*.txt *.dat);;All files (*)")
        if not paths:
            return
        shift, ok = QInputDialog.getDouble(self, "Time shift", "Time shift to apply to the fault files (ms).\nF3 dGB files need +4 ms.", 0.0, -1000, 1000, 1)
        if not ok:
            return
        for p in paths:
            try:
                self.faults.append(read_fault_sticks(p, self.g, time_shift_ms=shift))
            except Exception as exc:
                self._error(f"Could not read {p}: {exc}")
        self._refresh_all()

    def _clear_faults(self):
        self.faults = []
        self._refresh_all()

    def load_reference(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Reference horizon (x y time_ms)", self.settings.value("last_dir", ""), "Text (*.txt *.xyz *.dat);;All files (*)")
        if not path:
            return
        shift, ok = QInputDialog.getDouble(self, "Time shift", "Time shift for the reference horizon (ms).\nF3 dGB files need +4 ms.", 0.0, -1000, 1000, 1)
        if not ok:
            return
        self.reference = read_xyz_horizon(path, self.g, time_shift_ms=shift)
        self.statusBar().showMessage(f"Reference horizon loaded: {Path(path).name}")
        self._refresh_all()

    def export_horizon(self):
        if self.output is None:
            QMessageBox.information(self, APP_NAME, "Extract a horizon first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export horizon", self.settings.value("last_dir", "horizon.csv"), "CSV (*.csv)")
        if not path:
            return
        from ..export.horizon_export import export_horizon_csv

        o = self.output
        export_horizon_csv(path, o.result, o.geometry, confidence=o.confidence.confidence, extra_flags={"off_phase": o.phase["off_phase_mask"]})
        Path(path).with_suffix(".summary.json").write_text(json.dumps(o.summary, indent=2, default=float), encoding="utf-8")
        self.statusBar().showMessage(f"Exported horizon to {path} (+ provenance and summary)")

    def save_figures(self):
        d = QFileDialog.getExistingDirectory(self, "Save figures to folder", self.settings.value("last_dir", ""))
        if not d:
            return
        self.sec_canvas.fig.savefig(Path(d) / "section.png", dpi=150)
        self.map_canvas.fig.savefig(Path(d) / "map.png", dpi=150)
        self.ts_canvas.fig.savefig(Path(d) / "time_slice.png", dpi=150)
        self.statusBar().showMessage(f"Figures saved to {d}")

    def _need_volume(self) -> bool:
        if self.vol is None:
            QMessageBox.information(self, APP_NAME, "Open a seismic volume first (File menu).")
            return False
        return True

    # ------------------------------------------------------------------ picks
    def _section_click(self, ev):
        if self.vol is None or ev.inaxes is not self.sec_canvas.ax or ev.xdata is None:
            return
        if self.sec_canvas.toolbar_mode():
            return
        g = self.g
        inline_view = self.direction.currentIndex() == 0
        line = self._current_line()
        trace = int(round(ev.xdata))
        il, xl = (line, trace) if inline_view else (trace, line)
        if ev.button == 1:
            t = self._snap(il, xl, float(ev.ydata))
            p = SeedPick(il, xl, t, phase=self.snap.currentText() if self.snap.currentText() != "none" else "unspecified")
            self.picks.add(p); self.undo_stack.append(("add", p))
            self.statusBar().showMessage(f"Pick added: IL {il} XL {xl} {t:.1f} ms")
        elif ev.button == 3:
            on_line = [p for p in self.picks.picks if (p.inline == line if inline_view else p.crossline == line)]
            if not on_line:
                return
            d = [abs((p.crossline if inline_view else p.inline) - trace) / 5 + abs(p.time_ms - ev.ydata) / 20 for p in on_line]
            k = int(np.argmin(d))
            if d[k] < 3:
                p = on_line[k]; self.picks.picks.remove(p); self.undo_stack.append(("del", p))
                self.statusBar().showMessage("Pick removed")
        self._refresh_picks(); self.draw_section(); self.draw_map()

    def _snap(self, il, xl, t_ms) -> float:
        mode = self.snap.currentText()
        g = self.g
        if mode == "none":
            return t_ms
        i, j = int(round(g.il_index(il))), int(round(g.xl_index(xl)))
        tr = np.asarray(self.vol.data[i, j, :], float)
        k0 = int(round(float(g.sample_index(t_ms))))
        lo, hi = max(1, k0 - 4), min(tr.size - 2, k0 + 4)
        sgn = 1 if mode == "peak" else -1
        k = lo + int(np.argmax(sgn * tr[lo:hi + 1]))
        a, b, c = sgn * tr[k - 1], sgn * tr[k], sgn * tr[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        return float(g.time_ms(k + np.clip(off, -0.5, 0.5)))

    def undo(self):
        if not self.undo_stack:
            return
        kind, p = self.undo_stack.pop()
        if kind == "add" and p in self.picks.picks:
            self.picks.picks.remove(p)
        elif kind == "del":
            self.picks.add(p)
        self._refresh_picks(); self.draw_section(); self.draw_map()

    def _delete_selected(self):
        rows = sorted({i.row() for i in self.pick_table.selectedIndexes()}, reverse=True)
        for r in rows:
            p = self.picks.picks.pop(r); self.undo_stack.append(("del", p))
        self._refresh_all()

    def _clear_picks(self):
        if self.picks.picks and QMessageBox.question(self, APP_NAME, f"Remove all {len(self.picks.picks)} picks?") == QMessageBox.Yes:
            for p in self.picks.picks:
                self.undo_stack.append(("del", p))
            self.picks = PickSet()
            self._refresh_all()

    def _goto_pick(self, row, _col):
        p = self.picks.picks[row]
        self._goto(p.inline, p.crossline)

    # ------------------------------------------------------------------ navigation
    def _direction_changed(self):
        if self.vol is None:
            return
        g = self.g
        axis = g.ilines if self.direction.currentIndex() == 0 else g.xlines
        for w in (self.line_slider, self.line_spin):
            w.blockSignals(True)
        self.line_slider.setRange(0, axis.size - 1)
        self.line_spin.setRange(int(axis[0]), int(axis[-1]))
        self.line_slider.setValue(axis.size // 2)
        self.line_spin.setValue(int(axis[axis.size // 2]))
        for w in (self.line_slider, self.line_spin):
            w.blockSignals(False)
        self.draw_section()

    def _line_changed(self, idx):
        g = self.g
        axis = g.ilines if self.direction.currentIndex() == 0 else g.xlines
        self.line_spin.blockSignals(True); self.line_spin.setValue(int(axis[idx])); self.line_spin.blockSignals(False)
        self.draw_section()

    def _spin_changed(self, val):
        g = self.g
        axis = g.ilines if self.direction.currentIndex() == 0 else g.xlines
        idx = int(np.argmin(np.abs(axis - val)))
        self.line_slider.blockSignals(True); self.line_slider.setValue(idx); self.line_slider.blockSignals(False)
        self.draw_section()

    def _current_line(self) -> int:
        return int(self.line_spin.value())

    def _goto(self, il, xl):
        if self.direction.currentIndex() == 0:
            self.line_spin.setValue(int(il))
        else:
            self.line_spin.setValue(int(xl))
        self.tabs.setCurrentIndex(0)

    def _map_click(self, ev):
        if self.vol is None or ev.xdata is None or ev.ydata is None:
            return
        canvas = ev.canvas
        if canvas.toolbar_mode():
            return
        self._goto(int(round(ev.xdata)), int(round(ev.ydata)))

    def _roi_toggled(self, on):
        for w in (self.il_min, self.il_max, self.xl_min, self.xl_max):
            w.setEnabled(not on)

    def _manual_roi_slices(self):
        g = self.g
        il0, il1 = sorted((self.il_min.value(), self.il_max.value()))
        xl0, xl1 = sorted((self.xl_min.value(), self.xl_max.value()))
        i0, i1 = int(round(g.il_index(il0))), int(round(g.il_index(il1)))
        j0, j1 = int(round(g.xl_index(xl0))), int(round(g.xl_index(xl1)))
        return slice(max(0, i0), min(g.ilines.size, i1 + 1)), slice(max(0, j0), min(g.xlines.size, j1 + 1))

    def _tab_changed(self, idx):
        w = self.tabs.widget(idx)
        if w is self.view3d_holder:
            self.draw_3d()
        elif w is self.ts_tab:
            self.draw_timeslice()

    # ------------------------------------------------------------------ extraction
    def run_extraction(self):
        if not self._need_volume():
            return
        if not self.picks.approved():
            QMessageBox.information(self, APP_NAME, "Add seed picks first: left-click on the target reflection in the section view.")
            return
        g = self.g
        if self.auto_roi.isChecked():
            ils = [p.inline for p in self.picks.approved()]; xls = [p.crossline for p in self.picks.approved()]
            m = 25
            roi = ROI((max(g.ilines[0], min(ils) - m * g.il_step), min(g.ilines[-1], max(ils) + m * g.il_step)),
                      (max(g.xlines[0], min(xls) - m * g.xl_step), min(g.xlines[-1], max(xls) + m * g.xl_step)))
        else:
            roi = ROI((self.il_min.value(), self.il_max.value()), (self.xl_min.value(), self.xl_max.value()))
        il_s, xl_s, _ = roi.to_slices(g)
        n_cells = (il_s.stop - il_s.start) * (xl_s.stop - xl_s.start)
        if n_cells > 400_000 and QMessageBox.question(
                self, APP_NAME, f"The area has {n_cells:,} traces; this may take several minutes and a lot of memory. Continue?") != QMessageBox.Yes:
            return
        times = [p.time_ms for p in self.picks.approved()]
        roi = ROI(roi.inline_range, roi.crossline_range, (min(times) - self.margin.value(), max(times) + self.margin.value()))
        self._start(_extract, (self.vol, self.picks, roi, list(self.faults)), self._extraction_done, "Extracting horizon...")

    def _extraction_done(self, out):
        self.output = out
        shape = self.vol.live_mask.shape
        self.horizon_ms = out.full_grid(shape, "time_ms")
        self.conf_full = out.full_grid(shape, "confidence")
        s = out.summary
        lines = [
            f"<b>Horizon extracted</b> ({'converged' if s['converged'] else 'NOT converged'} in {s['iterations']} iterations, "
            f"{s['slopes_s'] + s['extract_s']:.0f} s)",
            f"Area: IL {s['roi']['inlines'][0]}–{s['roi']['inlines'][1]}, XL {s['roi']['crosslines'][0]}–{s['roi']['crosslines'][1]}",
            f"Picks used: {s['n_picks']} · fault cut-offs: {s['n_fault_cutoffs']}",
            f"In phase with the picked event: {s['fraction_in_phase']:.0%}",
            f"Mean confidence {s['confidence']['mean_confidence']:.2f}; {s['confidence']['fraction_reliable_ge_0.7']:.0%} of the area ≥ 0.7",
        ]
        if s["uninterpreted_live_cells"]:
            lines.append(f"<span style='color:#c0392b'>{s['uninterpreted_live_cells']} traces left blank (fault block without a pick)</span>")
        if self.reference is not None:
            from ..quality_control.horizon_residuals import compare_to_reference
            m = compare_to_reference(self.horizon_ms, self.g.time_ms(self.reference))
            lines.append(f"Vs reference: mean diff {m.mae_ms:.2f} ms, {m.within_10ms:.0%} within ±10 ms")
        for n in s["notes"]:
            lines.append(f"<i>{n}</i>")
        self.result_box.setHtml("<br>".join(lines))
        self.statusBar().showMessage("Horizon extracted. See the section, map and 3D tabs.")
        self._refresh_all()
        self.tabs.setCurrentIndex(2)

    # ------------------------------------------------------------------ drawing
    def _refresh_picks(self):
        t = self.pick_table
        t.setRowCount(len(self.picks.picks))
        for r, p in enumerate(self.picks.picks):
            for c, v in enumerate((f"{p.inline:g}", f"{p.crossline:g}", f"{p.time_ms:.1f}")):
                t.setItem(r, c, QTableWidgetItem(v))
        self.run_btn.setText(f"▶  Extract horizon ({len(self.picks.approved())} picks)")

    def _refresh_all(self):
        self._refresh_picks()
        self.fault_label.setText(
            "No fault sticks loaded." if not self.faults else
            f"{len(self.faults)} fault(s): " + ", ".join(f"{f.name} ({len(f.sticks)} sticks)" for f in self.faults)
            + "<br>They become hard barriers: the horizon is never smoothed across them.")
        self.draw_section()
        self.draw_map()

    def _section_data(self, inline_view: bool, idx: int) -> np.ndarray:
        key = (inline_view, idx)
        if key not in self._section_cache:
            if len(self._section_cache) > 40:
                self._section_cache.clear()
            self._section_cache[key] = np.asarray(self.vol.data[idx, :, :] if inline_view else self.vol.data[:, idx, :], np.float32)
        return self._section_cache[key]

    def draw_section(self):
        c = self.sec_canvas; ax = c.ax
        ax.clear()
        if self.vol is None:
            ax.text(0.5, 0.5, "File > Open SEG-Y... to start", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_axis_off(); c.draw_idle(); return
        ax.set_axis_on()
        g = self.g
        inline_view = self.direction.currentIndex() == 0
        idx = self.line_slider.value()
        line = int((g.ilines if inline_view else g.xlines)[idx])
        sec = self._section_data(inline_view, idx).T
        other = g.xlines if inline_view else g.ilines
        v = np.percentile(np.abs(sec), self.gain.value()) or 1.0
        ax.imshow(sec, cmap="gray_r", aspect="auto", vmin=-v, vmax=v, interpolation="bilinear",
                  extent=[other[0] - 0.5, other[-1] + 0.5, g.samples_ms[-1], g.samples_ms[0]])
        if self.reference is not None:
            r = self.reference[idx] if inline_view else self.reference[:, idx]
            ax.plot(other, g.time_ms(r), "--", color="#2ecc71", lw=1.2, label="reference")
        if self.horizon_ms is not None:
            h = self.horizon_ms[idx] if inline_view else self.horizon_ms[:, idx]
            ax.plot(other, h, "-", color="#f1c40f", lw=1.8, label="extracted horizon")
        for f in self.faults:
            for s in f.sticks:
                pos = s[:, 0] if inline_view else s[:, 1]
                m = np.abs(pos - idx) <= 3
                if m.sum() >= 2:
                    along = s[m, 1] if inline_view else s[m, 0]
                    ax.plot(other[0] + along * (g.xl_step if inline_view else g.il_step), g.time_ms(s[m, 2]), "-", color="#e74c3c", lw=1.5)
        on = [p for p in self.picks.picks if (p.inline == line if inline_view else p.crossline == line)]
        if on:
            ax.plot([p.crossline if inline_view else p.inline for p in on], [p.time_ms for p in on], "o",
                    mfc="#00d0ff", mec="k", ms=7, label="your picks")
        ax.set_xlabel("Crossline" if inline_view else "Inline"); ax.set_ylabel("TWT (ms)")
        ax.set_title(f"{'Inline' if inline_view else 'Crossline'} {line}")
        if self.horizon_ms is not None or self.reference is not None or on:
            ax.legend(loc="lower right", fontsize=8)
        if self.horizon_ms is not None:
            h = self.horizon_ms[idx] if inline_view else self.horizon_ms[:, idx]
            if np.isfinite(h).any():
                lo, hi = np.nanmin(h), np.nanmax(h)
                pad = max(150.0, (hi - lo) * 0.6)
                ax.set_ylim(min(g.samples_ms[-1], hi + pad), max(g.samples_ms[0], lo - pad))
        c.draw_idle()

    def draw_timeslice(self):
        c = self.ts_canvas; ax = c.ax; ax.clear()
        if self.vol is None:
            c.draw_idle(); return
        g = self.g
        k = self.t_slider.value()
        self.t_label.setText(f"{g.samples_ms[k]:.0f} ms")
        self.statusBar().showMessage("Reading time slice...")
        QApplication.processEvents()
        sl = np.asarray(self.vol.data[:, :, k], np.float32)
        v = np.percentile(np.abs(sl), 98) or 1.0
        ax.imshow(sl.T, origin="lower", cmap="gray_r", aspect="auto", vmin=-v, vmax=v,
                  extent=[g.ilines[0], g.ilines[-1], g.xlines[0], g.xlines[-1]])
        self._overlay_map(ax)
        ax.set_title(f"Time slice {g.samples_ms[k]:.0f} ms (click to jump to lines)")
        ax.set_xlabel("Inline"); ax.set_ylabel("Crossline")
        c.draw_idle()
        self.statusBar().showMessage("")

    def _overlay_map(self, ax):
        g = self.g
        if self.picks.picks:
            ax.plot([p.inline for p in self.picks.picks], [p.crossline for p in self.picks.picks], "+", color="#00d0ff", ms=7, mew=1.5)
        for f in self.faults:
            for s in f.sticks:
                ax.plot(g.ilines[0] + s[:, 0] * g.il_step, g.xlines[0] + s[:, 1] * g.xl_step, "-", color="#e74c3c", lw=0.8)
        if self.output is not None:
            i0, j0, _ = self.output.offsets
            for cut in self.output.cuts:
                ax.plot(g.ilines[0] + (cut[:, 0] + i0) * g.il_step, g.xlines[0] + (cut[:, 1] + j0) * g.xl_step, "-", color="#e74c3c", lw=2)
        if self.vol is not None:
            inline_view = self.direction.currentIndex() == 0
            line = self._current_line()
            if inline_view:
                ax.axvline(line, color="#f39c12", lw=0.8, ls=":")
            else:
                ax.axhline(line, color="#f39c12", lw=0.8, ls=":")

    def draw_map(self):
        c = self.map_canvas
        c.fig.clear(); ax = c.fig.add_subplot(111); c.ax = ax
        if self.vol is None:
            c.draw_idle(); return
        g = self.g
        kind = self.map_kind.currentText()
        ext = [g.ilines[0], g.ilines[-1], g.xlines[0], g.xlines[-1]]
        data, cmap, label, vlim = None, "viridis", "", (None, None)
        if kind == "Horizon time" and self.horizon_ms is not None:
            data, label = self.horizon_ms, "TWT (ms)"
        elif kind == "Confidence" and self.conf_full is not None:
            data, cmap, label, vlim = self.conf_full, "magma", "confidence (0-1, heuristic)", (0, 1)
        elif kind == "Difference to reference" and self.horizon_ms is not None and self.reference is not None:
            data, cmap, label, vlim = self.horizon_ms - g.time_ms(self.reference), "RdBu_r", "extracted − reference (ms)", (-12, 12)
        elif kind == "Reference horizon" and self.reference is not None:
            data, label = g.time_ms(self.reference), "TWT (ms)"
        if data is None:
            ax.text(0.5, 0.5, {"Horizon time": "Extract a horizon to see its map",
                               "Confidence": "Extract a horizon to see its confidence",
                               "Difference to reference": "Needs an extracted horizon and a reference horizon",
                               "Reference horizon": "Load a reference horizon (left panel)"}[kind],
                    ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=12)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        else:
            im = ax.imshow(data.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            c.fig.colorbar(im, ax=ax, label=label)
        self._overlay_map(ax)
        ax.set_xlabel("Inline"); ax.set_ylabel("Crossline"); ax.set_title(kind)
        c.draw_idle()

    def draw_3d(self):
        if self.plotter is None:
            try:
                from pyvistaqt import QtInteractor
            except Exception as exc:  # pragma: no cover
                self.view3d_holder.layout().addWidget(QLabel(f"3D view unavailable: {exc}")); return
            self.plotter = QtInteractor(self.view3d_holder)
            self.view3d_holder.layout().addWidget(self.plotter.interactor)
            row = QHBoxLayout()
            self.ve = QDoubleSpinBox(); self.ve.setRange(0.1, 50); self.ve.setValue(5); self.ve.setPrefix("vertical exaggeration ×")
            self.ve.valueChanged.connect(lambda _v: self.draw_3d())
            self.color3d = QComboBox(); self.color3d.addItems(["time", "confidence"])
            self.color3d.currentIndexChanged.connect(lambda _i: self.draw_3d())
            row.addWidget(self.ve); row.addWidget(QLabel("Colour by:")); row.addWidget(self.color3d); row.addStretch(1)
            self.view3d_holder.layout().addLayout(row)
        import pyvista as pv

        pl = self.plotter
        pl.clear()
        if self.horizon_ms is None or self.vol is None:
            pl.add_text("Extract a horizon to see it in 3D", font_size=12)
            pl.render(); return
        g = self.g
        o = self.output
        i0, j0, _ = o.offsets
        t = g.time_ms(o.result.t)
        ni, nj = t.shape
        ii, jj = np.meshgrid(np.arange(i0, i0 + ni), np.arange(j0, j0 + nj), indexing="ij")
        if g.affine is not None:
            x, y = g.world_xy(ii, jj)
        else:
            x, y = ii * 25.0, jj * 25.0
        tv = np.where(np.isfinite(t), t, np.nanmean(t))
        z = -tv * self.ve.value()
        grid = pv.StructuredGrid(x.astype(float), y.astype(float), z.astype(float))
        if self.color3d.currentText() == "time":
            grid["TWT (ms)"] = t.ravel(order="F")
            scal, cmap, clim = "TWT (ms)", "viridis_r", None
        else:
            grid["confidence"] = o.confidence.confidence.ravel(order="F")
            scal, cmap, clim = "confidence", "magma", (0, 1)
        bad = ~np.isfinite(t.ravel(order="F"))
        if bad.any():
            grid = grid.extract_points(~bad, adjacent_cells=False)
        pl.add_mesh(grid, scalars=scal, cmap=cmap, clim=clim, show_edges=False, nan_color="white", smooth_shading=True)
        for cut in o.cuts:
            cx, cy = (g.world_xy(cut[:, 0] + i0, cut[:, 1] + j0) if g.affine is not None else ((cut[:, 0] + i0) * 25.0, (cut[:, 1] + j0) * 25.0))
            ci = np.clip(np.rint(cut[:, 0]).astype(int), 0, ni - 1); cj = np.clip(np.rint(cut[:, 1]).astype(int), 0, nj - 1)
            cz = -tv[ci, cj] * self.ve.value() + 5
            pl.add_mesh(pv.lines_from_points(np.c_[cx, cy, cz]), color="red", line_width=4)
        pl.add_axes(); pl.show_bounds(grid="back", location="outer", xtitle="X", ytitle="Y", ztitle="−TWT × VE")
        pl.reset_camera(); pl.render()

    # ------------------------------------------------------------------ help
    def show_help(self):
        QMessageBox.information(self, "How to use", (
            "1. File > Open SEG-Y... (checked and converted once), or Open HDF5 volume...\n"
            "2. In 'Seismic section', move the slider to a line. Left-click on the target reflection to add a pick "
            "(it snaps to the nearest peak or trough). Right-click removes a pick; Ctrl+Z undoes.\n"
            "   Add picks on several inlines and crosslines, on both sides of any fault.\n"
            "3. Optional: 'Load fault sticks...' (faults become barriers) and 'Load reference horizon...' to compare.\n"
            "4. Press 'Extract horizon'. The area is chosen around your picks unless you set it yourself.\n"
            "5. Review the horizon on the sections, the 'Horizon map' (time, confidence, difference) and '3D view'.\n"
            "   Add or move picks where it is wrong and extract again.\n"
            "6. File > Export horizon (CSV) to use it in Petrel, OpendTect or Excel. File > Save picks to keep your work."))

    def closeEvent(self, ev):
        if self._thread is not None:
            QMessageBox.information(self, APP_NAME, "Please wait for the running task to finish.")
            ev.ignore(); return
        if self.vol:
            self.vol.close()
        if self.plotter is not None:
            self.plotter.close()
        ev.accept()


def _qc_markdown(qc: dict) -> str:
    lines = [f"# Seismic QC: {'PASSED' if qc.get('passed') else 'FAILED'}", "", "| Severity | Check | Finding |", "|---|---|---|"]
    for f in qc.get("findings", []):
        lines.append(f"| {f['severity']} | {f['code']} | {f['message']} |")
    geo = qc.get("stats", {}).get("geometry", {})
    if geo:
        lines += ["", "## Geometry", ""] + [f"- **{k}**: {v}" for k, v in geo.items()]
    return "\n".join(lines)


def _toolbar_mode(self) -> bool:
    tb = getattr(self, "toolbar", None)
    return bool(tb is not None and getattr(tb, "mode", ""))


FigureCanvasQTAgg.toolbar_mode = _toolbar_mode


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(make_icon())
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    args = (sys.argv if argv is None else argv)[1:]
    if args and args[0].lower().endswith((".h5", ".hdf5")):
        w.open_h5(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
