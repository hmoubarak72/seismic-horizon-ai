"""SEG-Y header scanning and streamed trace reading (segyio backend).

The reader never trusts segyio's automatic geometry inference: it scans the trace
headers itself (configurable byte locations), builds a :class:`SurveyGeometry`, and
hands everything to the QC module, which decides whether the file is safe to use.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import segyio

from ..config import SegyConfig
from ..logging_utils import get_logger
from .survey_geometry import SurveyGeometry

log = get_logger("seismic_io.segy_reader")

_CRS_PATTERNS = [
    re.compile(r"EPSG\s*[:=]?\s*(\d{4,6})", re.I),
    re.compile(r"\b(ED50|WGS\s*84|ETRS\s*89|NAD\s*27|NAD\s*83)\b[^\n]{0,40}?UTM[^\n]{0,20}?(\d{1,2}\s*[NS]?)", re.I),
]


@dataclasses.dataclass
class SegyScan:
    """Everything learned from the SEG-Y headers, before any amplitude is read."""

    path: Path
    n_traces: int
    n_samples: int
    sample_interval_us: float  # binary header (or segyio's best guess)
    trace_sample_intervals_us: np.ndarray  # unique per-trace values (byte 117)
    trace_sample_counts: np.ndarray  # unique per-trace values (byte 115)
    samples_ms: np.ndarray
    iline: np.ndarray
    xline: np.ndarray
    cdp_x: np.ndarray
    cdp_y: np.ndarray
    coord_scalars: np.ndarray  # unique raw scalars (byte 71)
    format_code: int
    measurement_system: str | None
    textual_header: str
    crs: str | None

    def build_geometry(self) -> tuple[SurveyGeometry, np.ndarray]:
        return SurveyGeometry.from_headers(
            self.iline,
            self.xline,
            self.samples_ms,
            self.cdp_x,
            self.cdp_y,
            crs=self.crs,
            measurement_system=self.measurement_system,
        )


def _apply_scalar(values: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Apply the SEG-Y coordinate scalar convention (negative = divide)."""
    v = values.astype(float)
    s = scalars.astype(float)
    out = np.where(s > 0, v * s, np.where(s < 0, v / np.abs(np.where(s == 0, 1, s)), v))
    return out


def _detect_crs(text: str) -> str | None:
    m = _CRS_PATTERNS[0].search(text)
    if m:
        return f"EPSG:{m.group(1)}"
    m = _CRS_PATTERNS[1].search(text)
    if m:
        return f"{m.group(1).upper()} UTM {m.group(2).strip().upper()} (from textual header; verify)"
    return None


def scan_segy(path: str | Path, cfg: SegyConfig | None = None) -> SegyScan:
    """Read all trace headers needed for geometry and QC (no amplitudes)."""
    cfg = cfg or SegyConfig()
    path = Path(path)
    with segyio.open(str(path), "r", ignore_geometry=True, endian=cfg.endian) as f:
        n = f.tracecount
        il = np.asarray(f.attributes(cfg.iline_byte)[:], dtype=np.int64)
        xl = np.asarray(f.attributes(cfg.xline_byte)[:], dtype=np.int64)
        raw_x = np.asarray(f.attributes(cfg.cdp_x_byte)[:], dtype=np.int64)
        raw_y = np.asarray(f.attributes(cfg.cdp_y_byte)[:], dtype=np.int64)
        scal = np.asarray(f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.int64)
        tr_dt = np.asarray(f.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[:], dtype=np.int64)
        tr_ns = np.asarray(f.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[:], dtype=np.int64)
        bin_dt = float(f.bin[segyio.BinField.Interval])
        try:
            dt_us = float(segyio.tools.dt(f, fallback_dt=0.0))
        except Exception:
            dt_us = bin_dt
        fmt = int(f.bin[segyio.BinField.Format])
        ms_code = int(f.bin[segyio.BinField.MeasurementSystem])
        try:
            text = segyio.tools.wrap(f.text[0])
        except Exception:
            text = ""
        n_samples = len(f.samples)
        t0_ms = float(f.samples[0]) if n_samples else 0.0
    samples = t0_ms + np.arange(n_samples) * (dt_us / 1000.0)
    scan = SegyScan(
        path=path,
        n_traces=n,
        n_samples=n_samples,
        sample_interval_us=dt_us,
        trace_sample_intervals_us=np.unique(tr_dt),
        trace_sample_counts=np.unique(tr_ns),
        samples_ms=samples,
        iline=il,
        xline=xl,
        cdp_x=_apply_scalar(raw_x, scal),
        cdp_y=_apply_scalar(raw_y, scal),
        coord_scalars=np.unique(scal),
        format_code=fmt,
        measurement_system={1: "meters", 2: "feet"}.get(ms_code),
        textual_header=text,
        crs=_detect_crs(text),
    )
    log.info("scanned SEG-Y", extra={"path": str(path), "n_traces": n, "n_samples": n_samples, "dt_us": dt_us})
    return scan


def iter_trace_blocks(
    path: str | Path, block_size: int = 4096, cfg: SegyConfig | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_trace_number, traces[block, n_samples])`` blocks (float32)."""
    cfg = cfg or SegyConfig()
    with segyio.open(str(path), "r", ignore_geometry=True, endian=cfg.endian) as f:
        n = f.tracecount
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            yield start, np.asarray(f.trace.raw[start:stop], dtype=np.float32).reshape(stop - start, -1)


def read_cube(
    path: str | Path,
    geometry: SurveyGeometry,
    il_slice: slice = slice(None),
    xl_slice: slice = slice(None),
    t_slice: slice = slice(None),
    cfg: SegyConfig | None = None,
) -> np.ndarray:
    """Read a (sub)cube into memory. Missing bins are zero-filled (see ``geometry.live_mask``).

    Intended for small volumes and ROIs; use :func:`..hdf5_manager.convert_segy_to_hdf5`
    for full surveys.
    """
    cfg = cfg or SegyConfig()
    tidx = geometry.trace_index[il_slice, xl_slice]
    n_t = len(range(*t_slice.indices(geometry.samples_ms.size)))
    cube = np.zeros(tidx.shape + (n_t,), dtype=np.float32)
    live = np.argwhere(tidx >= 0)
    if live.size == 0:
        return cube
    with segyio.open(str(path), "r", ignore_geometry=True, endian=cfg.endian) as f:
        for i, j in live:
            cube[i, j] = np.asarray(f.trace.raw[int(tidx[i, j])], dtype=np.float32)[t_slice]
    return cube


def write_segy(
    path: str | Path,
    cube: np.ndarray,
    dt_ms: float = 4.0,
    il0: int = 1,
    xl0: int = 1,
    il_step: int = 1,
    xl_step: int = 1,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    bin_size: tuple[float, float] = (25.0, 25.0),
    coord_scalar: int = -100,
    text_lines: dict[int, str] | None = None,
    skip_bins: set[tuple[int, int]] | None = None,
    duplicate_bins: set[tuple[int, int]] | None = None,
) -> Path:
    """Write a regular cube to SEG-Y (IEEE float). Used for tests and synthetic demos.

    ``skip_bins`` omits traces (missing traces) and ``duplicate_bins`` writes a second
    trace for the given bins, so QC can be exercised.
    """
    path = Path(path)
    n_il, n_xl, n_t = cube.shape
    skip_bins = skip_bins or set()
    duplicate_bins = duplicate_bins or set()
    bins = [(i, j) for i in range(n_il) for j in range(n_xl) if (i, j) not in skip_bins]
    bins += sorted(duplicate_bins)
    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(np.arange(n_t) * dt_ms)
    spec.tracecount = len(bins)
    spec.sorting = 2
    with segyio.create(str(path), spec) as f:
        lines = text_lines or {1: "SYNTHETIC VOLUME WRITTEN BY seismic_horizon_ai"}
        f.text[0] = segyio.tools.create_text_header({k: v[:76] for k, v in lines.items()})
        f.bin.update(hns=n_t, hdt=int(round(dt_ms * 1000)), format=5, mfeet=1)
        for k, (i, j) in enumerate(bins):
            x = origin_xy[0] + i * bin_size[0]
            y = origin_xy[1] + j * bin_size[1]
            scale = abs(coord_scalar) if coord_scalar < 0 else 1
            f.header[k] = {
                segyio.TraceField.INLINE_3D: il0 + i * il_step,
                segyio.TraceField.CROSSLINE_3D: xl0 + j * xl_step,
                segyio.TraceField.CDP_X: int(round(x * scale)),
                segyio.TraceField.CDP_Y: int(round(y * scale)),
                segyio.TraceField.SourceGroupScalar: coord_scalar,
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_t,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(round(dt_ms * 1000)),
            }
            f.trace[k] = np.ascontiguousarray(cube[i, j], dtype=np.float32)
    return path
