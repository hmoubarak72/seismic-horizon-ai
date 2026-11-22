"""Chunked HDF5 storage for validated seismic volumes (out-of-core, lazy reads).

Layout of a volume file::

    /seismic          float32 (n_il, n_xl, n_t), chunked, measured amplitudes
    /live_mask        bool    (n_il, n_xl), False for missing or dead traces
    /ilines, /xlines  int64 line numbers;  /samples_ms float64
    /cdp_x, /cdp_y    float64 (n_il, n_xl), NaN where missing
    attrs: geometry (json), qc_report (json), provenance (json), data_kind="measured"

Derived attributes are written to separate files/groups with ``data_kind="derived"``
so measured data and derived products are never confused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import segyio

from ..config import SegyConfig
from ..logging_utils import get_logger
from ..preprocessing.roi_manager import ROI
from ..preprocessing.seismic_qc import QCReport, run_segy_qc
from ..provenance import Provenance, sha256_file
from .survey_geometry import SurveyGeometry

log = get_logger("seismic_io.hdf5")


def _chunks(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(1, min(s, c)) for s, c in zip(shape, chunks))


def convert_segy_to_hdf5(
    segy_path: str | Path,
    out_path: str | Path,
    cfg: SegyConfig | None = None,
    roi: ROI | None = None,
    require_qc_pass: bool = True,
    inline_slab: int | None = None,
) -> tuple[Path, QCReport]:
    """QC a SEG-Y file and stream it into a chunked HDF5 volume.

    The conversion refuses to run (raises :class:`SeismicQCError`) when QC has critical
    findings, unless ``require_qc_pass=False`` is given explicitly.
    """
    cfg = cfg or SegyConfig()
    segy_path, out_path = Path(segy_path), Path(out_path)
    scan, geometry, report, dead = run_segy_qc(segy_path, cfg)
    if require_qc_pass:
        report.raise_if_failed()
    if geometry is None:
        raise RuntimeError("Geometry could not be established; see QC report.")
    il_s, xl_s, t_s = (roi or ROI()).to_slices(geometry)
    tidx = geometry.trace_index[il_s, xl_s]
    live = (tidx >= 0) & ~dead[il_s, xl_s]
    n_il, n_xl = tidx.shape
    n_t = len(range(*t_s.indices(geometry.samples_ms.size)))
    chunks = _chunks((n_il, n_xl, n_t), cfg.hdf5_chunks)
    slab = inline_slab or chunks[0]

    prov = Provenance(
        product="seismic_volume_hdf5",
        parameters={"segy_config": cfg.__dict__, "roi": roi.__dict__ if roi else None},
        inputs={str(segy_path.name): sha256_file(segy_path)},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5, segyio.open(str(segy_path), "r", ignore_geometry=True, endian=cfg.endian) as f:
        ds = h5.create_dataset("seismic", shape=(n_il, n_xl, n_t), dtype="f4", chunks=chunks, fillvalue=0.0)
        for i0 in range(0, n_il, slab):
            i1 = min(n_il, i0 + slab)
            buf = np.zeros((i1 - i0, n_xl, n_t), dtype=np.float32)
            for i, j in np.argwhere(live[i0:i1]):
                buf[i, j] = np.asarray(f.trace.raw[int(tidx[i0 + i, j])], dtype=np.float32)[t_s]
            ds[i0:i1] = buf
        h5.create_dataset("live_mask", data=live)
        h5.create_dataset("ilines", data=geometry.ilines[il_s])
        h5.create_dataset("xlines", data=geometry.xlines[xl_s])
        h5.create_dataset("samples_ms", data=geometry.samples_ms[t_s])
        h5.create_dataset("cdp_x", data=geometry.cdp_x[il_s, xl_s])
        h5.create_dataset("cdp_y", data=geometry.cdp_y[il_s, xl_s])
        h5.create_dataset("trace_index", data=tidx)
        geo = json.loads(geometry.to_json())
        if geometry.affine is not None:
            # shift the affine so that index 0 of the ROI maps correctly
            A = geometry.affine.copy()
            A[:, 2] = A[:, 0] * il_s.start + A[:, 1] * xl_s.start + A[:, 2]
            geo["affine"] = A.tolist()
        for k in ("ilines", "xlines", "samples_ms"):
            geo.pop(k)
        ds.attrs["units"] = "amplitude (as recorded; unit not encoded in SEG-Y)"
        h5.attrs["data_kind"] = "measured"
        h5.attrs["geometry"] = json.dumps(geo)
        h5.attrs["qc_report"] = report.to_json()
        h5.attrs["provenance"] = prov.to_json()
    log.info("wrote HDF5 volume", extra={"out": str(out_path), "shape": (n_il, n_xl, n_t), "chunks": chunks})
    return out_path, report


class SeismicVolume:
    """Lazy, read-only access to an HDF5 seismic volume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        self.data: h5py.Dataset = self._h5["seismic"]
        self.live_mask: np.ndarray = self._h5["live_mask"][...]
        geo = json.loads(self._h5.attrs["geometry"])
        affine = None if geo.get("affine") is None else np.asarray(geo["affine"])
        self.geometry = SurveyGeometry(
            ilines=self._h5["ilines"][...],
            xlines=self._h5["xlines"][...],
            samples_ms=self._h5["samples_ms"][...],
            trace_index=self._h5["trace_index"][...],
            cdp_x=self._h5["cdp_x"][...],
            cdp_y=self._h5["cdp_y"][...],
            affine=affine,
            affine_rms=geo.get("affine_rms"),
            crs=geo.get("crs"),
            domain=geo.get("domain", "time"),
            measurement_system=geo.get("measurement_system"),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def qc_report(self) -> dict[str, Any]:
        return json.loads(self._h5.attrs["qc_report"])

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._h5.attrs["provenance"])

    def load(self, il: slice = slice(None), xl: slice = slice(None), t: slice = slice(None)) -> np.ndarray:
        return np.asarray(self.data[il, xl, t], dtype=np.float32)

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "SeismicVolume":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_attribute_hdf5(path: str | Path, arrays: dict[str, np.ndarray], provenance: Provenance, units: dict[str, str] | None = None) -> Path:
    """Store derived attributes with units and provenance (``data_kind="derived"``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for name, arr in arrays.items():
            chunks = _chunks(arr.shape, (64,) * arr.ndim) if arr.ndim else None
            d = h5.create_dataset(name, data=arr, chunks=chunks)
            if units and name in units:
                d.attrs["units"] = units[name]
        h5.attrs["data_kind"] = "derived"
        h5.attrs["provenance"] = provenance.to_json()
    return path
