"""User-level workflows behind the CLI: make picks from an existing horizon, extract a horizon."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import PlatformConfig
from .horizon_engine.seed_picks import PickSet, SeedPick
from .preprocessing.roi_manager import ROI
from .seismic_io.hdf5_manager import SeismicVolume
from .seismic_io.interpretation_io import read_xyz_horizon


def picks_from_horizon(h5: str | Path, horizon_xyz: str | Path, out_csv: str | Path, spacing: int = 50,
                       time_shift_ms: float = 0.0, roi: ROI | None = None) -> int:
    """Sample an existing x y time horizon at every ``spacing``-th inline/crossline intersection.

    Picks are written as ``approved`` with ``source = "import:<file>"`` so their origin stays visible.
    """
    with SeismicVolume(h5) as v:
        g = v.geometry
        live = v.live_mask
    h = read_xyz_horizon(horizon_xyz, g, time_shift_ms=time_shift_ms)
    il_s, xl_s, _ = (roi or ROI()).to_slices(g)
    ps = PickSet()
    for i in range(il_s.start + spacing // 2, il_s.stop, spacing):
        for j in range(xl_s.start + spacing // 2, xl_s.stop, spacing):
            if np.isfinite(h[i, j]) and live[i, j]:
                ps.add(SeedPick(int(g.ilines[i]), int(g.xlines[j]), float(g.time_ms(h[i, j])),
                                source=f"import:{Path(horizon_xyz).name}"))
    ps.to_csv(out_csv)
    return len(ps.picks)


def extract_horizon(h5: str | Path, picks_csv: str | Path, out_dir: str | Path, cfg: PlatformConfig | None = None,
                    roi: ROI | None = None, fault_sticks: list[str] | None = None, fault_time_shift_ms: float = 0.0,
                    margin_ms: float = 120.0, show: bool = False) -> dict:
    """Seeds (+ optional fault sticks) -> slopes -> least-squares horizon -> QC -> CSV + figures.

    If no time range is given, the ROI is the pick time range +/- ``margin_ms``. Fault sticks
    are turned into horizon cut-offs using a first pass without barriers, then the horizon is
    re-extracted with the faults as hard barriers.
    """
    from .attributes.fault_probability import barriers_from_polylines
    from .attributes.structure_tensor import compute_slopes
    from .export.horizon_export import export_horizon_csv
    from .horizon_engine.horizon_3d import HorizonExtractor3D
    from .quality_control.phase_consistency import phase_consistency
    from .quality_control.uncertainty import confidence_map
    from .seismic_io.interpretation_io import read_fault_sticks

    cfg = cfg or PlatformConfig()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    picks = PickSet.from_csv(picks_csv)
    vol = SeismicVolume(h5)
    g = vol.geometry
    times = [p.time_ms for p in picks.approved()]
    roi = roi or ROI()
    if roi.time_range_ms is None:
        roi = ROI(roi.inline_range, roi.crossline_range, (min(times) - margin_ms, max(times) + margin_ms))
    il_s, xl_s, t_s = roi.to_slices(g)
    seis = vol.load(il_s, xl_s, t_s)
    live = vol.live_mask[il_s, xl_s]
    sub = type(g)(ilines=g.ilines[il_s], xlines=g.xlines[xl_s], samples_ms=g.samples_ms[t_s],
                  trace_index=g.trace_index[il_s, xl_s], cdp_x=g.cdp_x[il_s, xl_s], cdp_y=g.cdp_y[il_s, xl_s],
                  affine=None if g.affine is None else np.c_[g.affine[:, :2], g.affine[:, 0] * il_s.start + g.affine[:, 1] * xl_s.start + g.affine[:, 2]],
                  crs=g.crs)
    si, sj, st = picks.to_grid_constraints(sub)
    t0 = time.perf_counter()
    slopes = compute_slopes(seis, cfg.structure_tensor, live_mask=live)
    t_slopes = time.perf_counter() - t0

    barriers = None
    cuts = []
    if fault_sticks:
        first = HorizonExtractor3D(seis, slopes, cfg=cfg.horizon, live_mask=live).extract(si, sj, st)
        for f in fault_sticks:
            fs = read_fault_sticks(f, sub, time_shift_ms=fault_time_shift_ms)
            c = fs.cutoff_polyline(first.t)
            if len(c) >= 2:
                cuts.append(c)
        if cuts:
            barriers = barriers_from_polylines(*live.shape, cuts)
    t0 = time.perf_counter()
    res = HorizonExtractor3D(seis, slopes, cfg=cfg.horizon, live_mask=live, barriers=barriers).extract(si, sj, st)
    t_extract = time.perf_counter() - t0
    ph = phase_consistency(seis, res.t)
    cm = confidence_map(res, off_phase=ph["off_phase_mask"])
    export_horizon_csv(out / "horizon.csv", res, sub, confidence=cm.confidence, extra_flags={"off_phase": ph["off_phase_mask"]})
    res.provenance.write(out / "horizon.provenance.json")

    summary = {
        "roi": {"inlines": [int(sub.ilines[0]), int(sub.ilines[-1])], "crosslines": [int(sub.xlines[0]), int(sub.xlines[-1])],
                "time_ms": [float(sub.samples_ms[0]), float(sub.samples_ms[-1])]},
        "n_picks": int(si.size), "n_fault_cutoffs": len(cuts), "converged": res.converged,
        "iterations": len(res.iterations), "slopes_s": t_slopes, "extract_s": t_extract,
        "uninterpreted_live_cells": int(res.unconstrained.sum()),
        "phase_mean_deg": ph["circular_mean_deg"], "fraction_in_phase": ph["fraction_in_phase"],
        "confidence": cm.summary(), "notes": res.provenance.notes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _figures(out, seis, sub, res, cm, cuts, show)
    vol.close()
    return summary


def _figures(out: Path, seis, g, res, cm, cuts, show: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [g.ilines[0], g.ilines[-1], g.xlines[0], g.xlines[-1]]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    im = axs[0].imshow(g.time_ms(res.t).T, origin="lower", extent=ext, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=axs[0], label="TWT (ms)")
    si, sj, _ = res.seeds
    axs[0].plot(g.ilines[0] + si, g.xlines[0] + sj, "w+", ms=5, label="picks")
    for c in cuts:
        axs[0].plot(g.ilines[0] + c[:, 0], g.xlines[0] + c[:, 1], "r-", lw=1.5)
    axs[0].set_title("Extracted horizon"); axs[0].set_xlabel("inline"); axs[0].set_ylabel("crossline"); axs[0].legend(fontsize=8)
    im = axs[1].imshow(cm.confidence.T, origin="lower", extent=ext, aspect="auto", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[1]); axs[1].set_title("Heuristic confidence (review low areas)")
    p1 = out / "horizon_map.png"; fig.tight_layout(); fig.savefig(p1, dpi=110); plt.close(fig)

    i = seis.shape[0] // 2
    fig, ax = plt.subplots(figsize=(14, 5.5))
    sec = seis[i].T; vv = np.percentile(np.abs(sec), 98)
    ax.imshow(sec, cmap="gray_r", aspect="auto", vmin=-vv, vmax=vv, extent=[g.xlines[0], g.xlines[-1], g.samples_ms[-1], g.samples_ms[0]])
    ax.plot(g.xlines, g.time_ms(res.t[i]), "y-", lw=1.5, label="extracted horizon")
    ax.set_title(f"Inline {g.ilines[i]}"); ax.set_xlabel("crossline"); ax.set_ylabel("TWT (ms)"); ax.legend()
    p2 = out / "horizon_section.png"; fig.tight_layout(); fig.savefig(p2, dpi=110); plt.close(fig)
    if show:
        import os
        for p in (p1, p2):
            os.startfile(str(p.resolve()))  # type: ignore[attr-defined]
