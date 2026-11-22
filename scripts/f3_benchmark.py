"""F3 Netherlands benchmark: horizon FS8 across Fault A.

Usage (from the repository root):
    python scripts/f3_benchmark.py --h5 C:/SeismicData/F3/f3_seismic.h5 --f3 ../F3_Demo_2023 --out outputs/f3

Reference = dGB's interpreted FS8 horizon (expert interpretation, not ground truth).
The reference times are shifted by +4 ms: the raw SEG-Y starts at 4 ms (trace delay) while the
OpendTect exports count the first sample as 0 ms. Three independent horizons (FS8, MFS4,
Truncation) peak exactly one sample later than their file times, which is the evidence for this static.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from seismic_horizon_ai.attributes.fault_probability import barriers_from_polylines, discontinuity_proxy
from seismic_horizon_ai.attributes.instantaneous import dominant_period_samples
from seismic_horizon_ai.attributes.structure_tensor import compute_slopes
from seismic_horizon_ai.config import HorizonConfig
from seismic_horizon_ai.horizon_engine.horizon_3d import HorizonExtractor3D
from seismic_horizon_ai.horizon_engine.peak_tracking import peak_track
from seismic_horizon_ai.quality_control.horizon_residuals import compare_to_reference, cycle_skip_candidates
from seismic_horizon_ai.quality_control.phase_consistency import phase_consistency
from seismic_horizon_ai.quality_control.uncertainty import confidence_map
from seismic_horizon_ai.seismic_io.hdf5_manager import SeismicVolume
from seismic_horizon_ai.seismic_io.interpretation_io import read_fault_sticks, read_xyz_horizon

STATIC_MS = 4.0


def rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--f3", required=True)
    ap.add_argument("--out", default="outputs/f3")
    ap.add_argument("--il", nargs=2, type=int, default=[0, 350], help="inline index range of the ROI")
    ap.add_argument("--xl", nargs=2, type=int, default=[450, 951], help="crossline index range of the ROI")
    ap.add_argument("--seed-spacing", type=int, default=50, help="interpreter line spacing for seed picks (traces)")
    ap.add_argument("--open", action="store_true", help="open figures on screen as they are written")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.f3) / "Rawdata"

    def show(p: Path) -> None:
        if a.open:
            import os
            os.startfile(str(p.resolve()))  # type: ignore[attr-defined]

    vol = SeismicVolume(a.h5)
    g = vol.geometry
    ref_full = read_xyz_horizon(raw / "Surface_data" / "F3-Horizon-FS8.txt", g, time_shift_ms=STATIC_MS)
    sticks = read_fault_sticks(raw / "Faults" / "FaultA.txt", g, time_shift_ms=STATIC_MS)
    cut_full = sticks.cutoff_polyline(ref_full)

    il_s, xl_s = slice(*a.il), slice(*a.xl)
    ref_roi = ref_full[il_s, xl_s]
    t_lo = int(max(0, np.nanmin(ref_roi) - 30)); t_hi = int(min(g.samples_ms.size, np.nanmax(ref_roi) + 30))
    t_s = slice(t_lo, t_hi)
    t0 = time.perf_counter()
    seis = vol.load(il_s, xl_s, t_s)
    live = vol.live_mask[il_s, xl_s]
    t_load = time.perf_counter() - t0
    ref = ref_roi - t_lo
    cut = cut_full - np.array([il_s.start, xl_s.start])
    n_il, n_xl, n_t = seis.shape
    print(f"ROI {seis.shape}, inlines {g.ilines[il_s][0]}-{g.ilines[il_s][-1]}, crosslines {g.xlines[xl_s][0]}-{g.xlines[xl_s][-1]}, "
          f"{g.samples_ms[t_lo]:.0f}-{g.samples_ms[t_hi - 1]:.0f} ms")

    t0 = time.perf_counter()
    slopes = compute_slopes(seis, live_mask=live)
    t_slopes = time.perf_counter() - t0
    fprob = discontinuity_proxy(slopes.planarity)
    period = dominant_period_samples(seis)
    barriers = barriers_from_polylines(n_il, n_xl, [cut])

    # interpreter-style seeds: reference picks at intersections of every Nth inline and crossline,
    # avoiding the 4 traces either side of the fault cut-off (picks there are ambiguous)
    near_fault = ndimage.binary_dilation(barriers.cell_mask(), iterations=4)
    sp = a.seed_spacing
    gi, gj = np.meshgrid(np.arange(sp // 2, n_il, sp), np.arange(sp // 2, n_xl, sp), indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    keep = np.isfinite(ref[gi, gj]) & live[gi, gj] & ~near_fault[gi, gj]
    si, sj = gi[keep], gj[keep]
    st = ref[si, sj]
    print(f"{si.size} seeds (every {sp} lines), dominant period {period:.1f} samples, slopes {t_slopes:.1f} s")

    runs: dict[str, tuple[np.ndarray, float, object]] = {}
    t0 = time.perf_counter()
    k = int(np.argmin(np.hypot(si - n_il / 2, sj - n_xl / 2)))
    pk = peak_track(seis, (int(si[k]), int(sj[k]), float(st[k])), search=2, polarity=1, live=live)
    runs["Conventional peak tracking (1 seed)"] = (pk, time.perf_counter() - t0, None)
    variants = {
        "Least squares, no fault constraint": {},
        "Least squares + Fault A barrier": {"barriers": barriers},
        "Least squares + barrier + fault-probability weights": {"barriers": barriers, "fault_probability": fprob},
    }
    final = None
    for name, kw in variants.items():
        t0 = time.perf_counter()
        res = HorizonExtractor3D(seis, slopes, cfg=HorizonConfig(), live_mask=live, **kw).extract(si, sj, st)
        runs[name] = (res.t, time.perf_counter() - t0, res)
        print(f"{name}: {runs[name][1]:.1f} s, converged {res.converged} in {len(res.iterations)} it")
        if "barrier" in name and final is None:
            final = res

    dt = g.dt_ms
    ref_ms = np.where(live, ref, np.nan) * dt
    near = ndimage.binary_dilation(barriers.cell_mask(), iterations=10)
    rows = []
    for name, (t, secs, res) in runs.items():
        m_all = compare_to_reference(t * dt, ref_ms)
        m_near = compare_to_reference(np.where(near, t, np.nan) * dt, np.where(near, ref_ms, np.nan))
        cs = int(cycle_skip_candidates(res, slopes.p, slopes.q, period).sum()) if res is not None else None
        rows.append({"method": name, "all": m_all.as_dict(), "near_fault": m_near.as_dict(), "runtime_s": secs, "cycle_skip_cells": cs})

    ph = phase_consistency(seis, final.t, reference_phase=0.0)
    cm = confidence_map(final, off_phase=ph["off_phase_mask"])
    summary = {
        "roi": {"shape": seis.shape, "inlines": [int(g.ilines[il_s][0]), int(g.ilines[il_s][-1])],
                "crosslines": [int(g.xlines[xl_s][0]), int(g.xlines[xl_s][-1])],
                "time_ms": [float(g.samples_ms[t_lo]), float(g.samples_ms[t_hi - 1])]},
        "n_seeds": int(si.size), "seed_spacing": sp, "static_ms": STATIC_MS,
        "timings_s": {"load": t_load, "slopes": t_slopes}, "peak_rss_gb_approx": rss_gb(),
        "dominant_period_samples": period, "methods": rows,
        "final_qc": {"phase_circular_mean_deg": ph["circular_mean_deg"], "fraction_in_phase": ph["fraction_in_phase"],
                     "confidence": cm.summary(), "iterations": final.iterations[-1]},
    }
    (out / "f3_benchmark.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [g.ilines[il_s][0], g.ilines[il_s][-1], g.xlines[xl_s][0], g.xlines[xl_s][-1]]
    tms = lambda t: g.samples_ms[t_lo] + t * dt  # noqa: E731
    names = [n for n in runs if runs[n][2] is not None]
    fig, axs = plt.subplots(1, len(names) + 1, figsize=(5 * (len(names) + 1), 4.6))
    im = axs[0].imshow(tms(ref).T, origin="lower", extent=ext, aspect="auto", cmap="viridis")
    axs[0].set_title("Reference FS8 (dGB), ms"); fig.colorbar(im, ax=axs[0])
    axs[0].plot(g.ilines[il_s][0] + cut[:, 0], g.xlines[xl_s][0] + cut[:, 1], "r-", lw=1.5)
    axs[0].plot(g.ilines[il_s][0] + si, g.xlines[xl_s][0] + sj, "w+", ms=5)
    for ax, n in zip(axs[1:], names):
        err = (runs[n][0] - ref) * dt
        im = ax.imshow(err.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-12, vmax=12)
        ax.plot(g.ilines[il_s][0] + cut[:, 0], g.xlines[xl_s][0] + cut[:, 1], "k-", lw=1)
        ax.set_title(n.replace("Least squares", "LSQ"), fontsize=9)
    fig.colorbar(im, ax=axs[1:], label="extracted - reference (ms)", shrink=0.85)
    p = out / "02_f3_error_maps.png"; fig.savefig(p, dpi=110); plt.close(fig); show(p)

    il_sec = int(np.clip(np.median(cut[:, 0]), 0, n_il - 1))
    fig, ax = plt.subplots(figsize=(14, 6))
    sec = seis[il_sec].T; vv = np.percentile(np.abs(sec), 98)
    xls = g.xlines[xl_s]
    ax.imshow(sec, cmap="gray_r", aspect="auto", vmin=-vv, vmax=vv, extent=[xls[0], xls[-1], tms(n_t - 1), tms(0)])
    ax.plot(xls, tms(ref[il_sec]), "y-", lw=3, alpha=0.7, label="reference FS8")
    for (n, (t, _, _)), sty in zip(runs.items(), ["c:", "r--", "b-", "g-."]):
        ax.plot(xls, tms(t[il_sec]), sty, lw=1.3, label=n)
    ax.set_title(f"Inline {g.ilines[il_s][il_sec]} across Fault A"); ax.set_xlabel("crossline"); ax.set_ylabel("TWT (ms)")
    ax.legend(fontsize=8, loc="lower left")
    p = out / "03_f3_section.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); show(p)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    im = axs[0].imshow(cm.confidence.T, origin="lower", extent=ext, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axs[0].set_title("Heuristic confidence (final horizon)"); fig.colorbar(im, ax=axs[0])
    lim = np.where(np.isfinite(cm.confidence), cm.limiting_driver, np.nan)
    im = axs[1].imshow(lim.T, origin="lower", extent=ext, aspect="auto", cmap="tab10", vmin=-0.5, vmax=9.5)
    cb = fig.colorbar(im, ax=axs[1], ticks=range(len(cm.driver_names))); cb.ax.set_yticklabels(cm.driver_names, fontsize=7)
    axs[1].set_title("Limiting confidence driver")
    p = out / "04_f3_confidence.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); show(p)

    lines = ["| Method | Coverage | MAE ms | RMS ms | ±10 ms | ±20 ms | MAE near fault ms | Cycle-skip cells | Runtime s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, mn = r["all"], r["near_fault"]
        lines.append(f"| {r['method']} | {m['coverage']:.1%} | {m['mae_ms']:.2f} | {m['rms_ms']:.2f} | {m['within_10ms']:.1%} | "
                     f"{m['within_20ms']:.1%} | {mn['mae_ms']:.2f} | {r['cycle_skip_cells'] if r['cycle_skip_cells'] is not None else 'n/a'} | {r['runtime_s']:.1f} |")
    (out / "f3_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
