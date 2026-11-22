"""End-to-end synthetic benchmark: SEG-Y -> QC -> HDF5 -> slopes -> horizons -> QC -> export.

Compares conventional peak tracking with the least-squares engine with and without
fault constraints, against the *known* synthetic horizon. On real data (e.g. Volve
BCU) the reference is an expert interpretation, not ground truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .attributes.fault_probability import barriers_from_polylines, discontinuity_proxy
from .attributes.instantaneous import dominant_period_samples
from .attributes.structure_tensor import compute_slopes
from .config import HorizonConfig, PlatformConfig
from .export.horizon_export import export_horizon_csv
from .horizon_engine.horizon_3d import HorizonExtractor3D
from .horizon_engine.peak_tracking import peak_track
from .horizon_engine.seed_picks import PickSet, SeedPick
from .quality_control.horizon_residuals import compare_to_reference, cycle_skip_candidates, spikes
from .quality_control.phase_consistency import phase_consistency
from .quality_control.uncertainty import confidence_map
from .seismic_io.hdf5_manager import SeismicVolume, convert_segy_to_hdf5
from .seismic_io.segy_reader import write_segy
from .synthetic import make_synthetic

TAU = 80.0


def _seeds(sv, live, rng, n_each=8):
    from scipy import ndimage

    side = sv.fault_side
    band = ndimage.binary_dilation(side, iterations=4) & ~ndimage.binary_erosion(side, iterations=4)
    out = []
    for block in (side, ~side):
        cells = np.argwhere(block & ~band & live)
        out.append(cells[rng.choice(len(cells), n_each, replace=False)])
    return np.concatenate(out)


def run_demo(out: Path, seed: int = 11) -> str:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = PlatformConfig()
    sv = make_synthetic(n_il=80, n_xl=70, n_t=200, throw=7.0, noise=0.35, seed=seed)
    truth = sv.horizon(TAU)

    # 1. SEG-Y with realistic defects
    segy = write_segy(out / "synthetic_faulted.sgy", sv.seismic, dt_ms=sv.dt_ms, il0=1000, xl0=2000,
                      origin_xy=(435000.0, 6477000.0), bin_size=(25.0, 12.5),
                      skip_bins={(3, 60), (3, 61)}, duplicate_bins={(40, 10)},
                      text_lines={1: "SYNTHETIC FAULTED BENCHMARK (seismic_horizon_ai demo)", 2: "CRS EPSG:23031"})
    cube_with_dead = sv.seismic.copy()
    # 2. QC + conversion
    h5, report = convert_segy_to_hdf5(segy, out / "synthetic_faulted.h5", cfg.segy)
    (out / "seismic_qc_report.md").write_text(report.to_markdown(), encoding="utf-8")
    with SeismicVolume(h5) as vol:
        seis = vol.load()
        live = vol.live_mask
        geom = vol.geometry
    del cube_with_dead

    # 3. attributes
    t0 = time.perf_counter()
    slopes = compute_slopes(seis, cfg.structure_tensor, live_mask=live)
    t_slopes = time.perf_counter() - t0
    fprob = discontinuity_proxy(slopes.planarity)
    period = dominant_period_samples(seis)

    # 4. interpreter picks (+ one AI proposal that must NOT be used until approved)
    cells = _seeds(sv, live, rng)
    picks = PickSet([SeedPick(geom.ilines[i], geom.xlines[j], float(geom.time_ms(truth[i, j])), phase="trough",
                              interpreter="synthetic-truth") for i, j in cells])
    picks.add(SeedPick(geom.ilines[5], geom.xlines[5], float(geom.time_ms(truth[5, 5] + period)), status="proposed",
                       source="ai:waveform_similarity", phase="trough"))
    picks.to_csv(out / "seed_picks.csv")
    si, sj, st = picks.to_grid_constraints(geom)

    fl = sv.fault_line
    barriers = barriers_from_polylines(*live.shape, [np.c_[fl[0], fl[1]]])

    # 5. methods
    runs = {}
    t0 = time.perf_counter()
    pk = peak_track(seis, (int(si[0]), int(sj[0]), float(st[0])), search=3, polarity=-1, live=live)
    runs["Conventional peak tracking (1 seed)"] = (pk, time.perf_counter() - t0, None)
    variants = {
        "Least squares, no fault constraint": dict(),
        "Least squares + interpreted fault barrier": dict(barriers=barriers),
        "Least squares + barrier + fault-probability weights": dict(barriers=barriers, fault_probability=fprob),
    }
    final = None
    for name, kw in variants.items():
        t0 = time.perf_counter()
        res = HorizonExtractor3D(seis, slopes, cfg=HorizonConfig(), live_mask=live, **kw).extract(si, sj, st)
        runs[name] = (res.t, time.perf_counter() - t0, res)
        final = res

    rows = []
    truth_ms = geom.time_ms(np.where(live, truth, np.nan))
    for name, (t, secs, res) in runs.items():
        m = compare_to_reference(geom.time_ms(t), truth_ms)
        cs = cycle_skip_candidates(res, slopes.p, slopes.q, period).sum() if res is not None else None
        rows.append((name, m, secs, cs, res))

    # 6. QC of the final horizon
    ph = phase_consistency(seis, final.t, reference_phase=np.pi)
    cm = confidence_map(final, off_phase=ph["off_phase_mask"])
    sp = spikes(final.t, barrier_cells=final.barriers.cell_mask())
    export_horizon_csv(out / "horizon_final.csv", final, geom, confidence=cm.confidence,
                       extra_flags={"off_phase": ph["off_phase_mask"], "spike": sp})
    final.provenance.write(out / "horizon_final.provenance.json")

    # 7. figures
    figs = _figures(out, seis, truth, runs, final, cm, geom, live)

    # 8. report
    lines = [
        "# Synthetic benchmark: faulted volume", "",
        f"Volume {seis.shape} (inline x crossline x samples), dt {geom.dt_ms} ms, noise {sv.noise} x RMS, "
        f"fault throw {sv.throw} samples ({sv.throw * geom.dt_ms:.0f} ms), dominant period {period:.1f} samples.",
        f"Seeds: {len(si)} approved interpreter picks (8 per fault block); 1 AI-proposed pick ignored until approved.",
        f"Structure-tensor slopes: {t_slopes:.2f} s. SEG-Y QC: {'PASSED' if report.passed else 'FAILED'} with "
        f"{sum(f.severity == 'warning' for f in report.findings)} warnings (see seismic_qc_report.md).", "",
        "Reference = known synthetic horizon (on real data this would be an expert interpretation, not ground truth).", "",
        "| Method | Coverage | MAE (ms) | RMS (ms) | within 10 ms | within 20 ms | Cycle-skip cells | Runtime (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m, secs, cs, _ in rows:
        lines.append(f"| {name} | {m.coverage:.1%} | {m.mae_ms:.2f} | {m.rms_ms:.2f} | {m.within_10ms:.1%} | {m.within_20ms:.1%} | "
                     f"{'n/a' if cs is None else cs} | {secs:.2f} |")
    s = cm.summary()
    lines += [
        "", "## Final horizon QC", "",
        f"* Converged: {final.converged} in {len(final.iterations)} iterations; last max update "
        f"{final.iterations[-1]['max_update_samples']:.4f} samples.",
        f"* Correlation pairs used: {final.iterations[-1]['n_correlation_pairs']} "
        f"(rejected: {final.iterations[-1]['n_corr_rejected_barrier']} across faults, "
        f"{final.iterations[-1]['n_corr_rejected_low_corr']} low correlation, {final.iterations[-1]['n_corr_rejected_lag_limit']} at lag limit).",
        f"* Uninterpreted live cells (no seed in their domain, e.g. isolated by barriers or dead neighbours): {int(final.unconstrained.sum())}.",
        f"* Phase: circular mean {ph['circular_mean_deg']:.0f} deg, {ph['fraction_in_phase']:.1%} of cells within 45 deg of the picked trough.",
        f"* Spikes flagged: {int(sp.sum())}.",
        f"* Mean heuristic confidence {s['mean_confidence']:.2f}; {s['fraction_reliable_ge_0.7']:.1%} of cells >= 0.7. "
        f"Limiting driver shares: " + ", ".join(f"{k} {v:.0%}" for k, v in s["limiting_driver_share"].items() if v > 0.005) + ".",
        "", "## Figures", "",
    ] + [f"![{f.stem}]({f.name})" for f in figs]
    (out / "demo_report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {name: {"mae_ms": m.mae_ms, "rms_ms": m.rms_ms, "coverage": m.coverage, "runtime_s": secs} for name, m, secs, _, _ in rows}
    (out / "demo_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "\n".join(lines[:len(rows) + 10])


def _figures(out, seis, truth, runs, final, cm, geom, live):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return []
    figs = []
    j = seis.shape[1] // 2
    fig, ax = plt.subplots(figsize=(11, 5))
    sec = seis[:, j, :].T
    v = np.percentile(np.abs(sec), 98)
    ax.imshow(sec, cmap="gray_r", aspect="auto", vmin=-v, vmax=v, extent=[0, seis.shape[0], seis.shape[2], 0])
    ax.plot(np.arange(seis.shape[0]) + 0.5, truth[:, j], "y-", lw=3, alpha=0.6, label="true horizon")
    styles = ["c:", "r--", "b-", "g-"]
    for (name, (t, _, _)), sty in zip(runs.items(), styles):
        ax.plot(np.arange(seis.shape[0]) + 0.5, t[:, j], sty, lw=1.3, label=name)
    ax.set_ylim(np.nanmax(truth[:, j]) + 25, np.nanmin(truth[:, j]) - 25)
    ax.set_xlabel("inline index"); ax.set_ylabel("sample"); ax.set_title(f"Crossline index {j}: horizon picks across the fault")
    ax.legend(fontsize=7, loc="lower left")
    p = out / "fig_section.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs.append(p)

    names = [n for n in runs if runs[n][2] is not None]
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8))
    for ax, n in zip(axs, names):
        err = (runs[n][0] - truth) * geom.dt_ms
        im = ax.imshow(err.T, origin="lower", cmap="RdBu_r", vmin=-8, vmax=8)
        ax.set_title(n, fontsize=8); ax.set_xlabel("inline idx"); ax.set_ylabel("xline idx")
    fig.colorbar(im, ax=axs, label="error (ms)", shrink=0.8)
    p = out / "fig_error_maps.png"; fig.savefig(p, dpi=130); plt.close(fig); figs.append(p)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    im = axs[0].imshow(geom.time_ms(final.t).T, origin="lower", cmap="viridis"); axs[0].set_title("Final horizon (ms)")
    fig.colorbar(im, ax=axs[0])
    si, sj, _ = final.seeds
    axs[0].plot(si, sj, "w+", ms=6)
    im = axs[1].imshow(cm.confidence.T, origin="lower", cmap="magma", vmin=0, vmax=1); axs[1].set_title("Heuristic confidence")
    fig.colorbar(im, ax=axs[1])
    lim = np.where(np.isfinite(cm.confidence), cm.limiting_driver, np.nan)
    im = axs[2].imshow(lim.T, origin="lower", cmap="tab10", vmin=-0.5, vmax=9.5); axs[2].set_title("Limiting driver")
    cb = fig.colorbar(im, ax=axs[2], ticks=range(len(cm.driver_names))); cb.ax.set_yticklabels(cm.driver_names, fontsize=7)
    p = out / "fig_confidence.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs.append(p)
    return figs
