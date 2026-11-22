"""Survey and amplitude quality control for SEG-Y input.

QC is a gate: findings are graded ``critical`` / ``warning`` / ``info``. A report with
any critical finding fails, and :meth:`QCReport.raise_if_failed` stops the pipeline
with an actionable message instead of producing a silently wrong volume.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SegyConfig
from ..seismic_io.segy_reader import SegyScan, iter_trace_blocks, scan_segy
from ..seismic_io.survey_geometry import GeometryError, SurveyGeometry


class SeismicQCError(RuntimeError):
    """Raised when critical QC checks fail."""


@dataclasses.dataclass
class QCFinding:
    severity: str  # "critical" | "warning" | "info"
    code: str
    message: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class QCReport:
    source: str
    findings: list[QCFinding] = dataclasses.field(default_factory=list)
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "critical" for f in self.findings)

    def add(self, severity: str, code: str, message: str, **details: Any) -> None:
        self.findings.append(QCFinding(severity, code, message, details))

    def by_code(self, code: str) -> list[QCFinding]:
        return [f for f in self.findings if f.code == code]

    def raise_if_failed(self) -> None:
        if not self.passed:
            msgs = "; ".join(f"[{f.code}] {f.message}" for f in self.findings if f.severity == "critical")
            raise SeismicQCError(f"Seismic QC failed for {self.source}: {msgs}")

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "passed": self.passed, "findings": [dataclasses.asdict(f) for f in self.findings], "stats": self.stats}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_np_default)

    def to_markdown(self) -> str:
        lines = [f"# Seismic QC report: `{self.source}`", "", f"**Result:** {'PASSED' if self.passed else 'FAILED'}", ""]
        order = {"critical": 0, "warning": 1, "info": 2}
        lines += ["| Severity | Check | Finding |", "|---|---|---|"]
        for f in sorted(self.findings, key=lambda f: order.get(f.severity, 3)):
            lines.append(f"| {f.severity} | `{f.code}` | {f.message} |")
        lines += ["", "## Statistics", "", "```json", json.dumps(self.stats, indent=2, default=_np_default), "```"]
        return "\n".join(lines)


def _np_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- stats
class TraceStatsAccumulator:
    """Single-pass, out-of-core per-trace and global amplitude statistics."""

    def __init__(self, n_traces: int, reservoir: int = 200_000, seed: int = 12345) -> None:
        self.rms = np.zeros(n_traces, dtype=np.float32)
        self.dead = np.zeros(n_traces, dtype=bool)
        self.nonfinite = np.zeros(n_traces, dtype=bool)
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf
        self._rng = np.random.default_rng(seed)
        self._reservoir_size = reservoir
        self._samples: list[np.ndarray] = []
        self._n_traces = n_traces

    def update(self, start: int, block: np.ndarray) -> None:
        stop = start + block.shape[0]
        finite = np.isfinite(block)
        self.nonfinite[start:stop] = ~finite.all(axis=1)
        safe = np.where(finite, block, 0.0).astype(np.float64)
        self.rms[start:stop] = np.sqrt(np.mean(safe**2, axis=1))
        self.dead[start:stop] = (np.ptp(safe, axis=1) == 0) | self.nonfinite[start:stop]
        live = safe[~self.dead[start:stop]]
        if live.size:
            self.count += live.size
            self.sum += float(live.sum())
            self.sumsq += float((live**2).sum())
            self.min = min(self.min, float(live.min()))
            self.max = max(self.max, float(live.max()))
            k = max(1, int(self._reservoir_size * block.shape[0] / max(1, self._n_traces)))
            flat = live.ravel()
            self._samples.append(flat[self._rng.integers(0, flat.size, size=min(k, flat.size))])

    def summary(self) -> dict[str, Any]:
        s = np.concatenate(self._samples) if self._samples else np.zeros(0)
        mean = self.sum / self.count if self.count else float("nan")
        var = self.sumsq / self.count - mean**2 if self.count else float("nan")
        out: dict[str, Any] = {
            "n_live_samples": self.count,
            "mean": mean,
            "std": float(np.sqrt(max(var, 0.0))) if self.count else float("nan"),
            "min": self.min if self.count else float("nan"),
            "max": self.max if self.count else float("nan"),
        }
        if s.size:
            pct = np.percentile(s, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
            out["percentiles"] = dict(zip(["p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9"], pct.tolist()))
            maxabs = max(abs(self.min), abs(self.max))
            out["fraction_at_clip"] = float(np.mean(np.abs(s) >= 0.999 * maxabs)) if maxabs > 0 else 0.0
        return out


# ---------------------------------------------------------------------- checks
def header_checks(scan: SegyScan, geometry: SurveyGeometry, duplicates: np.ndarray, report: QCReport) -> None:
    if scan.n_samples < 2:
        report.add("critical", "samples.count", f"Traces have {scan.n_samples} samples; a 3D volume needs more than one.")
    if not np.isfinite(scan.sample_interval_us) or scan.sample_interval_us <= 0:
        report.add("critical", "samples.interval", "Sample interval is zero or missing in binary and trace headers.")
    tr_dt = scan.trace_sample_intervals_us[scan.trace_sample_intervals_us > 0]
    if tr_dt.size > 1:
        report.add("warning", "samples.interval_varies", "Trace headers disagree on sample interval.", values=tr_dt.tolist())
    elif tr_dt.size == 1 and abs(tr_dt[0] - scan.sample_interval_us) > 0.5:
        report.add(
            "warning",
            "samples.interval_mismatch",
            f"Trace-header interval {tr_dt[0]} us differs from binary header {scan.sample_interval_us} us; binary header used.",
        )
    ns = scan.trace_sample_counts[scan.trace_sample_counts > 0]
    if ns.size > 1:
        report.add("critical", "samples.count_varies", "Traces have different sample counts; volume is not a regular cube.", values=ns.tolist())

    n_bins = geometry.trace_index.size
    n_live_bins = int(geometry.live_mask.sum())
    missing = n_bins - n_live_bins
    report.stats["geometry"] = {
        "n_inlines": int(geometry.ilines.size),
        "n_crosslines": int(geometry.xlines.size),
        "n_samples": int(geometry.samples_ms.size),
        "inline_range": [int(geometry.ilines[0]), int(geometry.ilines[-1]), geometry.il_step],
        "crossline_range": [int(geometry.xlines[0]), int(geometry.xlines[-1]), geometry.xl_step],
        "sample_interval_ms": geometry.dt_ms,
        "time_range_ms": [float(geometry.samples_ms[0]), float(geometry.samples_ms[-1])],
        "n_traces_in_file": scan.n_traces,
        "n_bins": n_bins,
        "n_missing_bins": missing,
        "n_duplicate_traces": int(duplicates.size),
        "format_code": scan.format_code,
        "measurement_system": scan.measurement_system,
        "bin_spacing": geometry.bin_spacing(),
        "affine_rms": geometry.affine_rms,
        "crs": geometry.crs,
    }
    if missing:
        frac = missing / n_bins
        report.add(
            "critical" if frac > 0.5 else "warning",
            "geometry.missing_traces",
            f"{missing} of {n_bins} bins ({frac:.1%}) have no trace; they are masked, never interpolated.",
            fraction=frac,
        )
    if duplicates.size:
        report.add(
            "warning",
            "geometry.duplicate_traces",
            f"{duplicates.size} traces duplicate an occupied (inline, crossline) bin; the first trace is kept.",
            first_examples=duplicates[:10].tolist(),
        )
    if geometry.affine is None:
        report.add("warning", "geometry.no_coordinates", "CDP X/Y are missing or degenerate; outputs cannot be georeferenced until coordinates are supplied.")
    else:
        spacing = geometry.bin_spacing() or (1.0, 1.0)
        rms_bins = (geometry.affine_rms or 0.0) / max(min(spacing), 1e-9)
        if rms_bins > 0.25:
            report.add(
                "warning",
                "geometry.coordinates_irregular",
                f"Bin-to-world affine misfit is {rms_bins:.2f} bins RMS; check coordinate scalar and header bytes.",
                rms_bins=rms_bins,
            )
    if geometry.crs is None:
        report.add("warning", "geometry.crs_missing", "No coordinate reference system found in the textual header; the interpreter must supply one before export.")
    report.add("info", "data.polarity", "Polarity (SEG normal or reverse) cannot be determined from SEG-Y headers; the interpreter must confirm it.")
    report.add("info", "data.domain", "Vertical domain assumed to be TWT time in ms; confirm if the volume is depth-converted.")


def amplitude_checks(acc: TraceStatsAccumulator, geometry: SurveyGeometry, report: QCReport) -> np.ndarray:
    """Add amplitude findings and return the (n_il, n_xl) dead-trace mask."""
    summary = acc.summary()
    report.stats["amplitude"] = summary
    tidx = geometry.trace_index
    dead_bins = np.zeros(tidx.shape, dtype=bool)
    live = tidx >= 0
    dead_bins[live] = acc.dead[tidx[live]]
    n_dead = int(dead_bins.sum())
    n_live = int(live.sum())
    if n_dead:
        frac = n_dead / max(n_live, 1)
        report.add(
            "critical" if frac > 0.5 else "warning",
            "amplitude.dead_traces",
            f"{n_dead} of {n_live} traces ({frac:.1%}) are dead (constant, zero or non-finite); they are masked.",
            fraction=frac,
        )
    n_nonfinite = int(acc.nonfinite.sum())
    if n_nonfinite:
        report.add("warning", "amplitude.non_finite", f"{n_nonfinite} traces contain NaN or Inf samples; treated as dead.")
    if summary.get("fraction_at_clip", 0.0) > 0.001:
        report.add(
            "warning",
            "amplitude.clipping",
            f"{summary['fraction_at_clip']:.2%} of sampled amplitudes sit at the extreme value; data may be clipped.",
        )
    if acc.count == 0:
        report.add("critical", "amplitude.no_live_data", "No live samples found in the volume.")
    rms_live = acc.rms[~acc.dead]
    if rms_live.size > 10:
        med = float(np.median(rms_live))
        mad = float(np.median(np.abs(rms_live - med))) or 1e-12
        outliers = int(np.sum(np.abs(rms_live - med) > 10 * 1.4826 * mad))
        report.stats["trace_rms"] = {"median": med, "mad": mad, "n_outliers_10sigma": outliers}
        if outliers:
            report.add("warning", "amplitude.rms_outliers", f"{outliers} live traces have RMS amplitude more than 10 robust sigma from the median (noise bursts or bad traces).")
    return dead_bins


def run_segy_qc(path: str | Path, cfg: SegyConfig | None = None, block_size: int = 4096, seed: int = 12345) -> tuple[SegyScan | None, SurveyGeometry | None, QCReport, np.ndarray | None]:
    """Scan headers, build geometry and stream amplitudes once to grade the file.

    Returns ``(scan, geometry, report, dead_mask)``. Geometry failures are reported as
    critical findings (``geometry`` is then None) rather than raised, so the
    interpreter always receives a report.
    """
    cfg = cfg or SegyConfig()
    report = QCReport(source=str(path))
    scan = scan_segy(path, cfg)
    try:
        geometry, duplicates = scan.build_geometry()
    except GeometryError as exc:
        report.add("critical", "geometry.invalid", str(exc))
        header_only = QCReport(source=str(path), findings=report.findings)
        return scan, None, header_only, None
    header_checks(scan, geometry, duplicates, report)
    acc = TraceStatsAccumulator(scan.n_traces, seed=seed)
    for start, block in iter_trace_blocks(path, block_size, cfg):
        acc.update(start, block)
    dead = amplitude_checks(acc, geometry, report)
    return scan, geometry, report, dead
