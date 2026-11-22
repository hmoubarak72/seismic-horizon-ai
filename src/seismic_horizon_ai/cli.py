"""Command line entry point: ``shai qc``, ``shai convert``, ``shai demo``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .logging_utils import configure_logging


def _cmd_qc(a: argparse.Namespace) -> int:
    from .preprocessing.seismic_qc import run_segy_qc

    cfg = load_config(a.config)
    _, _, report, _ = run_segy_qc(a.segy, cfg.segy)
    md = report.to_markdown()
    if a.report:
        Path(a.report).write_text(md, encoding="utf-8")
        Path(a.report).with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    print(md)
    return 0 if report.passed else 2


def _cmd_convert(a: argparse.Namespace) -> int:
    from .preprocessing.roi_manager import ROI
    from .seismic_io.hdf5_manager import convert_segy_to_hdf5

    cfg = load_config(a.config)
    roi = ROI(
        tuple(a.inlines) if a.inlines else None,
        tuple(a.crosslines) if a.crosslines else None,
        tuple(a.time_ms) if a.time_ms else None,
    )
    out, report = convert_segy_to_hdf5(a.segy, a.out, cfg.segy, roi=roi, require_qc_pass=not a.force)
    print(f"wrote {out}  (QC {'passed' if report.passed else 'FAILED, forced'})")
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from .demo import run_demo

    summary = run_demo(Path(a.out), seed=a.seed)
    print(summary)
    return 0


def _roi(a: argparse.Namespace):
    from .preprocessing.roi_manager import ROI

    return ROI(tuple(a.inlines) if a.inlines else None, tuple(a.crosslines) if a.crosslines else None,
               tuple(a.time_ms) if getattr(a, "time_ms", None) else None)


def _cmd_picks(a: argparse.Namespace) -> int:
    from .workflow import picks_from_horizon

    n = picks_from_horizon(a.h5, a.horizon, a.out, spacing=a.spacing, time_shift_ms=a.time_shift_ms, roi=_roi(a))
    print(f"wrote {n} picks to {a.out}")
    return 0


def _cmd_extract(a: argparse.Namespace) -> int:
    import json

    from .workflow import extract_horizon

    s = extract_horizon(a.h5, a.picks, a.out, cfg=load_config(a.config), roi=_roi(a), fault_sticks=a.fault_sticks,
                        fault_time_shift_ms=a.fault_time_shift_ms, show=a.show)
    print(json.dumps({k: v for k, v in s.items() if k != "confidence"}, indent=2, default=float))
    print(f"results in {a.out}: horizon.csv, horizon.provenance.json, summary.json, horizon_map.png, horizon_section.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="shai", description="AI-assisted seismic horizon interpretation (physics-first core)")
    p.add_argument("--config", default=None, help="TOML configuration file")
    p.add_argument("--log", default=None, help="write JSON-lines log to this file")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qc", help="QC a SEG-Y file and print a report")
    q.add_argument("segy"); q.add_argument("--report", default=None)
    c = sub.add_parser("convert", help="QC and convert SEG-Y to chunked HDF5")
    c.add_argument("segy"); c.add_argument("out")
    c.add_argument("--inlines", nargs=2, type=int); c.add_argument("--crosslines", nargs=2, type=int)
    c.add_argument("--time-ms", nargs=2, type=float)
    c.add_argument("--force", action="store_true", help="convert even if QC has critical findings")
    d = sub.add_parser("demo", help="end-to-end synthetic benchmark with report and figures")
    d.add_argument("out"); d.add_argument("--seed", type=int, default=11)
    k = sub.add_parser("picks-from-horizon", help="make seed picks from an existing x y time horizon file")
    k.add_argument("h5"); k.add_argument("horizon"); k.add_argument("out")
    k.add_argument("--spacing", type=int, default=50, help="pick every Nth inline/crossline intersection")
    k.add_argument("--time-shift-ms", type=float, default=0.0)
    k.add_argument("--inlines", nargs=2, type=int); k.add_argument("--crosslines", nargs=2, type=int)
    e = sub.add_parser("extract", help="extract a horizon from seed picks (and optional fault sticks)")
    e.add_argument("h5"); e.add_argument("picks", help="CSV with columns inline,crossline,time_ms (others optional)")
    e.add_argument("out", help="output folder")
    e.add_argument("--inlines", nargs=2, type=int); e.add_argument("--crosslines", nargs=2, type=int)
    e.add_argument("--time-ms", nargs=2, type=float)
    e.add_argument("--fault-sticks", nargs="*", default=None, help="fault stick files: x y time_ms stick node")
    e.add_argument("--fault-time-shift-ms", type=float, default=0.0)
    e.add_argument("--show", action="store_true", help="open the figures on screen when done")
    a = p.parse_args(argv)
    configure_logging(path=a.log) if a.log else configure_logging(json_lines=False, level=30)
    return {"qc": _cmd_qc, "convert": _cmd_convert, "demo": _cmd_demo,
            "picks-from-horizon": _cmd_picks, "extract": _cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
