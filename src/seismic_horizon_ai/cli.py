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
    a = p.parse_args(argv)
    configure_logging(path=a.log) if a.log else configure_logging(json_lines=False, level=30)
    return {"qc": _cmd_qc, "convert": _cmd_convert, "demo": _cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
