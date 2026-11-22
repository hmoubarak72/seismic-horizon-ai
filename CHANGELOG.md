# Changelog

## 0.1.0 (2026-11-22) — M1 physics core

* SEG-Y header scan with configurable bytes, coordinate scalars, CRS detection from the textual header.
* Survey geometry with gap-preserving axes, duplicate detection and affine bin-to-world fit.
* QC gate: sample interval/count, missing, duplicate and dead traces, non-finite values, clipping,
  RMS outliers, georeferencing; Markdown/JSON reports.
* Chunked HDF5 conversion with ROI, live mask, QC report and provenance.
* Chunked 3D structure tensor (slopes, planarity, dip/azimuth) and 2D variant; instantaneous phase.
* Fault barriers from interpreter polylines or masks; planarity-based fault-probability proxy.
* Least-squares horizon engine: slopes + multigrid smooth-DTW correlations + exact seeds, fault-aware
  domains, direct/CG solvers, convergence diagnostics.
* 2D line tracking, line mistie check, peak-tracking baseline.
* QC: slope misfit, cycle-skip candidates, spikes, phase consistency, reference metrics,
  confidence with named drivers.
* CSV/XYZ export with flags and provenance sidecar; CLI (`shai qc|convert|demo`); synthetic benchmark.
