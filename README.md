# seismic-horizon-ai

Physics-first, AI-assisted 3D seismic horizon and fault interpretation.

The numerical engine interprets; AI (later milestones) only proposes. Version 0.1.0 delivers the
validated core: SEG-Y ingestion with a QC gate, chunked HDF5 storage, structure-tensor slopes, and a
Wu & Fomel (2018) style least-squares horizon extractor with multigrid DTW correlations and fault
barriers, plus QC, confidence drivers, export and provenance.

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture, algorithm, assumptions, staged plan,
validation plan and risk register.

## Install

```powershell
cd seismic-horizon-ai
python -m pip install -e .[test]
```

Requires Python ≥ 3.10, NumPy, SciPy, segyio, h5py. Matplotlib is used by the demo figures if present.

## Use

```powershell
# grade a SEG-Y file (exit code 2 if QC has critical findings)
shai qc path\to\survey.sgy --report qc.md

# QC + convert to chunked HDF5, optionally an ROI (line numbers, ms)
shai convert path\to\survey.sgy data\seismic_hdf5\survey.h5 --inlines 10000 10200 --time-ms 1500 3000

# end-to-end synthetic benchmark with report and figures
shai demo outputs\demo
```

Python API (sketch):

```python
from seismic_horizon_ai.seismic_io.hdf5_manager import SeismicVolume
from seismic_horizon_ai.attributes.structure_tensor import compute_slopes
from seismic_horizon_ai.attributes.fault_probability import barriers_from_polylines
from seismic_horizon_ai.horizon_engine.seed_picks import PickSet
from seismic_horizon_ai.horizon_engine.horizon_3d import HorizonExtractor3D

vol = SeismicVolume("survey.h5")
seis, live, geom = vol.load(), vol.live_mask, vol.geometry
slopes = compute_slopes(seis, live_mask=live)
si, sj, st = PickSet.from_csv("bcu_picks.csv").to_grid_constraints(geom)   # approved picks only
faults = barriers_from_polylines(*live.shape, fault_polylines_in_bin_indices)
result = HorizonExtractor3D(seis, slopes, live_mask=live, barriers=faults).extract(si, sj, st)
```

## Principles enforced in code

* Interpreter picks are exact constraints; AI-proposed picks are ignored until `status == "approved"`.
  Conflicting picks in one bin raise an error instead of being averaged away.
* Interpreted faults are hard barriers. A fault block without a seed stays uninterpreted (NaN) and is
  reported; the engine never bridges it.
* Missing and dead traces are masked, never interpolated. QC findings with severity `critical` stop
  conversion unless `--force` is given.
* Every product carries provenance: input SHA-256, parameters, software/algorithm/library versions,
  git commit, user, timestamp. Runs are deterministic (no unseeded randomness).
* Confidence is reported with its drivers and the limiting driver per cell; it is a review aid, not a
  calibrated probability. Extrapolated areas are flagged in every export row.

## Repository layout

```
src/seismic_horizon_ai/
  seismic_io/        segy_reader, survey_geometry, hdf5_manager
  preprocessing/     seismic_qc (QC gate), roi_manager
  attributes/        structure_tensor, instantaneous, fault_probability (barriers + proxy)
  horizon_engine/    seed_picks, surface_interpolation, multigrid_dtw, least_squares,
                     horizon_3d (MHE3-style driver), horizon_2d, peak_tracking (baseline)
  quality_control/   horizon_residuals, phase_consistency, uncertainty
  export/            horizon_export (CSV/XYZ + provenance sidecar)
  ai/ api/ visualization/   placeholders for M3-M5
  config.py provenance.py logging_utils.py synthetic.py demo.py cli.py
configs/default.toml   version-controlled parameters
tests/                 unit + integration tests on synthetic volumes
docs/DESIGN.md         design, plan, validation, risks
```

The layout follows the master prompt's proposed structure, nested under one package to avoid
top-level name clashes (`attributes`, `api`, ...).

## Tests

```powershell
python -m pytest -q
```

46 tests, about 30 s on a laptop.

## Limitations (v0.1.0)

* Validated on synthetic data only. No real survey or reference interpretation has been run yet.
* Correlations are windowed around the current horizon, not full-trace coarse-grid DTW as in the
  original MHE code; numerical parity with the Java reference has not been tested.
* The fault-probability attribute is a planarity-based proxy, not a trained model.
* One horizon per run; no GUI, API, AI components, ZMAP/VTK export or Zarr storage yet.
* Whole-volume arrays are loaded for attribute computation; ROIs keep this manageable on a laptop.
