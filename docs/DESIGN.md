# Technical design and staged build plan

AI-assisted 3D seismic horizon and fault interpretation. Version 0.1.0 (milestone M1, physics core).

## 1. Executive summary

The platform treats horizon picking as a constrained geophysical inverse problem. The core engine is a
least-squares horizon extractor in the spirit of Wu & Fomel (2018): a surface `t(il, xl)` is solved
globally from local reflection slopes (structure tensor), long-range multigrid trace correlations
(smooth DTW) and interpreter seed picks, with fault barriers that the solution may never cross. AI is
added later only to *propose* (seed candidates, fault probability, confidence) and to orchestrate tools;
it never writes a final pick.

Main technical risks: slope errors in steep/chaotic zones, cycle skips across low-signal gaps, fault
geometry that is not captured by the barrier model (e.g. small-throw faults below the probability
threshold), and the gap between a synthetic benchmark and real data (Volve BCU is still to be run).

## 2. Assumptions (the scoping questions are still open, so these are working defaults)

| # | Assumption | Why / how to change |
|---|---|---|
| A1 | Post-stack 3D, regular inline/crossline grid, time domain (TWT ms) | Depth volumes work if `domain` is set; QC states the assumption |
| A2 | Headers at SEG-Y rev1 bytes 189/193 (IL/XL) and 181/185 (CDP X/Y) | Configurable in `configs/default.toml` |
| A3 | Python 3.13 on a Windows laptop (12 cores, 16 GB RAM), CPU only for M1 | Numba/CuPy only after profiling shows the need |
| A4 | One target horizon per run, picked on a consistent phase (peak or trough) | Multi-horizon / RGT later |
| A5 | Fault cut-offs come from the interpreter as map-view polylines; fault probability is advisory | Hard barriers from probability only above `fault_hard_threshold` |
| A6 | Re-implement the method in Python rather than wrap the Java MHE code | Benchmark against `xinwucwp/mhe` in M2 before trusting numerical parity |
| A7 | Volve is the first real benchmark; it needs the person's acceptance of Equinor's licence | No data has been downloaded |
| A8 | Code is private, no licence chosen yet; local git only (no GitHub remote) | `pyproject.toml` licence field is a placeholder |
| A9 | HDF5 is the working store for M1; Zarr is an optional extra | Zarr when volumes exceed comfortable HDF5 use or cloud storage is needed |
| A10 | Pydantic/FastAPI arrive with the API layer; M1 uses typed dataclasses + TOML | Keeps the numerical core dependency-light |

## 3. Architecture

```
SEG-Y ──► seismic_io (scan headers, geometry, stream traces)
             │
             ▼
        preprocessing.seismic_qc  ── QC gate (critical findings stop the pipeline)
             │
             ▼
        HDF5 volume (measured data, live mask, geometry, QC report, provenance)
             │
             ▼
        attributes: structure tensor slopes p, q + planarity · instantaneous phase/envelope
                    · fault-probability proxy (later: 3D CNN) · interpreter fault polylines → barriers
             │
             ▼
        horizon_engine: seed picks (approved only) → per-fault-block initial surface
                        → [sample slopes on surface → edge weights & barriers → domains
                           → multigrid DTW pairs → sparse weighted LSQ (hard seeds)] × iterations
             │
             ▼
        quality_control: slope misfit · cycle-skip candidates · spikes · phase consistency
                         · reference metrics · confidence with named drivers
             │
             ▼
        export: CSV/XYZ with flags + provenance sidecar          (later: ZMAP, VTK, grids)

Later layers (separate packages, call the core through typed APIs):
  ai/  (seed proposals, fault segmentation, confidence model, LLM assistant with tool calling)
  api/ (FastAPI)   visualization/ (PySide6 + PyVista)
```

Data-kind separation: HDF5 files carry `data_kind = measured | derived`; horizons carry their provenance,
extrapolation and uncertainty flags; AI picks carry `status = proposed` and `source = ai:<model>`.

## 4. Core algorithm (as implemented)

Unknown: `t[i, j]`, fractional sample index of the horizon at each bin.

Equations, each multiplied by `sqrt(w)`:

* inline slope: `t[i+1, j] − t[i, j] = ½(p[i, j] + p[i+1, j])` sampled at the current surface
* crossline slope: `t[i, j+1] − t[i, j] = ½(q[i, j] + q[i, j+1])`
* correlation: `t[b] − t[a] = s_ab` for coarse-grid pairs at strides 4, 8, 16 traces
* seeds: eliminated as fixed unknowns (exact), or soft rows on request

Weights: `w = max(floor, planarity^2) · (1 − fault_prob)^2`; `w = 0` on interpreter barriers and where
fault probability ≥ 0.9. Domains = connected components of the non-zero edge graph. Domains without a
seed are returned as NaN and reported. Correlation pairs are rejected across barriers, across domains,
when the DTW lag hits its limit (possible cycle skip) or when the aligned correlation is below 0.6.

Correlation shifts: smooth DTW (forward + backward accumulation, strain ≤ 1) for a robust integer lag,
refined to sub-sample with a centre-tapered NCC parabola (two passes). Measured accuracy on synthetic
pairs: median error 0.04 samples.

Outer iteration: re-sample slopes on the new surface; correlations re-estimated for the first two passes,
then frozen; stop at max update < 0.02 samples. Normal equations solved by sparse direct factorisation
(≤ 300k unknowns) or Jacobi-preconditioned CG.

Deviations from Wu & Fomel (documented on purpose): correlations are windowed around the current horizon
rather than full-trace DTW on coarse grids; seeds are exact constraints by elimination; barriers and
domains are an addition for fault awareness.

## 5. Staged build plan

| Milestone | Scope | Acceptance criteria | Status |
|---|---|---|---|
| **M1 Physics core** | SEG-Y I/O + QC gate, HDF5, structure-tensor slopes, LSQ engine with multigrid DTW, fault barriers, 2D tracking, peak-tracking baseline, QC metrics, confidence drivers, CSV/XYZ export, provenance, tests, synthetic benchmark | All tests pass; synthetic faulted benchmark MAE < 1 ms with barrier; seeds honoured exactly; no bridging of seedless blocks | **Done (this commit)** |
| M2 Real-data baseline | Volve ingest (licence permitting), BCU benchmark vs reference interpretation, peak-tracking and original Java MHE comparison, profiling on full survey, Zarr option, ZMAP/VTK export | MAE/RMS/±10/20/40 ms table on BCU; runtime and memory measured; parity check vs Java MHE on a sub-volume | Next |
| M3 Interactive interpretation | PySide6 + PyVista viewer: inline/crossline/time slice/3D, pick editing, fault polyline editing, QC overlays, undo/redo of every edit | Interpreter can pick, run, inspect and correct without code | |
| M4 AI assistance | Waveform-similarity seed proposals (explainable ranking), 3D CNN fault probability (validated against interpreted faults), learned confidence, LLM assistant with tool calling + RAG over QC/provenance | Every proposal reviewable and reversible; models evaluated on held-out labelled data; assistant cites data and parameters | |
| M5 Service and deployment | FastAPI + Pydantic, Docker, MLflow, benchmark report, deployment guide | Reproducible container run of the Volve benchmark | |

## 6. Validation plan

* Unit tests: geometry parsing, QC grading, slopes on plane waves, DTW shift recovery, sparse solver
  exactness, barrier domains, seed authority, phase, metrics, export, config.
* Integration tests: synthetic volumes with known horizons, folds, noise and a 28 ms normal fault.
* Benchmarks (M2): Volve BCU; methods = reference interpretation, peak tracking, LSQ engine,
  LSQ + AI seeds + fault constraints; metrics = MAE, RMS, % within ±10/20/40 ms, phase consistency,
  cycle-skip count, runtime, memory, seeds needed, corrections needed, % area reliable.
* The reference interpretation is treated as expert opinion, not ground truth; well ties and geological
  plausibility are checked separately.

## 7. Risk register

| Risk | Type | Mitigation |
|---|---|---|
| Slopes wrong in steep, chaotic or low-SNR zones | Technical | Planarity weighting, multigrid correlations, confidence driver `slope_quality`, interpreter seeds |
| Cycle skip propagated by correlations | Geophysical | Lag-limit and correlation-threshold rejection; cycle-skip QC map; phase QC |
| Small faults below probability threshold get smoothed | Geological | Interpreter polylines are hard barriers; fault-proximity driver; QC spikes/misfit near faults |
| Fault-probability proxy fires on noise | Technical | Soft weights by default; hard only ≥ 0.9; CNN must be validated before use as barrier |
| Synthetic success does not transfer to real data | Validation | M2 Volve benchmark before any AI work; report limitations |
| Large surveys exceed laptop memory | Operational | Chunked HDF5, chunked tensor, ROI; profiling before Numba/GPU |
| Licence/confidentiality of data | Data governance | No data fetched without explicit consent; provenance hashes, no data in prompts to external LLMs without approval |
| LLM overstates or alters interpretations | AI safety | Assistant limited to calling validated tools; no write access without interpreter confirmation; citations required |
| Numerical differences vs original MHE | Technical | Parity test against Java reference on a sub-volume in M2 |

## 8. Next actions

1. Obtain a real volume (Volve with BCU reference, or a survey the person is authorised to use).
2. Run `shai qc` and `shai convert` on it and review the QC report together.
3. Benchmark BCU with peak tracking vs the LSQ engine; profile runtime/memory on the full survey.
4. Decide on interactive viewer priority vs AI seed proposals based on where interpreter time goes.
