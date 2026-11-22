# User guide

## What was built

An app that tracks a seismic horizon across a 3D volume from a handful of your picks, and refuses
to smooth it across faults you have interpreted. Version 0.1 runs from the command line (no
windowed GUI yet) and writes results as CSV files, figures and QC reports.

| Part | What it does for you |
|---|---|
| **SEG-Y reader + QC** | Reads your SEG-Y, works out the inline/crossline grid and coordinates, and checks for missing, duplicate, dead, noisy or clipped traces. If something critical is wrong it stops and says why. |
| **HDF5 converter** | Saves the checked volume in a fast, chunked format so later steps only read the part they need. |
| **Slope engine** | Measures the local dip of the reflections at every sample (structure tensor), plus how clean the reflections are. |
| **Horizon engine** | Solves for the whole horizon at once so it fits your picks exactly, follows the measured dips, and agrees with long-distance trace-to-trace correlations. Fault cut-offs are hard walls: nothing is smoothed across them, and a fault block with no pick is left blank. |
| **QC + confidence** | Checks the result for cycle skips, spikes and phase consistency, and draws a confidence map that names *why* an area is weak (few picks, near a fault, poor reflections...). |
| **Export** | Writes the horizon as CSV/XYZ with confidence and flags, plus a provenance file recording inputs, parameters and versions. |

## What was done so far

1. Built and tested the core (47 automated tests), first on synthetic volumes with known answers.
2. Downloaded the F3 Netherlands data you provided, QC'd its SEG-Y (passed) and converted it to
   `C:\SeismicData\F3\f3_seismic.h5`.
3. Tracked horizon FS8 across Fault A and compared it with dGB's own interpretation:
   1.25 ms mean difference, 99% of traces within ±10 ms (conventional peak tracking: 7.4 ms).

## How to use it (Windows)

Open PowerShell in the app folder:

```powershell
cd "C:\Users\HP\OneDrive\Documenten\Active_Confirmed work\__AI_PROJECTS\automated seismic horizon and fault picking\seismic-horizon-ai"
```

The app is already installed for your Python. On a new machine run `python -m pip install -e .[test]` once.

### 1. Check a SEG-Y file

```powershell
shai qc ..\F3_Demo_2023\Rawdata\Seismic_data.sgy --report outputs\my_qc.md
```

### 2. Convert it (once per survey)

```powershell
shai convert ..\F3_Demo_2023\Rawdata\Seismic_data.sgy C:\SeismicData\F3\f3_seismic.h5
```

Add `--inlines 100 449 --crosslines 750 1250 --time-ms 400 800` to keep only an area of interest.

### 3. Make seed picks

Either write a CSV yourself with at least these columns (times in ms):

```
inline,crossline,time_ms
125,775,607.1
125,825,588.6
```

or take picks from an existing horizon file (x y time) on every Nth line:

```powershell
shai picks-from-horizon C:\SeismicData\F3\f3_seismic.h5 ..\F3_Demo_2023\Rawdata\Surface_data\F3-Horizon-FS8.txt outputs\my_run\picks.csv --spacing 50 --time-shift-ms 4 --inlines 100 449 --crosslines 750 1250
```

(`--time-shift-ms 4` is needed for the F3 dGB files: they count time from 0 ms, the SEG-Y from 4 ms.)

### 4. Extract the horizon (optionally with faults)

```powershell
shai extract C:\SeismicData\F3\f3_seismic.h5 outputs\my_run\picks.csv outputs\my_run --inlines 100 449 --crosslines 750 1250 --fault-sticks ..\F3_Demo_2023\Rawdata\Faults\FaultA.txt --fault-time-shift-ms 4 --show
```

`--show` opens the map and section figures when it finishes (about a minute for this area).

### 5. Look at the results in `outputs\my_run`

| File | Content |
|---|---|
| `horizon_map.png` | Horizon time map with your picks and fault cut-offs; confidence map |
| `horizon_section.png` | Seismic section with the extracted horizon |
| `horizon.csv` | inline, crossline, x, y, time_ms, confidence, flags (load into Petrel/OpendTect/Excel) |
| `summary.json` | Run summary: convergence, timings, phase, confidence, warnings |
| `horizon.provenance.json` | Full record of inputs, parameters and versions |

### Other commands

```powershell
shai demo outputs\demo                         # synthetic test with known answer
python scripts\f3_benchmark.py --h5 C:\SeismicData\F3\f3_seismic.h5 --f3 ..\F3_Demo_2023 --out outputs\f3 --open
python -m pytest -q                            # run the test suite
```

## Good to know

* Picks you give are honoured exactly; check them, because a wrong pick is followed faithfully.
* Keep the area of interest moderate on a 16 GB laptop (a few hundred lines square); the
  full F3 survey at once needs much more memory.
* Blank areas in the result are deliberate: they had no pick in their fault block or no data.
* Confidence is a review aid, not a probability.
* Not built yet: the interactive viewer for picking with the mouse, AI pick suggestions, and the
  natural-language assistant (milestones 3 and 4 in `docs/DESIGN.md`).
