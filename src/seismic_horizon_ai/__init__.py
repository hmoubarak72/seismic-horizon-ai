"""seismic_horizon_ai: physics-first, AI-assisted seismic horizon interpretation.

The core geophysical engine is a least-squares horizon extractor in the spirit of
Wu & Fomel (2018), "Least-squares horizons with local slopes and multigrid
correlations", Geophysics 83(4), IM29-IM40. AI components (to be added in later
milestones) only *propose*; the numerical engine and the interpreter decide.

Conventions used throughout the package
---------------------------------------
* Seismic cubes are ``float32`` arrays of shape ``(n_il, n_xl, n_t)``
  (inline index, crossline index, time/depth sample), matching segyio's cube layout.
* Horizons are ``float64`` arrays of shape ``(n_il, n_xl)`` holding *fractional
  sample indices* along the time axis. ``NaN`` means "not interpreted" and is never
  silently filled.
* Slopes ``p = dt/d(il)`` and ``q = dt/d(xl)`` are in samples per trace.
"""

__version__ = "0.1.0"
ALGORITHM_VERSIONS = {
    "structure_tensor_slopes": "1.0",
    "smooth_dtw": "1.0",
    "lsq_horizon": "1.0",
    "surface_interpolation": "1.0",
}
