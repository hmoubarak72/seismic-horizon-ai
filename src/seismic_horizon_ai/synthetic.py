"""Synthetic 3D seismic volumes with known horizons and faults (for tests and demos).

Model: a layered earth in relative geologic time ``tau``. At bin (i, j) a reflector at
``tau`` sits at time ``t = tau * (1 + a(i, j)) + s(i, j)`` where ``s`` is a smooth
structural surface (folds) and ``a`` a mild thickness variation. A vertical normal
fault adds ``throw`` samples to ``s`` on its hanging-wall side. Amplitudes are a
Ricker-convolved sparse reflectivity series in ``tau`` plus seeded Gaussian noise,
so every horizon's true time is known analytically.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def ricker(peak_freq_cycles_per_sample: float, half_length: int) -> np.ndarray:
    t = np.arange(-half_length, half_length + 1, dtype=float)
    a = (np.pi * peak_freq_cycles_per_sample * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


@dataclasses.dataclass
class SyntheticVolume:
    seismic: np.ndarray  # (n_il, n_xl, n_t) float32
    tau_reflectors: np.ndarray  # tau of strong reflectors (candidate target horizons)
    structure: np.ndarray  # s(i, j) including the fault throw
    stretch: np.ndarray  # a(i, j)
    fault_side: np.ndarray  # bool, hanging wall
    fault_line: tuple[np.ndarray, np.ndarray] | None  # map-view fault trace (il_idx, xl_idx) endpoints
    throw: float
    dt_ms: float
    noise: float

    def horizon(self, tau: float) -> np.ndarray:
        """True horizon time (fractional samples) of the reflector at ``tau``."""
        return tau * (1 + self.stretch) + self.structure

    def true_slopes(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        h = self.horizon(tau)
        return np.gradient(h, axis=0), np.gradient(h, axis=1)


def make_synthetic(
    n_il: int = 60,
    n_xl: int = 50,
    n_t: int = 160,
    dt_ms: float = 4.0,
    peak_freq: float = 0.09,  # cycles/sample (~22 Hz at 4 ms) -> period ~11 samples
    fold_amplitude: float = 8.0,
    dip_il: float = 0.25,  # samples per trace
    dip_xl: float = -0.1,
    throw: float = 0.0,
    fault_position: float = 0.5,  # fraction of the inline axis where the fault crosses
    fault_angle_deg: float = 15.0,
    noise: float = 0.1,
    seed: int = 7,
    target_tau: float | None = 80.0,
) -> SyntheticVolume:
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(n_il, dtype=float), np.arange(n_xl, dtype=float), indexing="ij")
    s = (
        20.0
        + dip_il * ii
        + dip_xl * jj
        + fold_amplitude * np.sin(2 * np.pi * ii / (1.3 * n_il)) * np.cos(2 * np.pi * jj / (1.7 * n_xl))
    )
    a = 0.05 * np.sin(2 * np.pi * (ii + jj) / (n_il + n_xl))
    fault_side = np.zeros((n_il, n_xl), bool)
    fault_line = None
    if throw:
        ang = np.radians(fault_angle_deg)
        x0 = fault_position * n_il
        # fault trace: il = x0 + tan(ang) * (xl - n_xl/2)
        fault_side = ii > x0 + np.tan(ang) * (jj - n_xl / 2)
        s = s + throw * fault_side
        j_end = np.array([-1.0, n_xl + 1.0])
        fault_line = (x0 + np.tan(ang) * (j_end - n_xl / 2), j_end)

    # sparse reflectivity in tau with a few dominant reflectors
    n_tau = int(n_t * 1.6)
    refl = rng.normal(0, 0.15, n_tau) * (rng.random(n_tau) < 0.25)
    strong = np.array([40.0, 80.0, 115.0]) if target_tau is not None else np.array([])
    for k, tau in enumerate(strong):
        if tau < n_tau:
            refl[int(tau)] = 1.0 if k % 2 == 0 else -0.8
    w = ricker(peak_freq, int(3 / peak_freq))
    trace_tau = np.convolve(refl, w, mode="same")
    taus = np.arange(n_tau, dtype=float)
    t_axis = np.arange(n_t, dtype=float)
    tau_of_t = (t_axis[None, None, :] - s[..., None]) / (1 + a[..., None])
    seis = np.interp(tau_of_t, taus, trace_tau, left=0.0, right=0.0)
    rms = np.sqrt(np.mean(seis**2)) or 1.0
    seis = seis + rng.normal(0, noise * rms, seis.shape)
    return SyntheticVolume(
        seismic=seis.astype(np.float32), tau_reflectors=strong, structure=s, stretch=a,
        fault_side=fault_side, fault_line=fault_line, throw=throw, dt_ms=dt_ms, noise=noise,
    )
