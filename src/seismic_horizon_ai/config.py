"""Version-controlled configuration.

Parameters live in TOML files (see ``configs/default.toml``) and are parsed into typed
dataclasses. Unknown keys raise an error, so a typo can never silently fall back to a
default.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any, TypeVar


@dataclasses.dataclass
class SegyConfig:
    iline_byte: int = 189
    xline_byte: int = 193
    cdp_x_byte: int = 181
    cdp_y_byte: int = 185
    endian: str = "big"
    ignore_geometry: bool = True  # we build geometry ourselves and QC it
    hdf5_chunks: tuple[int, int, int] = (64, 64, 64)


@dataclasses.dataclass
class StructureTensorConfig:
    sigma_gradient: float = 1.0  # samples; derivative-of-Gaussian scale
    sigma_smooth: tuple[float, float, float] = (3.0, 3.0, 2.0)  # (il, xl, t) tensor smoothing
    chunk_inlines: int = 64  # inlines processed per chunk (halo added automatically)


@dataclasses.dataclass
class CorrelationConfig:
    enabled: bool = True
    strides: tuple[int, ...] = (4, 8, 16)  # multigrid coarse-grid strides (traces)
    half_window: int = 20  # samples either side of the horizon
    max_lag: int = 6  # samples; DTW shift limit for each correlation pair
    min_correlation: float = 0.6  # pairs below this normalised correlation are discarded
    weight: float = 1.0  # relative weight of correlation equations vs slope equations
    refresh_iterations: int = 2  # correlations are re-estimated in the first N outer iterations, then frozen


@dataclasses.dataclass
class HorizonConfig:
    max_iterations: int = 8
    tolerance: float = 0.02  # samples; stop when max |update| falls below this
    slope_weight_power: float = 2.0  # weight = planarity ** power
    min_slope_weight: float = 0.02  # floor so weak areas stay connected (soft, not a barrier)
    fault_soft_power: float = 2.0  # weight *= (1 - fault_probability) ** power
    fault_hard_threshold: float = 0.9  # probability at/above which an edge becomes a barrier
    seed_mode: str = "hard"  # "hard" (exact) or "soft"
    soft_seed_weight: float = 100.0
    solver: str = "auto"  # "direct", "cg" or "auto"
    correlation: CorrelationConfig = dataclasses.field(default_factory=CorrelationConfig)


@dataclasses.dataclass
class PlatformConfig:
    segy: SegyConfig = dataclasses.field(default_factory=SegyConfig)
    structure_tensor: StructureTensorConfig = dataclasses.field(default_factory=StructureTensorConfig)
    horizon: HorizonConfig = dataclasses.field(default_factory=HorizonConfig)
    random_seed: int = 12345


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any]) -> T:
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        f = fields[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore[misc]
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PlatformConfig:
    """Load a TOML configuration file. ``None`` returns the defaults."""
    if path is None:
        return PlatformConfig()
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return _build(PlatformConfig, data)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
