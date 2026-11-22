"""MHE3-style 3D horizon extraction (Wu & Fomel 2018), fault aware.

Algorithm (one call to :meth:`HorizonExtractor3D.extract`)

1. Build horizon domains from interpreter barriers (fault cut-offs) and live traces.
2. Initial surface from approved seeds, interpolated per domain (never across faults).
3. Repeat until the largest update is below ``tolerance``:
   a. sample slopes p, q, planarity and fault probability along the current surface;
   b. edge weights = max(floor, planarity^power) * (1 - fault_prob)^soft_power;
      edges at or above ``fault_hard_threshold`` or on interpreter barriers get weight 0;
   c. recompute domains; build multigrid DTW correlation pairs inside domains only
      (re-estimated during the first ``refresh_iterations`` passes, then frozen so the
      outer fixed-point iteration converges);
   d. solve the weighted least-squares system with seeds as hard constraints.
4. Report domains without seeds as unconstrained (NaN), never bridged.

All parameters, per-iteration diagnostics and input hashes go into the result's
provenance record.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
from scipy import ndimage

from ..attributes.fault_probability import FaultBarriers
from ..attributes.structure_tensor import SlopeVolume
from ..config import HorizonConfig, config_to_dict
from ..logging_utils import get_logger
from ..provenance import Provenance, sha256_array
from .least_squares import edge_components, solve_horizon_lsq
from .multigrid_dtw import CorrelationPairs, multigrid_correlations
from .surface_interpolation import InitialSurface, initial_surface

log = get_logger("horizon_engine.horizon_3d")


def sample_along(volume: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``volume[i, j, t[i, j]]``; NaN where t is NaN."""
    n_t = volume.shape[-1]
    valid = np.isfinite(t)
    tt = np.clip(np.where(valid, t, 0.0), 0, n_t - 1)
    k0 = np.floor(tt).astype(int)
    k1 = np.minimum(k0 + 1, n_t - 1)
    w = tt - k0
    v0 = np.take_along_axis(volume, k0[..., None], axis=-1)[..., 0]
    v1 = np.take_along_axis(volume, k1[..., None], axis=-1)[..., 0]
    out = (1 - w) * v0 + w * v1
    return np.where(valid, out, np.nan)


def fill_nearest(t: np.ndarray) -> np.ndarray:
    """Nearest-finite fill, used *only* to sample attributes at uninterpreted cells.

    Without it, a cell isolated by a fault barrier would sample fault probability at
    a meaningless time and could be reconnected in the next iteration.
    """
    bad = ~np.isfinite(t)
    if not bad.any() or bad.all():
        return t
    _, (ii, jj) = ndimage.distance_transform_edt(bad, return_indices=True)
    return t[ii, jj]


@dataclasses.dataclass
class HorizonResult:
    t: np.ndarray  # (n_il, n_xl) fractional samples; NaN = not interpreted
    labels: np.ndarray  # horizon domain per cell (-1 inactive)
    unconstrained: np.ndarray  # live cells in domains without any seed
    extrapolated: np.ndarray  # outside the pick hull of their domain in the initial surface
    seed_distance: np.ndarray  # bins to the nearest seed in the same domain
    clipped: np.ndarray  # cells pushed to the top/bottom of the volume
    slope_support: np.ndarray  # mean incident edge weight (0..1)
    correlation_support: np.ndarray  # summed weight of correlation pairs touching the cell
    barriers: FaultBarriers
    seeds: tuple[np.ndarray, np.ndarray, np.ndarray]
    iterations: list[dict[str, Any]]
    converged: bool
    provenance: Provenance
    correlations: CorrelationPairs

    def time_ms(self, t0_ms: float, dt_ms: float) -> np.ndarray:
        return t0_ms + self.t * dt_ms


class HorizonExtractor3D:
    """Least-squares horizon extraction from local slopes, multigrid correlations and seeds."""

    def __init__(
        self,
        seismic: np.ndarray,
        slopes: SlopeVolume,
        cfg: HorizonConfig | None = None,
        live_mask: np.ndarray | None = None,
        fault_probability: np.ndarray | None = None,
        barriers: FaultBarriers | None = None,
    ) -> None:
        self.seismic = np.asarray(seismic, dtype=np.float32)
        self.slopes = slopes
        self.cfg = cfg or HorizonConfig()
        n_il, n_xl, _ = self.seismic.shape
        if slopes.shape != self.seismic.shape:
            raise ValueError("Slope volume and seismic must have the same shape.")
        self.live = np.ones((n_il, n_xl), bool) if live_mask is None else np.asarray(live_mask, bool)
        self.fault_probability = fault_probability
        self.barriers = barriers or FaultBarriers.empty(n_il, n_xl)

    # ------------------------------------------------------------------ weights
    def _edge_weights(self, t: np.ndarray):
        c = self.cfg
        t = fill_nearest(t)
        P = np.nan_to_num(sample_along(self.slopes.p, t))
        Q = np.nan_to_num(sample_along(self.slopes.q, t))
        PL = np.nan_to_num(sample_along(self.slopes.planarity, t))
        p_edge = 0.5 * (P[:-1] + P[1:])
        q_edge = 0.5 * (Q[:, :-1] + Q[:, 1:])
        w_il = np.maximum(c.min_slope_weight, (0.5 * (PL[:-1] + PL[1:])) ** c.slope_weight_power)
        w_xl = np.maximum(c.min_slope_weight, (0.5 * (PL[:, :-1] + PL[:, 1:])) ** c.slope_weight_power)
        block_il = self.barriers.block_il.copy()
        block_xl = self.barriers.block_xl.copy()
        n_prob_blocked = 0
        if self.fault_probability is not None:
            F = np.nan_to_num(sample_along(self.fault_probability, t))
            f_il = np.maximum(F[:-1], F[1:])
            f_xl = np.maximum(F[:, :-1], F[:, 1:])
            w_il *= (1.0 - f_il) ** c.fault_soft_power
            w_xl *= (1.0 - f_xl) ** c.fault_soft_power
            hard_il = f_il >= c.fault_hard_threshold
            hard_xl = f_xl >= c.fault_hard_threshold
            n_prob_blocked = int((hard_il & ~block_il).sum() + (hard_xl & ~block_xl).sum())
            block_il |= hard_il
            block_xl |= hard_xl
        w_il[block_il] = 0.0
        w_xl[block_xl] = 0.0
        return p_edge, w_il, q_edge, w_xl, FaultBarriers(block_il, block_xl, self.barriers.source), n_prob_blocked

    @staticmethod
    def _cell_support(w_il: np.ndarray, w_xl: np.ndarray) -> np.ndarray:
        n_il, n_xl = w_xl.shape[0], w_il.shape[1]
        s = np.zeros((n_il, n_xl)); n = np.zeros((n_il, n_xl))
        s[:-1] += w_il; s[1:] += w_il; n[:-1] += 1; n[1:] += 1
        s[:, :-1] += w_xl; s[:, 1:] += w_xl; n[:, :-1] += 1; n[:, 1:] += 1
        return s / np.maximum(n, 1)

    # ------------------------------------------------------------------ extract
    def extract(
        self,
        seed_i: np.ndarray,
        seed_j: np.ndarray,
        seed_t: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> HorizonResult:
        c = self.cfg
        n_il, n_xl, n_t = self.seismic.shape
        seed_i = np.asarray(seed_i, int); seed_j = np.asarray(seed_j, int); seed_t = np.asarray(seed_t, float)
        if seed_i.size == 0:
            raise ValueError("At least one approved seed pick is required.")
        if not self.live[seed_i, seed_j].all():
            bad = np.c_[seed_i, seed_j][~self.live[seed_i, seed_j]]
            raise ValueError(f"Seed picks on dead or missing traces: {bad[:5].tolist()}")
        if (seed_t < 0).any() or (seed_t > n_t - 1).any():
            raise ValueError("Seed times outside the vertical extent of the volume/ROI.")

        bi = np.where(self.barriers.block_il, 0.0, 1.0)
        bx = np.where(self.barriers.block_xl, 0.0, 1.0)
        labels0 = edge_components(n_il, n_xl, bi, bx, self.live)
        init: InitialSurface = initial_surface((n_il, n_xl), seed_i, seed_j, seed_t, labels0)
        t = init.t.copy() if initial is None else np.asarray(initial, float).copy()
        t[~self.live] = np.nan
        t = np.clip(t, 0, n_t - 1)

        history: list[dict[str, Any]] = []
        converged = False
        corr = CorrelationPairs.empty()
        clipped = np.zeros((n_il, n_xl), bool)
        t_start = time.perf_counter()
        for it in range(c.max_iterations):
            p_edge, w_il, q_edge, w_xl, barriers, n_prob_blocked = self._edge_weights(t)
            labels = edge_components(n_il, n_xl, w_il, w_xl, self.live)
            refresh = it < max(1, c.correlation.refresh_iterations)
            if c.correlation.enabled and refresh:
                cc = c.correlation
                corr = multigrid_correlations(
                    self.seismic, t, labels, barriers.block_il, barriers.block_xl,
                    strides=tuple(cc.strides), half_window=cc.half_window, max_lag=cc.max_lag,
                    min_correlation=cc.min_correlation, weight=cc.weight,
                )
                corr_tuple = corr.as_tuple()
            elif c.correlation.enabled:
                # frozen correlations: drop pairs whose cells changed domain this iteration
                same = labels.ravel()[corr.a] == labels.ravel()[corr.b]
                corr_tuple = (corr.a[same], corr.b[same], corr.shift[same], corr.weight[same])
            else:
                corr_tuple = None
            t_new, labels, info = solve_horizon_lsq(
                p_edge, w_il, q_edge, w_xl, seed_i, seed_j, seed_t, self.live,
                corr=corr_tuple, seed_mode=c.seed_mode, soft_seed_weight=c.soft_seed_weight,
                solver=c.solver, x0=t, labels=labels,
            )
            clipped = np.isfinite(t_new) & ((t_new < 0) | (t_new > n_t - 1))
            t_new = np.where(np.isfinite(t_new), np.clip(t_new, 0, n_t - 1), np.nan)
            both = np.isfinite(t_new) & np.isfinite(t)
            update = float(np.max(np.abs(t_new[both] - t[both]))) if both.any() else float("inf")
            rec = {
                "iteration": it + 1,
                "max_update_samples": update,
                "rms_update_samples": float(np.sqrt(np.mean((t_new[both] - t[both]) ** 2))) if both.any() else None,
                "n_correlation_pairs": int(corr.a.size),
                "n_corr_candidates": corr.n_candidates,
                "n_corr_rejected_low_corr": corr.n_rejected_low_corr,
                "n_corr_rejected_lag_limit": corr.n_rejected_at_lag_limit,
                "n_corr_rejected_barrier": corr.n_rejected_barrier,
                "n_barrier_edges": barriers.n_blocked,
                "n_probability_barrier_edges": n_prob_blocked,
                "n_clipped": int(clipped.sum()),
                **dataclasses.asdict(info),
            }
            history.append(rec)
            log.info("horizon iteration", extra=rec)
            t = t_new
            if update < c.tolerance:
                converged = True
                break

        unconstrained = self.live & ~np.isfinite(t)
        corr_support = np.zeros(n_il * n_xl)
        np.add.at(corr_support, corr.a, corr.weight)
        np.add.at(corr_support, corr.b, corr.weight)
        prov = Provenance(
            product="horizon_3d",
            parameters={"horizon_config": config_to_dict(c), "slope_params": self.slopes.params,
                        "n_seeds": int(seed_i.size), "seed_mode": c.seed_mode},
            inputs={"seismic": sha256_array(self.seismic), "seeds": sha256_array(np.c_[seed_i, seed_j, seed_t])},
        )
        prov.parameters["runtime_seconds"] = time.perf_counter() - t_start
        prov.parameters["converged"] = converged
        if not converged:
            prov.notes.append(f"Did not converge in {c.max_iterations} iterations (last max update {history[-1]['max_update_samples']:.3f} samples).")
        if unconstrained.any():
            prov.notes.append(f"{int(unconstrained.sum())} live cells lie in fault blocks without seed picks and were left uninterpreted.")
        return HorizonResult(
            t=t, labels=labels, unconstrained=unconstrained, extrapolated=init.extrapolated & np.isfinite(t),
            seed_distance=init.seed_distance, clipped=clipped, slope_support=self._cell_support(w_il, w_xl),
            correlation_support=corr_support.reshape(n_il, n_xl), barriers=barriers,
            seeds=(seed_i, seed_j, seed_t), iterations=history, converged=converged, provenance=prov,
            correlations=corr,
        )
