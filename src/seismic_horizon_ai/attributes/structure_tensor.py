"""Gradient structure tensor: local reflection slopes, dip, azimuth and planarity.

For a locally planar reflector ``f(x, y, t) = g(t - p x - q y)`` the amplitude
gradient is parallel to the reflector normal ``u = (-p, -q, 1) / norm``. The eigenvector
of the smoothed gradient outer product with the largest eigenvalue estimates ``u``,
so ``p = -u_il / u_t`` and ``q = -u_xl / u_t`` (samples per trace).

Planarity ``(l1 - l2) / l1`` (l1 >= l2 >= l3) is near 1 for clean continuous
reflections and drops at faults, noise and chaotic zones; it is used as the slope
confidence (equation weight) in the horizon engine.

The volume is processed in inline chunks with a halo so memory stays bounded.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

from ..config import StructureTensorConfig


@dataclasses.dataclass
class SlopeVolume:
    """Derived structural attributes, all shaped like the seismic cube."""

    p: np.ndarray  # dt/d(il), samples per inline step
    q: np.ndarray  # dt/d(xl), samples per crossline step
    planarity: np.ndarray  # 0..1 slope confidence
    params: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.p.shape  # type: ignore[return-value]

    def dip_azimuth(self) -> tuple[np.ndarray, np.ndarray]:
        """Apparent dip magnitude (samples/trace) and azimuth (deg, grid frame, 0 = +inline index)."""
        return np.hypot(self.p, self.q), np.degrees(np.arctan2(self.q, self.p))


def _tensor_eigen(gi: np.ndarray, gx: np.ndarray, gt: np.ndarray, sigma_s: tuple[float, float, float], max_slope: float):
    comps = {
        "ii": gi * gi, "ix": gi * gx, "it": gi * gt,
        "xx": gx * gx, "xt": gx * gt, "tt": gt * gt,
    }
    for k in comps:
        comps[k] = ndimage.gaussian_filter(comps[k], sigma_s, mode="nearest")
    T = np.empty(gi.shape + (3, 3), dtype=np.float32)
    T[..., 0, 0] = comps["ii"]; T[..., 0, 1] = T[..., 1, 0] = comps["ix"]; T[..., 0, 2] = T[..., 2, 0] = comps["it"]
    T[..., 1, 1] = comps["xx"]; T[..., 1, 2] = T[..., 2, 1] = comps["xt"]; T[..., 2, 2] = comps["tt"]
    w, v = np.linalg.eigh(T)  # ascending eigenvalues
    l1, l2 = w[..., 2], w[..., 1]
    u = v[..., :, 2]
    ut = u[..., 2]
    sign = np.where(ut < 0, -1.0, 1.0).astype(np.float32)
    ui, ux, ut = u[..., 0] * sign, u[..., 1] * sign, ut * sign
    ut_safe = np.maximum(ut, 1.0 / np.sqrt(1.0 + 2 * max_slope**2))
    p = np.clip(-ui / ut_safe, -max_slope, max_slope)
    q = np.clip(-ux / ut_safe, -max_slope, max_slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        plan = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
    return p.astype(np.float32), q.astype(np.float32), np.clip(plan, 0, 1).astype(np.float32)


def compute_slopes(
    seismic: np.ndarray,
    cfg: StructureTensorConfig | None = None,
    live_mask: np.ndarray | None = None,
    max_slope: float = 8.0,
) -> SlopeVolume:
    """Estimate inline/crossline reflection slopes with a 3D structure tensor.

    Parameters
    ----------
    seismic : (n_il, n_xl, n_t) array (in memory or an h5py dataset; read chunk-wise).
    live_mask : optional (n_il, n_xl); planarity is forced to 0 on dead/missing traces.
    max_slope : |p|, |q| are clipped to this value (samples/trace).
    """
    cfg = cfg or StructureTensorConfig()
    n_il, n_xl, n_t = seismic.shape
    sg = float(cfg.sigma_gradient)
    ss = tuple(float(s) for s in cfg.sigma_smooth)
    halo = int(np.ceil(4 * (sg + ss[0])))
    p = np.zeros((n_il, n_xl, n_t), dtype=np.float32)
    q = np.zeros_like(p)
    plan = np.zeros_like(p)
    step = max(1, int(cfg.chunk_inlines))
    for i0 in range(0, n_il, step):
        i1 = min(n_il, i0 + step)
        a0, a1 = max(0, i0 - halo), min(n_il, i1 + halo)
        block = np.asarray(seismic[a0:a1], dtype=np.float32)
        gi = ndimage.gaussian_filter(block, sg, order=(1, 0, 0), mode="nearest")
        gx = ndimage.gaussian_filter(block, sg, order=(0, 1, 0), mode="nearest")
        gt = ndimage.gaussian_filter(block, sg, order=(0, 0, 1), mode="nearest")
        bp, bq, bpl = _tensor_eigen(gi, gx, gt, ss, max_slope)
        sl = slice(i0 - a0, i0 - a0 + (i1 - i0))
        p[i0:i1], q[i0:i1], plan[i0:i1] = bp[sl], bq[sl], bpl[sl]
    if live_mask is not None:
        plan[~live_mask] = 0.0
    return SlopeVolume(p=p, q=q, planarity=plan, params={"sigma_gradient": sg, "sigma_smooth": ss, "max_slope": max_slope, "halo": halo})


def compute_slopes_2d(section: np.ndarray, sigma_gradient: float = 1.0, sigma_smooth: tuple[float, float] = (3.0, 2.0), max_slope: float = 8.0):
    """2D structure-tensor slopes for a single (n_trace, n_t) section. Returns (p, planarity)."""
    s = np.asarray(section, dtype=np.float32)
    gx = ndimage.gaussian_filter(s, sigma_gradient, order=(1, 0), mode="nearest")
    gt = ndimage.gaussian_filter(s, sigma_gradient, order=(0, 1), mode="nearest")
    xx = ndimage.gaussian_filter(gx * gx, sigma_smooth, mode="nearest")
    xt = ndimage.gaussian_filter(gx * gt, sigma_smooth, mode="nearest")
    tt = ndimage.gaussian_filter(gt * gt, sigma_smooth, mode="nearest")
    tr, det = xx + tt, xx * tt - xt * xt
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    # eigenvector for l1: (xt, l1 - xx) or (l1 - tt, xt)
    ux, ut = l1 - tt, xt
    ux2, ut2 = xt, l1 - xx
    use2 = np.abs(ut2) > np.abs(ut)
    ux, ut = np.where(use2, ux2, ux), np.where(use2, ut2, ut)
    sign = np.where(ut < 0, -1.0, 1.0)
    ux, ut = ux * sign, ut * sign
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(ut > 1e-20, -ux / ut, 0.0)
        plan = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
    return np.clip(p, -max_slope, max_slope).astype(np.float32), np.clip(plan, 0, 1).astype(np.float32)
