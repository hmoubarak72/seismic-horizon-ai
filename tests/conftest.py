import numpy as np
import pytest

from seismic_horizon_ai.attributes.structure_tensor import compute_slopes
from seismic_horizon_ai.synthetic import make_synthetic

TARGET_TAU = 80.0


@pytest.fixture(scope="session")
def flat_case():
    sv = make_synthetic(throw=0.0, noise=0.1, seed=7)
    return sv, compute_slopes(sv.seismic)


@pytest.fixture(scope="session")
def fault_case():
    sv = make_synthetic(throw=7.0, noise=0.1, seed=11)
    return sv, compute_slopes(sv.seismic)


def random_seeds(sv, n=12, seed=0, mask=None):
    rng = np.random.default_rng(seed)
    n_il, n_xl, _ = sv.seismic.shape
    h = sv.horizon(TARGET_TAU)
    cells = np.argwhere(np.ones((n_il, n_xl), bool) if mask is None else mask)
    pick = cells[rng.choice(len(cells), size=n, replace=False)]
    return pick[:, 0], pick[:, 1], h[pick[:, 0], pick[:, 1]]


def fault_seeds(sv, n_each=6, seed=0):
    """Seeds on both fault blocks, kept away from the fault itself."""
    from scipy import ndimage

    side = sv.fault_side
    edge = ndimage.binary_dilation(side, iterations=3) & ~ndimage.binary_erosion(side, iterations=3)
    a = random_seeds(sv, n_each, seed, mask=side & ~edge)
    b = random_seeds(sv, n_each, seed + 1, mask=~side & ~edge)
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))
