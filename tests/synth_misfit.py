import math, numpy as np
from xrdcore.util import L1, dbl, tth
HEX_HK0 = {(1, 0): 900, (1, 1): 700, (2, 0): 260, (2, 1): 220, (3, 0): 160, (2, 2): 120}


TET_HK0 = {(2, 0): 700, (2, 2): 500, (4, 0): 200, (4, 2): 160, (4, 4): 90}


def misfit_pattern(seed=2, c=12.228, a_hex=3.442, a_tet=6.055, K=-0.05, bands=1.0, hex_hk0=None, tet_hk0=None):
    """00l superlattice (sharp) + hexagonal and tetragonal hk0 bands (broad); hex_hk0 = {(h, k): integrated intensity}."""
    rng = np.random.default_rng(seed)
    x = np.arange(5, 90, 0.01)
    y = 400 + 6000 * np.exp(-(x - 5) / 1.2) + 300 * np.exp(-(x - 20) ** 2 / 400)
    sh = lambda t: t + K * math.cos(math.radians(t / 2))
    for l in range(1, 15):
        t = float(tth(c / l, L1))
        if np.isfinite(t) and t < 90:
            y = y + dbl(x, sh(t), 30000 * math.exp(-0.18 * l) * (1.6 if l % 2 == 0 else 0.5), 0.10, 0.5)
    for (h, k), I in (hex_hk0 or HEX_HK0).items():
        d = a_hex / math.sqrt(4 / 3 * (h * h + h * k + k * k)); t = float(tth(d, L1))
        if np.isfinite(t) and t < 90: y = y + dbl(x, sh(t), I * bands, 0.8, 0.6)
    for (h, k), I in (tet_hk0 or TET_HK0).items():
        d = a_tet / math.hypot(h, k); t = float(tth(d, L1))
        if np.isfinite(t) and t < 90: y = y + dbl(x, sh(t), I * bands, 0.6, 0.6)
    return x, rng.poisson(y).astype(float)
