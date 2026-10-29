"""Constants and small numerical helpers shared by the indexing, figure and view modules."""
import math

import numpy as np

# Cu radiation (angstrom)
L1 = 1.540593      # Ka1
L2 = 1.544427      # Ka2
LB = 1.39222       # Kb
LW = 1.47642       # W La (tube contamination line)


def fsu(v, e):
    """Value with standard uncertainty in crystallographic notation, e.g. fsu(3.35926, 0.00029) -> '3.3593(3)'.

    One or two digits in the parentheses (two when the leading digit is 1). Without a usable esd the
    value is returned plainly.
    """
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    if e is None or not np.isfinite(e) or e <= 0:
        return f'{v:g}'
    dec = -int(math.floor(math.log10(e))) + (1 if f'{e:.1e}'[0] == '1' else 0)
    dec = max(dec, 0)
    u = round(e * 10 ** dec)
    if u >= 100:
        dec -= 1
        u = round(e * 10 ** dec)
    return f'{v:.{dec}f}({u})' if dec > 0 else f'{round(v)}({u})'


def tth(d, lam=L1):
    """2θ (degrees) of spacing d; NaN where the reflection is beyond the sphere."""
    s = lam / (2 * np.asarray(d, dtype=float))
    with np.errstate(invalid='ignore'):
        return np.where(s < 1, 2 * np.degrees(np.arcsin(np.clip(s, -1, 1))), np.nan)


def dsp(t, lam=L1):
    """d spacing of a 2θ position (degrees)."""
    return lam / (2 * np.sin(np.radians(np.asarray(t, dtype=float) / 2)))


def ka2(t1):
    """Ka2 position of a Ka1 line at 2θ = t1."""
    return 2 * np.degrees(np.arcsin(np.clip(L2 / L1 * np.sin(np.radians(np.asarray(t1) / 2)), -1, 1)))


def pv(x, c, h, w, eta):
    """Pseudo-Voigt with height h, FWHM w and Lorentzian fraction eta (broadcasts)."""
    z = (x - c) / w
    return h * (eta / (1 + 4 * z * z) + (1 - eta) * np.exp(-4 * np.log(2) * z * z))


def dbl(x, c, h, w, eta=0.5):
    """Ka1 + Ka2 doublet (Ka2 at half height)."""
    return pv(x, c, h, w, eta) + 0.5 * pv(x, ka2(c), h, w, eta)


def dbl_sum(x, c, h, w, eta):
    """Sum of several doublets; c, h, w, eta are 1-D arrays of equal length."""
    if len(c) == 0:
        return np.zeros_like(x)
    X = x[:, None]
    return dbl(X, np.asarray(c)[None, :], np.asarray(h)[None, :], np.asarray(w)[None, :], np.asarray(eta)[None, :]).sum(axis=1)


def rwp(yo, yc, m=slice(None)):
    """Weighted profile R factor (%), w = 1/yobs (Poisson)."""
    w = 1 / np.sqrt(np.maximum(yo, 1))
    return float(100 * np.sqrt((((yo - yc) * w)[m] ** 2).sum() / ((yo * w)[m] ** 2).sum()))


def sub_digits(s):
    """'VSe2' -> 'VSe₂' for display (digits after a letter become subscripts)."""
    out = []
    for i, ch in enumerate(s):
        if ch.isdigit() and i > 0 and (s[i - 1].isalpha() or s[i - 1] in '₀₁₂₃₄₅₆₇₈₉)'):
            out.append('₀₁₂₃₄₅₆₇₈₉'[int(ch)])
        else:
            out.append(ch)
    return ''.join(out)
