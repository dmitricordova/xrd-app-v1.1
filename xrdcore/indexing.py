"""Peak indexing of a lab Cu Kα pattern (Part A of the XRD workup).

Stage 1 (depends only on the pattern, slow part, cached by the app):
    background         - one of (BkgSettings.mode):
                           'step2'     refit: prior peak model held fixed (ycal - bkg, tapered to zero below the first
                                       degree), background = N-term Chebyshev (default 24) + 4 low-angle exponential
                                       decays (optional), weighted least squares; without a prior model the target is a
                                       3° rolling 5th percentile of yobs
                           'file'      the bkg column of the input file, as it is
                           'file+poly' the file's bkg plus a Chebyshev (+ decays) correction fitted to yobs - peaks - bkg
                         refine=True: the background polynomial and every fitted peak (pseudo-Voigt doublets) are
                         refined together by least squares over the whole pattern (bkgrefine.refine); the refined
                         peaks (still above 3σ) are the ones indexed
    peaks              - scipy find_peaks seeds, pseudo-Voigt Kα1+Kα2 doublets fitted cluster by cluster,
                         kept above 3σ of the integrated signal; the fitted position is Kα1
Stage 2 (phases, correction):
    hkl assignment     - every listed hkl of every enabled phase to the nearest sharp peak (0.15 → 0.08 → 0.06°)
    cells              - one joint least-squares fit of all cells + one angle correction on the lines that belong
                         to one phase only; σ = sqrt(diag[(JᵀJ)⁻¹ Σr²/(n − p)]); axes no assigned line
                         constrains are held at their start value and reported as "not determined"
    Cu Kβ              - every strong reflection fitted directly at its Kβ position
    tables             - peak table, refined-parameters table, text report, plot labels and tick rows
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as Cheb
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .phases import ANGLES, AXES, SYMBOL, hkl_label, unit
from .util import L1, LB, LW, dbl, dbl_sum, dsp, fsu, ka2, rwp, sub_digits, tth

KB_COLOR = '#7d3c98'
UN_COLOR = '#555555'


class IndexingError(RuntimeError):
    """The indexing could not run; the message says why."""


# --------------------------------------------------------------------------------------------- stage 1
def step2_background(x, yobs, peaks=None, order=24, decays=(0.3, 0.6, 1.2, 2.5), taper_width=0.4):
    """Returns (background, prior peaks used). peaks = prior peak model (ycal - bkg) or None."""
    w = 1 / np.sqrt(np.maximum(yobs, 1))
    if peaks is None:
        n = max(3, int(round(3 / np.median(np.diff(x)))))
        target = pd.Series(yobs).rolling(n, center=True, min_periods=1).quantile(0.05).values
        used = np.zeros_like(yobs)
    else:
        used = peaks * np.clip((x - (x.min() + 1.0)) / taper_width, 0, 1)
        target = yobs - used
    t = (2 * x - x.min() - x.max()) / (x.max() - x.min())
    A = np.column_stack([Cheb.chebvander(t, order - 1)] + [np.exp(-(x - x.min()) / s) for s in decays])
    c, *_ = np.linalg.lstsq(A * w[:, None], target * w, rcond=None)
    return A @ c, used


def fit_peaks(x, y, bkg, model=None, prom_factor=8.0):
    """Find and fit peaks. Returns a DataFrame t, h, w, eta, I, profile, atbound (sorted by 2θ)."""
    sig = y - bkg
    step = float(np.median(np.diff(x)))
    noise = 1.5 * np.median(np.abs(np.diff(sig)))
    p1, _ = find_peaks(sig, prominence=prom_factor * noise, distance=8)
    seeds = set(np.round(x[p1], 3))
    if model is not None:
        p2, _ = find_peaks(model, prominence=40, distance=8)
        seeds |= set(np.round(x[p2], 3))
    seeds = sorted(s for s in seeds if s > x.min() + 0.6)
    S = []
    for s in seeds:
        if not S or s - S[-1] >= 0.03:
            S.append(s)
    if not S:
        raise IndexingError('No peaks rise above the noise in this pattern.')

    def width_seed(c):
        if model is None:
            return 0.1
        i = int(np.argmin(abs(x - c)))
        m = model[i]
        if m <= 0:
            return 0.1
        j = i
        while j > 0 and model[j] > m / 2:
            j -= 1
        k = i
        while k < len(x) - 1 and model[k] > m / 2:
            k += 1
        return float(np.clip((k - j) * step, 0.04, 2.0))

    clusters, cur = [], [S[0]]
    for s in S[1:]:
        if s - cur[-1] < 0.8:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)
    def fit_cluster(seeds_, widths, lo_pad, hi_pad):
        m = (x >= seeds_[0] - lo_pad) & (x <= seeds_[-1] + hi_pad)
        xx, yy, ww = x[m], sig[m], 1 / np.sqrt(np.maximum(y[m], 1))
        p0, lb, ub = [], [], []
        for s, w0 in zip(seeds_, widths):
            i = int(np.argmin(abs(x - s)))
            h0 = max(sig[i], 10)
            p0 += [s, h0, w0, 0.5]
            lb += [s - 0.08, 0, 0.02, 0]
            ub += [s + 0.08, max(h0 * 5, 100), 2.5, 1]
        p0 += [0]; lb += [-1e5]; ub += [1e5]
        n = len(seeds_)

        def f(p):
            q = p[:4 * n].reshape(n, 4)
            return (p[-1] + dbl_sum(xx, q[:, 0], q[:, 1], q[:, 2], q[:, 3]) - yy) * ww

        return least_squares(f, np.clip(p0, lb, ub), bounds=(lb, ub), max_nfev=600).x

    out = []
    for cl in clusters:
        r = fit_cluster(cl, [width_seed(s) for s in cl], 0.35, 0.45)
        widths = r[2:4 * len(cl):4]
        if model is None and widths.max() > 0.3:
            # no prior model to seed widths: a broad band fills the small window and its width runs away.
            # Refit in a window sized to the band, with the neighbouring seeds inside it modelled too.
            pad = float(np.clip(2.5 * min(widths.max(), 1.2), 0.5, 3.0))
            inside = [s for s in S if cl[0] - pad <= s <= cl[-1] + pad][:10]
            if not all(s in inside for s in cl):
                inside = sorted(set(inside) | set(cl))
            w_in = [float(widths[cl.index(s)]) if s in cl else width_seed(s) for s in inside]
            r2 = fit_cluster(inside, w_in, pad, pad)
            pick = [inside.index(s) for s in cl]
            r = np.concatenate([r2[4 * i:4 * i + 4] for i in pick] + [r2[-1:]])
        for k in range(len(cl)):
            c, h, w, eta = r[4 * k:4 * k + 4]
            i = int(np.argmin(abs(x - c)))
            thr = 3 * np.sqrt(y[i]) / np.sqrt(max(w / step, 1))
            if h > thr:
                out.append(dict(t=c, h=h, w=w, eta=eta))
    if not out:
        raise IndexingError('Peaks were found but none passed the 3σ significance test.')
    R = pd.DataFrame(out).sort_values('h', ascending=False)
    keep = []
    for _, r in R.iterrows():
        bad = any(abs(r.t - k.t) < 0.06 or (abs(r.t - k.t) < 0.15 and r.h < 0.3 * k.h and abs(r.t - ka2(k.t)) < 0.03) for k in keep)
        if r.w < 0.05 and r.h < 400:
            bad = True
        if not bad:
            keep.append(r)
    P = pd.DataFrame(keep).sort_values('t').reset_index(drop=True)
    P['I'] = 100 * P.h / P.h.max()
    P['profile'] = np.where(P.w > 0.25, 'broad', 'sharp')
    P['atbound'] = (P.w > 2.49) | (P.w < 0.021)
    return P


BKG_MODES = {'step2': 'Refit: Chebyshev + low-angle decays', 'file': 'From the file, as it is',
             'file+poly': 'From the file + Chebyshev correction'}


@dataclass(frozen=True)
class BkgSettings:
    mode: str = 'step2'          # step2 | file | file+poly
    order: int = 24              # Chebyshev terms (of the background, or of the correction for file+poly)
    decays: bool = True          # four low-angle exponential decays (0.3, 0.6, 1.2, 2.5°)
    refine: bool = False         # refine background polynomial and peaks together before indexing
    cycles: int = 5              # effort of the joint refinement: at most 20 × cycles function evaluations

    def label(self):
        base = {'step2': f'Chebyshev {self.order}' + (' + decays' if self.decays else ''), 'file': 'from file',
                'file+poly': f'file + Chebyshev {self.order}' + (' + decays' if self.decays else '')}[self.mode]
        return base + (', refined' if self.refine and self.mode != 'file' else '')


def poly_background(x, y, target, order=24, decays=(0.3, 0.6, 1.2, 2.5)):
    """Weighted (1/√y) least-squares fit of an N-term Chebyshev (+ low-angle exponential decays) to target."""
    w = 1 / np.sqrt(np.maximum(y, 1))
    t = (2 * x - x.min() - x.max()) / (x.max() - x.min())
    A = np.column_stack([Cheb.chebvander(t, max(int(order), 1) - 1)] + [np.exp(-(x - x.min()) / s) for s in (decays or ())])
    c, *_ = np.linalg.lstsq(A * w[:, None], target * w, rcond=None)
    return A @ c


@dataclass
class Stage1:
    name: str
    x: np.ndarray
    yobs: np.ndarray
    bkg: np.ndarray
    ycal: np.ndarray
    calc_source: str
    peaks: pd.DataFrame
    rwp_prior: Optional[float]
    rwp_prior_lo: Optional[float]
    rwp_step2: float
    rwp_step2_lo: float
    has_prior: bool
    bkg_label: str = 'step-2'
    bkg_settings: Optional[BkgSettings] = None
    bkg_history: List[dict] = field(default_factory=list)

    @property
    def sig(self):
        return self.yobs - self.bkg

    def pattern_csv(self):
        df = pd.DataFrame({'#twotheta': self.x, 'yobs': self.yobs, 'ycal': self.ycal, 'bkg': self.bkg, 'diff': self.yobs - self.ycal})
        return df.to_csv(index=False, float_format='%.6e')


def background_and_peaks(pat, order=24, prom_factor=8.0, bkg=None):
    """Background (see BkgSettings) and fitted peaks. bkg: BkgSettings or None (= step-2 refit, Chebyshev order)."""
    B = bkg or BkgSettings(order=order)
    x, y = pat.x.astype(float), pat.y.astype(float)
    prior = pat.prior_peaks
    decays = (0.3, 0.6, 1.2, 2.5) if B.decays else ()
    if B.mode in ('file', 'file+poly') and pat.bkg is None:
        raise IndexingError('This file has no background column; choose "Refit" for the background.')
    base = pat.bkg.astype(float) if B.mode != 'step2' else np.zeros_like(y)
    used = np.zeros_like(y)
    if prior is not None:
        used = prior * np.clip((x - (x.min() + 1.0)) / 0.4, 0, 1)
    if B.mode == 'file':
        b = base.copy()
    elif B.mode == 'file+poly':
        target = (y - used - base) if prior is not None else \
            pd.Series(y - base).rolling(max(3, int(round(3 / np.median(np.diff(x))))), center=True, min_periods=1).quantile(0.05).values
        b = base + poly_background(x, y, target, B.order, decays)
    else:
        if decays == (0.3, 0.6, 1.2, 2.5):
            b, used = step2_background(x, y, prior, order=B.order)
        else:
            if prior is None:
                n = max(3, int(round(3 / np.median(np.diff(x)))))
                target = pd.Series(y).rolling(n, center=True, min_periods=1).quantile(0.05).values
            else:
                target = y - used
            b = poly_background(x, y, target, B.order, decays)
    P = fit_peaks(x, y, b, prior, prom_factor=prom_factor)
    fitted = lambda P_: dbl_sum(x, P_.t.values, P_.h.values, P_.w.values, P_.eta.values)
    hist = [dict(cycle=0, rwp=rwp(y, b + fitted(P)), shift_sigma=float('nan'), peaks=len(P))]
    refined = B.refine and B.mode != 'file'
    if refined:
        from . import bkgrefine
        J = bkgrefine.refine(x, y, base, b, P, order=B.order, decays=B.decays, max_nfev=B.cycles * 20)
        shift = float(np.sqrt(np.mean(((J['bkg'] - b) / np.sqrt(np.maximum(y, 1))) ** 2)))
        b = J['bkg']
        step = float(np.median(np.diff(x)))
        Q = pd.DataFrame(dict(t=J['t'], h=J['h'], w=J['w'], eta=J['eta']))
        thr = [3 * np.sqrt(y[int(np.argmin(abs(x - r.t)))]) / np.sqrt(max(r.w / step, 1)) for r in Q.itertuples()]
        Q = Q[Q.h.values > np.array(thr)].sort_values('t').reset_index(drop=True)
        Q['I'] = 100 * Q.h / Q.h.max()
        Q['profile'] = np.where(Q.w > 0.25, 'broad', 'sharp')
        Q['atbound'] = (Q.w > 2.49) | (Q.w < 0.021)
        P = Q
        hist.append(dict(cycle='joint', rwp=rwp(y, b + fitted(P)), shift_sigma=shift, peaks=len(P), nfev=J['nfev'],
                         converged=J['status'] > 0))
    if prior is not None and not refined:
        ycal, src = used + b, f'prior refinement peaks + background ({B.label()})'
    else:
        ycal, src = b + fitted(P), f'fitted peaks + background ({B.label()})'
    lo = x < 10
    rp = rpl = None
    if prior is not None:
        rp = rwp(y, pat.ycal)
        rpl = rwp(y, pat.ycal, lo) if lo.sum() > 10 else None
    return Stage1(name=pat.name, x=x, yobs=y, bkg=b, ycal=ycal, calc_source=src, peaks=P, rwp_prior=rp, rwp_prior_lo=rpl,
                  rwp_step2=rwp(y, ycal), rwp_step2_lo=rwp(y, ycal, lo) if lo.sum() > 10 else float('nan'), has_prior=prior is not None,
                  bkg_label=B.label(), bkg_settings=B, bkg_history=hist)


# --------------------------------------------------------------------------------------------- geometry
def _gstar(cell):
    a = cell['a']; b = cell.get('b', a); c = cell.get('c', a)
    al, be, ga = (math.radians(cell.get(x, 90.0)) for x in ANGLES)
    G = np.array([[a * a, a * b * math.cos(ga), a * c * math.cos(be)], [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
                  [a * c * math.cos(be), b * c * math.cos(al), c * c]])
    return G


def inv_d2(lat, cell, h, k, l):
    if lat in ('monoclinic', 'triclinic'):
        v = np.array([h, k, l], float)
        return float(v @ np.linalg.inv(_gstar(cell)) @ v)
    a = cell['a']; c = cell.get('c', a); b = cell.get('b', a)
    if lat == 'cubic':
        return (h * h + k * k + l * l) / a ** 2
    if lat == 'tetragonal':
        return (h * h + k * k) / a ** 2 + l * l / c ** 2
    if lat == 'hexagonal':
        return 4 / 3 * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2
    if lat == 'orthorhombic':
        return h * h / a ** 2 + k * k / b ** 2 + l * l / c ** 2
    raise ValueError('unknown lattice ' + lat)


def d_hkl(ph, hkl, cell=None):
    return 1 / math.sqrt(inv_d2(ph['lattice'], cell or ph['cell'], *hkl))


def cell_volume(lat, cell):
    if lat in ('monoclinic', 'triclinic'):
        return float(math.sqrt(np.linalg.det(_gstar(cell))))
    a = cell['a']; c = cell.get('c', a); b = cell.get('b', a)
    return {'cubic': a ** 3, 'tetragonal': a * a * c, 'hexagonal': math.sqrt(3) / 2 * a * a * c, 'orthorhombic': a * b * c}[lat]


def allowed(cen, h, k, l):
    if cen == 'I':
        return (h + k + l) % 2 == 0
    if cen == 'F':
        return len({h % 2, k % 2, l % 2}) == 1
    if cen == 'C':
        return (h + k) % 2 == 0
    if cen == 'R':
        return (-h + k + l) % 3 == 0
    return True


SERIES = ('all', '00l', 'hk0', '00l+hk0')


def in_series(hkl, series):
    """00l: only h = k = 0; hk0: only l = 0; 00l+hk0: either; all: every reflection."""
    h, k, l = hkl
    if series == '00l':
        return h == 0 and k == 0
    if series == 'hk0':
        return l == 0
    if series == '00l+hk0':
        return (h == 0 and k == 0) or l == 0
    return True


def hkl_list(ph):
    series = ph.get('series', 'all')
    if ph.get('hkl'):
        return [tuple(v) for v in ph['hkl'] if in_series(v, series)]
    cell = ph['cell']
    ab = max(cell['a'], cell.get('b', cell['a']))
    # index ranges from the cell: every reflection with d > λ/2 (capped)
    H = int(ph.get('hmax') or min(12, math.ceil(2 * ab / L1)))
    L = int(ph.get('lmax') or min(60, math.ceil(2 * cell.get('c', cell['a']) / L1)))
    if series == '00l':
        H = 0
    if series == 'hk0':
        L = 0
    lat, seen, out = ph['lattice'], set(), []
    low = lat in ('monoclinic', 'triclinic')
    rng_l = range(-L, L + 1) if ph.get('centering') == 'R' or low else range(0, L + 1)
    for h in range(-H if lat == 'triclinic' else 0, H + 1):
        for k in range(-H if lat == 'triclinic' else 0, H + 1):
            for l in rng_l:
                if h == k == l == 0 or not allowed(ph.get('centering', 'P'), h, k, l) or not in_series((h, k, l), series):
                    continue
                if lat in ('hexagonal', 'tetragonal') and k > h and ph.get('centering') != 'R':
                    continue
                if lat == 'cubic' and not (h >= k >= l >= 0):
                    continue
                d = round(d_hkl(ph, (h, k, l)), 5)
                if d < L1 / 2:
                    continue
                if d not in seen:
                    seen.add(d)
                    out.append((h, k, l))
    return out


def corr(t, model, k):
    """Angle correction added to a calculated 2θ: displacement K·cosθ, zero shift z, or none."""
    t = np.asarray(t, dtype=float)
    if model == 'displacement':
        return k * np.cos(np.radians(t / 2))
    if model == 'zero':
        return k + 0 * t
    return 0 * t


def observed_position(d, model, k, lam=L1):
    t = float(tth(d, lam))
    return t + float(corr(t, model, k)) if np.isfinite(t) else float('nan')


# --------------------------------------------------------------------------------------------- stage 2
def assign(P, phases, model, k, tol):
    """hkl -> nearest sharp peak within tol. The hk0 lines of a phase restricted to hk0 (turbostratic in-plane bands)
    may also take a broad peak, within half its width (at most 0.5°)."""
    A = []
    cand = P[(P.profile == 'sharp') & ~P.atbound]
    broad = P[~P.atbound]
    for pi, ph in enumerate(phases):
        bands = ph.get('series', 'all') in ('hk0', '00l+hk0')
        for hkl in hkl_list(ph):
            to = observed_position(d_hkl(ph, hkl), model, k)
            if not np.isfinite(to) or to < P.t.min() - 0.5 or to > P.t.max() + 0.5:
                continue
            if bands and hkl[2] == 0 and len(broad):
                reach = np.maximum(tol, np.minimum(0.5 * broad.w.values, 0.5))
                d = (broad.t - to).abs().values
                ok = d <= reach
                if ok.any():
                    A.append((pi, tuple(hkl), int(broad.index[np.where(ok)[0][np.argmin(d[ok])]])))
                continue
            if not len(cand):
                continue
            j = (cand.t - to).abs().idxmin()
            if abs(P.t[j] - to) < tol:
                A.append((pi, tuple(hkl), int(j)))
    return A


def _free_axes(phases, rows, model, k):
    """Axes each phase's assigned lines actually constrain (numerical rank test)."""
    free, held = [], {}
    for pi, ph in enumerate(phases):
        R = [hkl for qi, hkl, _ in rows if qi == pi]
        axes = AXES[ph['lattice']]
        if not R:
            for ax in axes:
                held[(pi, ax)] = 'no assigned lines'
            continue
        M = np.zeros((len(R), len(axes)))
        for j, ax in enumerate(axes):
            c1 = dict(ph['cell']); c1[ax] *= 1.0001
            for i, hkl in enumerate(R):
                M[i, j] = (observed_position(d_hkl(ph, hkl, c1), model, k) - observed_position(d_hkl(ph, hkl), model, k)) / (ph['cell'][ax] * 1e-4)
        order = np.argsort(-np.linalg.norm(M, axis=0))
        keep = []
        for j in order:
            trial = keep + [j]
            if np.linalg.matrix_rank(M[:, trial], tol=1e-6 * max(np.abs(M).max(), 1e-12)) == len(trial):
                keep = trial
        for j, ax in enumerate(axes):
            if j in keep:
                free.append((pi, ax))
            else:
                held[(pi, ax)] = 'no assigned line depends on it' if np.allclose(M[:, j], 0) else 'not independent of the other axes with these lines'
    return free, held


def fit_cells(P, phases, A, model, exclude=(), k0=0.0):
    rows = [a for a in A if a[2] not in exclude]
    keys, held = _free_axes(phases, rows, model, k0)
    fit_corr = model != 'none'
    n = len(rows)
    if fit_corr and n < len(keys) + 2:
        fit_corr = False
    p0 = [phases[pi]['cell'][ax] for pi, ax in keys] + ([k0] if fit_corr else [])
    if not rows or not p0:
        return dict(cells={}, held=held, corr=(0.0, float('nan')) if model != 'none' else (0.0, 0.0), corr_fitted=False, rms=float('nan'),
                    n=n, used={pi: 0 for pi in range(len(phases))}, resid={})
    tobs = np.array([P.t[j] for *_, j in rows])
    wts = np.array([min(1.0, 0.25 / max(P.w[j], 0.25)) for *_, j in rows])      # broad bands count less
    def calc(p):
        out = np.empty(len(rows))
        cells = [dict(ph['cell']) for ph in phases]
        for v, (pi, ax) in zip(p, keys):
            cells[pi][ax] = v
        kk = p[-1] if fit_corr else (k0 if model != 'none' else 0.0)
        for i, (pi, hkl, _) in enumerate(rows):
            t = float(tth(d_hkl(phases[pi], hkl, cells[pi])))
            out[i] = t + float(corr(t, model, kk))
        return out

    R = least_squares(lambda p: (calc(p) - tobs) * wts, p0, x_scale='jac')
    dof = max(n - len(p0), 1)
    cov = np.linalg.pinv(R.jac.T @ R.jac) * np.sum(R.fun ** 2) / dof
    raw = calc(R.x) - tobs
    e = np.sqrt(np.abs(np.diag(cov)))
    cells = {}
    for (pi, ax), v, u in zip(keys, R.x, e):
        cells.setdefault(pi, {})[ax] = (float(v), float(u) if n > len(p0) else float('nan'))
    if fit_corr:
        cp = (float(R.x[-1]), float(e[-1]))
    else:
        cp = (k0 if model != 'none' else 0.0, float('nan') if model != 'none' else 0.0)
    used = {pi: sum(1 for a in rows if a[0] == pi) for pi in range(len(phases))}
    return dict(cells=cells, held=held, corr=cp, corr_fitted=fit_corr, rms=float(np.sqrt(np.mean(raw ** 2))), n=n, used=used,
                resid={(pi, hkl, j): float(r) for (pi, hkl, j), r in zip(rows, raw)})


def line_test(x, y, sig, P, tpos, width=0.12, free=0.0, win=0.55, exclude_near=0.03):
    """Fit a doublet at a fixed (or nearly fixed, ±free) position beside the known peaks of the window.
    Returns dict(pos, h, e) - height and its esd in counts above the step-2 background."""
    m = (x > tpos - win) & (x < tpos + win)
    xx, yy, ww = x[m], sig[m], 1 / np.sqrt(np.maximum(y[m], 1))
    if len(xx) < 8:
        return dict(pos=tpos, h=float('nan'), e=float('nan'))
    # neighbours stay in the model; a sharp peak right at the test position is left out (it is what is tested),
    # a broad band there stays in, so the narrow test line cannot absorb it
    kn = P[(P.t > tpos - win - 0.3) & (P.t < tpos + win + 0.3) & (((P.t - tpos).abs() > exclude_near) | (P.w > 0.25))]
    kt, kh, kw, ke = kn.t.values, kn.h.values, kn.w.values, kn.eta.values
    nk = len(kt)

    def mk(p):
        mm = p[0] + p[1] * (xx - tpos)
        if nk:
            q = p[2:2 + 3 * nk].reshape(nk, 3)
            mm = mm + dbl_sum(xx, kt + q[:, 0], q[:, 1], q[:, 2], ke)
        i = 2 + 3 * nk
        return mm + dbl(xx, tpos + (p[i + 1] if free else 0), p[i], width)

    p0, lb, ub = [0.0, 0.0], [-1e5, -1e5], [1e5, 1e5]
    for h_, w_ in zip(kh, kw):
        p0 += [0, h_, w_]; lb += [-.04, 0, .02]; ub += [.04, max(h_ * 3, 50), 2.5]
    p0 += [10]; lb += [-1e4]; ub += [1e5]
    if free:
        p0 += [0]; lb += [-free]; ub += [free]
    p0 = np.clip(p0, lb, ub)
    # capped: a broad hump in the window can leave a flat direction that never meets the tolerance,
    # while the tested height has long settled (checked: 300 vs 1500 evaluations change it by < 0.02 esd)
    R = least_squares(lambda p: (mk(p) - yy) * ww, p0, bounds=(lb, ub), max_nfev=300)
    cov = np.linalg.pinv(R.jac.T @ R.jac) * np.sum(R.fun ** 2) / max(len(xx) - len(R.x), 1)
    i = 2 + 3 * nk
    return dict(pos=float(tpos + (R.x[i + 1] if free else 0)), h=float(R.x[i]), e=float(np.sqrt(abs(cov[i, i]))))


@dataclass
class IndexResult:
    stage1: Stage1
    phases: List[dict]
    correction: str
    radius: float
    corr: tuple
    corr_fitted: bool
    height: Optional[tuple]
    cells: Dict[str, dict]
    cells_all: Dict[str, dict]
    held: Dict[str, dict]
    used: Dict[str, int]
    matched: Dict[str, tuple]
    rms: float
    n_lines: int
    shared: List[float]
    table: pd.DataFrame
    kbeta: pd.DataFrame
    wla: dict
    params: pd.DataFrame
    report: str
    labels: List[tuple]
    ticks: List[dict]
    unindexed: List[float]
    warnings: List[str] = field(default_factory=list)

    def shift(self, t):
        return float(corr(t, self.correction, self.corr[0]))

    def cell_text(self, name, with_shared=False):
        c = (self.cells_all if with_shared else self.cells).get(name, {})
        ph = next(p for p in self.phases if p['name'] == name)
        parts = []
        for ax in AXES[ph['lattice']]:
            if ax in c:
                parts.append(f'{SYMBOL[ax]} = {fsu(*c[ax])} {unit(ax)}')
            else:
                parts.append(f"{SYMBOL[ax]} = {ph['cell'][ax]:g} {unit(ax)} (held)")
        return ', '.join(parts)

    def test_positions(self, positions, width=0.12):
        """Fixed-position line tests at observed-scale 2θ values. Returns a DataFrame."""
        S, P = self.stage1, self.stage1.peaks
        out = []
        for t0 in positions:
            r = line_test(S.x, S.yobs, S.sig, P, float(t0), width=width)
            near = P[(P.t - t0).abs() < 0.08]
            pct, epct = 100 * r['h'] / P.h.max(), 100 * r['e'] / P.h.max()
            verdict = 'present' if r['h'] > 3 * r['e'] else ('absent' if r['h'] < 2 * r['e'] else 'weak, < 3σ')
            out.append({'2θ': round(float(t0), 3), 'height (counts)': round(r['h'], 1), 'esd': round(r['e'], 1),
                        '% of max': round(pct, 3), 'esd (%)': round(epct, 3), 'verdict': verdict,
                        'observed peak within 0.08°': ', '.join(f'{t:.3f}' for t in near.t)})
        return pd.DataFrame(out)

    def predicted(self, phase_dict):
        """Observed-scale positions of a phase's listed hkl with this result's correction."""
        rows = []
        S = self.stage1
        for hkl in hkl_list(phase_dict):
            t = observed_position(d_hkl(phase_dict, hkl), self.correction, self.corr[0])
            if np.isfinite(t) and S.x.min() < t < S.x.max():
                rows.append((t, hkl_label(hkl)))
        return sorted(rows)


def index_pattern(stage1, phases, correction='displacement', radius=240.0):
    """Stage 2. phases: validated phase dicts (disabled ones are skipped). Returns IndexResult."""
    S, P = stage1, stage1.peaks
    x, y, sig = S.x, S.yobs, S.sig
    phases = [dict(p, cell=dict(p['cell'])) for p in phases if p.get('enabled', True)]
    warnings = []
    k = 0.0
    A, shared, F = [], set(), None
    if phases:
        for tol in (0.15, 0.08, 0.06):
            A = assign(P, phases, correction, k, tol)
            cnt = pd.Series([j for *_, j in A], dtype=int).value_counts()
            shared = set(int(i) for i in cnt[cnt > 1].index)
            F = fit_cells(P, phases, A, correction, exclude=shared, k0=k)
            for pi, c in F['cells'].items():
                for ax, (v, u) in c.items():
                    phases[pi]['cell'][ax] = v
            k = F['corr'][0]
        Fall = fit_cells(P, phases, A, correction, k0=k)
    else:
        F = Fall = dict(cells={}, held={}, corr=(0.0, float('nan')), corr_fitted=False, rms=float('nan'), n=0, used={}, resid={})
        warnings.append('No phases are switched on; only peaks, background and Cu Kβ checks were computed.')
    if correction != 'none' and not F['corr_fitted']:
        warnings.append('Too few assigned lines to refine the angle correction; it was held at 0.')
    k, ek = F['corr']
    height = None
    if correction == 'displacement' and F['corr_fitted']:
        height = (-math.radians(k) * radius / 2, math.radians(ek) * radius / 2)
    names = [p['name'] for p in phases]
    cells = {names[pi]: c for pi, c in F['cells'].items()}
    cells_all = {names[pi]: c for pi, c in Fall['cells'].items()}
    held = {}
    for (pi, ax), why in F['held'].items():
        held.setdefault(names[pi], {})[ax] = why
    used = {names[pi]: n for pi, n in F['used'].items()}
    shift = lambda t: float(corr(t, correction, k))

    # how many of each phase's predicted lines (in range, at sharp-peak positions) were matched
    matched = {}
    for pi, ph in enumerate(phases):
        pred = [observed_position(d_hkl(ph, h), correction, k) for h in hkl_list(ph)]
        pred = [t for t in pred if np.isfinite(t) and x.min() + 0.6 < t < x.max()]
        nm = sum(1 for a in A if a[0] == pi)
        matched[ph['name']] = (nm, len(pred))
        if pred and nm < max(2, 0.25 * len(pred)):
            warnings.append(f"{ph['name']}: only {nm} of {len(pred)} listed reflections in range match a peak. "
                            'If the phase is not there, switch it off: lines it claims are left out of the other cells as shared.')

    lab = {j: [] for j in P.index}
    for pi, hkl, j in A:
        lab[j].append((phases[pi]['name'], hkl_label(hkl)))
    notes = {j: [] for j in P.index}
    for j in shared:
        notes[j].append('shared line (' + ' / '.join(f'{n} {h}' for n, h in lab[j]) + '); left out of the cell fits')
    for j, p in P.iterrows():
        if p.atbound:
            notes[j].append('width at fit bound (broad hump)')
    for (pi, hkl, j), r in F['resid'].items():
        if abs(r) > 0.04:
            notes[j].append(f'{phases[pi]["name"]} {hkl_label(hkl)} misfit {r:+.3f}°')

    # Cu Kβ check
    good = P[~P.atbound]
    floor = float(good.I.min()) if len(good) else 1.0
    thr = min(floor / 0.015, 20)
    kb_rows, kb_ticks, extra = [], [], []
    short = lambda j: '/'.join(next(p['prefix'] for p in phases if p['name'] == n) + h for n, h in lab[j]) if lab[j] else f'{P.t[j]:.2f}°'
    for j, p in P[(P.I >= thr) & (P.profile == 'sharp')].iterrows():
        tb = observed_position(dsp(p.t - shift(p.t)), correction, k, LB)
        if not (x.min() + 0.3 < tb < x.max() - 0.3):
            continue
        kb_ticks.append((tb, 'β' + short(j), float(dsp(tb - shift(tb), LB))))
        r = line_test(x, y, sig, P, tb, width=p.w, free=0.04, exclude_near=0.06)
        ratio = r['h'] / p.h
        ok = r['h'] > 3 * r['e'] and 0.003 < ratio < 0.03
        on = P[(P.t - tb).abs() < 0.06]
        parent = ('; '.join(f'{n} {h}' for n, h in lab[j])) or f'{p.t:.2f}° peak'
        if len(on):
            jj = int(on.index[0])
            if ok and not lab[jj]:
                status = 'confirmed (on a found peak)'
                lab[jj].append(('Kβ of', parent))
                notes[jj].append(f'Kβ/Kα {100 * ratio:.2f} % from a direct fit')
            elif ok:
                status = 'confirmed, overlaps an indexed line'
                notes[jj].append(f'Kβ of {parent} lies under it ({100 * ratio:.2f} %)')
            else:
                status = f'under the {P.t[jj]:.3f}° peak (not separable)'
                notes[jj].append(f'Kβ of {parent} lies under it')
        elif ok:
            status = 'confirmed (new line)'
            extra.append({'2θ obs': round(r['pos'], 3), '2θ corr': round(r['pos'] - shift(r['pos']), 3),
                          'd (Å)': round(float(dsp(r['pos'] - shift(r['pos']))), 4), 'FWHM': round(p.w, 3),
                          'I (%)': round(100 * r['h'] / P.h.max(), 3), 'profile': 'sharp', 'index': f'Kβ of {parent}',
                          'note': f'found by a direct fit at the Kβ position; Kβ/Kα {100 * ratio:.2f} %'})
        else:
            status = 'not detected'
        kb_rows.append({'parent 2θ': round(p.t, 3), 'parent': parent, 'parent I (%)': round(p.I, 2), 'Kβ 2θ': round(tb, 3),
                        'fitted height': round(r['h'], 1), 'esd': round(r['e'], 1), 'Kβ/Kα (%)': round(100 * ratio, 2), 'status': status})
    kbeta = pd.DataFrame(kb_rows, columns=['parent 2θ', 'parent', 'parent I (%)', 'Kβ 2θ', 'fitted height', 'esd', 'Kβ/Kα (%)', 'status'])
    jmax = int(P.h.idxmax())
    tw = observed_position(dsp(P.t[jmax] - shift(P.t[jmax])), correction, k, LW)
    rw = line_test(x, y, sig, P, tw, free=0.03) if x.min() < tw < x.max() else dict(h=float('nan'), e=float('nan'))
    wla = dict(pos=tw, pct=100 * rw['h'] / P.h.max(), epct=100 * rw['e'] / P.h.max())

    # peak table
    T = []
    for j, p in P.iterrows():
        tc = p.t - shift(p.t)
        if lab[j]:
            ix = '; '.join(f'{n} {h}' for n, h in lab[j])
        else:
            ix = 'hump' if (p.profile == 'broad' or p.atbound) else '—'
        T.append({'2θ obs': round(p.t, 3), '2θ corr': round(tc, 3), 'd (Å)': round(float(dsp(tc)), 4), 'FWHM': round(p.w, 3),
                  'I (%)': round(p.I, 3), 'profile': p.profile, 'index': ix, 'note': '; '.join(notes[j])})
    T = pd.DataFrame(T + extra).sort_values('2θ obs').reset_index(drop=True)
    un = [float(t) for t, ix, pr in zip(T['2θ obs'], T['index'], T['profile']) if ix == '—' and pr == 'sharp']

    # labels and tick rows for the plots
    pref = {p['name']: p['prefix'] for p in phases}
    col = {p['name']: p['color'] for p in phases}
    labels = []
    for _, r in T.iterrows():
        ix = r['index']
        if ix == 'hump':
            continue
        if ix == '—':
            labels.append((r['2θ obs'], '*', UN_COLOR, r['I (%)']))
            continue
        parts, c0 = [], None
        for part in ix.split(';'):
            part = part.strip()
            if part.startswith('Kβ of'):
                rest = part[6:].split(';')[0].strip()
                if ' ' in rest and rest.rsplit(' ', 1)[0] in pref:
                    nm, h = rest.rsplit(' ', 1)
                    parts.append('β' + pref[nm] + h)
                else:
                    parts.append('β')
                c0 = c0 or KB_COLOR
            else:
                nm, h = part.rsplit(' ', 1)
                parts.append(pref.get(nm, nm[:1]) + h)
                c0 = c0 or col.get(nm, '#1f3a93')
        labels.append((r['2θ obs'], '/'.join(parts), c0, r['I (%)']))
    ticks = []
    for ph in phases:
        rows = _predicted_rows(ph, correction, k, x)
        ticks.append(dict(name=ph['name'], label=sub_digits(ph['name']) + (f" · {ph['space_group']}" if ph['space_group'] else ''),
                          color=ph['color'], val=None, rows=rows))
    ticks.append(dict(name='Cu Kβ', label='Cu Kβ of strong reflections', color=KB_COLOR, val='', rows=sorted(kb_ticks)))
    if un:
        ticks.append(dict(name='Unindexed', label='Unindexed', color=UN_COLOR, val=f'{len(un)} sharp lines',
                          rows=[(t, '*', round(float(dsp(t - shift(t))), 4)) for t in un]))

    res = IndexResult(stage1=S, phases=phases, correction=correction, radius=radius, corr=(k, ek), corr_fitted=F['corr_fitted'], height=height,
                      cells=cells, cells_all=cells_all, held=held, used=used, matched=matched, rms=F['rms'], n_lines=F['n'],
                      shared=sorted(float(P.t[j]) for j in shared), table=T, kbeta=kbeta, wla=wla, params=pd.DataFrame(),
                      report='', labels=labels, ticks=ticks, unindexed=un, warnings=warnings)
    for t in res.ticks[:len(phases)]:
        t['val'] = res.cell_text(t['name'])
    res.params = _params_table(res)
    res.report = _report(res)
    return res


def _predicted_rows(ph, model, k, x):
    """(observed-scale 2θ, hkl label, d) for each listed reflection inside the scan."""
    out = []
    for hkl in hkl_list(ph):
        d = d_hkl(ph, hkl)
        t = observed_position(d, model, k)
        if np.isfinite(t) and x.min() < t < x.max():
            out.append((round(t, 3), hkl_label(hkl), round(d, 4)))
    return sorted(out)


# --------------------------------------------------------------------------------------------- tables
def _row(phase, sg, parameter, value, esd=None, unit='', status='refined', basis='', note=''):
    return dict(part='A', phase=phase, space_group=sg, parameter=parameter, value=value, esd=esd,
                value_su=fsu(value, esd) if status in ('refined', 'derived') and value is not None else ('' if value is None else f'{value:g}'),
                unit=unit, status=status, basis=basis, note=note)


def _params_table(r):
    S = r.stage1
    R = []
    corr_txt = {'displacement': 'sample-displacement correction', 'zero': 'zero-shift correction', 'none': 'no angle correction'}[r.correction]
    for ph in r.phases:
        nm, sg, lat = ph['name'], ph['space_group'], ph['lattice']
        c, ca, h = r.cells.get(nm, {}), r.cells_all.get(nm, {}), r.held.get(nm, {})
        nm_m, nm_p = r.matched.get(nm, (0, 0))
        basis = f"n = {r.used.get(nm, 0)} lines of this phase ({nm_m} of {nm_p} listed reflections matched), joint fit of {r.n_lines} lines, rms {r.rms:.4f}°, {corr_txt}"
        for ax in AXES[lat]:
            if ax in c:
                R.append(_row(nm, sg, ax, c[ax][0], c[ax][1], unit(ax), 'refined', basis))
            else:
                R.append(_row(nm, sg, ax, ph['cell'][ax], None, unit(ax), 'not determined', basis, f'held at the start value: {h.get(ax, "not refined")}'))
        for ax in AXES[lat]:
            if ax in ca:
                R.append(_row(nm + ' (incl. shared lines)', sg, ax, ca[ax][0], ca[ax][1], unit(ax), 'refined', 'all assigned lines, shared ones included'))
        if all(ax in c for ax in AXES[lat]):
            cell = {ax: c[ax][0] for ax in c}
            V = cell_volume(lat, cell)
            mult = {'cubic': [3], 'tetragonal': [2, 1], 'hexagonal': [2, 1], 'orthorhombic': [1, 1, 1]}.get(lat, [1, 1, 1])
            eV = V * math.sqrt(sum((m * c[ax][1] / c[ax][0]) ** 2 for m, ax in zip(mult, AXES[lat][:3]) if np.isfinite(c[ax][1])))
            R.append(_row(nm, sg, 'cell volume', V, eV, 'Å³', 'derived', 'from the refined axes', 'esd neglects correlations between axes'))
    if r.correction != 'none':
        k, ek = r.corr
        if r.corr_fitted:
            R.append(_row('instrument', '', 'K (2θobs = 2θcalc + K·cosθ)' if r.correction == 'displacement' else 'zero shift z', k, ek, '°', 'refined', corr_txt))
        else:
            R.append(_row('instrument', '', 'angle correction', 0.0, None, '°', 'fixed', '', 'too few lines to refine it'))
    if r.height:
        R.append(_row('instrument', '', 'height error s ≈ −K·R/2', r.height[0], r.height[1], 'mm', 'derived', f'R = {r.radius:g} mm'))
    conf = r.kbeta[r.kbeta.status.str.startswith('confirmed')] if len(r.kbeta) else r.kbeta
    if len(conf):
        R.append(_row('Cu Kβ', '', 'Kβ/Kα1 height ratio', None, None, '%', 'statistic',
                      ', '.join(f"{v:.2f}" for v in conf['Kβ/Kα (%)']) + ' % on ' + ', '.join(f'{t:.2f}°' for t in conf['parent 2θ'])))
    else:
        R.append(_row('Cu Kβ', '', 'Kβ/Kα1 height ratio', None, None, '', 'not determined', '', 'no Kβ copy passed the 3σ test'))
    if r.unindexed:
        R.append(_row('unassigned lines', '', f'{len(r.unindexed)} sharp lines', None, None, '', 'not determined', ', '.join(f'{t:.2f}°' for t in r.unindexed)))
    if S.rwp_prior is not None:
        R.append(_row('fit', '', 'Rwp prior refinement', round(S.rwp_prior, 3), None, '%', 'statistic'))
    R.append(_row('fit', '', 'Rwp (background + peaks)', round(S.rwp_step2, 3), None, '%', 'statistic', S.calc_source))
    R.append(_row('fit', '', 'background', None, None, '', 'setting', '', getattr(S, 'bkg_label', 'step-2')))
    R.append(_row('fit', '', 'peaks found', len(S.peaks), None, '', 'statistic'))
    return pd.DataFrame(R, columns=['part', 'phase', 'space_group', 'parameter', 'value', 'esd', 'value_su', 'unit', 'status', 'basis', 'note'])


def _report(r):
    S, L = r.stage1, []
    L.append(f'XRD peak indexing - {S.name}')
    L.append(f'{len(S.x)} points, {S.x.min():.2f}–{S.x.max():.2f}° 2θ, Cu Kα1 {L1} Å')
    if S.rwp_prior is not None:
        L.append(f'prior refinement Rwp {S.rwp_prior:.2f}%' + (f' (<10°: {S.rwp_prior_lo:.2f}%)' if S.rwp_prior_lo else ''))
    L.append(f'background: {getattr(S, "bkg_label", "step-2")}; Rwp {S.rwp_step2:.2f}% ({S.calc_source})')
    L.append(f'{len(S.peaks)} peaks kept ({int((S.peaks.profile == "sharp").sum())} sharp)')
    L.append('')
    if r.phases:
        L.append(f'Cells (joint fit of {r.n_lines} lines that belong to one phase, rms {r.rms:.4f}°):')
        for ph in r.phases:
            nm = ph['name']
            L.append(f'  {nm:12s} {r.cell_text(nm)}   [n = {r.used.get(nm, 0)}; {r.matched[nm][0]} of {r.matched[nm][1]} listed reflections matched]')
            if nm in r.cells_all:
                L.append(f'  {"":12s} with shared lines: {r.cell_text(nm, True)}')
            for ax, why in r.held.get(nm, {}).items():
                L.append(f'  {"":12s} {ax} not determined: {why}')
    k, ek = r.corr
    if r.correction == 'displacement' and r.corr_fitted:
        L.append(f'  displacement K = {fsu(k, ek)}° (2θobs = 2θcalc + K·cosθ); height error ≈ {fsu(*r.height)} mm (R = {r.radius:g} mm)')
    elif r.correction == 'zero' and r.corr_fitted:
        L.append(f'  zero shift z = {fsu(k, ek)}°')
    if r.shared:
        L.append('  shared lines (left out): ' + ', '.join(f'{t:.3f}°' for t in r.shared))
    L.append('')
    L.append('Cu Kβ check:')
    for _, q in r.kbeta.iterrows():
        L.append(f"  {q['parent']:24s} {q['parent 2θ']:7.3f}° → {q['Kβ 2θ']:7.3f}°: {q['fitted height']:8.1f} ± {q['esd']:6.1f} ({q['Kβ/Kα (%)']:5.2f} %)  {q.status}")
    L.append(f"  W Lα of the strongest line at {r.wla['pos']:.3f}°: {r.wla['pct']:.3f} ± {r.wla['epct']:.3f} % of max")
    L.append('')
    L.append(f'Unindexed sharp lines ({len(r.unindexed)}): ' + ', '.join(f'{t:.3f}' for t in r.unindexed))
    for w in r.warnings:
        L.append('Note: ' + w)
    return '\n'.join(L) + '\n'
