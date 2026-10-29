"""Candidate phases from the Materials Project, screened against an indexing result.

    from xrdcore import mpsearch
    cands = mpsearch.fetch_candidates(['Sn', 'V', 'Se', 'O'], api_key)        # experimentally made entries only
    table = mpsearch.screen(index_result, cands)                               # one row per entry, with a verdict
    phase = mpsearch.candidate_phase(table.iloc[0], existing_names=[...])      # ready for the phase list

Needs the optional packages in requirements-mp.txt (mp-api, which brings pymatgen) and a Materials Project API key.

fetch_candidates: every chemical system made of the chosen elements (each element alone, every pair, ...), entries with
    theoretical = False (an ICSD or other experimental structure exists). Solid phases of the gaseous elements
    (O, N, H, F, Cl, noble gases) are skipped. Structures are the conventional standard cell at MP's own symmetry
    tolerance (0.1 Å), so the crystal system matches MP's record.

screen, for each entry and for random-powder and (00l)-oriented intensities:
  1. reflections with |F|² × multiplicity from pymatgen (Lorentz-polarisation removed, re-applied at each cell)
  2. grid scan of the cell lengths, each axis 0.93-1.02 of the DFT value (DFT cells are too large, by up to ~6 % across
     van der Waals gaps), maximising the share of strong-line (≥ 2 %) intensity within 0.10° of an observed sharp peak
  3. robust least-squares refinement of the free cell parameters (angles too for monoclinic / triclinic)
  4. coverage = share of predicted strong intensity observed (sharp peaks within 0.08°, broad features within half
     their width); phase amount k from the unindexed lines it explains (median Iobs/Icalc), or from its weakest
     observed line; a predicted line ≥ 2 × the weakest kept peak with nothing observed there is tested with a
     fixed-position line fit and counts as missing only if it is > 3σ and > 5× below the prediction
  5. chance tests (25 random sets): coverage against random peak lists of the same size; unindexed lines explained
     against lists where only the unindexed peaks are moved to random free positions
Verdicts: explains unindexed lines | possible, check | same as <phase> in your list | consistent, adds nothing |
not supported, each with its reason.

Validation (see tests): with VSe2 + SnSe2 listed, corundum V2O3 is the only strong candidate for the 13 unindexed lines
of SnSeVSe22_27Aug26 (6 explained, cell −0.4 %); with all three listed, all three come back as "same as"; on a synthetic
Si pattern, Fd-3m Si is found and the other 11 experimental Si polymorphs are rejected.
"""
import itertools
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .phases import PALETTE

L1 = 1.540593
TOL = 0.08
GRID = np.round(np.arange(0.93, 1.0201, 0.004), 4)
GASES = {'H', 'N', 'O', 'F', 'Cl', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}
MP_URL = 'https://next-gen.materialsproject.org/materials/'


class MPSearchError(RuntimeError):
    """The search could not run; the message says why and what to do."""


def available():
    """(True, '') when mp-api and pymatgen import, else (False, how to install)."""
    try:
        import mp_api.client  # noqa: F401
        import pymatgen.core  # noqa: F401
        return True, ''
    except Exception as e:  # pragma: no cover - depends on the installation
        return False, f'The Materials Project search needs the optional packages: pip install -r requirements-mp.txt ({e.__class__.__name__}: {e})'


def parse_elements(text):
    """'Sn, V Se;O' -> (['O', 'Se', 'Sn', 'V'], errors)."""
    from pymatgen.core import Element
    toks = [t for t in re.split(r'[\s,;/+-]+', text or '') if t]
    out, bad = [], []
    for t in toks:
        sym = t[:1].upper() + t[1:].lower()
        try:
            Element(sym)
            if sym not in out:
                out.append(sym)
        except Exception:
            bad.append(t)
    errors = []
    if bad:
        errors.append('Not element symbols: ' + ', '.join(bad) + '.')
    if not out and not bad:
        errors.append('Enter at least one element, e.g. Sn V Se O.')
    if len(out) > 7:
        errors.append(f'{len(out)} elements give {2 ** len(out) - 1} chemical systems; use at most 7.')
    return sorted(out), errors


def elements_of(names):
    """Elements in phase names that are formulas (VSe2 -> V, Se)."""
    from pymatgen.core import Composition
    found = []
    for n in names:
        try:
            for el in Composition(re.sub(r'\s*\(.*\)', '', n)).elements:
                if el.symbol not in found:
                    found.append(el.symbol)
        except Exception:
            continue
    return found


def chemical_systems(elements):
    return ['-'.join(sorted(c)) for r in range(1, len(elements) + 1) for c in itertools.combinations(sorted(elements), r)]


def fetch_candidates(elements, api_key, skip_gases=True):
    """Experimentally made Materials Project entries in every chemical system of the elements.
    Returns a list of dicts: mpid, formula, chemsys, sg, system, e_hull, icsd, structure (conventional)."""
    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    if not api_key:
        raise MPSearchError('Enter your Materials Project API key (next-gen.materialsproject.org → your profile → API key).')
    systems = chemical_systems(elements)
    fields = ['material_id', 'formula_pretty', 'chemsys', 'symmetry', 'energy_above_hull', 'theoretical', 'database_IDs', 'structure']
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with MPRester(api_key, mute_progress_bars=True) as mpr:
                docs = mpr.materials.summary.search(chemsys=systems, theoretical=False, fields=fields)
    except Exception as e:
        msg = str(e)
        if 'API key' in msg or '401' in msg or '403' in msg or 'nauthorized' in msg:
            raise MPSearchError('The Materials Project rejected the API key. Copy it again from your profile page.') from None
        raise MPSearchError(f'The Materials Project could not be reached ({e.__class__.__name__}). Check the network and try again.') from None
    out = []
    for d in docs:
        if skip_gases and d.chemsys in GASES:
            continue
        try:
            sga = SpacegroupAnalyzer(d.structure, symprec=0.1)
            conv = sga.get_conventional_standard_structure()
            system = sga.get_crystal_system()
        except Exception:
            continue
        icsd = (d.database_IDs or {}).get('icsd', [])
        out.append(dict(mpid=str(d.material_id), formula=d.formula_pretty, chemsys=d.chemsys, sg=d.symmetry.symbol, system=system,
                        e_hull=float(d.energy_above_hull or 0.0), icsd=';'.join(map(str, icsd[:3])), structure=conv))
    return sorted(out, key=lambda c: (c['chemsys'], c['formula'], c['e_hull']))


# ------------------------------------------------------------------------------------------- pattern of a candidate
def _lp(t):
    th = np.radians(t / 2)
    return (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))


def _gstar(p):
    a, b, c, al, be, ga = p
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return np.linalg.inv(np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]]))


def _two_theta(p, H):
    q = np.einsum('ni,ij,nj->n', H, _gstar(p), H)
    s = L1 * np.sqrt(np.maximum(q, 0)) / 2
    return np.where(s < 1, 2 * np.degrees(np.arcsin(np.clip(s, 0, 1))), np.nan)


_CALC = None
MODES = ('powder', '00l', 'hk0', '00l+hk0')
MODE_LABEL = {'powder': 'random powder', '00l': '00l only (oriented)', 'hk0': 'hk0 only (in-plane bands)', '00l+hk0': '00l + hk0'}
SERIES_OF = {'powder': 'all', '00l': '00l', 'hk0': 'hk0', '00l+hk0': '00l+hk0'}
# strong lines a mode needs, and unindexed lines it must explain: an in-plane (hk0) family has only a few lines in range,
# while a 00l series lands on other 00l series easily, so it must explain three
NEED = {'powder': 2, '00l': 3, 'hk0': 2, '00l+hk0': 3}


def _reflections(structure, mode='powder'):
    global _CALC
    if _CALC is None:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        _CALC = XRDCalculator(wavelength='CuKa1')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = _CALC.get_pattern(structure, two_theta_range=(1, 150))
    H = np.array([(h[0]['hkl'][0], h[0]['hkl'][1], h[0]['hkl'][-1]) for h in p.hkls], float)
    F = np.array(p.y) / _lp(np.array(p.x))
    if mode != 'powder':
        is00l = (H[:, 0] == 0) & (H[:, 1] == 0)
        ishk0 = H[:, 2] == 0
        k = is00l if mode == '00l' else ishk0 if mode == 'hk0' else (is00l | ishk0)
        H, F = H[k], F[k]
    return H, F


def _lines(p, H, F, lo, hi):
    t = _two_theta(p, H)
    ok = np.isfinite(t)
    I = np.zeros(len(H))
    I[ok] = F[ok] * _lp(t[ok])
    I = 100 * I / I.max() if I.max() > 0 else I
    m = ok & (t > lo) & (t < hi)
    return t[m], I[m], np.where(m)[0]


def _axes(system, mode):
    """Length axes (0 a, 1 b, 2 c) scanned and refined independently, ties (index -> source), and refined angles."""
    s = system.lower()
    if s == 'cubic':                                         # one axis in every mode
        return [0], {1: 0, 2: 0}, []
    same_ab = s in ('tetragonal', 'hexagonal', 'trigonal')
    if mode == '00l':
        return [2], {}, []
    if mode == 'hk0':
        return ([0], {1: 0}, []) if same_ab else ([0, 1], {}, [])
    if mode == '00l+hk0':
        return ([0, 2], {1: 0}, []) if same_ab else ([0, 1, 2], {}, [])
    if s == 'cubic':
        return [0], {1: 0, 2: 0}, []
    if same_ab:
        return [0, 2], {1: 0}, []
    if s == 'orthorhombic':
        return [0, 1, 2], {}, []
    if s == 'monoclinic':
        return [0, 1, 2], {}, ['mono']
    return [0, 1, 2], {}, [3, 4, 5]


def _nearest(t, ref_sorted):
    i = np.clip(np.searchsorted(ref_sorted, t), 1, len(ref_sorted) - 1)
    return np.minimum(np.abs(t - ref_sorted[i - 1]), np.abs(t - ref_sorted[i]))


def _grid_scan(p0, system, mode, H, F, ref_sorted, lo, hi, tol=0.10):
    _, I0, idx = _lines(p0, H, F, lo - 5, hi + 5)
    k = I0 >= 2
    if not k.any() or len(ref_sorted) < 2:
        return None, 0.0
    Hs, w = H[idx[k]], I0[k]
    M = np.einsum('ni,nj,ij->nij', Hs, Hs, _gstar(p0))
    lens, ties, _ = _axes(system, mode)
    S = np.ones((len(GRID) ** len(lens), 3))
    for col, vals in zip(lens, zip(*itertools.product(GRID, repeat=len(lens)))):
        S[:, col] = vals
    for i, src in ties.items():
        S[:, i] = S[:, src]
    best = (-1.0, None)
    for chunk in np.array_split(S, max(1, len(S) // 4000)):
        q = np.einsum('nij,mij->mn', M, 1 / (chunk[:, :, None] * chunk[:, None, :]))
        tt = 2 * np.degrees(np.arcsin(np.clip(L1 * np.sqrt(np.maximum(q, 1e-12)) / 2, 0, 1)))
        inr = (tt > lo) & (tt < hi)
        d = _nearest(tt.ravel(), ref_sorted).reshape(tt.shape)
        v = ((d <= tol) & inr) @ w / np.maximum(inr @ w, 1e-9)
        j = int(np.argmax(v))
        if v[j] > best[0]:
            best = (float(v[j]), chunk[j])
    p = np.array(p0, float)
    p[:3] *= best[1]
    return p, best[0]


def _refine(p, p0, system, mode, H, F, ref, reach, lo, hi):
    lens, ties, angs = _axes(system, mode)
    if angs == ['mono']:
        angs = [i for i in (3, 4, 5) if abs(p0[i] - 90) > 0.01][:1]
    free = lens + angs
    for tol in (0.10, 0.08):
        t, I, idx = _lines(p, H, F, lo, hi)
        st = I >= 1.0
        tt, hh = t[st], H[idx[st]]
        if not len(tt):
            return p
        d = np.abs(tt[:, None] - ref[None, :])
        j = d.argmin(1)
        ok = d.min(1) <= np.maximum(tol, reach[j])
        if ok.sum() < len(free) + 2:
            return p
        Hs, tgt = hh[ok], ref[j[ok]]
        wt = np.sqrt(I[st][ok] / 100) * np.minimum(1.0, 0.1 / np.maximum(reach[j[ok]], 0.1))

        def full(x, p=p):
            q = np.array(p, float)
            for v, i in zip(x, free):
                q[i] = v
            for i, src in ties.items():
                q[i] = q[src]
            return q
        lb = np.array([p0[i] * 0.92 if i < 3 else p0[i] - 4 for i in free])
        ub = np.array([p0[i] * 1.03 if i < 3 else p0[i] + 4 for i in free])
        x0 = np.clip([p[i] for i in free], lb + 1e-9, ub - 1e-9)
        p = full(least_squares(lambda x: (_two_theta(full(x), Hs) - tgt) * wt, x0, bounds=(lb, ub), loss='soft_l1', f_scale=0.02).x)
    return p


# ------------------------------------------------------------------------------------------- observed side
class _Obs:
    def __init__(self, res):
        T = res.table
        self.res = res
        t_all = T['2θ corr'].values.astype(float)
        w_all = T['FWHM'].values.astype(float)
        I_all = T['I (%)'].values.astype(float)
        lab = T['index'].astype(str).values
        real = w_all < 2.49                                   # humps stuck at the width bound are background artefacts
        self.t_cov, self.w_cov = t_all[real], w_all[real]
        sharp = (T.profile == 'sharp').values
        band = real & ~sharp
        self.sharp = dict(t=t_all[sharp], I=I_all[sharp], reach=np.full(sharp.sum(), TOL), un=lab[sharp] == '—')
        # sharp peaks plus broad bands (for modes with hk0 lines); an unassigned band ('hump') counts as unindexed
        # band modes compare integrated intensities (height × width, in sharp-line units): a 0.8° band has a far lower
        # height than a sharp line of the same area
        w_ref = float(np.median(w_all[sharp])) if sharp.any() else 0.12
        A_all = I_all * w_all / w_ref
        self.bands = dict(t=np.concatenate([t_all[sharp], t_all[band]]), I=np.concatenate([A_all[sharp], A_all[band]]),
                          reach=np.concatenate([np.full(sharp.sum(), TOL), np.minimum(0.5 * w_all[band], 0.5)]),
                          un=np.concatenate([lab[sharp] == '—', np.isin(lab[band], ['hump', '—'])]))
        self.floor = float(I_all[sharp].min()) if sharp.any() else 0.05
        self.lo, self.hi = float(res.stage1.x.min()) + 0.5, float(res.stage1.x.max())
        self.Imax = float(res.stage1.peaks.h.max())

    def targets(self, mode):
        return self.bands if mode in ('hk0', '00l+hk0') else self.sharp

    def covered(self, t, band=False):
        """Is a predicted line at t accounted for by an observed feature? A predicted band (band=True) may also hide
        within 0.4° of any feature, e.g. under a sharp line."""
        if not len(self.t_cov):
            return np.zeros(len(t), bool)
        d = np.abs(t[:, None] - self.t_cov[None, :])
        reach = np.maximum(1.5 * TOL, self.w_cov / 2)
        if band:
            reach = np.maximum(reach, 0.4)
        return (d <= reach[None, :]).any(1)

    def line_height(self, t_corr):
        from .indexing import line_test
        S = self.res.stage1
        r = line_test(S.x, S.yobs, S.sig, S.peaks, t_corr + self.res.shift(t_corr))
        return 100 * r['h'] / self.Imax, 100 * r['e'] / self.Imax


def _evaluate(structure, system, mode, O, test_lines=True, tg=None):
    tg = tg or O.targets(mode)
    t, tI, reach, un = tg['t'], tg['I'], tg['reach'], tg['un']
    p0 = np.array(structure.lattice.parameters)
    H, F = _reflections(structure, mode)
    if not len(H) or len(t) < 2:
        return None
    pg, _ = _grid_scan(p0, system, mode, H, F, np.sort(t), O.lo, O.hi)
    if pg is None:
        return None
    p = _refine(pg, p0, system, mode, H, F, t, reach, O.lo, O.hi)
    tl, I, idx = _lines(p, H, F, O.lo, O.hi)
    strong = I >= 2
    if strong.sum() < NEED[mode]:
        return None
    ts, Is = tl[strong], I[strong]
    d = np.abs(ts[:, None] - t[None, :])
    jo = d.argmin(1)
    seen = d.min(1) <= np.maximum(TOL, reach[jo])
    hk0_strong = (H[idx][strong][:, 2] == 0) if mode in ('hk0', '00l+hk0') else np.zeros(len(ts), bool)
    if tg is O.targets(mode):
        cov_any = seen | O.covered(ts) | (hk0_strong & O.covered(ts, band=True))
    else:
        cov_any = seen
    cover = float(Is[cov_any].sum() / Is.sum())
    strongest_seen = bool(cov_any[np.argmax(Is)])
    k_all = float(np.min(tI[jo[seen]] / Is[seen])) if seen.any() else 0.0
    ex = seen & un[jo]
    ratios = tI[jo[ex]] / Is[ex] if ex.any() else np.zeros(0)
    k_un = float(np.median(ratios)) if len(ratios) else 0.0
    spread = float(ratios.max() / ratios.min()) if len(ratios) >= 2 and ratios.min() > 0 else 1.0
    pred = max(k_all, k_un) * I
    is_band = tg is O.bands or mode in ('hk0', '00l+hk0')
    hk0_line = H[idx][:, 2] == 0 if mode in ('hk0', '00l+hk0') else np.zeros(len(tl), bool)
    cand = (pred >= 2 * O.floor) & ~O.covered(tl, band=False) & ~(hk0_line & O.covered(tl, band=True))
    unpeaks = sorted(set(np.round(t[jo[seen]][un[jo[seen]]], 3)))
    in_play = (len(unpeaks) >= 2 or cover >= 0.8) and cand.sum() <= 6
    contra = []
    for i in np.where(cand)[0]:
        if test_lines and in_play and not hk0_line[i]:
            h, e = O.line_height(tl[i])
            if h < pred[i] - 3 * e and h < pred[i] / 5:
                contra.append((round(float(tl[i]), 3), round(float(pred[i]), 3), round(h, 3), round(e, 3)))
        else:
            contra.append((round(float(tl[i]), 3), round(float(pred[i]), 3), None, None))
    return dict(p0=p0, p=p, H=H, F=F, n_strong=int(strong.sum()), n_seen=int(seen.sum()), cover=cover, strongest_seen=strongest_seen,
                spread=spread, contra=contra, unexpl=len(unpeaks), unpeaks=[float(v) for v in unpeaks])


def same_compound(formula, names):
    from pymatgen.core import Composition
    try:
        f = Composition(formula).reduced_formula
    except Exception:
        return None
    for n in names:
        try:
            if Composition(re.sub(r'\s*\(.*\)', '', n)).reduced_formula == f:
                return n
        except Exception:
            continue
    return None


def screen(res, cands, modes=('powder', '00l'), n_random=25, seed=0, progress=None):
    """Screen candidates against an IndexResult in the chosen intensity modes (powder, 00l, hk0, 00l+hk0).
    Returns a DataFrame, one row per entry (its best mode), best verdict first."""
    names = [p['name'] for p in res.phases]
    O = _Obs(res)
    if len(O.sharp['t']) < 3:
        raise MPSearchError('The pattern has fewer than three sharp peaks; there is nothing to screen against.')
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise MPSearchError('Choose at least one of: ' + ', '.join(MODE_LABEL[m] for m in MODES) + '.')
    rng = np.random.default_rng(seed)
    rand_sets = {}
    for key, tg in (('sharp', O.sharp), ('bands', O.bands)):
        un = tg['un']
        sets = []
        if un.any():
            U = tg['t'][un]
            for _ in range(n_random):
                r = None
                for _ in range(1000):
                    r = rng.uniform(max(O.lo, U.min() - 10), min(O.hi, U.max() + 10), size=len(U))
                    if not (~un).any() or np.all(np.abs(r[:, None] - tg['t'][~un][None, :]).min(1) > TOL):
                        break
                t2 = tg['t'].copy()
                t2[un] = r
                sets.append(dict(tg, t=t2))
        rand_sets[key] = sets
    full_sets = [np.sort(rng.uniform(O.lo, O.hi, size=len(O.sharp['t']))) for _ in range(n_random)]
    rows = []
    for n, c in enumerate(cands):
        if progress:
            progress(n, len(cands), c)
        for mode in modes:
            ev = _evaluate(c['structure'], c['system'], mode, O)
            if ev is None:
                continue
            row = dict(mpid=c['mpid'], formula=c['formula'], sg=c['sg'], system=c['system'], e_hull=c.get('e_hull', 0.0),
                       source=c.get('source', 'mp'), file=c.get('file', ''),
                       mode=MODE_LABEL[mode], mode_key=mode, **ev, p_cover=np.nan, p_unexpl=np.nan)
            ok_shape = (not ev['contra'] or (len(ev['contra']) == 1 and ev['contra'][0][2] is not None)) and ev['strongest_seen']
            if ok_shape and ev['cover'] >= 0.8:
                hits = sum(_grid_scan(ev['p0'], c['system'], mode, ev['H'], ev['F'], ref, O.lo, O.hi)[1] >= ev['cover'] - 1e-9 for ref in full_sets)
                row['p_cover'] = hits / n_random
            sets = rand_sets['bands' if mode in ('hk0', '00l+hk0') else 'sharp']
            if ok_shape and ev['unexpl'] >= NEED[mode] and ev['spread'] <= 5 and sets:
                hits = 0
                for tg in sets:
                    e2 = _evaluate(c['structure'], c['system'], mode, O, test_lines=False, tg=tg)
                    hits += e2 is not None and e2['unexpl'] >= ev['unexpl']
                row['p_unexpl'] = hits / n_random
            rows.append(row)
    if progress:
        progress(len(cands), len(cands), None)
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    V = D.apply(lambda r: _verdict(r, names), axis=1, result_type='expand')
    D['verdict'], D['reason'] = V[0], V[1]
    D['cell_change_pct'] = [np.round(100 * (r.p[:3] / r.p0[:3] - 1), 1) for r in D.itertuples()]
    order = {'explains unindexed lines': 0, 'possible, check': 1}
    D['_o'] = [order.get(v, 2 if v.startswith('same as') else 3 if v.startswith('consistent') else 4) for v in D.verdict]
    D['_n'] = D.contra.map(len)
    D['_m'] = D.mode_key.map({'powder': 0, '00l+hk0': 1, '00l': 2, 'hk0': 3})
    D = D.sort_values(['_o', '_n', 'cover', '_m'], ascending=[True, True, False, True]).drop_duplicates('mpid', keep='first')
    return D.drop(columns=['_o', '_n', '_m']).reset_index(drop=True)


def _verdict(r, names):
    same = same_compound(r.formula, names)
    note = ''
    if len(r.contra) == 1 and r.contra[0][2] is not None:
        c0 = r.contra[0]
        note = f'; one predicted line weaker than expected ({c0[0]:.2f}°: {c0[2]:.2f} ± {c0[3]:.2f} % vs {c0[1]:.2f} %; texture?)'
    if len(r.contra) >= 2 or (r.contra and r.contra[0][2] is None):
        c0 = r.contra[0]
        fit = f', fitted {c0[2]:.2f} ± {c0[3]:.2f} %' if c0[2] is not None else ', no peak there'
        return 'not supported', f'{len(r.contra)} predicted line(s) missing, e.g. {c0[0]:.2f}° (expected {c0[1]:.2f} % of the strongest peak{fit})'
    if not r.strongest_seen:
        return 'not supported', 'its strongest line is not observed'
    need = NEED[r.mode_key]
    if r.unexpl >= need and r.p_unexpl <= 0.05 and r.spread <= 5:
        ch = 100 * (r.p[:3] / r.p0[:3] - 1)
        axes = _axes(r.system, r.mode_key)[0]
        worst = max((ch[i] for i in axes), key=abs)
        text = f'{r.unexpl} unindexed lines: ' + ', '.join(f'{v:.2f}°' for v in r.unpeaks) + note
        doubts = []
        if any(ch[i] < -4.5 or ch[i] > 2.5 for i in axes):
            doubts.append(f'cell changed {worst:+.1f} % from the CIF value' if getattr(r, 'source', 'mp') == 'cif' else
                          f'cell changed {worst:+.1f} % from the DFT value (typical DFT error is −4…+2 %)')
        if r.e_hull > 0.1:
            doubts.append(f'entry {r.e_hull:.2f} eV/atom above the hull')
        return ('possible, check', text + '; ' + '; '.join(doubts)) if doubts else ('explains unindexed lines', text)
    if r.mode_key != 'powder' and r.unexpl >= need and r.spread <= 5:
        ch = 100 * (r.p[:3] / r.p0[:3] - 1)
        axes = _axes(r.system, r.mode_key)[0]
        if all(-4.5 <= ch[i] <= 2.5 for i in axes) and r.e_hull <= 0.1:
            kind = 'bands' if r.mode_key in ('hk0', '00l+hk0') else 'lines'
            p = f' (P = {r.p_unexpl:.2f})' if np.isfinite(r.p_unexpl) else ''
            return 'possible, check', (f'{r.unexpl} unindexed {kind}: ' + ', '.join(f'{v:.2f}°' for v in r.unpeaks) +
                                       f'; random {kind} are matched as often{p}, so this is weak evidence on its own' + note)
    if same and r.cover >= 0.8 and r.p_cover <= 0.05:
        return f'same as {same} in your list', f'{100 * r.cover:.0f} % of its predicted intensity observed ({r.n_strong} strong lines)' + note
    if r.cover >= 0.8 and r.p_cover <= 0.05:
        return 'consistent, adds nothing', 'every matched line is already indexed' if r.unexpl == 0 else (
            f"explains only {r.unexpl} unindexed line{'s' if r.unexpl != 1 else ''}")
    if r.unexpl >= 2 and r.spread > 5:
        return 'not supported', f'explains {r.unexpl} unindexed lines, but their intensities disagree with its pattern by a factor {r.spread:.0f}'
    if r.unexpl >= 2:
        p = f' (P = {r.p_unexpl:.2f})' if np.isfinite(r.p_unexpl) else ''
        return 'not supported', f'explains {r.unexpl} unindexed lines, but random lines are explained as often{p}'
    return 'not supported', f'{100 * r.cover:.0f} % of its predicted intensity observed ({r.n_strong} strong lines)' + (
        f', P = {r.p_cover:.2f}' if np.isfinite(r.p_cover) else '')


# ------------------------------------------------------------------------------------------- results
def display_table(D):
    """Readable columns for the app and the CSV export."""
    if not len(D):
        return pd.DataFrame()
    cif = (D['source'] == 'cif') if 'source' in D else pd.Series(False, index=D.index)
    return pd.DataFrame({
        'verdict': D.verdict, 'formula': D.formula, 'space group': D.sg,
        'mp-id': np.where(cif, D.get('file', D.mpid), D.mpid), 'link': np.where(cif, None, MP_URL + D.mpid),
        'E above hull (eV/atom)': D.e_hull.round(3).where(~cif), 'intensities': D['mode'],
        'cell change (%)': [', '.join(f"{'abc'[i]} {c[i]:+.1f}" for i in _axes(sy, m)[0]) for c, m, sy in zip(D.cell_change_pct, D.mode_key, D.system)],
        'predicted intensity observed (%)': (100 * D.cover).round(0), 'strong lines': D.n_strong,
        'unindexed lines explained': D.unexpl, 'chance P': D[['p_cover', 'p_unexpl']].min(axis=1, skipna=True).round(2),
        'reason': D.reason})


def _standard_hkl(h, lat, sg):
    """An equivalent hkl with non-negative h, k (same d), the way reflections are usually labelled: 2,-1,0 -> 110."""
    hh, kk, ll = h
    if lat == 'hexagonal':
        n = hh * hh + hh * kk + kk * kk
        for a in range(0, 13):
            for b in range(0, a + 1):
                if a * a + a * b + b * b == n:
                    if sg.startswith('R'):                       # keep -h+k+l = 3n in the obverse setting
                        for x, y, z in ((a, b, ll), (b, a, ll), (a, b, -ll), (b, a, -ll)):
                            if (-x + y + z) % 3 == 0:
                                return [x, y, z]
                        continue
                    return [a, b, abs(ll)]
        return [hh, kk, ll]
    if lat == 'tetragonal':
        a, b = sorted((abs(hh), abs(kk)), reverse=True)
        return [a, b, abs(ll)]
    if lat == 'cubic':
        return sorted((abs(hh), abs(kk), abs(ll)), reverse=True)
    if lat == 'orthorhombic':
        return [abs(hh), abs(kk), abs(ll)]
    return [hh, kk, ll]


def candidate_phase(row, existing_names=(), color_index=0, min_intensity=2.0, x_range=(5.0, 90.0), series=None, start_cell=False):
    """Phase dict for the phase list: refined cell from the screen, hkl of the lines ≥ min_intensity in the scan.
    series: all | 00l | hk0 | 00l+hk0 (default: the family of the mode the entry was screened in).
    start_cell: keep the entry's own cell (the CIF or DFT cell) instead of the one refined to the peaks."""
    p, sysname, sg = np.array(row.p0 if start_cell else row.p, float), str(row.system).lower(), str(row.sg)
    a, b, c, al, be, ga = p
    if sysname == 'cubic':
        lat, cell = 'cubic', {'a': a}
    elif sysname == 'tetragonal':
        lat, cell = 'tetragonal', {'a': a, 'c': c}
    elif sysname in ('hexagonal', 'trigonal'):
        lat, cell = 'hexagonal', {'a': a, 'c': c}
    elif sysname == 'orthorhombic':
        lat, cell = 'orthorhombic', {'a': a, 'b': b, 'c': c}
    elif sysname == 'monoclinic' and abs(al - 90) < 0.01 and abs(ga - 90) < 0.01:
        lat, cell = 'monoclinic', {'a': a, 'b': b, 'c': c, 'beta': be}
    else:
        lat, cell = 'triclinic', {'a': a, 'b': b, 'c': c, 'alpha': al, 'beta': be, 'gamma': ga}
    t, I, idx = _lines(p, row.H, row.F, x_range[0], x_range[1])
    order = np.argsort(t)
    hkl = []
    for i in order:
        if I[i] >= min_intensity:
            h = _standard_hkl([int(v) for v in row.H[idx[i]]], lat, sg)
            if h not in hkl:
                hkl.append(h)
    name = row.formula
    if name in existing_names:
        name = f'{row.formula} ({sg})'
    k = 2
    while name in existing_names:
        name = f'{row.formula} ({sg}) {k}'
        k += 1
    letters = re.sub(r'[^A-Za-z]', '', row.formula)
    return dict(name=name, space_group=sg, lattice=lat, centering='R' if sg.startswith('R') and lat == 'hexagonal' else 'P',
                cell={k_: round(float(v), 5) for k_, v in cell.items()}, hkl=hkl, color=PALETTE[color_index % len(PALETTE)],
                prefix=letters[:3], enabled=True, series=series or SERIES_OF.get(getattr(row, 'mode_key', 'powder'), 'all'),
                source=(f'CIF {getattr(row, "file", "") or row.mpid}' if getattr(row, 'source', 'mp') == 'cif' else f'Materials Project {row.mpid}')
                + (' (cell as in the entry)' if start_cell else ' (cell refined to this pattern)'))
