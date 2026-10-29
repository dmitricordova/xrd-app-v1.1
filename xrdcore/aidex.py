"""Unit cells from AIdex-r2 for the unindexed lines, checked against the peaks.

AIdex-r2 (https://aidex-r2-web.hf.space/; K. Shu, D.-Y. Gui, W.-X. Yan, C.-H. Wang, J. Chem. Inf. Model. 2025, 65, 10025)
predicts a unit cell and extinction group from up to the first 20 low-angle peak positions of ONE phase and returns five
candidates with a score. It is a cell finder, not a phase identifier, and needs ideally 15-20 lines of a single phase.

    lines = aidex.peak_list(index_result)                 # sharp unindexed lines, corrected 2θ, lowest first, ≤ 20
    cands = aidex.predict_online(lines)                   # calls the AIdex-r2 web service directly (needs internet)
    cands, errors = aidex.parse_results(pasted_text)      # or: the table AIdex shows, pasted (EG, a, b, c, α, β, γ, score)
    js = aidex.fill_script(lines)                         # or: a one-line script that fills AIdex-r2's 20 peak boxes
    table = aidex.check(cands, lines, index_result)       # refined cell, misfit, line density, M(n), chance P, verdict
    phase = aidex.to_phase(table.iloc[0], 'AIdex cell 1')

check(), for each candidate:
  - lattice type from the cell metric (a = b = c, angles 90 / 120 ...) and centering from the extinction-group letter
  - all reflections the centering allows (no other extinctions, so the test is generous to the candidate), d > λ/2
  - cell refined to the entered lines (nearest-line assignment, 0.15 → 0.06°, least squares on the free parameters)
  - rms and largest misfit; number of distinct calculated lines in the entered 2θ window (line density); de Wolff M(n)
    = Q_n / (2 ε N_n); impossible when the largest calculated d is shorter than the lowest-angle entered line
  - chance P: the same refinement against random line sets of the same size and range; P = share fitted as well
  - Niggli-reduced cell compared with the listed phases (same lattice in another setting), when pymatgen is installed
verdict: credible (rms ≤ 0.03°, M(n) ≥ 10, P ≤ 0.05), possible, or rejected, with the reason.
"""
import json
import math
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

L1 = 1.540593
URL = 'https://aidex-r2-web.hf.space/'
MAX_PEAKS = 20
EMPTY_PEAK = 180.0          # AIdex-r2's value for an unused peak box

# AIdex-r2 extinction-group number -> (space-group numbers, lattice centering, crystal system, space groups).
# From the EG list of the AIdex-r2 web page; centering and system of each EG from its centrosymmetric space group.
EG = {
    1: ((1, 2), 'P', 'triclinic', 'P1, P-1'),
    2: ((3, 6, 10), 'P', 'monoclinic', 'P121, P1m1, P12/m1'),
    3: ((4, 11), 'P', 'monoclinic', 'P12_11, P12_1/m1'),
    4: ((5, 8, 12), 'C', 'monoclinic', 'C121, C1m1, C12/m1'),
    5: ((7, 13), 'P', 'monoclinic', 'P1c1, P12/c1'),
    6: ((14,), 'P', 'monoclinic', 'P12_1/c1'),
    7: ((9, 15), 'C', 'monoclinic', 'C1c1, C12/c1'),
    8: ((17,), 'P', 'orthorhombic', 'P222_1'),
    9: ((18,), 'P', 'orthorhombic', 'P2_12_12'),
    10: ((19,), 'P', 'orthorhombic', 'P2_12_12_1'),
    11: ((20,), 'C', 'orthorhombic', 'C222_1'),
    12: ((43,), 'F', 'orthorhombic', 'Fdd2'),
    13: ((16, 25, 47), 'P', 'orthorhombic', 'P222, Pmm2, Pmmm'),
    14: ((48,), 'P', 'orthorhombic', 'Pnnn'),
    15: ((27, 49), 'P', 'orthorhombic', 'Pcc2, Pccm'),
    16: ((50,), 'P', 'orthorhombic', 'Pban'),
    17: ((26, 28, 51), 'P', 'orthorhombic', 'Pmc2_1, Pma2, Pmma'),
    18: ((52,), 'P', 'orthorhombic', 'Pnna'),
    19: ((30, 53), 'P', 'orthorhombic', 'Pnc2, Pmna'),
    20: ((54,), 'P', 'orthorhombic', 'Pcca'),
    21: ((32, 55), 'P', 'orthorhombic', 'Pba2, Pbam'),
    22: ((56,), 'P', 'orthorhombic', 'Pccn'),
    23: ((29, 57), 'P', 'orthorhombic', 'Pca2_1, Pbcm'),
    24: ((34, 58), 'P', 'orthorhombic', 'Pnn2, Pnnm'),
    25: ((31, 59), 'P', 'orthorhombic', 'Pmn2_1, Pmmn'),
    26: ((60,), 'P', 'orthorhombic', 'Pbcn'),
    27: ((61,), 'P', 'orthorhombic', 'Pbca'),
    28: ((33, 62), 'P', 'orthorhombic', 'Pna2_1, Pnma'),
    29: ((36, 40, 63), 'C', 'orthorhombic', 'Cmc2_1, Ama2, Cmcm'),
    30: ((41, 64), 'C', 'orthorhombic', 'Aea2, Cmce'),
    31: ((21, 35, 38, 65), 'C', 'orthorhombic', 'C222, Cmm2, Amm2, Cmmm'),
    32: ((37, 66), 'C', 'orthorhombic', 'Ccc2, Cccm'),
    33: ((39, 67), 'C', 'orthorhombic', 'Aem2, Cmme'),
    34: ((68,), 'C', 'orthorhombic', 'Ccce'),
    35: ((22, 42, 69), 'F', 'orthorhombic', 'F222, Fmm2, Fmmm'),
    36: ((70,), 'F', 'orthorhombic', 'Fddd'),
    37: ((23, 24, 44, 71), 'I', 'orthorhombic', 'I222, I2_12_12_1, Imm2, Immm'),
    38: ((45, 72), 'I', 'orthorhombic', 'Iba2, Ibam'),
    39: ((73,), 'I', 'orthorhombic', 'Ibca'),
    40: ((46, 74), 'I', 'orthorhombic', 'Ima2, Imma'),
    41: ((86,), 'P', 'tetragonal', 'P4_2/n'),
    42: ((88,), 'I', 'tetragonal', 'I4_1/a'),
    43: ((77, 84, 93), 'P', 'tetragonal', 'P4_2, P4_2/m, P4_222'),
    44: ((94,), 'P', 'tetragonal', 'P4_22_12'),
    45: ((76, 78, 91, 95), 'P', 'tetragonal', 'P4_1, P4_3, P4_122, P4_322'),
    46: ((92, 96), 'P', 'tetragonal', 'P4_12_12, P4_32_12'),
    47: ((80, 98), 'I', 'tetragonal', 'I4_1, I4_122'),
    48: ((110,), 'I', 'tetragonal', 'I4_1cd'),
    49: ((90, 113), 'P', 'tetragonal', 'P42_12, P-42_1m'),
    50: ((114,), 'P', 'tetragonal', 'P-42_1c'),
    51: ((109, 122), 'I', 'tetragonal', 'I4_1md, I-42d'),
    52: ((75, 81, 83, 89, 99, 111, 115, 123), 'P', 'tetragonal', 'P4, P-4, P4/m, P422, P4mm, P-42m, P-4m2, P4/mmm'),
    53: ((103, 124), 'P', 'tetragonal', 'P4cc, P4/mcc'),
    54: ((125,), 'P', 'tetragonal', 'P4/nbm'),
    55: ((126,), 'P', 'tetragonal', 'P4/nnc'),
    56: ((100, 117, 127), 'P', 'tetragonal', 'P4bm, P-4b2, P4/mbm'),
    57: ((104, 128), 'P', 'tetragonal', 'P4nc, P4/mnc'),
    58: ((85, 129), 'P', 'tetragonal', 'P4/n, P4/nmm'),
    59: ((130,), 'P', 'tetragonal', 'P4/ncc'),
    60: ((105, 112, 131), 'P', 'tetragonal', 'P4_2mc, P-42c, P4_2/mmc'),
    61: ((101, 116, 132), 'P', 'tetragonal', 'P4_2cm, P-4c2, P4_2/mcm'),
    62: ((133,), 'P', 'tetragonal', 'P4_2/nbc'),
    63: ((134,), 'P', 'tetragonal', 'P4_2/nnm'),
    64: ((106, 135), 'P', 'tetragonal', 'P4_2bc, P4_2/mbc'),
    65: ((102, 118, 136), 'P', 'tetragonal', 'P4_2nm, P-4n2, P4_2/mnm'),
    66: ((137,), 'P', 'tetragonal', 'P4_2/nmc'),
    67: ((138,), 'P', 'tetragonal', 'P4_2/ncm'),
    68: ((79, 82, 87, 97, 107, 119, 121, 139), 'I', 'tetragonal', 'I4, I-4, I4/m, I422, I4mm, I-4m2, I-42m, I4/mmm'),
    69: ((108, 120, 140), 'I', 'tetragonal', 'I4cm, I-4c2, I4/mcm'),
    70: ((141,), 'I', 'tetragonal', 'I4_1/amd'),
    71: ((142,), 'I', 'tetragonal', 'I4_1/acd'),
    72: ((144, 145, 151, 152, 153, 154), 'P', 'trigonal', 'P3_1, P3_2, P3_112, P3_121, P3_212, P3_221'),
    73: ((159, 163), 'P', 'trigonal', 'P31c, P-31c'),
    74: ((143, 147, 149, 150, 156, 157, 162, 164), 'P', 'trigonal', 'P3, P-3, P312, P321, P3m1, P31m, P-31m, P-3m1'),
    75: ((158, 165), 'P', 'trigonal', 'P3c1, P-3c1'),
    76: ((146, 148, 155, 160, 166), 'R', 'trigonal', 'R3, R-3, R32, R3m, R-3m'),
    77: ((161, 167), 'R', 'trigonal', 'R3c, R-3c'),
    78: ((169, 170, 178, 179), 'P', 'hexagonal', 'P6_1, P6_5, P6_122, P6_522'),
    79: ((171, 172, 180, 181), 'P', 'hexagonal', 'P6_2, P6_4, P6_222, P6_422'),
    80: ((173, 176, 182), 'P', 'hexagonal', 'P6_3, P6_3/m, P6_322'),
    81: ((168, 174, 175, 177, 183, 187, 189, 191), 'P', 'hexagonal', 'P6, P-6, P6/m, P622, P6mm, P-6m2, P-62m, P6/mmm'),
    82: ((184, 192), 'P', 'hexagonal', 'P6cc, P6/mcc'),
    83: ((185, 188, 193), 'P', 'hexagonal', 'P6_3cm, P-6c2, P6_3/mcm'),
    84: ((186, 190, 194), 'P', 'hexagonal', 'P6_3mc, P-62c, P6_3/mmc'),
    85: ((205,), 'P', 'cubic', 'Pa-3'),
    86: ((206,), 'I', 'cubic', 'Ia-3'),
    87: ((198, 208), 'P', 'cubic', 'P2_13, P4_232'),
    88: ((210,), 'F', 'cubic', 'F4_132'),
    89: ((212, 213), 'P', 'cubic', 'P4_332, P4_132'),
    90: ((214,), 'I', 'cubic', 'I4_132'),
    91: ((220,), 'I', 'cubic', 'I-43d'),
    92: ((195, 200, 207, 215, 221), 'P', 'cubic', 'P23, Pm-3, P432, P-43m, Pm-3m'),
    93: ((222,), 'P', 'cubic', 'Pn-3n'),
    94: ((218, 223), 'P', 'cubic', 'P-43n, Pm-3n'),
    95: ((201, 224), 'P', 'cubic', 'Pn-3, Pn-3m'),
    96: ((196, 202, 209, 216, 225), 'F', 'cubic', 'F23, Fm-3, F432, F-43m, Fm-3m'),
    97: ((219, 226), 'F', 'cubic', 'F-43c, Fm-3c'),
    98: ((203, 227), 'F', 'cubic', 'Fd-3, Fd-3m'),
    99: ((228,), 'F', 'cubic', 'Fd-3c'),
    100: ((197, 199, 204, 211, 217, 229), 'I', 'cubic', 'I23, I2_13, Im-3, I432, I-43m, Im-3m'),
    101: ((230,), 'I', 'cubic', 'Ia-3d'),
}

CITATION = 'K. Shu, D.-Y. Gui, W.-X. Yan, C.-H. Wang, J. Chem. Inf. Model. 2025, 65, 10025 (AIdex-r2)'


# ------------------------------------------------------------------------------------------- input / output
def peak_list(res, max_n=20):
    """Sharp unindexed lines on the corrected 2θ scale, lowest angles first (no Kβ copies, humps or bands)."""
    T = res.table
    m = (T['index'] == '—') & (T.profile == 'sharp')
    return sorted(float(v) for v in T.loc[m, '2θ corr'])[:max_n]


def eg_label(n):
    """'I · EG 69 (I4cm, I-4c2, I4/mcm)': starts with the centering letter, which check() reads."""
    n = int(n)
    if n not in EG:
        return 'P · EG ' + str(n)
    _, cen, _, syms = EG[n]
    return f'{cen} · EG {n} ({syms})'


def fill_script(lines):
    """JavaScript that fills AIdex-r2's peak boxes (run in the browser console on the AIdex-r2 page)."""
    v = ','.join(f'{x:.3f}' for x in sorted(lines)[:MAX_PEAKS])
    return ("(()=>{const v=[" + v + "];const b=[...document.querySelectorAll('.peak-input')];"
            "b.forEach((e,i)=>{e.value=i<v.length?v[i].toFixed(3):'180.0';"
            "e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))});"
            "return v.length+' peaks entered in '+b.length+' boxes'})()")


class AIdexError(RuntimeError):
    pass


def predict_online(lines, timeout=90, url=None):
    """Send up to 20 lines to the AIdex-r2 web service (POST /api/predict, unused boxes = 180.0, as its page does).
    Returns candidates in the parse_results format. Raises AIdexError when the service cannot be reached."""
    peaks = sorted(float(x) for x in lines)[:MAX_PEAKS]
    if len(peaks) < 3:
        raise AIdexError('Give AIdex-r2 at least 3 lines.')
    peaks += [EMPTY_PEAK] * (MAX_PEAKS - len(peaks))
    req = urllib.request.Request((url or URL).rstrip('/') + '/api/predict', data=json.dumps({'peaks': peaks}).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'xrd-app-v1'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AIdexError(f'AIdex-r2 could not be reached from this computer ({getattr(e, "reason", e)}). '
                         'Use Option B (fill AIdex-r2 in your browser) instead.') from None
    except ValueError:
        raise AIdexError('AIdex-r2 sent a reply that is not JSON; its web service may have changed.') from None
    if not isinstance(data, dict) or data.get('error'):
        raise AIdexError(f"AIdex-r2 returned an error: {data.get('error') if isinstance(data, dict) else data}")
    out = []
    for r_ in data.get('results') or []:
        try:
            out.append(dict(eg=eg_label(r_['eg']), a=float(r_['a']), b=float(r_['b']), c=float(r_['c']), alpha=float(r_['alpha']),
                            beta=float(r_['beta']), gamma=float(r_['gamma']), score=100 * float(r_.get('score', float('nan')))))
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise AIdexError('AIdex-r2 returned no candidate cells.')
    return out


def parse_results(text):
    """Read the AIdex-r2 results table as copied from the page (any spacing / tabs / | separators).
    Each row: extinction group (its number, as the page shows it, or a symbol), a, b, c, α, β, γ and, optionally,
    a rank before and a score (%) after."""
    out, errors = [], []
    for ln in (text or '').splitlines():
        s = ln.strip().strip('|').strip()
        if not s:
            continue
        toks = [t for t in re.split(r'[\s|,;]+', s) if t]
        nums, eg = [], None
        for t in toks:
            v = t.rstrip('%').lstrip('<')
            try:
                nums.append(float(v))
            except ValueError:
                if eg is None and re.match(r'^[PIFCABRH]', t):
                    eg = t
        if len(nums) < 6:
            if re.search(r'\d', s) and not re.search(r'^(EG|Rank|#)', s, re.I):
                errors.append(f'Could not read a cell from: "{s[:60]}"')
            continue
        # the page shows the extinction group as a number (1-101): EG, a, b, c, α, β, γ, score (+ an optional rank first)
        if eg is None and len(nums) >= 9 and float(nums[1]).is_integer() and 1 <= nums[1] <= 101:
            nums = nums[1:]                                   # rank, EG, cell, score
        if eg is None and len(nums) >= 8 and float(nums[0]).is_integer() and 1 <= nums[0] <= 101 and nums[3] >= 0.5 and 30 < nums[4] < 150:
            eg = eg_label(int(nums[0]))
            nums = nums[1:]
        # a leading rank shows as: 8 numbers, or 7 numbers whose fourth is a length (c) rather than an angle (α)
        elif len(nums) >= 8 or (len(nums) == 7 and nums[3] < 30):
            nums = nums[1:]
        a, b, c, al, be, ga = nums[:6]
        score = nums[6] if len(nums) > 6 else float('nan')
        if not (0.5 < min(a, b, c) and max(a, b, c) < 200 and 30 < min(al, be, ga) and max(al, be, ga) < 150):
            errors.append(f'Not a unit cell (lengths 0.5-200 Å, angles 30-150°): "{s[:60]}"')
            continue
        out.append(dict(eg=eg or 'P', a=a, b=b, c=c, alpha=al, beta=be, gamma=ga, score=score))
    if not out and not errors:
        errors.append('Paste the table AIdex-r2 shows (one row per candidate: EG, a, b, c, α, β, γ, score).')
    return out, errors


# ------------------------------------------------------------------------------------------- geometry
def _gstar(p):
    a, b, c, al, be, ga = p
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return np.linalg.inv(np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]]))


def _volume(p):
    return float(1 / math.sqrt(np.linalg.det(_gstar(p))))


def lattice_type(p, tol=0.002, atol=0.05):
    a, b, c, al, be, ga = p
    eq = lambda x, y: abs(x - y) <= tol * max(x, y)
    r90 = lambda x: abs(x - 90) <= atol
    if all(r90(v) for v in (al, be, ga)):
        if eq(a, b) and eq(b, c):
            return 'cubic'
        if eq(a, b):
            return 'tetragonal'
        return 'orthorhombic'
    if r90(al) and r90(be) and abs(ga - 120) <= atol and eq(a, b):
        return 'hexagonal'
    if eq(a, b) and eq(b, c) and abs(al - be) <= atol and abs(be - ga) <= atol:
        return 'rhombohedral'
    if sum(r90(v) for v in (al, be, ga)) == 2:
        return 'monoclinic'
    return 'triclinic'


def _free(p, lat):
    """free parameter indices and ties (index -> source)"""
    if lat == 'cubic':
        return [0], {1: 0, 2: 0}
    if lat in ('tetragonal', 'hexagonal'):
        return [0, 2], {1: 0}
    if lat == 'rhombohedral':
        return [0, 3], {1: 0, 2: 0, 4: 3, 5: 3}
    if lat == 'orthorhombic':
        return [0, 1, 2], {}
    if lat == 'monoclinic':
        return [0, 1, 2] + [i for i in (3, 4, 5) if abs(p[i] - 90) > 0.05][:1], {}
    return [0, 1, 2, 3, 4, 5], {}


def _allowed(cen, H):
    h, k, l = H[:, 0], H[:, 1], H[:, 2]
    if cen == 'I':
        return (h + k + l) % 2 == 0
    if cen == 'F':
        return ((h % 2) == (k % 2)) & ((k % 2) == (l % 2))
    if cen == 'C':
        return (h + k) % 2 == 0
    if cen == 'A':
        return (k + l) % 2 == 0
    if cen == 'B':
        return (h + l) % 2 == 0
    if cen == 'R':
        return (-h + k + l) % 3 == 0
    return np.ones(len(H), bool)


def _hkl(p, cen, tmax):
    """All reflections (centering only) with 2θ < tmax, one representative per d."""
    dmin = L1 / (2 * math.sin(math.radians(min(tmax, 170) / 2)))
    G = _gstar(p)
    lim = [int(math.ceil(1 / (dmin * math.sqrt(G[i, i])))) + 1 for i in range(3)]
    lim = [min(v, 25) for v in lim]
    r = [np.arange(-v, v + 1) for v in lim]
    H = np.array(np.meshgrid(*r, indexing='ij')).reshape(3, -1).T.astype(float)
    H = H[np.any(H != 0, axis=1)]
    H = H[_allowed(cen, H.astype(int))]
    q = np.einsum('ni,ij,nj->n', H, G, H)
    H, q = H[q <= 1 / dmin ** 2], q[q <= 1 / dmin ** 2]
    _, keep = np.unique(np.round(q, 7), return_index=True)
    return H[keep]


def _tth(p, H):
    q = np.einsum('ni,ij,nj->n', H, _gstar(p), H)
    s = L1 * np.sqrt(np.maximum(q, 0)) / 2
    return np.where(s < 1, 2 * np.degrees(np.arcsin(np.clip(s, 0, 1))), np.nan)


def _refine(p0, lat, cen, obs, tmax):
    free, ties = _free(p0, lat)
    p = np.array(p0, float)

    def full(x):
        q = np.array(p, float)
        for v, i in zip(x, free):
            q[i] = v
        for i, src in ties.items():
            q[i] = q[src]
        return q
    for tol in (0.15, 0.10, 0.06):
        H = _hkl(p, cen, tmax + 1)
        t = _tth(p, H)
        ok = np.isfinite(t)
        H, t = H[ok], t[ok]
        if not len(t):
            break
        d = np.abs(obs[:, None] - t[None, :])
        j = d.argmin(1)
        use = d.min(1) <= tol
        if use.sum() < len(free) + 1:
            break
        Hs, tgt = H[j[use]], obs[use]
        lb = [p[i] * 0.97 if i < 3 else p[i] - 3 for i in free]
        ub = [p[i] * 1.03 if i < 3 else p[i] + 3 for i in free]
        r = least_squares(lambda x: _tth(full(x), Hs) - tgt, np.clip([p[i] for i in free], np.array(lb) + 1e-9, np.array(ub) - 1e-9),
                          bounds=(lb, ub))
        p = full(r.x)
    return p


def _score(p, cen, obs):
    tmax = obs.max() + 0.5
    H = _hkl(p, cen, tmax + 1)
    t = _tth(p, H)
    t = np.sort(t[np.isfinite(t)])
    if not len(t):
        return dict(rms=np.inf, maxmis=np.inf, n_lines=0, M=0.0)
    dev = np.array([np.min(np.abs(t - o)) for o in obs])
    win = t[(t > obs.min() - 0.5) & (t < tmax)]
    # de Wolff M(n): Q of the last entered line / (2 · mean |ΔQ| · number of calculated lines up to it)
    Q = lambda tt: (2 * np.sin(np.radians(np.asarray(tt) / 2)) / L1) ** 2
    nearest = np.array([t[np.argmin(np.abs(t - o))] for o in obs])
    eps = float(np.mean(np.abs(Q(obs) - Q(nearest))))
    N = int(np.sum(t <= obs.max() + 1e-9))
    M = float(Q(obs.max()) / (2 * max(eps, 1e-9) * max(N, 1)))
    return dict(rms=float(np.sqrt(np.mean(dev ** 2))), maxmis=float(dev.max()), n_lines=int(len(win)), M=M, dev=dev)


_TO_PRIMITIVE = {   # rows: primitive vectors in terms of the conventional ones
    'P': np.eye(3),
    'F': np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
    'I': np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]),
    'C': np.array([[.5, .5, 0], [-.5, .5, 0], [0, 0, 1]]),
    'A': np.array([[1, 0, 0], [0, .5, .5], [0, -.5, .5]]),
    'B': np.array([[.5, 0, .5], [0, 1, 0], [-.5, 0, .5]]),
    'R': np.array([[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]]),
}


def _niggli(p, cen='P'):
    """Niggli-reduced cell of the primitive lattice (centred cells are reduced first), or None without pymatgen."""
    try:
        from pymatgen.core import Lattice
    except Exception:
        return None
    try:
        conv = Lattice.from_parameters(*p)
        prim = Lattice(_TO_PRIMITIVE.get(cen, np.eye(3)) @ conv.matrix)
        lat = prim.get_niggli_reduced_lattice()
        return np.array(list(lat.abc) + list(lat.angles))
    except Exception:
        return None


def _same(n1, n2, ltol=0.01, atol=1.0):
    if n1 is None or n2 is None:
        return False
    if np.any(np.abs(n1[:3] - n2[:3]) > ltol * n2[:3]):
        return False
    return all(abs(x - y) <= atol or abs(x - (180 - y)) <= atol for x, y in zip(n1[3:], n2[3:]))


def _phase_params(ph):
    c = ph['cell']
    lat = ph['lattice']
    a = c['a']
    b = c.get('b', a)
    cc = c.get('c', a)
    al = c.get('alpha', 90.0)
    be = c.get('beta', 90.0)
    ga = c.get('gamma', 120.0 if lat == 'hexagonal' else 90.0)
    return [a, b, cc, al, be, ga]


def check(cands, peaks, res=None, n_random=100, seed=0):
    """Check AIdex-r2 candidates against the entered lines. Returns a DataFrame (one row per candidate)."""
    obs = np.sort(np.asarray(peaks, float))
    if len(obs) < 3:
        raise ValueError('At least three lines are needed to test a cell.')
    rng = np.random.default_rng(seed)
    lo, hi = float(obs.min()), float(obs.max())
    span = (max(2.0, lo - 2), hi + 2)
    rsets = [np.sort(rng.uniform(*span, size=len(obs))) for _ in range(n_random)]
    listed = []
    if res is not None:
        for ph in res.phases:
            try:
                listed.append((ph['name'], _niggli(_phase_params(ph), ph.get('centering', 'P'))))
            except Exception:
                pass
    d_first = L1 / (2 * math.sin(math.radians(lo / 2)))
    rows = []
    for i, c in enumerate(cands, 1):
        p0 = np.array([c['a'], c['b'], c['c'], c['alpha'], c['beta'], c['gamma']], float)
        cen = (c.get('eg') or 'P')[0].upper()
        if cen == 'H':
            cen = 'P'
        lat = lattice_type(p0)
        p = _refine(p0, lat, cen, obs, hi + 0.5)
        sc = _score(p, cen, obs)
        Hd = _hkl(p, cen, 60)
        dmax = float(1 / np.sqrt(np.min(np.einsum('ni,ij,nj->n', Hd, _gstar(p), Hd)))) if len(Hd) else 0.0
        impossible = dmax < d_first * 0.995
        hits = 0
        for r in rsets:
            pr = _refine(p0, lat, cen, r, r.max() + 0.5)
            hits += _score(pr, cen, r)['rms'] <= sc['rms'] + 1e-9
        pch = hits / n_random
        nig = _niggli(p, cen)
        same = [n for n, nl in listed if _same(nig, nl)]
        reasons = []
        if impossible:
            verdict = 'rejected'
            reasons.append(f'its largest d ({dmax:.3f} Å) is shorter than the lowest-angle line ({d_first:.3f} Å)')
        elif sc['rms'] <= 0.03 and sc['M'] >= 10 and pch <= 0.05:
            verdict = 'credible'
            reasons.append(f"rms {sc['rms']:.3f}°, M({len(obs)}) = {sc['M']:.0f}, random lines fit as well in {100 * pch:.0f} % of tries")
        elif sc['rms'] <= 0.05 and pch <= 0.2:
            verdict = 'possible'
            reasons.append(f"rms {sc['rms']:.3f}°, M({len(obs)}) = {sc['M']:.1f}; random lines fit as well in {100 * pch:.0f} % of tries")
        else:
            verdict = 'rejected'
            why = []
            if sc['rms'] > 0.05:
                why.append(f"misfit too large (rms {sc['rms']:.3f}°, worst {sc['maxmis']:.3f}°)")
            if pch > 0.2:
                why.append(f"random lines fit as well in {100 * pch:.0f} % of tries ({sc['n_lines']} calculated lines in the window)")
            if sc['M'] < 10 and not why:
                why.append(f"M({len(obs)}) = {sc['M']:.1f} is low")
            reasons.extend(why or ['does not fit the lines'])
        if same:
            reasons.append('same lattice as ' + ', '.join(same) + ' in your list')
        rows.append(dict(rank=i, eg=c.get('eg') or 'P', aidex_score=c.get('score'), lattice=lat, centering=cen,
                         a=round(p[0], 5), b=round(p[1], 5), c=round(p[2], 5), alpha=round(p[3], 3), beta=round(p[4], 3), gamma=round(p[5], 3),
                         volume=round(_volume(p), 2), rms=round(sc['rms'], 4), max_misfit=round(sc['maxmis'], 4), lines_in_window=sc['n_lines'],
                         M=round(sc['M'], 1), p_chance=round(pch, 3), same_as=', '.join(same), verdict=verdict, reason='; '.join(reasons),
                         start=[c['a'], c['b'], c['c'], c['alpha'], c['beta'], c['gamma']],
                         niggli=None if nig is None else [round(float(v), 4) for v in nig]))
    order = {'credible': 0, 'possible': 1, 'rejected': 2}
    D = pd.DataFrame(rows)
    return D.assign(_o=D.verdict.map(order)).sort_values(['_o', 'rms']).drop(columns='_o').reset_index(drop=True)


def to_phase(row, name, color='#b03a7a', prefix='X'):
    """Phase dict for the phase list. Reflections: every one the centering allows ('auto'); the joint fit refines it."""
    lat = row.lattice
    p = [row.a, row.b, row.c, row.alpha, row.beta, row.gamma]
    cen = row.centering if row.centering in ('P', 'I', 'F', 'C', 'R') else 'P'
    if lat == 'rhombohedral':                                    # to the hexagonal setting
        a, al = p[0], math.radians(p[3])
        ah = 2 * a * math.sin(al / 2)
        ch = a * math.sqrt(3 * (1 + 2 * math.cos(al)))
        lat, cell, cen = 'hexagonal', {'a': ah, 'c': ch}, 'R'
    elif lat == 'cubic':
        cell = {'a': p[0]}
    elif lat in ('tetragonal', 'hexagonal'):
        cell = {'a': p[0], 'c': p[2]}
    elif lat == 'orthorhombic':
        cell = {'a': p[0], 'b': p[1], 'c': p[2]}
    elif lat == 'monoclinic' and abs(p[3] - 90) < 0.05 and abs(p[5] - 90) < 0.05:
        cell = {'a': p[0], 'b': p[1], 'c': p[2], 'beta': p[4]}
    else:
        lat, cell = 'triclinic', {'a': p[0], 'b': p[1], 'c': p[2], 'alpha': p[3], 'beta': p[4], 'gamma': p[5]}
    if cen in ('A', 'B') or (lat == 'triclinic' and cen != 'P'):
        cen = 'P'
    return dict(name=name, space_group=f'AIdex {row.eg}', lattice=lat, centering=cen, cell={k: round(float(v), 5) for k, v in cell.items()},
                hkl=[], series='all', color=color, prefix=prefix, enabled=True)
