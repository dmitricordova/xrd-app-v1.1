"""Phase definitions: the example set, the bundled library, hkl text parsing and validation.

A phase is a dict:
    {"name": "VSe2", "space_group": "P-3m1", "lattice": "hexagonal", "centering": "P",
     "cell": {"a": 3.356, "c": 6.104}, "hkl": [[0,0,1], [1,0,1], ...], "color": "#1f3a93", "prefix": "V",
     "enabled": true}
lattice: cubic (a) | tetragonal (a, c) | hexagonal (a, c) | orthorhombic (a, b, c) | monoclinic (a, b, c, beta; unique axis b)
         | triclinic (a, b, c, alpha, beta, gamma); angles in degrees
centering: P | I | F | C | R (R: hexagonal setting, -h+k+l = 3n)
series: all | 00l | hk0 | 00l+hk0 - restrict the phase to its 00l lines (stacking period only), its hk0 lines (in-plane
        cell only; these may match broad turbostratic bands), or both. The axes no kept line depends on are held.
hkl: the reflections to look for (only their positions are used). Empty -> every hkl allowed by the
centering up to hmax / lmax, which lets weak or extinct reflections claim real lines.
The same JSON ({"phases": [...]}) is read by cli.py and by the older xrd_index_local.py.
"""
import copy
import json
import os
import re

LATTICES = ['hexagonal', 'tetragonal', 'cubic', 'orthorhombic', 'monoclinic', 'triclinic']
SERIES = ['all', '00l', 'hk0', '00l+hk0']
SERIES_LABEL = {'all': 'all reflections', '00l': '00l only', 'hk0': 'hk0 only', '00l+hk0': '00l + hk0'}
CENTERINGS = ['P', 'I', 'F', 'C', 'R']
AXES = {'cubic': ['a'], 'tetragonal': ['a', 'c'], 'hexagonal': ['a', 'c'], 'orthorhombic': ['a', 'b', 'c'],
        'monoclinic': ['a', 'b', 'c', 'beta'], 'triclinic': ['a', 'b', 'c', 'alpha', 'beta', 'gamma']}
ANGLES = ('alpha', 'beta', 'gamma')
SYMBOL = {'a': 'a', 'b': 'b', 'c': 'c', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}


def unit(ax):
    return '°' if ax in ANGLES else 'Å'
PALETTE = ['#1f3a93', '#1e7d4f', '#c25a12', '#8e44ad', '#2c3e50', '#16a085', '#a04000', '#7f8c8d', '#b03a7a', '#2e86c1']

_V = [[0, 0, 1], [0, 0, 2], [1, 0, 1], [1, 0, 2], [0, 0, 3], [1, 1, 0], [1, 0, 3], [1, 1, 1], [0, 0, 4], [1, 1, 2], [2, 0, 0],
      [2, 0, 1], [1, 0, 4], [2, 0, 2], [1, 1, 3], [2, 0, 3], [1, 1, 4], [1, 0, 5]]
_S = [[0, 0, 1], [0, 0, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [0, 0, 3], [1, 1, 0], [1, 1, 1], [1, 0, 3], [2, 0, 1], [0, 0, 4],
      [2, 0, 2], [1, 1, 3], [2, 0, 3], [2, 1, 1], [1, 1, 4], [2, 1, 2], [1, 0, 5], [3, 0, 0]]

EXAMPLE_PHASES = [
    {"name": "VSe2", "space_group": "P-3m1", "lattice": "hexagonal", "centering": "P", "cell": {"a": 3.356, "c": 6.104},
     "hkl": _V, "color": "#1f3a93", "prefix": "V", "enabled": True},
    {"name": "SnSe2", "space_group": "P-3m1", "lattice": "hexagonal", "centering": "P", "cell": {"a": 3.811, "c": 6.137},
     "hkl": _S, "color": "#1e7d4f", "prefix": "S", "enabled": True},
    {"name": "V2O3", "space_group": "R-3c", "lattice": "hexagonal", "centering": "R", "cell": {"a": 4.952, "c": 14.003},
     "hkl": [[0, 1, 2], [1, 0, 4], [1, 1, 3], [0, 2, 4], [1, 1, 6]], "color": "#c25a12", "prefix": "O", "enabled": True},
]


def library():
    """Bundled phase library (list of phase dicts)."""
    with open(os.path.join(os.path.dirname(__file__), 'phase_library.json'), encoding='utf-8') as fh:
        lib = json.load(fh)['phases']
    for p in lib:
        p.setdefault('enabled', True)
    return lib


def parse_hkl(text):
    """'001 002 101 10-2 1,0,10' -> [[0,0,1], [0,0,2], [1,0,1], [1,0,-2], [1,0,10]].
    Empty text or 'auto' -> [] (generate all allowed hkl). Raises ValueError naming the bad entries."""
    text = (text or '').strip()
    if text.lower() in ('', 'auto', 'all'):
        return []
    out, bad = [], []
    for tok in re.split(r'[\s;]+', text):
        if not tok:
            continue
        if ',' in tok:
            parts = [p for p in tok.strip('()[]').split(',') if p != '']
            try:
                v = [int(p) for p in parts]
            except ValueError:
                v = []
        else:
            v = [int(m) for m in re.findall(r'-?\d', tok)] if re.fullmatch(r'(-?\d){3}', tok) else []
        if len(v) == 3 and any(v):
            out.append(v)
        else:
            bad.append(tok)
    if bad:
        raise ValueError('cannot read ' + ', '.join(f'"{b}"' for b in bad[:5]) +
                         ' as hkl. Write each reflection as three digits (101, 10-2) or with commas (1,0,10).')
    return out


def format_hkl(hkl):
    if not hkl:
        return 'auto'
    return ' '.join(''.join(str(v) for v in h) if all(-9 <= v <= 9 for v in h) else ','.join(str(v) for v in h) for h in hkl)


def hkl_label(h):
    return ''.join(str(v) for v in h) if all(-9 <= v <= 9 for v in h) else ','.join(str(v) for v in h)


def validate(phases):
    """Return (clean phases, list of error strings). Clean phases have every field filled."""
    errors, clean, names = [], [], set()
    for i, p in enumerate(phases):
        p = copy.deepcopy(p)
        nm = str(p.get('name') or '').strip()
        where = f'Phase {i + 1}' + (f' ({nm})' if nm else '')
        if not nm:
            errors.append(f'{where}: give it a name.')
            continue
        if nm in names:
            errors.append(f'{where}: the name is used twice; names must be unique.')
            continue
        names.add(nm)
        lat = str(p.get('lattice', 'hexagonal')).lower()
        if lat not in AXES:
            errors.append(f'{where}: lattice must be one of {", ".join(LATTICES)}.')
            continue
        cen = str(p.get('centering', 'P')).upper() or 'P'
        if cen not in CENTERINGS:
            errors.append(f'{where}: centering must be one of {", ".join(CENTERINGS)}.')
            continue
        cell = {}
        for ax in AXES[lat]:
            v = (p.get('cell') or {}).get(ax)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = None
            if ax in ANGLES:
                if v is None or not (60 < v < 150):
                    errors.append(f'{where}: {SYMBOL[ax]} must be an angle between 60 and 150°.')
                    break
            elif v is None or not (0.5 < v < 100):
                errors.append(f'{where}: {ax} must be a length in Å between 0.5 and 100.')
                break
            cell[ax] = v
        else:
            hkl = p.get('hkl') or []
            if isinstance(hkl, str):
                try:
                    hkl = parse_hkl(hkl)
                except ValueError as e:
                    errors.append(f'{where}: {e}')
                    continue
            color = str(p.get('color') or PALETTE[i % len(PALETTE)])
            if not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
                color = PALETTE[i % len(PALETTE)]
            series = str(p.get('series') or 'all').replace(' ', '')
            if series not in SERIES:
                errors.append(f'{where}: reflections must be one of {", ".join(SERIES)}.')
                continue
            clean.append(dict(name=nm, space_group=str(p.get('space_group') or ''), lattice=lat, centering=cen, cell=cell,
                              hkl=[list(map(int, h)) for h in hkl], color=color,
                              prefix=str(p.get('prefix') or nm[:2]).strip() or nm[:2],
                              enabled=bool(p.get('enabled', True)), series=series,
                              hmax=int(p['hmax']) if p.get('hmax') else None, lmax=int(p['lmax']) if p.get('lmax') else None))
    return clean, errors


def to_json(phases):
    keep = ('name', 'space_group', 'lattice', 'centering', 'series', 'cell', 'hkl', 'color', 'prefix', 'enabled')
    text = json.dumps({'phases': [{k: p[k] for k in keep if k in p} for p in phases]}, indent=1, ensure_ascii=False)
    text = re.sub(r'\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\]', r'[\1, \2, \3]', text)        # one hkl per line
    return re.sub(r'"cell": \{\s*([^{}]*?)\s*\}', lambda m: '"cell": {' + re.sub(r'\s*\n\s*', ' ', m.group(1)) + '}', text)


def from_json(text):
    """Read a phases JSON ({"phases": [...]} or a bare list). Returns (phases, errors)."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        return [], [f'The phases file is not valid JSON ({e.msg}, line {e.lineno}).']
    lst = d.get('phases') if isinstance(d, dict) else d
    if not isinstance(lst, list):
        return [], ['The phases file must contain a list under "phases".']
    return validate(lst)
