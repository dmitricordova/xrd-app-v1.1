"""Reading powder patterns: refinement CSV exports (2θ, yobs, ycal, bkg, diff), two- or three-column
.xy / .xye / .txt / .dat files, with or without '#' header lines.

    from xrdcore.io import load_pattern
    pat = load_pattern(open('sample.csv', 'rb').read(), 'sample.csv')
"""
import io as _io
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class PatternError(ValueError):
    """The file could not be read as a powder pattern; the message says why and what to change."""


RATE_UNITS = {'cps', 'counts/s', 'count/s', 'c/s', 'counts per second', 'counts/sec'}

_ALIASES = {
    'twotheta': ('twotheta', '2theta', '2θ', 'tth', '2th', 'angle', 'two_theta', '2-theta', 'twotheta(deg)'),
    'yobs': ('yobs', 'iobs', 'intensity', 'counts', 'int', 'y_obs', 'y', 'i', 'cps', 'obs'),
    'ycal': ('ycal', 'ycalc', 'icalc', 'calc', 'y_calc', 'ycalculated'),
    'bkg': ('bkg', 'back', 'background', 'bgr', 'bg', 'ybkg'),
    'diff': ('diff', 'difference', 'delta', 'yobs-ycal'),
    'esd': ('esd', 'sigma', 'err', 'error', 'sig', 'e', 'uncertainty'),
}


@dataclass
class Pattern:
    name: str
    x: np.ndarray
    y: np.ndarray
    ycal: Optional[np.ndarray] = None
    bkg: Optional[np.ndarray] = None
    esd: Optional[np.ndarray] = None
    header: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    unit: str = 'counts'
    unit_stated: bool = False
    skipped_lines: int = 0
    sorted_input: bool = False

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.name))[0]

    @property
    def unit_is_rate(self):
        return self.unit.strip().lower() in RATE_UNITS

    @property
    def prior_peaks(self):
        """Peak model of the prior refinement (ycal - bkg), or None for a plain pattern."""
        if self.ycal is None or self.bkg is None:
            return None
        return self.ycal - self.bkg

    def stats(self):
        dx = np.diff(self.x)
        step = float(np.median(dx))
        uniform = bool(np.all(np.abs(dx - step) <= 1e-6 + 1e-4 * abs(step)))
        i_max, i_min = int(np.argmax(self.y)), int(np.argmin(self.y))
        return dict(n=len(self.x), xmin=float(self.x[0]), xmax=float(self.x[-1]), step=step, uniform=uniform,
                    ymax=float(self.y[i_max]), x_at_max=float(self.x[i_max]),
                    ymin=float(self.y[i_min]), x_at_min=float(self.x[i_min]))

    def raw_csv(self):
        """Two-column CSV of the data exactly as read (2θ, intensity)."""
        buf = _io.StringIO()
        buf.write(f'twotheta,intensity_{self.unit.replace(" ", "_")}\n')
        for a, b in zip(self.x, self.y):
            buf.write(f'{a:.6g},{b:.8g}\n')
        return buf.getvalue()


def _numbers(line):
    parts = [p for p in re.split(r'[,;\s]+', line.strip()) if p]
    if len(parts) < 2:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _role(name):
    n = name.strip().lower().lstrip('#').strip()
    n = re.sub(r'\s*\(.*?\)\s*', '', n).replace(' ', '')
    for role, names in _ALIASES.items():
        if n in names:
            return role
    if '2th' in n or 'theta' in n or '2θ' in n:
        return 'twotheta'
    if 'calc' in n:
        return 'ycal'
    if 'back' in n or 'bkg' in n:
        return 'bkg'
    if 'obs' in n or 'intens' in n or 'count' in n:
        return 'yobs'
    return None


def load_pattern(data, name='pattern.csv'):
    """Read a pattern from bytes or text. Raises PatternError with a readable message."""
    if isinstance(data, bytes):
        text = data.decode('utf-8-sig', errors='replace')
    else:
        text = data
    lines = text.splitlines()
    header, rows, skipped, ncol = [], [], 0, None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        nums = None if s.startswith(('#', '!', '//')) else _numbers(s)
        if nums is None:
            if rows:
                skipped += 1
            else:
                header.append(s)
            continue
        if ncol is None:
            ncol = len(nums)
        if len(nums) != ncol:
            skipped += 1
            continue
        rows.append(nums)
    if not rows:
        raise PatternError(f'{name}: no numeric data rows found. The file should have one 2θ value and one '
                           'intensity per line, separated by commas, tabs or spaces.')
    if len(rows) < 50:
        raise PatternError(f'{name}: only {len(rows)} data rows found; a powder pattern needs at least 50 points.')
    arr = np.array(rows, dtype=float)

    # column names from the last header line that has one name per column
    names = []
    for h in reversed(header):
        parts = [p.strip() for p in re.split(r'[,;\t]+' if re.search(r'[,;\t]', h) else r'\s+', h.lstrip('#!/ ').strip()) if p.strip()]
        if len(parts) == ncol and any(re.search(r'[A-Za-zθ]', p) for p in parts):
            names = parts
            break
    roles = [_role(n) for n in names] if names else []
    col = {}
    for i, r in enumerate(roles):
        if r and r not in col:
            col[r] = i
    if 'twotheta' not in col or 'yobs' not in col:
        col = {'twotheta': 0, 'yobs': 1}
        if ncol == 3:
            col['esd'] = 2
        if not names:
            names = ['twotheta', 'intensity'] + (['esd'] if ncol == 3 else []) + [f'col{i + 1}' for i in range(3, ncol)]

    x = arr[:, col['twotheta']]
    y = arr[:, col['yobs']]
    get = lambda r: arr[:, col[r]] if r in col else None
    ycal, bkg, esd = get('ycal'), get('bkg'), get('esd')

    ok = np.isfinite(x) & np.isfinite(y)
    sorted_input = False
    if not np.all(np.diff(x[ok]) > 0):
        order = np.argsort(x[ok], kind='stable')
        sorted_input = True
    else:
        order = slice(None)
    pick = lambda a: None if a is None else a[ok][order]
    x, y, ycal, bkg, esd = pick(x), pick(y), pick(ycal), pick(bkg), pick(esd)
    if x.min() < 0 or x.max() > 180:
        raise PatternError(f'{name}: the first column runs from {x.min():g} to {x.max():g}; it should be 2θ in degrees '
                           '(0–180). Check the column order.')
    if np.any(np.diff(x) == 0):
        keep = np.concatenate([[True], np.diff(x) > 0])
        x, y = x[keep], y[keep]
        ycal, bkg, esd = [None if a is None else a[keep] for a in (ycal, bkg, esd)]

    unit, stated = 'counts', False
    for h in header:
        m = re.search(r'unit\w*\s*[=:]\s*([^,;]+)', h, re.I)
        if m:
            unit, stated = m.group(1).strip(), True
            break
    if not stated and names and 'yobs' in col and 'cps' in names[col['yobs']].lower():
        unit, stated = 'cps', True

    return Pattern(name=name, x=x, y=y, ycal=ycal, bkg=bkg, esd=esd, header=header, columns=names, unit=unit,
                   unit_stated=stated, skipped_lines=skipped, sorted_input=sorted_input)
