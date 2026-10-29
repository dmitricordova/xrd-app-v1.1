"""Le Bail whole-pattern fit with rietx (https://rietx.org), started from an indexing result.

    from xrdcore import lebail
    lb = lebail.run(index_result, quality='fast')      # ~45 s on an 8500-point pattern with 3 phases
    lb.cells, lb.stats, lb.table_vs_indexing(), lb.fit_csv(), lb.params

What it does:
  - background: the step-2 background of the indexing result, held exactly as it is (the one background of the workup)
  - phases: every switched-on phase, starting from its indexed cell; cell lengths bounded to ±0.5 %, angles ±0.5°
      all       -> a full cell (the phase's space group when rietx knows it, else the highest symmetry of its lattice and centering)
      00l       -> P4/mmm with a = b = 0.8 Å held, so only (00l) falls in the scan; c refined
      hk0       -> c = 0.8 Å held, so only (hk0) falls in the scan; a (b) refined, broad-band size, added late
      00l+hk0   -> both of the above as two rietx phases
    A phase whose strongest indexed line is weaker than `weak_percent` of the strongest peak keeps its indexed cell
    (Le Bail, with every reflection intensity free, cannot fix the cell of a ~1 % phase); the fit still models it.
  - unindexed sharp lines -> free extra peaks; unassigned broad features (not stuck at the width bound) -> humps
  - Cu Kβ: a held Kβ line at the measured Kβ/Kα ratio (median of the confirmed copies; 0.008 if none)
  - angle correction: as in the indexing (sample displacement in mm, or zero shift)
  - plan: correction -> first cell -> widths -> extras -> all cells -> profile + strain (-> late hk0 phases);
    'fast' = 1 Le Bail pass, intermediate ftol 1e-4, no closing stage; 'thorough' = 2 passes, rietx defaults, closing stage
  - esds: rietx esd × Bérar-Lelann factor sqrt((1 + ρ)/(1 − ρ)), ρ = 1 − DW/2 (raw esds on these films are 3-7× too small)
"""
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .indexing import corr as angle_corr
from .phases import AXES, SYMBOL, unit
from .util import L1, fsu

LOR0 = 0.05
GENERIC_SG = {
    ('cubic', 'P'): 'Pm-3m', ('cubic', 'I'): 'Im-3m', ('cubic', 'F'): 'Fm-3m',
    ('tetragonal', 'P'): 'P4/mmm', ('tetragonal', 'I'): 'I4/mmm',
    ('hexagonal', 'P'): 'P6/mmm', ('hexagonal', 'R'): 'R-3m',
    ('orthorhombic', 'P'): 'Pmmm', ('orthorhombic', 'I'): 'Immm', ('orthorhombic', 'F'): 'Fmmm', ('orthorhombic', 'C'): 'Cmmm',
    ('monoclinic', 'P'): 'P2/m', ('monoclinic', 'C'): 'C2/m', ('monoclinic', 'I'): 'I2/m',
    ('triclinic', 'P'): 'P-1',
}

READING = {
    'CELL_RUNAWAY': 'a cell parameter ran far from its start; check that placeholder axes of 00l/hk0 series are held',
    'LOW_ANGLE_UNMODELLED': 'intensity at low angle is not modelled; raise the lower 2θ limit or check the background',
    'SIZE_UNUSUALLY_SMALL': 'a very small crystallite size, expected for turbostratic hk0 bands; their size is not a measurement',
    'BOUND_HIT': 'a parameter stopped at a bound; its value is not determined by the data',
    'EXTRA_PEAK_ON_REFLECTION': 'an unindexed extra peak sits on a phase reflection; it may belong to that phase',
    'EXTRA_PEAK_NO_INTENSITY': 'an extra peak fitted to zero; the line is not there at this level',
    'HUMP_TOO_NARROW': 'a hump narrowed to a sharp line; it is probably a peak',
    'RESOLUTION_UNCONSTRAINED': 'profile widths are not determined; quote cells only',
    'RESOLUTION_NOT_POSITIVE': 'profile widths are not physical; quote cells only',
    'COVARIANCE_DIRECTION_DISCARDED': 'some parameter combinations are not determined; quote cells only',
    'LEBAIL_ALTERNATION_STOPPED': 'the intensity/parameter alternation stopped early; the result depends on the start',
    'HIGH_CORRELATION': 'two parameters are strongly correlated; their esds are large together',
    'FLOOR_HIT': 'an intensity or width went to its floor; that component is not needed or not determined',
    'SPACE_GROUP_SETTING_ASSUMED': 'rietx chose the standard setting of the space group',
    'FLAT_DIRECTION': 'the fit is flat along some parameter combination; those parameters are not determined',
    'STRAIN_UNUSUALLY_LARGE': 'a large microstrain; on films it often stands in for size broadening or stacking disorder',
}


class LeBailError(RuntimeError):
    """The fit could not run; the message says why."""


def available():
    try:
        import rietx  # noqa: F401
        return True, ''
    except Exception as e:  # pragma: no cover
        return False, f'The Le Bail fit needs rietx: pip install -r requirements-lebail.txt ({e.__class__.__name__}: {e})'


def _gstar(p):
    a, b, c, al, be, ga = p
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return np.linalg.inv(np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]]))


def _cell6(ph):
    c = ph['cell']
    lat = ph['lattice']
    a = c['a']
    return [a, c.get('b', a), c.get('c', a), c.get('alpha', 90.0), c.get('beta', 90.0), c.get('gamma', 120.0 if lat == 'hexagonal' else 90.0)]


def _rx_phases(ph):
    """[(rietx name, space group, cell6, free axes, held axes, late, role)] for one app phase."""
    s = ph.get('series', 'all')
    lat, cen = ph['lattice'], ph.get('centering', 'P')
    p6 = _cell6(ph)
    out = []
    if s in ('00l', '00l+hk0'):
        nm = ph['name'] if s == '00l' else ph['name'] + ' (00l)'
        out.append((nm, 'P4/mmm', [0.8, 0.8, p6[2], 90, 90, 90], ['c'], ['a'], False, '00l'))
    if s in ('hk0', '00l+hk0'):
        nm = ph['name'] if s == 'hk0' else ph['name'] + ' (hk0)'
        if lat in ('hexagonal',):
            sg, cell, free = 'P6/mmm', [p6[0], p6[0], 0.8, 90, 90, 120], ['a']
        elif lat in ('tetragonal', 'cubic'):
            sg, cell, free = 'P4/mmm', [p6[0], p6[0], 0.8, 90, 90, 90], ['a']
        else:
            sg, cell, free = 'Pmmm', [p6[0], p6[1], 0.8, 90, 90, 90], ['a', 'b']
        out.append((nm, sg, cell, free, ['c'], True, 'hk0'))
    if s == 'all':
        sg = (ph.get('space_group') or '').strip()
        out.append((ph['name'], sg, p6, [ax for ax in AXES[lat]], [], False, 'all'))
    return out


@dataclass
class LeBailResult:
    name: str
    x: np.ndarray
    yobs: np.ndarray
    ycalc: np.ndarray
    ybkg: np.ndarray
    stats: dict
    correction: dict
    cells: Dict[str, dict]
    ticks: List[dict]
    extras: pd.DataFrame
    diagnostics: List[dict]
    settings: dict
    params: pd.DataFrame
    elapsed: float
    notes: List[str] = field(default_factory=list)

    def as_view(self, res):
        """An IndexResult look-alike for views.indexed_view / figures.indexed_figure: Le Bail curves and ticks, indexing labels."""
        from types import SimpleNamespace
        st = SimpleNamespace(x=self.x, yobs=self.yobs, ycal=self.ycalc, bkg=self.ybkg, rwp_step2=self.stats['rwp'], name=self.name)
        return SimpleNamespace(stage1=st, ticks=self.ticks, labels=list(res.labels))

    def fit_csv(self):
        return pd.DataFrame({'twotheta': self.x, 'yobs': self.yobs, 'ycalc': self.ycalc, 'bkg': self.ybkg,
                             'diff': self.yobs - self.ycalc}).to_csv(index=False, float_format='%.6g')

    def reflections_csv(self):
        rows = [dict(phase=t['name'], hkl=h, d=d, two_theta=tt) for t in self.ticks for tt, h, d in t['rows']]
        return pd.DataFrame(rows).to_csv(index=False)

    def table_vs_indexing(self, res):
        rows = []
        for ph in res.phases:
            for nm, info in self.cells.items():
                if info['phase'] != ph['name']:
                    continue
                for ax, d in info['axes'].items():
                    a_idx = res.cells.get(ph['name'], {}).get(ax)
                    lb_v, lb_e = d['value'], d['esd']
                    if d['status'] != 'refined' or a_idx is None or not np.isfinite(a_idx[1]) or lb_e is None:
                        agree = '—'
                    else:
                        z = abs(lb_v - a_idx[0]) / math.hypot(lb_e, a_idx[1])
                        agree = 'agree (< 2σ)' if z < 2 else 'check (2–3σ)' if z < 3 else 'differ (> 3σ)'
                    rows.append({'Phase': nm, 'Axis': SYMBOL[ax],
                                 'Le Bail': (f'{fsu(lb_v, lb_e)} {unit(ax)}' if d['status'] == 'refined' else f'{lb_v:.4f} {unit(ax)} ({d["status"]})'),
                                 'raw rietx esd': fsu(lb_v, d['esd_raw']) if d.get('esd_raw') else '—',
                                 'Peak positions': (fsu(*a_idx) + ' ' + unit(ax)) if a_idx else '—', 'Agreement': agree})
        return pd.DataFrame(rows)


def run(res, quality='fast', weak_percent=3.0, refine_weak=False, two_theta_limits=None, progress=None, cancel=None):
    """Le Bail fit of an IndexResult. progress: callable(text) per stage. Returns LeBailResult."""
    ok, msg = available()
    if not ok:
        raise LeBailError(msg)
    import rietx as rx
    from rietx.schemas.structure import lebail_scaffold

    S = res.stage1
    if not res.phases:
        raise LeBailError('Switch on at least one phase in the indexing tab first.')
    x, y = S.x.astype(float), S.yobs.astype(float)
    lim = tuple(two_theta_limits) if two_theta_limits else (float(x.min()), float(x.max()))
    data = rx.PatternData(two_theta=x.tolist(), intensity=y.tolist())
    inst = rx.Instrument.bragg_brentano(radiation='CuKa', goniometer_radius_mm=float(res.radius))
    conf = res.kbeta[res.kbeta.status.str.startswith('confirmed')]['Kβ/Kα (%)'] if len(res.kbeta) else []
    kbw = float(np.clip(np.median(conf) / 100, 0.003, 0.015)) if len(conf) else 0.008
    inst.source.lines.append(rx.EmissionLine(wavelength=rx.Parameter(value=1.39222), weight=rx.Parameter(value=kbw)))
    inst.background = rx.BackgroundFixedPlusChebyshev(
        fixed_two_theta=x.tolist(), fixed_intensity=S.bkg.tolist(), fixed_source='background of the indexing (' + getattr(S, 'bkg_label', 'step-2') + ')',
        chebyshev=rx.BackgroundChebyshev(coefficients=[rx.Parameter(value=0.0, vary=False)]))
    T = res.table
    un = T[(T['index'] == '—') & (T.profile == 'sharp')]
    hm = T[(T['index'] == 'hump') & (T.FWHM < 2.49) & (T['2θ obs'] > lim[0] + 0.5)]
    extras_meta = []
    for _, r in un.iterrows():
        t0 = float(r['2θ obs'])
        inst.extra_components.append(rx.PeakComponent(
            center=rx.Parameter(value=t0, min=t0 - 0.15, max=t0 + 0.15), fwhm=rx.Parameter(value=float(min(max(r.FWHM, 0.05), 0.4)), min=0.02, max=3.0),
            area=rx.Parameter(value=100.0, min=0.0), eta=rx.Parameter(value=0.5, min=0, max=1), label=f'peak {t0:.2f}'))
        extras_meta.append(('peak', t0))
    for _, r in hm.iterrows():
        t0 = float(r['2θ obs'])
        inst.extra_components.append(rx.HumpComponent(
            position=rx.Parameter(value=t0, min=t0 - 1, max=t0 + 1), fwhm=rx.Parameter(value=float(min(max(r.FWHM, 0.6), 3.0)), min=0.6, max=8.0),
            height=rx.Parameter(value=50.0, min=0.0), label=f'hump {t0:.1f}'))
        extras_meta.append(('hump', t0))

    # phases
    strongest = {}
    for nm, hkl, inten in _assignments(res):
        strongest[nm] = max(strongest.get(nm, 0.0), inten)
    specs, rx_phases, notes = [], [], []
    for pi, ph in enumerate(res.phases):
        weak = strongest.get(ph['name'], 0.0) < weak_percent
        for nm, sg, cell, free, held, late, role in _rx_phases(ph):
            built = None
            for cand in [sg, GENERIC_SG.get((ph['lattice'] if role == 'all' else ('tetragonal' if sg == 'P4/mmm' else 'hexagonal' if sg == 'P6/mmm' else 'orthorhombic'),
                                              ph.get('centering', 'P') if role == 'all' else 'P'))]:
                if not cand:
                    continue
                try:
                    built = lebail_scaffold(cand, cell, name=nm).phases[0]
                    sg = cand
                    break
                except Exception:
                    continue
            if built is None:
                raise LeBailError(f'{nm}: rietx could not build a cell for space group "{sg}". Set a valid space group or lattice.')
            built.lor_size.value = 0.5 if late else LOR0
            hold_cell = weak and not refine_weak and role != 'hk0'
            for ax, i in zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), range(6)):
                q = getattr(built.cell, ax, None)
                if q is None or not hasattr(q, 'min'):
                    continue
                if i < 3:
                    q.min, q.max = q.value * 0.995, q.value * 1.005
                else:
                    q.min, q.max = q.value - 0.5, q.value + 0.5
            if hold_cell:
                notes.append(f'{nm}: its strongest indexed line is {strongest.get(ph["name"], 0):.1f} % of the strongest peak, below '
                             f'{weak_percent:g} %; its cell is kept at the indexed value (Le Bail cannot fix it).')
            specs.append(dict(name=nm, phase=ph['name'], sg=sg, free=[] if hold_cell else free, held=held, late=late, role=role,
                              weak_held=hold_cell, lattice=ph['lattice']))
            rx_phases.append(built)
    # reference phase first (strongest, refined, not late)
    order = sorted(range(len(specs)), key=lambda i: (specs[i]['late'], not specs[i]['free'], -strongest.get(specs[i]['phase'], 0)))
    specs = [specs[i] for i in order]
    rx_phases = [rx_phases[i] for i in order]
    st = rx.Structure(phases=rx_phases)
    ref = rx.Refinement(st, inst, history=False)
    for i, sp in enumerate(specs):
        lat_axes = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        frees = set(sp['free'])
        for ax in lat_axes:
            if ax not in frees:
                try:
                    ref.hold(f'phases.{i}.cell.{ax}')
                except Exception:
                    pass
        if sp['late']:
            try:
                ref.hold(f'phases.{i}.lor_strain')
            except Exception:
                pass
    corr_path = {'displacement': 'instrument.geometry.sample_displacement', 'zero': 'instrument.zero_shift'}.get(res.correction)
    Sg = rx.Stage
    early = [i for i, sp in enumerate(specs) if not sp['late']]
    stages = []
    if corr_path:
        stages.append(Sg('correction', [corr_path]))
    if early and specs[early[0]]['free']:
        stages.append(Sg('first cell', [f'phases.{early[0]}.cell.*']))
    stages += [Sg('widths', ['instrument.profile.w'] + [f'phases.{i}.lor_size' for i in early])]
    if inst.extra_components:
        stages.append(Sg('extra peaks', ['instrument.extra_components.*']))
    if any(specs[i]['free'] for i in early):
        stages.append(Sg('all cells', [f'phases.{i}.cell.*' for i in early if specs[i]['free']]))
    stages.append(Sg('profile', ['instrument.profile.u', 'instrument.profile.v', 'instrument.profile.x', 'instrument.profile.y'] +
                     [f'phases.{i}.lor_strain' for i in early], max_iter=150))
    for i, sp in enumerate(specs):
        if sp['late']:
            stages.append(Sg(f'late {sp["name"]}', ([f'phases.{i}.cell.*'] if sp['free'] else []) + [f'phases.{i}.lor_size'], max_iter=150))
    if quality == 'thorough':
        plan = rx.RefinementPlan(stages=stages, lebail_passes=2)
    else:
        plan = rx.RefinementPlan(stages=stages, lebail_passes=1, intermediate_ftol=1e-4, closing_window_tol=None)

    class _P:
        def write(self, s):
            if progress and s.strip():
                progress(s.strip())

        def flush(self):
            pass
    t0 = time.time()
    try:
        r = ref.fit(data, mode='lebail', plan=plan, two_theta_limits=lim, telemetry=False, progress=_P() if progress else None, cancel=cancel)
    except Exception as e:
        raise LeBailError(f'rietx stopped: {e.__class__.__name__}: {str(e)[:300]}') from None
    elapsed = time.time() - t0
    stt = r.statistics
    dw = float(getattr(stt, 'durbin_watson', float('nan')))
    rho = min(max(1 - dw / 2, 0.0), 0.99) if np.isfinite(dw) else 0.0
    berar = float(math.sqrt((1 + rho) / (1 - rho)))
    stats = dict(rwp=100 * stt.rwp, rexp=100 * stt.rexp, gof=float(stt.gof), dw=dw, berar=berar, status=str(r.status),
                 rwp_indexing=float(S.rwp_step2), rwp_prior=S.rwp_prior)

    def par(path):
        try:
            return r.parameter(path)
        except Exception:
            return None
    cdict = {}
    if corr_path:
        q = par(corr_path)
        if q is not None:
            cdict = dict(kind=res.correction, value=float(q.value), esd=float(q.stderr * berar) if q.stderr else None,
                         unit='mm' if res.correction == 'displacement' else '°')
    cells = {}
    for i, sp in enumerate(specs):
        axes = {}
        for ax in AXES[sp['lattice']] if sp['role'] == 'all' else (['c'] if sp['role'] == '00l' else (['a', 'b'] if 'b' in sp['free'] else ['a'])):
            q = par(f'phases.{i}.cell.{ax}')
            if q is None:          # held parameters are not in the result; report the start value
                from types import SimpleNamespace
                q = SimpleNamespace(value=float(getattr(rx_phases[i].cell, ax).value), stderr=None, at_bound=False)
            if sp['weak_held']:
                status = 'held, weak phase'
            elif ax not in sp['free']:
                status = 'held'
            elif getattr(q, 'at_bound', False):
                status = 'at bound (not determined)'
            else:
                status = 'refined'
            axes[ax] = dict(value=float(q.value), esd=float(q.stderr * berar) if q.stderr and status == 'refined' else None,
                            esd_raw=float(q.stderr) if q.stderr else None, status=status)
        cells[sp['name']] = dict(phase=sp['phase'], sg=sp['sg'], role=sp['role'], axes=axes)
    # tick marks: Kα1 position of each hkl rietx used
    ticks = []
    tt_r, hk_r = getattr(r, 'ticks', {}), getattr(r, 'tick_hkl', {})
    pal = {p['name']: p['color'] for p in res.phases}
    for i, sp in enumerate(specs):
        p6 = [cells[sp['name']]['axes'].get(ax, {}).get('value') for ax in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')]
        base = rx_phases[i].cell
        p6 = [v if v is not None else float(getattr(base, ax).value) for v, ax in zip(p6, ('a', 'b', 'c', 'alpha', 'beta', 'gamma'))]
        if sp['lattice'] == 'hexagonal' or sp['sg'] in ('P6/mmm', 'R-3m') or p6[5] == 120:
            p6[1] = p6[0]
        G = _gstar(p6)
        rows, seen = [], set()
        for t, h in zip(tt_r.get(sp['name'], []), hk_r.get(sp['name'], [])):
            v = np.array(h, float)
            d = 1 / math.sqrt(float(v @ G @ v))
            s1 = L1 / (2 * d)
            if s1 >= 1:
                continue
            t1 = 2 * math.degrees(math.asin(s1))
            key = tuple(int(u) for u in h)
            if abs(t - t1) > 0.6 or key in seen or not (lim[0] <= t <= lim[1]):
                continue
            seen.add(key)
            rows.append((round(float(t), 3), ''.join(str(u) for u in key) if all(-9 <= u <= 9 for u in key) else ','.join(map(str, key)), round(d, 4)))
        ticks.append(dict(name=sp['name'], label=sp['name'], color=pal.get(sp['phase'], '#444444'), rows=sorted(rows),
                          val=', '.join(f"{SYMBOL[ax]} = {fsu(d['value'], d['esd'])} {unit(ax)}" if d['status'] == 'refined' else f"{SYMBOL[ax]} = {d['value']:.4f} {unit(ax)} ({d['status']})"
                                        for ax, d in cells[sp['name']]['axes'].items())))
    # extras
    ex_rows = []
    for k, (kind, t0) in enumerate(extras_meta):
        names = ('center', 'area', 'fwhm') if kind == 'peak' else ('position', 'height', 'fwhm')
        vals = [par(f'instrument.extra_components.{k}.{n}') for n in names]
        pos = vals[0].value if vals[0] is not None else t0
        inten = vals[1].value if vals[1] is not None else float('nan')
        e_int = vals[1].stderr if vals[1] is not None and vals[1].stderr else float('nan')
        w = vals[2].value if vals[2] is not None else float('nan')
        determined = bool(np.isfinite(e_int) and inten > 3 * e_int and (0.021 < w < 2.99 if kind == 'peak' else 0.61 < w < 7.99))
        ex_rows.append({'kind': kind, 'start 2θ': round(t0, 3), '2θ': round(float(pos), 3), 'intensity': round(float(inten), 1),
                        'intensity esd': round(float(e_int * berar), 1) if np.isfinite(e_int) else None,
                        'unit': 'area (counts·°)' if kind == 'peak' else 'height (counts)', 'FWHM': round(float(w), 3), 'determined': determined})
    extras = pd.DataFrame(ex_rows)
    # diagnostics
    diags, seen_codes = [], {}
    for d in getattr(r, 'diagnostics', []) or []:
        code = str(getattr(d, 'code', ''))
        if code in seen_codes:
            seen_codes[code]['count'] += 1
            continue
        seen_codes[code] = dict(level=str(getattr(d, 'level', '')), code=code, count=1, message=str(getattr(d, 'message', ''))[:400],
                                reading=READING.get(code, ''))
        diags.append(seen_codes[code])
    # parameters table (same columns as the indexing's refined-parameters file)
    prow = []
    for nm, info in cells.items():
        for ax, d in info['axes'].items():
            prow.append(dict(part='B (Le Bail)', phase=nm, space_group=info['sg'], parameter=ax, value=d['value'], esd=d['esd'],
                             value_su=fsu(d['value'], d['esd']) if d['esd'] else f"{d['value']:g}", unit=unit(ax),
                             status='refined' if d['status'] == 'refined' else 'fixed' if d['status'].startswith('held') else 'not determined',
                             basis=f"rietx Le Bail ({quality}), indexing background held, Bérar factor ×{berar:.2f}",
                             note=('' if d['status'] == 'refined' else d['status']) + (f"; raw rietx esd {d['esd_raw']:.2g}" if d.get('esd_raw') else '')))
    if cdict:
        prow.append(dict(part='B (Le Bail)', phase='instrument', space_group='', parameter='sample displacement' if cdict['kind'] == 'displacement' else 'zero shift',
                         value=cdict['value'], esd=cdict['esd'], value_su=fsu(cdict['value'], cdict['esd']), unit=cdict['unit'], status='refined',
                         basis='rietx Le Bail', note=''))
    for k, lab in (('rwp', 'Rwp'), ('rexp', 'Rexp'), ('gof', 'GoF'), ('dw', 'Durbin–Watson')):
        prow.append(dict(part='B (Le Bail)', phase='fit', space_group='', parameter=lab, value=round(stats[k], 4), esd=None,
                         value_su=f'{stats[k]:.3f}', unit='%' if k in ('rwp', 'rexp') else '', status='statistic', basis='', note=''))
    params = pd.DataFrame(prow)
    return LeBailResult(name=S.name, x=np.asarray(r.two_theta, float), yobs=np.asarray(r.y_obs, float), ycalc=np.asarray(r.y_calc, float),
                        ybkg=np.asarray(r.y_background, float), stats=stats, correction=cdict, cells=cells, ticks=ticks, extras=extras,
                        diagnostics=diags, settings=dict(quality=quality, kbeta=kbw, weak_percent=weak_percent, refine_weak=refine_weak,
                                                         limits=lim, n_extra_peaks=len(un), n_humps=len(hm)),
                        params=params, elapsed=elapsed, notes=notes)


def _assignments(res):
    """(phase name, hkl, I %) for the indexed lines of the peak table (Kβ copies skipped)."""
    out = []
    names = {p['name'] for p in res.phases}
    for _, r in res.table.iterrows():
        ix = str(r['index'])
        if ix in ('—', 'hump') or ix.startswith('Kβ'):
            continue
        for part in ix.split(';'):
            nm, _, hkl = part.strip().rpartition(' ')
            if nm in names:
                out.append((nm, hkl, float(r['I (%)'])))
    return out
