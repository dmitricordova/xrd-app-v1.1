"""XRD Workup - raw pattern plotting and peak indexing of lab Cu Kα powder patterns.

    pip install -r requirements.txt
    streamlit run app.py
"""
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from xrdcore import __version__, figures, views
from xrdcore.indexing import BKG_MODES, BkgSettings, IndexingError, background_and_peaks, index_pattern
from xrdcore.io import PatternError, load_pattern
from xrdcore import aidex, cifphase, lebail, mpsearch
from xrdcore.phases import (ANGLES, AXES, CENTERINGS, EXAMPLE_PHASES, LATTICES, PALETTE, format_hkl, from_json, library,
                            parse_hkl, to_json, validate)
from xrdcore.phases import SERIES, SERIES_LABEL, SYMBOL, unit
from xrdcore.util import L1, L2, fsu, sub_digits

HERE = Path(__file__).parent
APP_TITLE = 'MaXXXLab Jr XRD Work-up App'
ICON = HERE / 'assets' / 'icon.png'
EXAMPLE = HERE / 'examples' / 'SnSeVSe22_27Aug26.csv'
FILE_TYPES = ['csv', 'xy', 'xye', 'txt', 'dat']

st.set_page_config(page_title=APP_TITLE, page_icon=str(ICON) if ICON.exists() else ':material/ssid_chart:', layout='wide')


# ------------------------------------------------------------------------------------------- cached work
@st.cache_data(show_spinner=False, max_entries=32)
def load(name, data):
    return load_pattern(data, name)


@st.cache_data(show_spinner=False, max_entries=32)
def stage1(name, data, bkg_json='{}'):
    return background_and_peaks(load_pattern(data, name), bkg=BkgSettings(**json.loads(bkg_json)))


@st.cache_data(show_spinner=False, max_entries=64)
def stage2(name, data, phases_json, correction, radius, bkg_json='{}'):
    phases = json.loads(phases_json)['phases']
    return index_pattern(stage1(name, data, bkg_json), phases, correction, radius)


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=16)
def mp_fetch(elements, key_hash, _api_key):
    """Experimentally made MP entries for these elements, cached for a day per key (only a hash of the key is kept)."""
    return mpsearch.fetch_candidates(list(elements), _api_key)


def fig_bytes(make, fmt):
    return figures.to_bytes(make(), fmt)


# ------------------------------------------------------------------------------------------- phases state
COL = {'a': 'a (Å)', 'b': 'b (Å)', 'c': 'c (Å)', 'alpha': 'α (°)', 'beta': 'β (°)', 'gamma': 'γ (°)'}
EDITOR_COLS = ['Use', 'Phase', 'Space group', 'Lattice', 'Centering', 'Reflections'] + list(COL.values()) + ['Label', 'Colour', 'hkl']


def phases_to_df(phases):
    rows = []
    for p in phases:
        c = p['cell']
        row = {'Use': bool(p.get('enabled', True)), 'Phase': p['name'], 'Space group': p.get('space_group', ''),
               'Lattice': p['lattice'], 'Centering': p.get('centering', 'P'), 'Reflections': p.get('series', 'all'),
               'Label': p.get('prefix', p['name'][:2]), 'Colour': p.get('color', PALETTE[len(rows) % len(PALETTE)]),
               'hkl': format_hkl(p.get('hkl') or [])}
        for ax, col in COL.items():
            row[col] = c.get(ax) if ax in AXES[p['lattice']] else None
        rows.append(row)
    return pd.DataFrame(rows, columns=EDITOR_COLS)


def df_to_phases(df):
    out, errors = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        if not str(r.get('Phase') or '').strip() and pd.isna(r.get('a (Å)')):
            continue                                            # empty row the editor added
        lat = str(r.get('Lattice') or 'hexagonal')
        cell = {ax: r.get(COL[ax]) for ax in AXES.get(lat, ['a'])}
        try:
            hkl = parse_hkl(str(r.get('hkl') or ''))
        except ValueError as e:
            errors.append(f'Phase {i + 1} ({r.get("Phase")}): {e}')
            hkl = []
        out.append(dict(name=str(r.get('Phase') or '').strip(), space_group=str(r.get('Space group') or ''), lattice=lat,
                        centering=str(r.get('Centering') or 'P'), series=str(r.get('Reflections') or 'all'), cell=cell, hkl=hkl,
                        color=str(r.get('Colour') or ''),
                        prefix=str(r.get('Label') or ''), enabled=bool(r.get('Use', True))))
    clean, err2 = validate(out)
    return clean, errors + err2


ss = st.session_state
ss.setdefault('phases_df', phases_to_df(EXAMPLE_PHASES))
ss.setdefault('editor_ver', 0)
ss.setdefault('line_tests', {})


def reset_editor(df):
    ss.phases_df = df
    ss.editor_ver += 1


# ------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header('Patterns')
    uploads = st.file_uploader('Add pattern files', type=FILE_TYPES, accept_multiple_files=True,
                               help='Refinement CSV exports (2θ, yobs, ycal, bkg, diff) or two/three-column .xy, .xye, .txt, .dat '
                                    'files. Header lines starting with # are read for column names and the intensity unit.')
    use_example = st.toggle('Include the example pattern', value=not uploads,
                            help='SnSeVSe22_27Aug26.csv: VSe₂ + SnSe₂ film with some V₂O₃, Cu Kα, 5–90°.')
    files = [(u.name, u.getvalue()) for u in (uploads or [])]
    if use_example and EXAMPLE.exists():
        files.insert(0, (EXAMPLE.name, EXAMPLE.read_bytes()))
    good, bad = [], []
    for name, data in files:
        try:
            load(name, data)
            good.append((name, data))
        except PatternError as e:
            bad.append(str(e))
    for msg in bad:
        st.error(msg)
    sample = None
    if good:
        names = [n for n, _ in good]
        pick = st.selectbox('Sample', names, index=0, help='Both tabs show this pattern.')
        sample = dict(good)[pick], pick
    st.header('Instrument')
    st.caption(f'Cu Kα1 {L1} Å, Kα2 {L2} Å (Kα2/Kα1 = 0.5). Kβ 1.39222 Å for the Kβ check.')
    radius = st.number_input('Goniometer radius (mm)', 100.0, 600.0, 240.0, 5.0,
                             help='Only used to turn the displacement K into a sample height error.')
    st.caption(f'{APP_TITLE} {__version__}')

_t = st.columns([0.07, 0.93], vertical_alignment='center')
if ICON.exists():
    _t[0].image(str(ICON), width=72)
_t[1].title(APP_TITLE, anchor=False)
if sample is None:
    st.info('Add a pattern file in the sidebar, or switch on "Include the example pattern", to start.')
    st.stop()

data, name = sample
pat = load(name, data)
stem = pat.stem
st.caption(f'{name} · {len(pat.x)} points · Cu Kα · one background (chosen in the Peak indexing tab) used for all indexing results')

tab_raw, tab_idx, tab_lb, tab_exp, tab_help = st.tabs(['Raw pattern', 'Peak indexing', 'Le Bail', 'Export figures', 'Help'])

# ------------------------------------------------------------------------------------------- raw pattern
with tab_raw:
    s = pat.stats()
    st.subheader(name, anchor=False)
    st.caption('As recorded. Nothing subtracted, smoothed or corrected.')
    m = st.columns(4)
    m[0].metric('2θ range', f"{s['xmin']:.2f}–{s['xmax']:.2f}°", f"{s['n']} points", delta_color='off', delta_arrow='off')
    m[1].metric('Step', f"{s['step']:.4g}°", 'uniform' if s['uniform'] else 'not uniform', delta_color='off', delta_arrow='off')
    m[2].metric('Strongest point', f"{s['ymax']:,.0f}", f"at {s['x_at_max']:.2f}°", delta_color='off', delta_arrow='off')
    m[3].metric('Weakest point', f"{s['ymin']:,.0f}", f"at {s['x_at_min']:.2f}°", delta_color='off', delta_arrow='off')

    c = st.columns([1.3, 1, 1, 3])
    raw_scale = c[0].segmented_control('Intensity scale', ['Linear', 'Log', '√'], default='Linear', key=f'raw_scale_{name}') or 'Linear'
    raw_points = c[1].toggle('Show data points', key=f'raw_pts_{name}')
    raw_pub = c[2].toggle('Publication style', key=f'raw_pub_{name}')
    raw_xr = c[3].slider('2θ range (°)', float(s['xmin']), float(s['xmax']), (float(s['xmin']), float(s['xmax'])), 0.01,
                         key=f'raw_xr_{name}', help='Sets the range of the plot and of the exported figure.')
    sc = {'Linear': 'linear', 'Log': 'log', '√': 'sqrt'}[raw_scale]
    st.plotly_chart(views.raw_view(pat, sc, raw_points, raw_xr, raw_pub, revision=f'{name}{raw_xr}{sc}'),
                    theme=None if raw_pub else 'streamlit', key=f'raw_plot_{name}')

    st.caption('Publication figures of this pattern are made in the Export figures tab.')
    st.download_button('Raw data · CSV', pat.raw_csv(), f'{stem}_raw.csv', 'text/csv', key='raw_csv')

    with st.container(border=True):
        st.markdown('**About this file**')
        roles = ', '.join(pat.columns) if pat.columns else 'none (two columns read as 2θ, intensity)'
        st.markdown(f'- Columns: {roles}. This tab plots the observed intensity only.\n'
                    f'- Intensity unit: **{pat.unit}**' + ('' if pat.unit_stated else ' (not stated in the file; assumed)') + '.\n'
                    + (f'- {pat.skipped_lines} non-numeric or short lines after the data were skipped.\n' if pat.skipped_lines else '')
                    + ('- The 2θ values were not in increasing order; they were sorted.\n' if pat.sorted_input else ''))
        if pat.unit_is_rate:
            st.warning('The file stores a count rate. Weights (1/√y), Rwp and every esd in the indexing tab assume counts; '
                       'multiply by the counting time per step for counting-statistics weights.')
        if pat.header:
            st.code('\n'.join(pat.header[:20]), language=None)

# ------------------------------------------------------------------------------------------- help
with tab_help:
    st.markdown((HERE / 'HELP.md').read_text(encoding='utf-8') if (HERE / 'HELP.md').exists() else 'See README.md.')

# ------------------------------------------------------------------------------------------- peak indexing
with tab_idx:
    # ------------------------------------------------------------------ background
    st.subheader('Background', anchor=False)
    has_file_bkg = pat.bkg is not None
    modes = ['step2'] + (['file', 'file+poly'] if has_file_bkg else [])
    bc = st.columns([2.2, 1, 1.1, 1.6])
    b_mode = bc[0].selectbox('Background', modes, format_func=lambda m: BKG_MODES[m], key=f'bkg_mode_{name}',
                             help='Refit: an N-term Chebyshev polynomial (+ four low-angle exponential decays) fitted to the data '
                                  '(to yobs minus the prior peak model when the file has one, otherwise to a rolling 5th percentile). '
                                  'From the file: the bkg column of the input file, as it is. From the file + correction: that column plus '
                                  'a Chebyshev correction fitted to what it misses.')
    if not has_file_bkg:
        bc[0].caption('This file has no background column, so only the refit is offered.')
    b_order = bc[1].number_input('Chebyshev terms', 1, 60, 24 if b_mode == 'step2' else 8, 1, key=f'bkg_order_{name}_{b_mode}',
                                 disabled=b_mode == 'file',
                                 help='Number of polynomial terms; for "file + correction" the terms of the correction only.')
    bc[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    b_decays = bc[2].toggle('Low-angle decays', True, key=f'bkg_decays_{name}', disabled=b_mode == 'file',
                            help='Four exponential decays (0.3, 0.6, 1.2, 2.5°) for the steep rise at low angle (air scatter, beam spill).')
    bc[3].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    b_refine = bc[3].toggle('Refine with the peaks', False, key=f'bkg_refine_{name}', disabled=b_mode == 'file',
                            help='Refines the polynomial and every fitted peak together over the whole pattern (weighted least squares, '
                                 'analytic Jacobian) before indexing; the refined peaks are the ones indexed. A few seconds to ~40 s.')
    bkg_set = BkgSettings(mode=b_mode, order=int(b_order), decays=bool(b_decays), refine=bool(b_refine))
    bkg_json = json.dumps(bkg_set.__dict__, sort_keys=True)

    st.subheader('Phases', anchor=False)
    st.caption('Each phase\'s listed reflections are matched to the fitted peaks; all cells and one angle correction are refined together. '
               'hkl: three digits per reflection (101, 10-2) or commas (1,0,10); "auto" lists every reflection the centering allows. '
               'Lines claimed by two phases are left out of both cells.')
    edited = st.data_editor(
        ss.phases_df, key=f'phase_editor_{ss.editor_ver}', num_rows='dynamic', hide_index=True,
        column_config={
            'Use': st.column_config.CheckboxColumn(width='small'),
            'Phase': st.column_config.TextColumn(required=True, width='small'),
            'Space group': st.column_config.TextColumn(width='small'),
            'Lattice': st.column_config.SelectboxColumn(options=LATTICES, required=True, width='small'),
            'Centering': st.column_config.SelectboxColumn(options=CENTERINGS, required=True, width='small',
                                                          help='R = rhombohedral in the hexagonal setting (−h+k+l = 3n)'),
            'a (Å)': st.column_config.NumberColumn(format='%.4f', min_value=0.5, max_value=100.0, width='small'),
            'b (Å)': st.column_config.NumberColumn(format='%.4f', min_value=0.5, max_value=100.0, width='small', help='orthorhombic, monoclinic, triclinic'),
            'c (Å)': st.column_config.NumberColumn(format='%.4f', min_value=0.5, max_value=100.0, width='small', help='not for cubic'),
            'α (°)': st.column_config.NumberColumn(format='%.3f', min_value=60.0, max_value=150.0, width='small', help='triclinic only'),
            'β (°)': st.column_config.NumberColumn(format='%.3f', min_value=60.0, max_value=150.0, width='small', help='monoclinic (unique axis b) and triclinic'),
            'γ (°)': st.column_config.NumberColumn(format='%.3f', min_value=60.0, max_value=150.0, width='small', help='triclinic only'),
            'Label': st.column_config.TextColumn(width='small', help='Short prefix for peak labels, e.g. V for V101'),
            'Colour': st.column_config.TextColumn(width='small', help='Hex colour, e.g. #1f3a93'),
            'Reflections': st.column_config.SelectboxColumn(options=SERIES, required=True, width='small',
                                                            help='all, 00l only (stacking period), hk0 only (in-plane cell; may match '
                                                                 'broad turbostratic bands), or 00l + hk0'),
            'hkl': st.column_config.TextColumn(width='large'),
        })
    lib = library()
    lib_names = [p['name'] for p in lib]
    r1 = st.columns([2, 1.2, 1, 2, 1, 1])
    add = r1[0].selectbox('Add a phase from the library', lib_names, index=None, placeholder='Choose a phase…', key='lib_add',
                          help='Literature starting cells; hkl lists are the reflections with ≥ 2 % calculated intensity.')
    lib_series = r1[1].selectbox('Reflections', SERIES, format_func=lambda v: SERIES_LABEL[v], key='lib_series',
                                 help='Use only its 00l lines (stacking period), only its hk0 lines (in-plane cell), or both.')
    r1[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if r1[2].button('Add phase', disabled=add is None, width='stretch'):
        p = dict(next(q for q in lib if q['name'] == add), series=lib_series)
        cur = edited.copy()
        if lib_series != 'all':
            p['name'] = f"{add} ({SERIES_LABEL[lib_series]})"
        if p['name'] in set(cur['Phase']):
            st.toast(f"{p['name']} is already in the list.")
        else:
            reset_editor(pd.concat([cur, phases_to_df([p])], ignore_index=True))
            st.rerun()
    r1 = [None, None] + list(r1[3:])          # uploader, example set, clear sit in the last three columns
    up = r1[2].file_uploader('Load phases (.json)', type=['json'], key=f'phase_upload_{ss.editor_ver}', label_visibility='visible')
    if up is not None:
        ph, errs = from_json(up.getvalue().decode('utf-8', errors='replace'))
        if errs:
            for e in errs:
                st.error(e)
        else:
            reset_editor(phases_to_df(ph))
            st.rerun()
    if r1[3].button('Example set', width='stretch', help='Back to VSe₂, SnSe₂ and V₂O₃.'):
        reset_editor(phases_to_df(EXAMPLE_PHASES))
        st.rerun()
    if r1[4].button('Clear', width='stretch', help='Remove every phase.'):
        reset_editor(pd.DataFrame(columns=EDITOR_COLS))
        st.rerun()

    with st.expander('Add a 00l or hk0 series (a phase known only by its stacking period or in-plane cell)'):
        st.caption('For layered and misfit films: a 00l series refines only c (a sharp superlattice), an hk0 series only the in-plane '
                   'cell (often broad turbostratic bands, which it may match within half their width). Reflections are generated '
                   'automatically; the axes the series cannot see are held and reported as not determined.')
        sc_ = st.columns([1.6, 1.6, 1, 1, 1])
        s_kind = sc_[0].selectbox('Series', ['00l superlattice', 'hexagonal hk0', 'tetragonal hk0', 'rectangular hk0'], key='ser_kind')
        s_name = sc_[1].text_input('Name', value={'00l superlattice': '00l superlattice', 'hexagonal hk0': 'Hexagonal hk0',
                                                  'tetragonal hk0': 'Tetragonal hk0', 'rectangular hk0': 'Rectangular hk0'}[s_kind], key=f'ser_name_{s_kind}')
        if s_kind == '00l superlattice':
            s_c = sc_[2].number_input('c (Å)', 2.0, 100.0, 12.0, 0.01, format='%.4f', key='ser_c')
            s_a = s_b = None
        else:
            s_a = sc_[2].number_input('a (Å)', 1.0, 50.0, 3.4 if s_kind == 'hexagonal hk0' else 6.0, 0.001, format='%.4f', key='ser_a')
            s_b = sc_[3].number_input('b (Å)', 1.0, 50.0, 4.0, 0.001, format='%.4f', key='ser_b') if s_kind == 'rectangular hk0' else None
            s_c = None
        sc_[4].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if sc_[4].button('Add series', width='stretch', key='ser_add'):
            if s_kind == '00l superlattice':
                newp = dict(name=s_name, lattice='tetragonal', centering='P', series='00l', cell={'a': 3.0, 'c': s_c}, prefix='')
            elif s_kind == 'hexagonal hk0':
                newp = dict(name=s_name, lattice='hexagonal', centering='P', series='hk0', cell={'a': s_a, 'c': 10.0}, prefix='H')
            elif s_kind == 'tetragonal hk0':
                newp = dict(name=s_name, lattice='tetragonal', centering='P', series='hk0', cell={'a': s_a, 'c': 10.0}, prefix='T')
            else:
                newp = dict(name=s_name, lattice='orthorhombic', centering='P', series='hk0', cell={'a': s_a, 'b': s_b, 'c': 10.0}, prefix='R')
            cur = edited.copy()
            newp.update(hkl=[], color=PALETTE[len(cur) % len(PALETTE)], enabled=True)
            if s_name in set(cur['Phase'].astype(str)):
                st.error(f'{s_name} is already in the list; give the series another name.')
            else:
                reset_editor(pd.concat([cur, phases_to_df([newp])], ignore_index=True))
                st.rerun()

    phases, errors = df_to_phases(edited)
    for e in errors:
        st.error(e)
    rc = st.columns([2.8, 1, 1.6, 2.6])
    corr_label = rc[0].segmented_control('Angle correction', ['Sample displacement', 'Zero shift', 'None'], default='Sample displacement',
                                         help='Displacement: 2θobs = 2θcalc + K·cosθ. Zero shift: 2θobs = 2θcalc + z. '
                                              'With few lines the two are strongly correlated; fit only one.') or 'Sample displacement'
    correction = {'Sample displacement': 'displacement', 'Zero shift': 'zero', 'None': 'none'}[corr_label]
    rc[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    auto = rc[2].toggle('Refine automatically', key='auto_refine', help='Refit after every edit instead of waiting for Refine.')
    if errors:
        st.stop()
    edit_json = json.dumps({'phases': phases}, sort_keys=True)
    if 'applied' not in ss or auto:
        ss.applied = (edit_json, correction, bkg_json)
    if len(ss.applied) == 2:
        ss.applied = (*ss.applied, bkg_json)
    pending = ss.applied != (edit_json, correction, bkg_json)
    rc[1].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if rc[1].button('Refine', type='primary', disabled=not pending, width='stretch', icon=':material/play_arrow:',
                    help='Refit the peaks, cells and angle correction with the phase list and correction above.'):
        ss.applied = (edit_json, correction, bkg_json)
        st.rerun()
    if pending:
        rc[3].warning('Edits not refined yet: the results below are for the last refined background and phase list. Click **Refine**.', icon=':material/pending:')
    phases_json, correction, bkg_applied = ss.applied
    try:
        BkgSettings(**json.loads(bkg_applied))
        if json.loads(bkg_applied).get('mode') != 'step2' and not has_file_bkg:
            raise ValueError
    except Exception:
        bkg_applied = json.dumps(BkgSettings().__dict__, sort_keys=True)

    try:
        with st.spinner('Fitting the background and peaks…'):
            res = stage2(name, data, phases_json, correction, float(radius), bkg_applied)
    except IndexingError as e:
        st.error(f'The indexing could not run: {e}')
        st.stop()
    S = res.stage1
    for w in res.warnings:
        st.warning(w)

    st.subheader('Result', anchor=False)
    m = st.columns(4)
    m[0].metric('Rwp, background + peaks', f'{S.rwp_step2:.2f} %',
                f'prior refinement {S.rwp_prior:.2f} %' if S.rwp_prior is not None else 'no prior model in the file', delta_color='off', delta_arrow='off')
    if correction == 'displacement' and res.corr_fitted:
        m[1].metric('Displacement K', f'{fsu(*res.corr)}°', f'≈ {fsu(*res.height)} mm height error', delta_color='off', delta_arrow='off')
    elif correction == 'zero' and res.corr_fitted:
        m[1].metric('Zero shift', f'{fsu(*res.corr)}°', 'K·cosθ model not used', delta_color='off', delta_arrow='off')
    else:
        m[1].metric('Angle correction', 'none', 'held at 0', delta_color='off', delta_arrow='off')
    m[2].metric('Peaks found', f'{len(S.peaks)}', f'{len(res.unindexed)} sharp unindexed', delta_color='off', delta_arrow='off')
    m[3].metric('Joint cell fit', f'rms {res.rms:.4f}°' if np.isfinite(res.rms) else '—', f'{res.n_lines} lines', delta_color='off', delta_arrow='off')

    bh = S.bkg_history
    btxt = f'Background: {S.bkg_label}.'
    if len(bh) > 1:
        jh = bh[-1]
        btxt += (f" Joint refinement of the background and {jh['peaks']} peaks: Rwp {bh[0]['rwp']:.2f} → {jh['rwp']:.2f} % "
                 f"(background + fitted peaks), background moved {jh['shift_sigma']:.2f} σ (rms)"
                 + ('' if jh.get('converged', True) else '; stopped at the evaluation limit, not fully converged') + '.')
    st.caption(btxt + ' ' + S.calc_source[0].upper() + S.calc_source[1:] + ' give the Rwp above.')
    c = st.columns([1.3, 4])
    idx_scale = c[0].segmented_control('Intensity scale', ['Log', 'Linear', '√'], default='Log', key=f'idx_scale_{name}') or 'Log'
    tg = c[1].columns(5)
    show_labels = tg[0].toggle('Peak labels', True, key='idx_labels')
    show_calc = tg[1].toggle('Calculated', True, key='idx_calc')
    show_bkg = tg[2].toggle('Background', True, key='idx_bkg')
    show_diff = tg[3].toggle('Difference', True, key='idx_diff')
    idx_pub = tg[4].toggle('Publication style', False, key='idx_pub')
    row_names = [t['name'] for t in res.ticks]
    row_label = {t['name']: t['label'] for t in res.ticks}
    vis = st.pills('Tick rows', row_names, selection_mode='multi', default=row_names, format_func=lambda n: row_label[n],
                   key=f'idx_rows_{name}_{"|".join(row_names)}') or []
    xmin, xmax = float(S.x.min()), float(S.x.max())
    idx_xr = st.slider('2θ range (°)', xmin, xmax, (xmin, xmax), 0.01, key=f'idx_xr_{name}',
                       help='Sets the range of the plot and of the exported figure. Click a peak in the table to zoom to it.')
    zoom_key = f'zoom_{name}'
    zoom = ss.get(zoom_key)
    xr_plot = idx_xr
    if zoom is not None:
        xr_plot = (max(xmin, zoom - 1.5), min(xmax, zoom + 1.5))
        zc = st.columns([4, 1])
        zc[0].caption(f'Zoomed to the peak at {zoom:.3f}° (±1.5°).')
        if zc[1].button('Show the full range', width='stretch'):
            ss[zoom_key] = None
            st.rerun()
    sc = {'Linear': 'linear', 'Log': 'log', '√': 'sqrt'}[idx_scale]
    fig = views.indexed_view(res, sc, xr_plot, set(vis), show_labels, show_calc, show_bkg, show_diff, idx_pub, highlight=zoom,
                             revision=f'{name}{xr_plot}{sc}{correction}')
    st.plotly_chart(fig, theme=None if idx_pub else 'streamlit', key=f'idx_plot_{name}')
    st.caption('Labels: phase label + hkl (V101), β = Cu Kβ copy, * = unindexed. Hover a tick for its hkl, d and 2θ.')

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown('##### Lattice constants')
        corr_txt = {'displacement': f"Sample displacement K = {fsu(*res.corr)}° (2θobs = 2θcalc + K·cosθ)",
                    'zero': f"Zero shift z = {fsu(*res.corr)}°", 'none': 'No angle correction.'}[correction]
        if correction != 'none' and not res.corr_fitted:
            corr_txt = 'Angle correction held at 0 (too few lines).'
        st.caption(corr_txt + (f'; height error ≈ {fsu(*res.height)} mm for R = {radius:g} mm.' if res.height else '.'))
        cells_rows = []
        for ph in res.phases:
            nm = ph['name']
            for ax in AXES[ph['lattice']]:
                v = res.cells.get(nm, {}).get(ax)
                va = res.cells_all.get(nm, {}).get(ax)
                cells_rows.append({'Phase': sub_digits(nm), 'Axis': SYMBOL[ax],
                                   'Value': (fsu(*v) if v else f"{ph['cell'][ax]:g} (held)") + ' ' + unit(ax),
                                   'With shared lines': fsu(*va) if va else '—',
                                   'Lines': f"{res.used.get(nm, 0)} ({res.matched[nm][0]}/{res.matched[nm][1]} matched)",
                                   'Note': res.held.get(nm, {}).get(ax, '')})
        if cells_rows:
            st.dataframe(pd.DataFrame(cells_rows), hide_index=True)
        else:
            st.caption('No phases switched on.')
        if res.shared:
            st.caption('Shared lines left out: ' + ', '.join(f'{t:.3f}°' for t in res.shared))
    with right:
        st.markdown(f'##### Peaks ({len(res.table)})')
        sel = st.dataframe(res.table, hide_index=True, height=380, on_select='rerun', selection_mode='single-row',
                           key=f'peak_table_{name}_{correction}',
                           column_config={'note': st.column_config.TextColumn(width='large'), 'index': st.column_config.TextColumn(width='medium')})
        rows_sel = sel.selection.rows if sel and sel.selection else []
        if rows_sel:
            t_sel = float(res.table.iloc[rows_sel[0]]['2θ obs'])
            if ss.get(zoom_key) != t_sel:
                ss[zoom_key] = t_sel
                st.rerun()

    with st.expander(f'Cu Kβ check ({int(res.kbeta.status.str.startswith("confirmed").sum()) if len(res.kbeta) else 0} confirmed)'):
        st.caption('Every reflection strong enough to leave a detectable Kβ copy (Kβ/Kα ≈ 0.4–1.5 % with a Ni filter) is fitted directly at '
                   'its Kβ position. Confirmed = height > 3σ and 0.3 % < Kβ/Kα < 3 %.')
        st.dataframe(res.kbeta, hide_index=True)
        st.caption(f"W Lα of the strongest line at {res.wla['pos']:.3f}°: {res.wla['pct']:.3f} ± {res.wla['epct']:.3f} % of the strongest peak.")

    with st.expander('Line tests: is a line from a suspected phase there?'):
        st.caption('Fits a line at a fixed 2θ beside the known peaks of the window and reports its height with an esd. '
                   'Positions are on the observed scale (the angle correction is already in them).')
        lt = st.columns([2, 2, 1])
        pos_txt = lt[0].text_input('2θ positions (°)', placeholder='30.38 26.52', key='lt_pos')
        test_phase = lt[1].selectbox('or every listed line of', ['—'] + [p['name'] for p in res.phases] + [n for n in lib_names if n not in {p['name'] for p in res.phases}],
                                     key='lt_phase', help='Library phases use their literature cells with this result\'s angle correction.')
        if lt[2].button('Run tests', width='stretch'):
            tests = []
            try:
                tests = [float(v) for v in pos_txt.replace(',', ' ').split()] if pos_txt.strip() else []
            except ValueError:
                st.error('Write the positions as numbers separated by spaces, e.g. 30.38 26.52.')
            out = []
            if tests:
                out.append(res.test_positions(tests).assign(source='entered'))
            if test_phase != '—':
                ph = next((p for p in res.phases if p['name'] == test_phase), None) or validate([next(q for q in lib if q['name'] == test_phase)])[0][0]
                pred = res.predicted(ph)
                if pred:
                    t = res.test_positions([p[0] for p in pred])
                    t.insert(1, 'hkl', [p[1] for p in pred])
                    out.append(t.assign(source=test_phase))
            if out:
                ss.line_tests[name] = pd.concat(out, ignore_index=True)
        if name in ss.line_tests:
            st.dataframe(ss.line_tests[name], hide_index=True)
            st.download_button('Line tests · CSV', ss.line_tests[name].to_csv(index=False), f'{stem}_line_tests.csv', 'text/csv', key='lt_csv')

    with st.expander('Report'):
        st.code(res.report, language=None)

    # --------------------------------------------------------------------------------- Materials Project candidates
    st.subheader('Candidate phases from the Materials Project', anchor=False)
    mp_ok, mp_msg = mpsearch.available()
    if not mp_ok:
        st.info(mp_msg)
    else:
        st.caption('Searches every chemical system made of the elements you give (each element alone, every pair, and so on) for '
                   'entries the Materials Project marks as experimentally made, then tests each structure against this pattern. '
                   'Its cell is refined to the peaks first, because DFT cells are typically 1–6 % too large.')
        env_key = os.environ.get('MP_API_KEY', '')
        if not env_key:
            try:
                env_key = st.secrets.get('MP_API_KEY', '')
            except Exception:
                env_key = ''
        suggested = ' '.join(mpsearch.elements_of([p['name'] for p in res.phases]))
        mc = st.columns([2.2, 2.2, 1])
        el_txt = mc[0].text_input('Elements likely present', value=suggested, key=f'mp_elements_{name}',
                                  placeholder='e.g. Sn V Se O',
                                  help='Element symbols separated by spaces or commas. Include O if the film saw air, and the '
                                       'elements of transport agents, precursors or the substrate if they could react.')
        if env_key:
            mp_key = env_key
            mc[1].caption('Using the API key set on the server (MP_API_KEY).')
        else:
            mp_key = mc[1].text_input('Materials Project API key', type='password', key='mp_key',
                                      help='Free at next-gen.materialsproject.org → your profile → API key. '
                                           'Kept only for this browser session and sent only to the Materials Project.')
        mc[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        go = mc[2].button('Search', type='primary', width='stretch', key='mp_go')
        mp_modes = st.multiselect('Intensity models to screen', list(mpsearch.MODES), default=['powder', '00l'], key='mp_modes',
                                  format_func=lambda m: mpsearch.MODE_LABEL[m],
                                  help='random powder: all reflections. 00l only: a strongly c-oriented film or a stacking series. '
                                       'hk0 only: in-plane reflections, e.g. turbostratic bands (matched within half their width). '
                                       '00l + hk0: an oriented film that also shows its in-plane bands. Each extra model adds time.')
        els, el_err = mpsearch.parse_elements(el_txt)
        if els and not el_err:
            st.caption(f'{len(els)} element(s): {", ".join(els)} → {2 ** len(els) - 1} chemical systems.')
        if go:
            if el_err:
                for e in el_err:
                    st.error(e)
            elif not mp_key:
                st.error('Enter your Materials Project API key first.')
            elif not mp_modes:
                st.error('Choose at least one intensity model.')
            else:
                try:
                    with st.status('Searching the Materials Project…', expanded=True) as status:
                        cands = mp_fetch(tuple(els), hashlib.sha256(mp_key.encode()).hexdigest(), mp_key)
                        status.write(f'{len(cands)} experimentally made entries found. Screening them against this pattern…')
                        bar = st.progress(0.0)
                        table = mpsearch.screen(res, cands, modes=tuple(mp_modes), progress=lambda n, N, c: bar.progress(
                            min(n / max(N, 1), 1.0), text=f'{n} of {N}' + (f' · {c["formula"]} {c["sg"]}' if c else '')))
                        status.update(label=f'Screened {len(cands)} entries.', state='complete', expanded=False)
                    ss.mp_result = dict(sample=name, phases_json=phases_json, correction=correction, elements=els,
                                        n=len(cands), table=table)
                except mpsearch.MPSearchError as e:
                    st.error(str(e))
        mr = ss.get('mp_result')
        if mr and mr['sample'] == name:
            D = mr['table']
            if mr['phases_json'] != phases_json or mr['correction'] != correction:
                st.warning('These results were screened against an earlier phase list or correction. Search again to update them.')
            st.markdown(f"{mr['n']} experimentally made entries in the {', '.join(mr['elements'])} systems were screened.")
            if not len(D):
                st.info('No entry gave enough lines in the scanned range to test.')
            else:
                tier = D.verdict.map(lambda v: 'new' if v in ('explains unindexed lines', 'possible, check')
                                     else 'same' if v.startswith('same as') else 'adds' if v.startswith('consistent') else 'no')
                link_cfg = {'link': st.column_config.LinkColumn('Materials Project', display_text='open'),
                            'reason': st.column_config.TextColumn(width='large')}
                new = D[tier == 'new']
                if len(new):
                    st.markdown('**Candidates for the unindexed lines**')
                    st.dataframe(mpsearch.display_table(new), hide_index=True, column_config=link_cfg)
                else:
                    st.success('No entry explains two or more unindexed lines better than random lines would.')
                for t_, label in (('same', 'Same compound as a phase in your list'), ('adds', 'Consistent with the pattern but adds nothing'),
                                  ('no', 'Not supported')):
                    part = D[tier == t_]
                    if len(part):
                        with st.expander(f'{label} ({len(part)})'):
                            st.dataframe(mpsearch.display_table(part), hide_index=True, column_config=link_cfg)
                opts = list(D.index)
                pick = st.multiselect('Add to the phase list', opts, key=f'mp_pick_{name}',
                                      format_func=lambda i: f"{D.loc[i, 'formula']} {D.loc[i, 'sg']} ({D.loc[i, 'mpid']}) · {D.loc[i, 'verdict']}",
                                      help='Adds the phase with its cell refined to this pattern and its reflections of ≥ 2 % '
                                           'calculated intensity; the joint fit then refines it with the other phases.')
                ac = st.columns([1.4, 1, 1, 1.6])
                add_series = ac[0].selectbox('Reflections for added phases', ['as screened'] + SERIES, key='mp_add_series',
                                             format_func=lambda v: v if v == 'as screened' else SERIES_LABEL[v],
                                             help='"as screened" keeps the family of the intensity model each entry matched best '
                                                  '(e.g. 00l only for an oriented-film match).')
                ac[1].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                ac[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                if ac[1].button('Add selected phases', disabled=not pick, width='stretch', key='mp_add'):
                    cur = edited.copy()
                    names_now = list(cur['Phase'].astype(str))
                    added = []
                    for n_, i in enumerate(pick):
                        ph = mpsearch.candidate_phase(D.loc[i], names_now, len(names_now) + n_, x_range=(float(S.x.min()), float(S.x.max())),
                                                      series=None if add_series == 'as screened' else add_series)
                        names_now.append(ph['name'])
                        added.append(ph)
                    reset_editor(pd.concat([cur, phases_to_df(added)], ignore_index=True))
                    st.toast('Added ' + ', '.join(p['name'] for p in added) + ' to the phase list.')
                    st.rerun()
                ac[2].download_button('Candidates · CSV', mpsearch.display_table(D).to_csv(index=False),
                                      f"{stem}_mp_candidates.csv", 'text/csv', key='mp_csv', width='stretch')
                st.caption('Structures and energies: Materials Project (CC BY 4.0). Cell changes are relative to the DFT cell; '
                           'chance P is the share of 25 random line sets that the entry fits as well.')



    # --------------------------------------------------------------------------------- CIF candidates
    st.subheader('Candidate phases from CIF files', anchor=False)
    cif_ok, cif_msg = cifphase.available()
    if not cif_ok:
        st.info(cif_msg)
    else:
        st.caption('Upload crystal structures (.cif) of phases you suspect, e.g. from ICSD, COD or your own refinement. Each structure is '
                   'tested against this pattern the same way as the Materials Project entries: its cell is refined to the peaks and its '
                   'calculated pattern is compared with the lines found. You can also add a CIF directly with its own cell.')
        cif_files = st.file_uploader('CIF files', type=['cif'], accept_multiple_files=True, key=f'cif_up_{name}',
                                     help='One phase per data block. Symmetry, cell and atom sites (with occupancies) are read with pymatgen.')
        cif_in = [(u.name, u.getvalue()) for u in (cif_files or [])]
        cif_sig = hashlib.md5(b''.join(n.encode() + d for n, d in cif_in)).hexdigest() if cif_in else ''
        if cif_in:
            if ss.get('cif_read', {}).get('sig') != cif_sig:
                cc, ce = cifphase.read_cifs(cif_in)
                ss.cif_read = dict(sig=cif_sig, cands=cc, errors=ce)
            CR = ss.cif_read
            for e_ in CR['errors']:
                st.warning(e_)
            if CR['cands']:
                st.dataframe(pd.DataFrame([{'file': c['file'], 'formula': c['formula'], 'space group': c['sg'], 'crystal system': c['system'],
                                            'a (Å)': round(c['structure'].lattice.a, 4), 'b (Å)': round(c['structure'].lattice.b, 4),
                                            'c (Å)': round(c['structure'].lattice.c, 4), 'α': round(c['structure'].lattice.alpha, 2),
                                            'β': round(c['structure'].lattice.beta, 2), 'γ': round(c['structure'].lattice.gamma, 2),
                                            'atoms (conv. cell)': len(c['structure'])} for c in CR['cands']]), hide_index=True)
                cm = st.columns([2.6, 1, 1.4])
                cif_modes = cm[0].multiselect('Intensity models to screen', list(mpsearch.MODES), default=['powder', '00l'], key='cif_modes',
                                              format_func=lambda m: mpsearch.MODE_LABEL[m],
                                              help='random powder: all reflections. 00l only: a strongly c-oriented film. hk0 only: in-plane '
                                                   'reflections (turbostratic bands). 00l + hk0: both.')
                cm[1].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                if cm[1].button('Screen', type='primary', width='stretch', key='cif_go', disabled=not cif_modes):
                    try:
                        with st.spinner(f"Screening {len(CR['cands'])} structure(s) against this pattern…"):
                            ss.cif_result = dict(sample=name, sig=cif_sig, phases_json=phases_json, correction=correction,
                                                 table=mpsearch.screen(res, CR['cands'], modes=tuple(cif_modes)))
                    except mpsearch.MPSearchError as e_:
                        st.error(str(e_))
                cr = ss.get('cif_result')
                screened = bool(cr and cr['sample'] == name and cr['sig'] == cif_sig)
                if screened:
                    Dc = cr['table']
                    if cr['phases_json'] != phases_json or cr['correction'] != correction:
                        st.warning('These structures were screened against an earlier phase list or correction. Screen again to update.')
                    if not len(Dc):
                        st.info('No structure gave enough lines in the scanned range to test.')
                    else:
                        Tc = mpsearch.display_table(Dc).rename(columns={'mp-id': 'file', 'cell change (%)': 'cell change from CIF (%)'})
                        Tc = Tc.drop(columns=['link', 'E above hull (eV/atom)'])
                        st.dataframe(Tc, hide_index=True, column_config={'reason': st.column_config.TextColumn(width='large')})
                        st.download_button('CIF screen · CSV', Tc.to_csv(index=False), f'{stem}_cif_candidates.csv', 'text/csv', key='cif_csv')
                labels_c = {c['mpid']: f"{c['file']} · {c['formula']} {c['sg']}" for c in CR['cands']}
                if screened and len(cr['table']):
                    vmap = dict(zip(cr['table'].mpid, cr['table'].verdict))
                    labels_c = {k_: v + (f' · {vmap[k_]}' if k_ in vmap else ' · not screened (too few lines)') for k_, v in labels_c.items()}
                pick_c = st.multiselect('Add to the phase list', list(labels_c), key=f'cif_pick_{name}_{cif_sig[:8]}', format_func=lambda k_: labels_c[k_])
                ac = st.columns([1.4, 1.6, 1])
                cif_series = ac[0].selectbox('Reflections', ['as screened'] + SERIES, key='cif_series',
                                             format_func=lambda v: v if v == 'as screened' else SERIES_LABEL[v],
                                             help='"as screened" keeps the family of the intensity model the structure matched best; '
                                                  'without a screen it means all reflections.')
                cell_opts = ['refined', 'cif'] if screened else ['cif']
                cif_cell = ac[1].selectbox('Starting cell', cell_opts, key=f'cif_cell_{screened}',
                                           format_func=lambda v: 'refined to the peaks (from the screen)' if v == 'refined' else 'as in the CIF',
                                           help='The joint cell fit refines the cell either way; this is only where it starts.')
                ac[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                if ac[2].button('Add selected', disabled=not pick_c, width='stretch', key='cif_add'):
                    cur = edited.copy()
                    names_now = list(cur['Phase'].astype(str))
                    added, problems = [], []
                    xr_ = (float(S.x.min()), float(S.x.max()))
                    rows_ = cr['table'].set_index('mpid') if screened and len(cr['table']) else None
                    for k_ in pick_c:
                        cand = next(c for c in CR['cands'] if c['mpid'] == k_)
                        ser = None if cif_series == 'as screened' else cif_series
                        try:
                            if rows_ is not None and k_ in rows_.index:
                                row_ = rows_.loc[k_].copy()
                                row_['mpid'] = k_
                                ph = mpsearch.candidate_phase(row_, names_now, len(names_now), x_range=xr_, series=ser,
                                                              start_cell=(cif_cell == 'cif'))
                            else:
                                ph = cifphase.direct_phase(cand, names_now, len(names_now), x_range=xr_, series=ser or 'all')
                        except cifphase.CIFError as e_:
                            problems.append(str(e_))
                            continue
                        names_now.append(ph['name'])
                        added.append(ph)
                    for pmsg in problems:
                        st.error(pmsg)
                    if added:
                        reset_editor(pd.concat([cur, phases_to_df(added)], ignore_index=True))
                        st.toast('Added ' + ', '.join(p['name'] for p in added) + ' to the phase list. Click Refine to fit it.')
                        st.rerun()

    # --------------------------------------------------------------------------------- AIdex-r2
    st.subheader('Unit cell from AIdex-r2', anchor=False)
    if st.toggle('Show the AIdex-r2 unit-cell tool', value=False, key='show_aidex',
                 help='Prepares the unindexed lines for AIdex-r2 and checks the cells it returns.'):
        st.caption('AIdex-r2 predicts a unit cell from up to 20 low-angle line positions of one phase and returns five candidates. '
                   'It is a cell finder, not a phase identifier. Every cell it returns is checked here against the lines: refined, '
                   'misfit, line density, de Wolff M(n), and how often random lines fit as well. ' + aidex.CITATION + '.')
        all_lines = aidex.peak_list(res)
        if not all_lines:
            st.info('There are no sharp unindexed lines to index.')
        else:
            a_lines = st.multiselect('Lines to send (sharp, unindexed, corrected 2θ, lowest first; one phase only)', all_lines,
                                     default=all_lines, key=f'aidex_lines_{name}_{hashlib.md5(str(all_lines).encode()).hexdigest()[:8]}', format_func=lambda v: f'{v:.3f}°',
                                     help='Leave out lines you think belong to another phase. Kβ copies, humps and bands are not offered.')
            a_lines = sorted(a_lines)[:20]
            if len(a_lines) < 15:
                st.warning(f'{len(a_lines)} line(s). AIdex-r2 works best with 15–20 lines of a single phase; with fewer, or with lines '
                           'from several phases, expect no reliable cell. The checks below will say so.')
            def _check(cands_a, how):
                with st.spinner('Refining each cell and running the chance test…'):
                    ss.aidex_result = dict(sample=name, lines=a_lines, phases_json=phases_json, how=how,
                                           table=aidex.check(cands_a, a_lines, res))

            st.markdown('**Option A · send the lines directly** (needs internet access from the computer running this app)')
            oc = st.columns([1.3, 3])
            if oc[0].button('Run AIdex-r2', type='primary', icon=':material/auto_awesome:', key='aidex_run', width='stretch',
                            disabled=len(a_lines) < 3,
                            help='Sends the selected lines to the AIdex-r2 web service (the same request its Predict button makes; '
                                 'unused peak boxes are 180°), then checks the five cells it returns.'):
                try:
                    with st.spinner('AIdex-r2 is predicting (usually a few seconds)…'):
                        cands_a = aidex.predict_online(a_lines)
                    _check(cands_a, 'sent directly')
                except aidex.AIdexError as e_:
                    st.error(str(e_))
            oc[1].caption(f'{len(a_lines)} line(s), λ = 1.5406 Å. Only the line positions are sent, to {aidex.URL}')

            with st.expander('Option B · fill AIdex-r2 in your browser (when this computer has no internet access)'):
                st.markdown('1. Click **Open AIdex-r2**.\n'
                            '2. On that page open the browser console: **F12** (Windows/Linux) or **⌥⌘J** (Chrome on Mac), **⌥⌘C** (Safari, '
                            'with the Develop menu on), then the **Console** tab.\n'
                            '3. Copy the line below (copy icon at its right), paste it into the console and press **Enter**. All peak boxes '
                            'are filled at once (unused boxes are set to 180, as AIdex-r2 expects). Chrome asks you to type `allow pasting` '
                            'the first time.\n'
                            '4. Click **Predict** on the AIdex-r2 page, copy its results table and paste it below.')
                st.code(aidex.fill_script(a_lines), language='javascript', wrap_lines=True)
                st.link_button('Open AIdex-r2', aidex.URL)
                st.caption('The plain list, for typing the values in by hand:')
                st.code('\n'.join(f'{v:.3f}' for v in a_lines), language=None)
                pasted = st.text_area('AIdex-r2 results table', key=f'aidex_paste_{name}', height=140,
                                      placeholder='EG\ta (Å)\tb (Å)\tc (Å)\tα (°)\tβ (°)\tγ (°)\tScore\n69\t4.302\t4.302\t15.894\t90.00\t90.00\t90.00\t38.6%')
                if st.button('Check the cells', key='aidex_check', disabled=not pasted.strip() or len(a_lines) < 3):
                    cands_a, perr = aidex.parse_results(pasted)
                    for e in perr:
                        st.warning(e)
                    if cands_a:
                        _check(cands_a, 'pasted')
            ar = ss.get('aidex_result')
            if ar and ar['sample'] == name:
                A_ = ar['table']
                if ar['lines'] != a_lines:
                    st.warning('These cells were checked against a different line selection. Run AIdex-r2 or check them again.')
                st.markdown(f"**AIdex-r2 cells** ({ar.get('how', 'pasted')}), checked against the {len(ar['lines'])} lines")
                show = A_[['verdict', 'eg', 'aidex_score', 'lattice', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'rms', 'max_misfit',
                           'lines_in_window', 'M', 'p_chance', 'same_as', 'reason']].rename(columns={
                    'eg': 'EG', 'aidex_score': 'AIdex score', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'volume': 'V (Å³)', 'rms': 'rms (°)',
                    'max_misfit': 'worst (°)', 'lines_in_window': 'calc. lines in window', 'M': 'M(n)', 'p_chance': 'chance P', 'same_as': 'same lattice as'})
                st.dataframe(show, hide_index=True, column_config={'reason': st.column_config.TextColumn(width='large')})
                if not (A_.verdict == 'credible').any():
                    st.info('No candidate passes: none fits the lines better than random lines would, with a misfit near the peak-position '
                            'precision. A cell found this way would not be an identification.')
                pick_a = st.multiselect('Add a cell to the phase list', list(A_.index), key=f'aidex_pick_{name}',
                                        format_func=lambda i: f"#{A_.loc[i, 'rank']} {A_.loc[i, 'lattice']} {A_.loc[i, 'a']:.4f}, {A_.loc[i, 'b']:.4f}, "
                                                              f"{A_.loc[i, 'c']:.4f} Å · {A_.loc[i, 'verdict']}",
                                        help='Adds it with every reflection its centering allows; the joint fit then refines it with the other phases.')
                if st.button('Add selected cells', disabled=not pick_a, key='aidex_add'):
                    cur = edited.copy()
                    names_now = list(cur['Phase'].astype(str))
                    added = []
                    for i in pick_a:
                        nm, k_ = f"AIdex cell {A_.loc[i, 'rank']}", 2
                        while nm in names_now:
                            nm, k_ = f"AIdex cell {A_.loc[i, 'rank']} ({k_})", k_ + 1
                        names_now.append(nm)
                        added.append(aidex.to_phase(A_.loc[i], nm, color=PALETTE[len(names_now) % len(PALETTE)], prefix='X'))
                    reset_editor(pd.concat([cur, phases_to_df(added)], ignore_index=True))
                    st.toast('Added ' + ', '.join(p['name'] for p in added) + ' to the phase list.')
                    st.rerun()

    st.markdown('##### Data files')
    st.caption('Figures are made in the Export figures tab.')
    e = st.columns(5)
    e[4].download_button('Report · TXT', res.report, f'{stem}_report.txt', 'text/plain', key='idx_rep')
    e[0].download_button('Peak table · CSV', res.table.to_csv(index=False), f'{stem}_peak_indexing.csv', 'text/csv', key='idx_peaks')
    e[1].download_button('Refined parameters · CSV', res.params.to_csv(index=False), f'{stem}_refined_parameters.csv', 'text/csv', key='idx_params')
    e[2].download_button('Pattern + background · CSV', S.pattern_csv(), f'{stem}_bkg24.csv', 'text/csv', key='idx_bkg_csv')
    e[3].download_button('Phases · JSON', to_json(phases), f'{stem}_phases.json', 'application/json', key='idx_phases')


# ------------------------------------------------------------------------------------------- Le Bail
lb_export = None
with tab_lb:
    st.subheader('Le Bail whole-pattern fit', anchor=False)
    st.caption('Fits the whole pattern with the refined phase list of the Peak indexing tab, using rietx. Every reflection intensity is free, '
               'so the fit refines cells, profile widths and the angle correction against all the data, not only the peak positions. '
               'The background of the Peak indexing tab is held as it is; unindexed sharp lines become free extra peaks and unassigned broad '
               'features become humps. 00l and hk0 series are fitted with their unseen axes held.')
    lb_ok, lb_msg = lebail.available()
    if not lb_ok:
        st.info(lb_msg)
    elif not res.phases:
        st.info('Switch on at least one phase in the Peak indexing tab and click Refine first.')
    else:
        if pending:
            st.warning('The phase list has edits that are not refined yet. The Le Bail fit uses the last refined list; click **Refine** '
                       'in the Peak indexing tab first if you want the edits in it.', icon=':material/pending:')
        lc = st.columns([2.3, 1.1, 1.5, 2.3])
        quality = lc[0].segmented_control('Plan', ['fast', 'thorough'], default='fast', key='lb_quality',
                                          format_func=lambda q: {'fast': 'Fast (1–2 min)', 'thorough': 'Thorough (5–10 min)'}[q],
                                          help='Fast: one Le Bail intensity pass and no closing stage; on the example the cells agree with the '
                                               'thorough plan within their esds. Thorough: two passes and the rietx closing stage.') or 'fast'
        weak_pct = lc[1].number_input('Weak phase below (% I)', 0.0, 50.0, 3.0, 0.5, key='lb_weak',
                                      help='A phase whose strongest indexed line is weaker than this keeps its indexed cell: with every '
                                           'intensity free, Le Bail cannot fix the cell of a ~1 % phase (it drifts to the bound).')
        lc[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        refine_weak = lc[2].toggle('Refine weak cells too', key='lb_refine_weak')
        xmin_, xmax_ = float(S.x.min()), float(S.x.max())
        lb_xr = lc[3].slider('Fit range 2θ (°)', xmin_, xmax_, (xmin_, xmax_), 0.1, key=f'lb_xr_{name}')
        lb_key = hashlib.md5(json.dumps([name, hashlib.md5(data).hexdigest(), phases_json, correction, float(radius), quality,
                                         weak_pct, refine_weak, [round(v, 2) for v in lb_xr]]).encode()).hexdigest()
        ss.setdefault('lebail', {})
        have = ss.lebail.get(lb_key)
        if st.button('Run Le Bail fit' if have is None else 'Run again', type='primary', icon=':material/play_arrow:', key='lb_run'):
            try:
                with st.status('Le Bail fit with rietx…', expanded=True) as status:
                    status.write(f"{len(res.phases)} phase(s), {quality} plan. Stages:")
                    lb_new = lebail.run(res, quality=quality, weak_percent=weak_pct, refine_weak=refine_weak, two_theta_limits=lb_xr,
                                        progress=lambda t: status.write(t))
                    status.update(label=f"Le Bail fit done in {lb_new.elapsed:.0f} s · Rwp {lb_new.stats['rwp']:.2f} %",
                                  state='complete', expanded=False)
                ss.lebail = {k: v for k, v in list(ss.lebail.items())[-3:]}
                ss.lebail[lb_key] = lb_new
                ss['lebail_last'] = (name, lb_key)
                have = lb_new
            except lebail.LeBailError as e:
                st.error(str(e))
        if have is None:
            last = ss.get('lebail_last')
            if last and last[0] == name and last[1] in ss.lebail:
                have = ss.lebail[last[1]]
                st.warning('This fit was run with an earlier phase list or different settings. Run it again to update it.')
        if have is None:
            st.caption('Nothing fitted yet. The fast plan takes about a minute on the example (8500 points, 3 phases).')
        else:
            lb = have
            lb_export = lb
            stt = lb.stats
            m = st.columns(4)
            m[0].metric('Rwp (Le Bail)', f"{stt['rwp']:.2f} %", f"Rexp {stt['rexp']:.2f} % · peak fit {stt['rwp_indexing']:.2f} %",
                        delta_color='off', delta_arrow='off')
            m[1].metric('GoF', f"{stt['gof']:.2f}", f"Durbin–Watson {stt['dw']:.2f}", delta_color='off', delta_arrow='off')
            m[2].metric('esd factor (Bérar–Lelann)', f"×{stt['berar']:.2f}", 'applied to every esd below', delta_color='off', delta_arrow='off')
            if lb.correction:
                cr = lb.correction
                m[3].metric('Sample displacement' if cr['kind'] == 'displacement' else 'Zero shift', f"{fsu(cr['value'], cr['esd'])} {cr['unit']}",
                            f"{lb.elapsed:.0f} s · {lb.settings['quality']} plan", delta_color='off', delta_arrow='off')
            else:
                m[3].metric('Angle correction', 'none', f"{lb.elapsed:.0f} s · {lb.settings['quality']} plan", delta_color='off', delta_arrow='off')
            for n_ in lb.notes:
                st.info(n_, icon=':material/info:')

            view = lb.as_view(res)
            c = st.columns([1.3, 4])
            lb_scale = c[0].segmented_control('Intensity scale', ['Log', 'Linear', '√'], default='Log', key=f'lb_scale_{name}') or 'Log'
            tg = c[1].columns(4)
            lb_labels = tg[0].toggle('Peak labels', False, key='lb_labels')
            lb_bkg = tg[1].toggle('Background', True, key='lb_bkg')
            lb_diff = tg[2].toggle('Difference', True, key='lb_diff')
            lb_pub = tg[3].toggle('Publication style', False, key='lb_pub')
            lsc = {'Linear': 'linear', 'Log': 'log', '√': 'sqrt'}[lb_scale]
            xr_lb = (float(lb.x.min()), float(lb.x.max()))
            st.plotly_chart(views.indexed_view(view, lsc, xr_lb, None, lb_labels, True, lb_bkg, lb_diff, lb_pub, revision=f'lb{name}{lsc}{lb_key}'),
                            theme=None if lb_pub else 'streamlit', key=f'lb_plot_{name}')
            st.caption('Red: observed; line: Le Bail fit; Δ = observed − fit. Ticks are the reflections rietx used (Kα1). Labels, when shown, '
                       'are those of the peak indexing.')

            st.markdown('##### Lattice constants: Le Bail vs peak positions')
            st.caption('Le Bail esds are the rietx esds × the Bérar–Lelann factor (serially correlated residuals make raw esds several '
                       'times too small). Agreement: |difference| / combined esd.')
            cmp = lb.table_vs_indexing(res)
            st.dataframe(cmp, hide_index=True, column_config={'Agreement': st.column_config.TextColumn(width='small')})

            with st.expander(f"Extra peaks and humps ({len(lb.extras)})"):
                st.caption('Unindexed sharp lines and unassigned broad features, fitted as free components. "determined" = intensity '
                           'above 3 esd and width not at a bound. An undetermined component is not needed at this level.')
                st.dataframe(lb.extras, hide_index=True)
            with st.expander(f"rietx diagnostics ({len(lb.diagnostics)})"):
                if lb.diagnostics:
                    st.dataframe(pd.DataFrame(lb.diagnostics), hide_index=True,
                                 column_config={'message': st.column_config.TextColumn(width='large'), 'reading': st.column_config.TextColumn(width='large')})
                else:
                    st.caption('None.')
            with st.expander('Settings of this fit'):
                st.json(lb.settings, expanded=True)

            st.markdown('##### Data files')
            st.caption('Figures are made in the Export figures tab.')
            e = st.columns(5)
            e[0].download_button('Fit · CSV', lb.fit_csv(), f'{stem}_lebail_fit.csv', 'text/csv', key='lb_fit')
            e[1].download_button('Parameters · CSV', lb.params.to_csv(index=False), f'{stem}_lebail_parameters.csv', 'text/csv', key='lb_params')
            e[2].download_button('Reflections · CSV', lb.reflections_csv(), f'{stem}_lebail_reflections.csv', 'text/csv', key='lb_refl')
            e[3].download_button('Comparison · CSV', cmp.to_csv(index=False), f'{stem}_lebail_vs_peaks.csv', 'text/csv', key='lb_cmp')
            e[4].download_button('Extras · CSV', lb.extras.to_csv(index=False), f'{stem}_lebail_extras.csv', 'text/csv', key='lb_ex')
            st.caption('Fitted with rietx (rietx.org), Le Bail mode.')

# ------------------------------------------------------------------------------------------- export figures
with tab_exp:
    st.subheader('Export figures', anchor=False)
    st.caption('Publication figures (matplotlib): black axes, inward ticks on all sides, no grid. Choose the figure, what it shows and its '
               'layout; the preview updates as you change them. PDF embeds the fonts; SVG keeps the text as text.')
    kinds = ['Raw pattern', 'Indexed pattern'] + (['Le Bail fit'] if lb_export is not None else [])
    kind = st.segmented_control('Figure', kinds, default='Indexed pattern', key='exp_kind') or 'Indexed pattern'
    if lb_export is None:
        st.caption('The Le Bail figure appears here once a Le Bail fit has been run for this pattern.')

    left, right = st.columns([1, 1.9], gap='large')
    with left:
        # ---------------- content
        st.markdown('**Content**')
        xmin_, xmax_ = float(pat.x.min()), float(pat.x.max())
        if kind == 'Le Bail fit':
            xmin_, xmax_ = float(lb_export.x.min()), float(lb_export.x.max())
        ex_scale = st.segmented_control('Intensity scale', ['Linear', 'Log', '√'], default='Linear' if kind == 'Raw pattern' else 'Log',
                                        key=f'exp_scale_{kind}') or 'Log'
        ex_sc = {'Linear': 'linear', 'Log': 'log', '√': 'sqrt'}[ex_scale]
        ex_xr = st.slider('2θ range (°)', xmin_, xmax_, (xmin_, xmax_), 0.01, key=f'exp_xr_{kind}_{name}')
        if kind == 'Raw pattern':
            ex_points = st.toggle('Show data points', False, key='exp_points')
        else:
            src = res if kind == 'Indexed pattern' else lb_export.as_view(res)
            row_names = [t['name'] for t in src.ticks]
            row_label = {t['name']: t['label'] for t in src.ticks}
            ex_rows = st.pills('Tick rows', row_names, selection_mode='multi', default=row_names, format_func=lambda n: row_label[n],
                               key=f'exp_rows_{kind}_{name}_{"|".join(row_names)}') or []
            tg = st.columns(2)
            ex_labels = tg[0].toggle('Peak labels', kind == 'Indexed pattern', key=f'exp_labels_{kind}')
            ex_calc = tg[1].toggle('Calculated', True, key=f'exp_calc_{kind}')
            ex_bkg = tg[0].toggle('Background', True, key=f'exp_bkg_{kind}')
            ex_diff = tg[1].toggle('Difference (Δ)', True, key=f'exp_diff_{kind}')

        # ---------------- layout
        st.markdown('**Layout**')
        D = figures.Layout()

        def _apply_preset():
            p = ss['exp_preset']
            if p in figures.PRESETS:
                ss['exp_w'], ss['exp_h'], ss['exp_pt'] = figures.PRESETS[p]

        def _to_custom():
            cur = (ss['exp_w'], ss['exp_h'], ss['exp_pt'])
            match = [k_ for k_, v in figures.PRESETS.items() if all(abs(a_ - b_) < 1e-9 for a_, b_ in zip(v, cur))]
            ss['exp_preset'] = match[0] if match else 'Custom'

        for k_, v in (('exp_preset', 'Double column'), ('exp_w', D.width), ('exp_h', D.height), ('exp_pt', D.font_size), ('exp_font', D.font),
                      ('exp_lw', D.line_width), ('exp_fw', D.frame_width), ('exp_ms', D.marker_size), ('exp_dr', D.diff_ratio),
                      ('exp_tr', 0.30), ('exp_tr_auto', True), ('exp_side', True)):
            ss.setdefault(k_, v)
        Lm = figures.LIMITS
        st.selectbox('Size preset', list(figures.PRESETS) + ['Custom'], key='exp_preset', on_change=_apply_preset,
                     format_func=lambda p: p if p == 'Custom' else f'{p} · {figures.PRESETS[p][0]} × {figures.PRESETS[p][1]} in, {figures.PRESETS[p][2]:g} pt')
        c = st.columns(2)
        c[0].number_input('Width (in)', *Lm['width'], step=0.1, key='exp_w', on_change=_to_custom)
        c[1].number_input('Height (in)', *Lm['height'], step=0.1, key='exp_h', on_change=_to_custom)
        c[0].selectbox('Font', list(figures.FONTS), key='exp_font',
                       help='Where a font is not installed, a metric-compatible free font stands in (Liberation Sans for Arial, '
                            'Liberation Serif for Times New Roman, Carlito for Calibri, Caladea for Cambria).')
        c[1].number_input('Font size (pt)', *Lm['font'], step=0.5, key='exp_pt', on_change=_to_custom,
                          help='Axis labels and tick labels; legend, peak labels and phase names scale with it.')
        c[0].number_input('Line thickness (pt)', *Lm['line'], step=0.1, key='exp_lw',
                          help='Calculated (or raw) curve; background, Δ and tick marks scale with it.')
        c[1].number_input('Frame thickness (pt)', *Lm['frame'], step=0.1, key='exp_fw', help='Axes and tick marks of the frame.')
        c[0].number_input('Marker size (pt)', *Lm['marker'], step=0.1, key='exp_ms', help='Observed-data circles; 0 draws a line instead.')
        if kind != 'Raw pattern':
            st.slider('Δ panel height (× main panel)', *Lm['diff_ratio'], step=0.01, key='exp_dr', disabled=not ex_diff,
                      help='Height of the residual panel relative to the main panel.')
            st.toggle('Tick-row panel height automatic', key='exp_tr_auto', help='Automatic: room for the phase name and cell (two text lines) per row at the chosen font size.')
            st.slider('Tick-row panel height (× main panel)', *Lm['ticks_ratio'], step=0.01, key='exp_tr', disabled=ss['exp_tr_auto'],
                      help='Height of the peak-position (tick-row) panel relative to the main panel.')
            st.toggle('Legend, Rwp and phase names at the right', key='exp_side',
                      help='Switch off for a narrow figure: the plot then uses the full width.')
        layout = figures.Layout(width=ss['exp_w'], height=ss['exp_h'], font=ss['exp_font'], font_size=ss['exp_pt'],
                                line_width=ss['exp_lw'], frame_width=ss['exp_fw'], marker_size=ss['exp_ms'], diff_ratio=ss['exp_dr'],
                                ticks_ratio=None if ss['exp_tr_auto'] else ss['exp_tr'], side_text=ss['exp_side'])

    if kind == 'Raw pattern':
        make_fig = partial(figures.raw_figure, pat, scale=ex_sc, xrange=ex_xr, points=ex_points, layout=layout)
        base = f'{stem}_pattern_figure'
    else:
        make_fig = partial(figures.indexed_figure, src, ex_sc, ex_xr, set(ex_rows), ex_labels, ex_calc, ex_bkg, ex_diff, layout=layout)
        base = f'{stem}_indexed_figure' if kind == 'Indexed pattern' else f'{stem}_lebail_figure'

    with right:
        st.markdown(f'**Preview** · {layout.width:g} × {layout.height:g} in, {layout.font} {layout.font_size:g} pt')
        with st.spinner('Drawing…'):
            prev_dpi = max(60, min(140, int(1000 / layout.width)))
            st.image(figures.to_bytes(make_fig(), 'png', dpi=prev_dpi), width=int(layout.width * prev_dpi))
        d = st.columns([1, 1, 1, 1.2])
        dpi = d[3].selectbox('PNG resolution', [300, 600, 1200], index=1, key='exp_dpi', format_func=lambda v: f'{v} dpi')
        d[0].download_button('PDF', partial(fig_bytes, make_fig, 'pdf'), f'{base}.pdf', 'application/pdf', key='exp_pdf',
                             icon=':material/download:', width='stretch')
        d[1].download_button('PNG', partial(lambda m, r: figures.to_bytes(m(), 'png', dpi=r), make_fig, dpi), f'{base}.png', 'image/png',
                             key='exp_png', icon=':material/download:', width='stretch')
        d[2].download_button('SVG', partial(fig_bytes, make_fig, 'svg'), f'{base}.svg', 'image/svg+xml', key='exp_svg',
                             icon=':material/download:', width='stretch')
        st.download_button('Layout settings · JSON', json.dumps(layout.as_dict(), indent=1), f'{stem}_figure_layout.json', 'application/json',
                           key='exp_layout_json', help='A record of the layout used, to reproduce the figure later.')
