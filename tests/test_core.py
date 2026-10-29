"""Tests: python3 -m pytest -q  (from the app folder)."""
import io
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from xrdcore import figures  # noqa: E402
from xrdcore.indexing import background_and_peaks, index_pattern, line_test  # noqa: E402
from xrdcore.io import PatternError, load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, from_json, library, parse_hkl, to_json, validate  # noqa: E402
from xrdcore.util import L1, dbl, fsu, tth  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'SnSeVSe22_27Aug26.csv')


@pytest.fixture(scope='module')
def example():
    pat = load_pattern(open(EXAMPLE, 'rb').read(), 'SnSeVSe22_27Aug26.csv')
    S = background_and_peaks(pat)
    R = index_pattern(S, validate(EXAMPLE_PHASES)[0])
    return pat, S, R


# ------------------------------------------------------------------ helpers and reading
def test_fsu():
    assert fsu(3.35926, 0.00029) == '3.3593(3)'
    assert fsu(6.0056, 0.0151) == '6.006(15)'
    assert fsu(14.0003, 0.0058) == '14.000(6)'
    assert fsu(1.5, None) == '1.5'


def test_read_refinement_csv(example):
    pat, _, _ = example
    assert len(pat.x) == 8499 and pat.ycal is not None and pat.bkg is not None
    assert pat.columns[:2] == ['twotheta', 'yobs'] and pat.unit == 'counts' and not pat.unit_stated


def test_read_xy_with_rate_header():
    x = np.arange(10, 20, 0.02)
    text = '#Sample=test\n#Intensity_unit=cps\n' + '\n'.join(f'{a:.2f}\t{100 + a:.1f}' for a in x)
    pat = load_pattern(text, 'scan.xy')
    assert pat.unit == 'cps' and pat.unit_is_rate and len(pat.x) == len(x)


def test_read_unsorted_and_three_columns():
    x = np.arange(10, 20, 0.05)[::-1]
    text = '\n'.join(f'{a:.2f} {5 + a:.1f} {1.0:.1f}' for a in x)
    pat = load_pattern(text, 'scan.xye')
    assert pat.sorted_input and np.all(np.diff(pat.x) > 0) and pat.esd is not None


def test_read_bad_files():
    with pytest.raises(PatternError, match='no numeric data'):
        load_pattern('hello\nworld\n', 'bad.txt')
    with pytest.raises(PatternError, match='at least 50'):
        load_pattern('\n'.join(f'{i} {i}' for i in range(10)), 'short.xy')
    with pytest.raises(PatternError, match='2θ in degrees'):
        load_pattern('\n'.join(f'{i * 10} {i}' for i in range(100)), 'swapped.xy')


def test_parse_hkl():
    assert parse_hkl('001 10-2 1,0,10') == [[0, 0, 1], [1, 0, -2], [1, 0, 10]]
    assert parse_hkl('auto') == []
    with pytest.raises(ValueError, match='1x1'):
        parse_hkl('101 1x1')


def test_phases_json_roundtrip():
    ph, err = validate(EXAMPLE_PHASES)
    ph2, err2 = from_json(to_json(ph))
    assert not err and not err2 and [p['hkl'] for p in ph] == [p['hkl'] for p in ph2]
    assert len(library()) >= 8


# ------------------------------------------------------------------ indexing, reference pattern
def test_example_cells(example):
    _, S, R = example
    ref = {'VSe2': {'a': 3.3592, 'c': 6.0980}, 'SnSe2': {'a': 3.8103, 'c': 6.1321}, 'V2O3': {'a': 4.9530, 'c': 14.000}}
    for ph, axes in ref.items():
        for ax, v in axes.items():
            val, esd = R.cells[ph][ax]
            assert abs(val - v) < max(3 * esd, 0.0006), (ph, ax, val, esd)
    assert abs(R.corr[0] + 0.066) < 0.015
    assert abs(S.rwp_step2 - 1.30) < 0.02
    assert R.used['VSe2'] >= 14 and R.used['SnSe2'] >= 15


def test_kbeta_found(example):
    _, _, R = example
    conf = R.kbeta[R.kbeta.status.str.startswith('confirmed')]
    assert len(conf) >= 4
    assert any(abs(t - 14.396) < 0.01 for t in conf['parent 2θ'])


def test_line_test_fast_and_absent(example):
    _, S, R = example
    t = time.time()
    r = line_test(S.x, S.yobs, S.sig, S.peaks, 30.38)
    assert time.time() - t < 3
    assert r['h'] < 2 * r['e']            # SnSe 111 is not there


def test_axis_held_when_unconstrained(example):
    _, S, _ = example
    only00l = [dict(validate(EXAMPLE_PHASES)[0][0], hkl=[[0, 0, l] for l in range(1, 7)])]
    R = index_pattern(S, only00l)
    assert 'a' in R.held['VSe2'] and 'c' in R.cells['VSe2']
    c, ec = R.cells['VSe2']['c']
    assert abs(c - 6.098) < 3 * ec        # c and K are strongly correlated with one 00l series


# ------------------------------------------------------------------ indexing, synthetic pattern with known answer
def test_synthetic_si_no_prior_model():
    rng = np.random.default_rng(3)
    a_true, K_true = 5.43102, 0.05
    si = next(p for p in library() if p['name'].startswith('Si'))
    rel = [100, 55, 30, 6, 11, 12, 6, 3, 7]
    x = np.arange(10, 100, 0.01)
    y = 300 + 2000 * np.exp(-(x - 10) / 3)
    for hkl, I in zip(si['hkl'], rel):
        t = float(tth(a_true / math.sqrt(sum(v * v for v in hkl)), L1))
        y = y + dbl(x, t + K_true * math.cos(math.radians(t / 2)), 80 * I, 0.09, 0.5)
    y = rng.poisson(y).astype(float)
    pat = load_pattern('\n'.join(f'{a:.2f} {b:.0f}' for a, b in zip(x, y)), 'si_synthetic.xy')
    start = dict(si, cell={'a': 5.40})
    R = index_pattern(background_and_peaks(pat), validate([start])[0])
    a, ea = R.cells['Si (standard)']['a']
    assert abs(a - a_true) < max(4 * ea, 0.0005), (a, ea)
    assert abs(R.corr[0] - K_true) < 0.01
    assert not R.unindexed


# ------------------------------------------------------------------ figures and CLI
def test_figure_sizes(example):
    from PIL import Image
    pat, _, R = example
    png = figures.to_bytes(figures.raw_figure(pat, 3.5, 2.6, 8), 'png')
    assert Image.open(io.BytesIO(png)).size == (2100, 1560)
    png = figures.to_bytes(figures.indexed_figure(R), 'png')
    assert Image.open(io.BytesIO(png)).size == (4200, 2580)
    pdf = figures.to_bytes(figures.raw_figure(pat, 5, 10, 8), 'pdf')
    assert b'/MediaBox [0 0 360 720]' in pdf or b'/MediaBox [ 0 0 360 720 ]' in pdf


def test_figure_layout(example):
    """Every layout option takes effect: size, font, line widths, panel ratios, side text; out-of-range values are clipped."""
    from PIL import Image
    pat, _, R = example
    L = figures.Layout(width=5.0, height=6.0, font='Times New Roman', font_size=11, line_width=1.5, frame_width=1.2,
                       marker_size=0, diff_ratio=0.5, ticks_ratio=0.8, side_text=False)
    fig = figures.indexed_figure(R, 'linear', layout=L)
    assert Image.open(io.BytesIO(figures.to_bytes(fig, 'png', dpi=100))).size == (500, 600)
    a, ad, at = fig.axes[:3]
    h = [ax.get_position().height for ax in (a, ad, at)]
    assert abs(h[1] / h[0] - 0.5) < 0.01 and abs(h[2] / h[0] - 0.8) < 0.01
    assert abs(a.lines[0].get_linewidth() - 1.5) < 1e-9 and a.get_legend() is None and not at.texts
    assert not [t for t in a.texts if t.get_position()[0] < 0.05 and t.get_transform() == a.transAxes]   # no corner label
    assert abs(a.spines['left'].get_linewidth() - 1.2) < 1e-9
    big = figures.Layout(width=99, font_size=1, diff_ratio=9).clipped()
    assert big.width == figures.LIMITS['width'][1] and big.font_size == figures.LIMITS['font'][0] and big.diff_ratio == 1.0
    raw = figures.raw_figure(pat, layout=figures.Layout(width=3.5, height=2.6, line_width=1.0))
    assert abs(raw.axes[0].lines[0].get_linewidth() - 1.0) < 1e-9 and not raw.axes[0].texts


def test_cli(tmp_path):
    import cli
    out = tmp_path / 'run'
    assert cli.main([EXAMPLE, '--out', str(out), '--no-figure', '--test', '30.38']) == 0
    for suf in ('_bkg24.csv', '_peak_indexing.csv', '_refined_parameters.csv', '_report.txt'):
        assert (tmp_path / ('run' + suf)).stat().st_size > 100
    assert 'Line tests' in (tmp_path / 'run_report.txt').read_text(encoding='utf-8')


def test_app_runs():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label: m.value for m in at.metric}
    assert labels['Rwp, background + peaks'] == '1.30 %'
