"""00l / hk0 series (indexing and Materials Project screen) and the AIdex-r2 cell check, on synthetic patterns with known answers."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from synth_misfit import misfit_pattern  # noqa: E402
from xrdcore import aidex  # noqa: E402
from xrdcore.indexing import background_and_peaks, hkl_list, index_pattern  # noqa: E402
from xrdcore.io import load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, library, validate  # noqa: E402
from xrdcore.util import L1, dbl, tth  # noqa: E402


def as_pattern(x, y, name='synthetic.xy'):
    return load_pattern('\n'.join(f'{a:.2f} {b:.0f}' for a, b in zip(x, y)), name)


@pytest.fixture(scope='module')
def misfit():
    x, y = misfit_pattern()
    return background_and_peaks(as_pattern(x, y))


def test_series_filter_and_auto_range():
    ph = validate([dict(name='SL', lattice='tetragonal', cell={'a': 3.0, 'c': 24.5}, series='00l')])[0][0]
    H = hkl_list(ph)
    assert all(h == 0 and k == 0 for h, k, _ in H) and max(l for *_, l in H) >= 30      # long c: all orders to d > λ/2
    hex_ = validate([dict(name='H', lattice='hexagonal', cell={'a': 3.44, 'c': 6.1}, series='hk0')])[0][0]
    assert all(l == 0 for *_, l in hkl_list(hex_)) and (1, 1, 0) in hkl_list(hex_)
    listed = validate([dict(EXAMPLE_PHASES[0], series='00l')])[0][0]
    assert hkl_list(listed) == [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]


def test_misfit_series_cells(misfit):
    """Sharp 00l superlattice + broad hexagonal and tetragonal hk0 bands: each series recovers its own axis."""
    ph = [dict(name='SL', lattice='tetragonal', cell={'a': 3.0, 'c': 12.20}, series='00l'),
          dict(name='H', lattice='hexagonal', cell={'a': 3.43, 'c': 6.0}, series='hk0'),
          dict(name='T', lattice='tetragonal', cell={'a': 6.04, 'c': 10.0}, series='hk0')]
    R = index_pattern(misfit, validate(ph)[0])
    c, ec = R.cells['SL']['c']
    assert abs(c - 12.228) < max(3 * ec, 0.002)
    assert abs(R.cells['T']['a'][0] - 6.055) < 0.006
    assert abs(R.cells['H']['a'][0] - 3.442) < 0.003
    assert 'a' in R.held['SL'] and 'c' in R.held['H'] and 'c' in R.held['T']
    assert abs(R.corr[0] + 0.05) < 0.01


def test_screen_hk0_mode_finds_rock_salt_layer():
    """Tetragonal (rock-salt) bands left out of the list: a rock-salt candidate with a DFT-like cell (+1 %) explains them in an
    in-plane mode; the same structure 6 % too large is only 'possible, check'."""
    pytest.importorskip('pymatgen')
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Lattice, Structure
    from xrdcore import mpsearch as M
    mk = lambda a: Structure.from_spacegroup('Fm-3m', Lattice.cubic(a), ['Pb', 'Se'], [[0, 0, 0], [.5, .5, .5]])
    p = XRDCalculator('CuKa1').get_pattern(mk(6.055), two_theta_range=(5, 95))
    tet = {}
    for I, h in zip(p.y, p.hkls):
        a = sorted((abs(v) for v in h[0]['hkl']), reverse=True)
        if a[2] == 0:
            tet[(a[0], a[1])] = I
    x, y = misfit_pattern(tet_hk0={k: v * 60 * 0.10 / 0.6 for k, v in tet.items()})
    S = background_and_peaks(as_pattern(x, y))
    ph = [dict(name='SL', lattice='tetragonal', cell={'a': 3.0, 'c': 12.20}, series='00l'),
          dict(name='H', lattice='hexagonal', cell={'a': 3.43, 'c': 6.0}, series='hk0')]
    R = index_pattern(S, validate(ph)[0])
    D = M.screen(R, [dict(mpid='rs-good', formula='PbSe', sg='Fm-3m', system='cubic', e_hull=0.0, structure=mk(6.12)),
                     dict(mpid='rs-big', formula='SnTe', sg='Fm-3m', system='cubic', e_hull=0.0, structure=mk(6.45))], modes=M.MODES).set_index('mpid')
    assert D.loc['rs-good', 'verdict'] == 'explains unindexed lines' and D.loc['rs-good', 'mode_key'] != 'powder'
    assert D.loc['rs-big', 'verdict'] != 'explains unindexed lines'
    phase = M.candidate_phase(D.reset_index().query("mpid == 'rs-good'").iloc[0], ['SL', 'H'], 2, x_range=(5, 90))
    assert phase['series'] != 'all' and [2, 2, 0] in phase['hkl']
    R2 = index_pattern(S, validate(ph + [phase])[0])
    assert abs(R2.cells['PbSe']['a'][0] - 6.055) < 0.01


# ------------------------------------------------------------------ AIdex-r2
def test_parse_aidex_tables():
    txt = ('EG\ta\tb\tc\tα\tβ\tγ\tScore\nF---\t5.4312\t5.4312\t5.4312\t90.00\t90.00\t90.00\t71.2%\n'
           '1 | P--- | 3.8405 | 3.8405 | 3.8405 | 60.00 | 60.00 | 60.00 | 12.0\n3 P-1 6.951 8.269 16.033 80.3 79.8 69.5 8.4\n'
           'P--- 2.50 2.50 2.50 90 90 90\nnot a row')
    C, err = aidex.parse_results(txt)
    assert [c['eg'] for c in C] == ['F---', 'P---', 'P-1', 'P---'] and not err
    assert C[1]['a'] == 3.8405 and C[1]['alpha'] == 60 and C[2]['c'] == 16.033 and C[0]['score'] == 71.2


@pytest.fixture(scope='module')
def si_lines():
    rng = np.random.default_rng(3)
    si = next(p for p in library() if p['name'].startswith('Si'))
    x = np.arange(10, 100, 0.01)
    y = 300 + 2000 * np.exp(-(x - 10) / 3)
    for hkl, I in zip(si['hkl'], [100, 55, 30, 6, 11, 12, 6, 3, 7]):
        t = float(tth(5.43102 / math.sqrt(sum(v * v for v in hkl)), L1))
        y = y + dbl(x, t, 80 * I, 0.09, 0.5)
    S = background_and_peaks(as_pattern(x, rng.poisson(y)))
    return S, si, aidex.peak_list(index_pattern(S, []))


def test_aidex_check_si(si_lines):
    S, si, lines = si_lines
    C, _ = aidex.parse_results('F--- 5.4312 5.4312 5.4312 90 90 90 71\nP--- 3.8405 3.8405 3.8405 60 60 60 12\n'
                               'P-1 6.951 8.269 16.033 80.3 79.8 69.5 8\nP--- 2.50 2.50 2.50 90 90 90 1')
    D = aidex.check(C, lines, index_pattern(S, validate([si])[0])).set_index('rank')
    assert D.loc[1, 'verdict'] == 'credible' and abs(D.loc[1, 'a'] - 5.431) < 0.002
    assert D.loc[2, 'verdict'] == 'credible' and D.loc[2, 'lattice'] == 'rhombohedral'
    assert D.loc[3, 'verdict'] == 'rejected' and D.loc[4, 'verdict'] == 'rejected'
    assert 'shorter than the lowest-angle line' in D.loc[4, 'reason']
    pytest.importorskip('pymatgen')
    assert 'Si' in D.loc[1, 'same_as'] and 'Si' in D.loc[2, 'same_as']     # F-cubic and its primitive rhombohedral cell


def test_aidex_phase_indexes(si_lines):
    S, _, lines = si_lines
    C, _ = aidex.parse_results('P--- 3.8405 3.8405 3.8405 60 60 60 12')
    D = aidex.check(C, lines, None, n_random=20)
    ph = aidex.to_phase(D.iloc[0], 'AIdex cell 1')
    assert ph['lattice'] == 'hexagonal' and ph['centering'] == 'R'
    R = index_pattern(S, validate([ph])[0])
    assert not R.unindexed and R.rms < 0.01


def test_aidex_numeric_eg_and_fill_script():
    """The page shows the extinction group as a number; its centering must survive the paste."""
    C, err = aidex.parse_results('EG\ta (Å)\tb (Å)\tc (Å)\tα (°)\tβ (°)\tγ (°)\tScore\tSG List\n'
                                 '69\t4.302\t4.302\t15.894\t90.00\t90.00\t90.00\t38.6%\tView\n'
                                 '76\t3.840\t3.840\t9.400\t90.00\t90.00\t120.00\t<0.1%\tView\n'
                                 '96\t5.431\t5.431\t5.431\t90.00\t90.00\t90.00\t71.0%\tView')
    assert not err and [c['eg'][0] for c in C] == ['I', 'R', 'F'] and C[0]['score'] == 38.6 and C[1]['c'] == 9.4
    assert 'Fm-3m' in C[2]['eg'] and len(aidex.EG) == 101
    js = aidex.fill_script([24.302, 12.029])
    assert js.startswith('(()=>{const v=[12.029,24.302];') and ".peak-input" in js and "'180.0'" in js


def test_aidex_predict_online_mock():
    """predict_online sends 20 sorted peaks (unused = 180) to /api/predict and reads the JSON reply."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen['path'] = self.path
            seen['body'] = _json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            out = _json.dumps({'results': [{'a': 5.43, 'b': 5.43, 'c': 5.43, 'alpha': 90, 'beta': 90, 'gamma': 90, 'eg': 96, 'score': 0.71}]})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass
    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        C = aidex.predict_online([47.3, 28.4, 56.1], url=f'http://127.0.0.1:{srv.server_port}/')
    finally:
        srv.shutdown()
    assert seen['path'] == '/api/predict' and seen['body']['peaks'][:4] == [28.4, 47.3, 56.1, 180.0] and len(seen['body']['peaks']) == 20
    assert C[0]['eg'].startswith('F · EG 96') and abs(C[0]['score'] - 71) < 1e-9
    with pytest.raises(aidex.AIdexError):
        aidex.predict_online([28.4, 47.3, 56.1], url='http://127.0.0.1:9/', timeout=2)
