"""Materials Project candidate search. Offline tests use saved MP structures in tests/data/mp (CC BY 4.0);
the live test runs only when MP_API_KEY is set. Skipped entirely without the optional packages (requirements-mp.txt)."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
pytest.importorskip('pymatgen')

from pymatgen.core import Structure  # noqa: E402

from xrdcore import mpsearch  # noqa: E402
from xrdcore.indexing import background_and_peaks, index_pattern  # noqa: E402
from xrdcore.io import load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, library, validate  # noqa: E402
from xrdcore.util import L1, dbl, tth  # noqa: E402

DATA = os.path.join(HERE, 'data', 'mp')
EXAMPLE = os.path.join(ROOT, 'examples', 'SnSeVSe22_27Aug26.csv')


def fixtures(si=False):
    """Saved MP entries: the Si polymorphs (si=True) or the Sn-V-Se-O ones."""
    idx = json.load(open(os.path.join(DATA, 'index.json')))['entries']
    return [dict(r, structure=Structure.from_file(os.path.join(DATA, r['cif']))) for r in idx if (r['chemsys'] == 'Si') == si]


@pytest.fixture(scope='module')
def stage1():
    pat = load_pattern(open(EXAMPLE, 'rb').read(), 'SnSeVSe22_27Aug26.csv')
    return pat, background_and_peaks(pat)


def test_parse_elements():
    assert mpsearch.parse_elements('sn, V se;O') == (['O', 'Se', 'Sn', 'V'], [])
    els, err = mpsearch.parse_elements('Sn Qq')
    assert els == ['Sn'] and 'Qq' in err[0]
    assert mpsearch.elements_of(['VSe2', 'V2O3', 'Si (standard)']) == ['V', 'Se', 'O', 'Si']
    assert len(mpsearch.chemical_systems(['Sn', 'V', 'Se', 'O'])) == 15


def test_finds_missing_v2o3(stage1):
    """VSe2 + SnSe2 listed: corundum V2O3 must be the strong candidate, the listed phases 'same as', high-pressure Sn not strong."""
    pat, S = stage1
    two = [p for p in validate(EXAMPLE_PHASES)[0] if p['name'] != 'V2O3']
    R = index_pattern(S, two)
    D = mpsearch.screen(R, fixtures(False)).set_index('mpid')
    assert D.loc['mp-18937', 'verdict'] == 'explains unindexed lines'
    assert D.loc['mp-18937', 'unexpl'] >= 5
    assert D.loc['mp-694', 'verdict'] == 'same as VSe2 in your list'
    assert D.loc['mp-665', 'verdict'] == 'same as SnSe2 in your list'
    assert D.loc['mp-623511', 'verdict'] != 'explains unindexed lines'
    # add it and refine: the cell must agree with the reference indexing
    row = D.reset_index().query("mpid == 'mp-18937'").iloc[0]
    ph = mpsearch.candidate_phase(row, [p['name'] for p in two], 2, x_range=(float(pat.x.min()), float(pat.x.max())))
    assert ph['lattice'] == 'hexagonal' and ph['centering'] == 'R' and len(ph['hkl']) >= 10
    R2 = index_pattern(S, validate(two + [ph])[0])
    a, ea = R2.cells['V2O3']['a']
    c, ec = R2.cells['V2O3']['c']
    assert abs(a - 4.953) < max(3 * ea, 0.002) and abs(c - 14.000) < max(3 * ec, 0.006)
    assert len(R2.unindexed) < len(R.unindexed)


def test_monoclinic_candidate_phase(stage1):
    _, S = stage1
    R = index_pattern(S, validate(EXAMPLE_PHASES)[0])
    D = mpsearch.screen(R, [c for c in fixtures(False) if c['mpid'] == 'mp-541404'], n_random=5)
    ph = mpsearch.candidate_phase(D.iloc[0], [], 0)
    assert ph['lattice'] == 'monoclinic' and set(ph['cell']) == {'a', 'b', 'c', 'beta'}
    assert validate([ph])[1] == []


def test_synthetic_si():
    """Nothing listed: Fd-3m Si explains the lines, the other polymorphs do not."""
    rng = np.random.default_rng(3)
    si = next(p for p in library() if p['name'].startswith('Si'))
    x = np.arange(10, 100, 0.01)
    y = 300 + 2000 * np.exp(-(x - 10) / 3)
    for hkl, I in zip(si['hkl'], [100, 55, 30, 6, 11, 12, 6, 3, 7]):
        t = float(tth(5.43102 / math.sqrt(sum(v * v for v in hkl)), L1))
        y = y + dbl(x, t + 0.05 * math.cos(math.radians(t / 2)), 80 * I, 0.09, 0.5)
    pat = load_pattern('\n'.join(f'{u:.2f} {v:.0f}' for u, v in zip(x, rng.poisson(y))), 'si.xy')
    R = index_pattern(background_and_peaks(pat), [])
    D = mpsearch.screen(R, fixtures(True)).set_index('mpid')
    assert D.loc['mp-149', 'verdict'] == 'explains unindexed lines'
    assert all(D.loc[m, 'verdict'] == 'not supported' for m in ('mp-27', 'mp-165', 'mp-168'))


@pytest.mark.skipif(not os.environ.get('MP_API_KEY'), reason='set MP_API_KEY to run the live Materials Project test')
def test_live_fetch():
    C = mpsearch.fetch_candidates(['V', 'Se'], os.environ['MP_API_KEY'])
    ids = {c['mpid'] for c in C}
    assert 'mp-694' in ids and all(c['chemsys'] in ('V', 'Se', 'Se-V') for c in C)
