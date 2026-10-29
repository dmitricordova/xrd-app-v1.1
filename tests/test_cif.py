"""CIF candidates: reading (good, bad, multi-block), screening against the example pattern, adding as a phase."""
import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
pytest.importorskip('pymatgen')

from xrdcore import cifphase, mpsearch  # noqa: E402
from xrdcore.indexing import background_and_peaks, index_pattern  # noqa: E402
from xrdcore.io import load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, validate  # noqa: E402

CIFS = os.path.join(HERE, 'data', 'mp')
EXAMPLE = os.path.join(ROOT, 'examples', 'SnSeVSe22_27Aug26.csv')


def _read(*names):
    return [(n, open(os.path.join(CIFS, n), 'rb').read()) for n in names]


def test_read_cifs():
    two = open(os.path.join(CIFS, 'V2O3_mp-18937.cif')).read() + '\n' + open(os.path.join(CIFS, 'Si_mp-149.cif')).read()
    C, E = cifphase.read_cifs(_read('VSe2_mp-694.cif') + [('both.cif', two.encode()), ('junk.cif', b'not a cif'),
                                                          ('nocell.cif', b'data_x\n_cell_length_a 3\n')])
    assert [c['formula'] for c in C] == ['VSe2', 'V2O3', 'Si']
    assert C[0]['sg'] == 'P-3m1' and C[0]['system'] == 'trigonal' and C[0]['source'] == 'cif'
    assert C[1]['file'].startswith('both [') and C[1]['mpid'] != C[2]['mpid']
    assert len(E) == 2 and 'no data_ block' in E[0] and 'nocell.cif' in E[1]


@pytest.fixture(scope='module')
def two_phase():
    pat = load_pattern(open(EXAMPLE, 'rb').read(), 'SnSeVSe22_27Aug26.csv')
    S = background_and_peaks(pat)
    two = [p for p in validate(EXAMPLE_PHASES)[0] if p['name'] != 'V2O3']
    return S, two, index_pattern(S, two)


def test_screen_and_add(two_phase):
    S, two, R = two_phase
    C, _ = cifphase.read_cifs(_read(*[os.path.basename(f) for f in glob.glob(os.path.join(CIFS, '*.cif'))]))
    D = mpsearch.screen(R, C).set_index('formula')
    assert D.loc['V2O3', 'verdict'] == 'explains unindexed lines' and D.loc['V2O3', 'source'] == 'cif'
    T = mpsearch.display_table(D.reset_index())
    assert T['link'].isna().all() and T['E above hull (eV/atom)'].isna().all()
    row = D.reset_index().query("formula == 'V2O3'").iloc[0]
    ph = mpsearch.candidate_phase(row, [p['name'] for p in two], 2, x_range=(5, 90))
    assert ph['source'].startswith('CIF V2O3_mp-18937') and abs(ph['cell']['a'] - 4.953) < 0.01
    R2 = index_pattern(S, validate(two + [ph])[0])
    assert abs(R2.cells['V2O3']['a'][0] - 4.953) < 0.003 and len(R2.unindexed) < len(R.unindexed)
    ph_cif = mpsearch.candidate_phase(row, [], 2, x_range=(5, 90), start_cell=True)
    assert abs(ph_cif['cell']['a'] - 4.973) < 0.01 and 'as in the entry' in ph_cif['source']


def test_direct_phase():
    C, _ = cifphase.read_cifs(_read('V2O3_mp-18937.cif'))
    ph = cifphase.direct_phase(C[0], ['V2O3'], 0, x_range=(5, 90))
    assert ph['name'] != 'V2O3' and ph['lattice'] == 'hexagonal' and ph['centering'] == 'R' and len(ph['hkl']) >= 10
    assert validate([ph])[1] == []
    with pytest.raises(cifphase.CIFError):
        cifphase.direct_phase(C[0], [], 0, x_range=(5, 6))
