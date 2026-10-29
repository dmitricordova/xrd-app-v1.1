"""Le Bail fit (rietx): phase mapping, a synthetic Si pattern with a known cell, and (XRD_SLOW=1) the example pattern."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from xrdcore import lebail  # noqa: E402
from xrdcore.indexing import background_and_peaks, index_pattern  # noqa: E402
from xrdcore.io import load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, library, validate  # noqa: E402
from xrdcore.util import L1, dbl, tth  # noqa: E402

pytest.importorskip('rietx')


def test_series_mapping():
    ph = validate([dict(name='M', lattice='hexagonal', cell={'a': 3.44, 'c': 12.2}, series='00l+hk0')])[0][0]
    out = lebail._rx_phases(ph)
    assert [o[0] for o in out] == ['M (00l)', 'M (hk0)']
    l00, hk0 = out
    assert l00[1] == 'P4/mmm' and l00[2][:2] == [0.8, 0.8] and l00[3] == ['c'] and not l00[5]
    assert hk0[1] == 'P6/mmm' and hk0[2][2] == 0.8 and hk0[3] == ['a'] and hk0[5]
    full = lebail._rx_phases(validate([EXAMPLE_PHASES[0]])[0][0])[0]
    assert full[1] and full[6] == 'all'


def test_si_cell():
    """Si (a = 5.43102 Å) with K = −0.04°: the Le Bail cell agrees with the true value and with the peak-position cell."""
    rng = np.random.default_rng(5)
    si = next(p for p in library() if p['name'].startswith('Si'))
    x = np.arange(20, 100, 0.01)
    y = 300 + 400 * np.exp(-(x - 20) / 30)
    for hkl, I in zip(si['hkl'], [100, 55, 30, 6, 11, 12, 6, 3, 7]):
        t = float(tth(5.43102 / math.sqrt(sum(v * v for v in hkl)), L1))
        t += -0.04 * math.cos(math.radians(t / 2))
        y = y + dbl(x, t, 60 * I, 0.08, 0.5)
    pat = load_pattern('\n'.join(f'{a:.2f} {b:.0f}' for a, b in zip(x, rng.poisson(y))), 'si.xy')
    R = index_pattern(background_and_peaks(pat), validate([si])[0], 'displacement')
    lb = lebail.run(R, quality='fast')
    a = lb.cells[si['name']]['axes']['a']
    assert a['status'] == 'refined' and abs(a['value'] - 5.43102) < max(3 * a['esd'], 2e-4)
    assert lb.stats['rwp'] < 25 and lb.ticks[0]['rows']
    T = lb.table_vs_indexing(R)
    assert T.Agreement.iloc[0].startswith('agree')
    assert 'twotheta,yobs,ycalc' in lb.fit_csv().splitlines()[0]


@pytest.mark.skipif(not os.environ.get('XRD_SLOW'), reason='about 80 s; set XRD_SLOW=1')
def test_example_pattern():
    p = os.path.join(ROOT, 'examples', 'SnSeVSe22_27Aug26.csv')
    R = index_pattern(background_and_peaks(load_pattern(open(p, 'rb').read(), os.path.basename(p))), validate(EXAMPLE_PHASES)[0], 'displacement')
    lb = lebail.run(R)
    T = lb.table_vs_indexing(R)
    assert (T.Agreement[T.Phase != 'V2O3'] == 'agree (< 2σ)').all()
    assert lb.cells['V2O3']['axes']['a']['status'] == 'held, weak phase'
    assert 2.0 < lb.stats['rwp'] < 3.5
