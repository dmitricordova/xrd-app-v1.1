"""Background options: file background as is, file + correction, refit with chosen order/decays, joint refinement."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from xrdcore import bkgrefine  # noqa: E402
from xrdcore.indexing import BkgSettings, IndexingError, background_and_peaks, index_pattern  # noqa: E402
from xrdcore.io import load_pattern  # noqa: E402
from xrdcore.phases import EXAMPLE_PHASES, validate  # noqa: E402
from xrdcore.util import dbl_sum, rwp  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'SnSeVSe22_27Aug26.csv')


@pytest.fixture(scope='module')
def pats():
    p = load_pattern(open(EXAMPLE, 'rb').read(), 'SnSeVSe22_27Aug26.csv')
    q = load_pattern('\n'.join(f'{a} {b}' for a, b in zip(p.x, p.y)), 'plain.xy')
    return p, q


def test_file_background_as_is(pats):
    p, q = pats
    S = background_and_peaks(p, bkg=BkgSettings(mode='file'))
    assert np.array_equal(S.bkg, p.bkg.astype(float)) and S.bkg_label == 'from file'
    with pytest.raises(IndexingError):
        background_and_peaks(q, bkg=BkgSettings(mode='file'))


def test_file_plus_correction_and_order(pats):
    p, _ = pats
    S = background_and_peaks(p, bkg=BkgSettings(mode='file+poly', order=6, decays=False))
    assert not np.array_equal(S.bkg, p.bkg) and 'file + Chebyshev 6' in S.bkg_label
    S12 = background_and_peaks(p, bkg=BkgSettings(order=12, decays=False))
    assert S12.bkg_label == 'Chebyshev 12'
    R = index_pattern(S, validate(EXAMPLE_PHASES)[0])
    assert abs(R.cells['VSe2']['a'][0] - 3.3592) < 0.0005


def test_joint_refinement_plain_pattern(pats):
    """Without a prior model the rolling-percentile background is rough; refining it with the peaks must lower Rwp
    a lot and keep the reference cells."""
    _, q = pats
    S0 = background_and_peaks(q)
    S = background_and_peaks(q, bkg=BkgSettings(refine=True))
    assert S.rwp_step2 < 0.4 * S0.rwp_step2 and S.bkg_history[-1]['cycle'] == 'joint'
    R = index_pattern(S, validate(EXAMPLE_PHASES)[0])
    assert abs(R.cells['VSe2']['a'][0] - 3.3592) < 0.0006 and abs(R.cells['SnSe2']['c'][0] - 6.1321) < 0.0012


def test_bkgrefine_recovers_synthetic():
    """Known quadratic background + 3 doublets: the joint fit recovers the background within noise."""
    rng = np.random.default_rng(1)
    x = np.arange(10, 60, 0.02)
    btrue = 500 + 3 * (x - 30) + 0.05 * (x - 30) ** 2
    y = rng.poisson(btrue + dbl_sum(x, np.array([20., 33., 47.]), np.array([3000., 1500., 800.]), np.full(3, 0.1), np.full(3, 0.5)))
    import pandas as pd
    P = pd.DataFrame(dict(t=[20.02, 32.97, 47.03], h=[2500., 1300., 700.], w=[0.12, 0.12, 0.12], eta=[0.5] * 3))
    r = bkgrefine.refine(x, y, np.zeros_like(x), np.full_like(x, float(np.median(y))), P, order=4, decays=False)
    assert np.sqrt(np.mean(((r['bkg'] - btrue) / np.sqrt(btrue)) ** 2)) < 0.3
    assert np.allclose(r['t'], [20, 33, 47], atol=0.005) and r['rwp'] < r['rwp_start']
