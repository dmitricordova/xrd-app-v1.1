"""Joint least-squares refinement of the background polynomial and every fitted peak, before indexing.

model(2θ) = base + Σ c_j T_j(t) [+ Σ d_k exp(-(2θ - 2θmin)/s_k)] + Σ_peaks pseudo-Voigt Kα1/Kα2 doublets
weights 1/√y; peak positions within ±0.08° of their start, FWHM 0.02-2.5°, 0 ≤ η ≤ 1, heights ≥ 0.
Analytic, sparse Jacobian (each peak touches only its own window), so a few hundred parameters refine in seconds.
"""
import numpy as np
from numpy.polynomial import chebyshev as Cheb
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack

from .util import dbl_sum, ka2

LN2_4 = 4 * np.log(2)
DECAYS = (0.3, 0.6, 1.2, 2.5)


def basis(x, order, decays=True):
    t = (2 * x - x.min() - x.max()) / (x.max() - x.min())
    cols = [Cheb.chebvander(t, max(int(order), 1) - 1)]
    if decays:
        cols += [np.exp(-(x - x.min()) / s)[:, None] for s in DECAYS]
    return np.hstack(cols)


def _pv_parts(x, c, h, w, eta):
    z = (x - c) / w
    L = 1 / (1 + 4 * z * z)
    G = np.exp(-LN2_4 * z * z)
    f = h * (eta * L + (1 - eta) * G)
    dfdz = h * (eta * (-8 * z * L * L) + (1 - eta) * (-2 * LN2_4 * z * G))
    return f, eta * L + (1 - eta) * G, h * (L - G), dfdz, z


def refine(x, y, base, bkg0, peaks, order=24, decays=True, max_nfev=80):
    """Returns dict(bkg, peaks (t, h, w, eta arrays), rwp_start, rwp, nfev, status)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    wt = 1 / np.sqrt(np.maximum(y, 1))
    A = basis(x, order, decays)
    m = A.shape[1]
    c0, *_ = np.linalg.lstsq(A * wt[:, None], (bkg0 - base) * wt, rcond=None)
    t0, h0, w0, e0 = (np.asarray(peaks[k], float) for k in ('t', 'h', 'w', 'eta'))
    n = len(t0)
    lb = np.column_stack([t0 - 0.08, np.zeros(n), np.full(n, 0.02), np.zeros(n)]).ravel()
    ub = np.column_stack([t0 + 0.08, np.maximum(h0 * 5, 100), np.full(n, 2.5), np.ones(n)]).ravel()
    q0 = np.clip(np.column_stack([t0, h0, w0, e0]).ravel(), lb + 1e-9, ub - 1e-9)
    # rows each peak can touch (Kα1 and Kα2, Lorentzian tails to ~30 FWHM)
    span = np.maximum(30 * np.maximum(w0, 0.05), 2.0)
    lo_i = np.searchsorted(x, t0 - span)
    hi_i = np.searchsorted(x, ka2(t0) + span)
    dk2 = (ka2(t0 + 1e-4) - ka2(t0 - 1e-4)) / 2e-4
    Aw = csr_matrix(A * wt[:, None])

    def model(v):
        q = v[m:].reshape(n, 4)
        return base + A @ v[:m] + dbl_sum(x, q[:, 0], q[:, 1], q[:, 2], q[:, 3])

    def fun(v):
        return (model(v) - y) * wt

    def jac(v):
        q = v[m:].reshape(n, 4)
        rows, cols, vals = [], [], []
        for k in range(n):
            c, h, w, eta = q[k]
            sl = slice(lo_i[k], hi_i[k])
            xs = x[sl]
            _, g1, de1, dz1, z1 = _pv_parts(xs, c, h, w, eta)
            c2 = ka2(c)
            _, g2, de2, dz2, z2 = _pv_parts(xs, c2, 0.5 * h, w, eta)
            ws = wt[sl]
            dc = (dz1 * (-1 / w) + dz2 * (-1 / w) * dk2[k]) * ws
            dh = (g1 + 0.5 * g2) * ws
            dw = (dz1 * (-z1 / w) + dz2 * (-z2 / w)) * ws
            de = (de1 + de2) * ws
            r = np.arange(lo_i[k], hi_i[k])
            for j, d in enumerate((dc, dh, dw, de)):
                rows.append(r)
                cols.append(np.full(len(r), 4 * k + j))
                vals.append(d)
        Jp = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(x), 4 * n))
        return hstack([Aw, Jp], format='csr')

    v0 = np.concatenate([c0, q0])
    L = np.concatenate([np.full(m, -np.inf), lb])
    U = np.concatenate([np.full(m, np.inf), ub])
    r0 = fun(v0)
    res = least_squares(fun, v0, jac=jac, bounds=(L, U), x_scale='jac', max_nfev=max_nfev, method='trf', tr_solver='lsmr')
    q = res.x[m:].reshape(n, 4)
    rw = lambda r: 100 * float(np.sqrt(np.sum(r ** 2) / np.sum((y * wt) ** 2)))
    return dict(bkg=base + A @ res.x[:m], t=q[:, 0], h=q[:, 1], w=q[:, 2], eta=q[:, 3], rwp_start=rw(r0), rwp=rw(res.fun),
                nfev=int(res.nfev), status=int(res.status))
