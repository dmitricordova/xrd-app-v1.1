"""Publication figures (matplotlib): the raw pattern, and the indexed (or Le Bail) pattern with Δ and tick-row panels.

Everything about the look is in a Layout: size, font family and size, curve and frame thickness, marker size, the
heights of the Δ and tick-row panels relative to the main panel, and whether the legend and phase names sit at the
right. Black axes, inward ticks on all four sides with minor ticks, no grid, open red circles for observed data.
PNG at 600 dpi (or any dpi), PDF with embedded TrueType fonts, SVG with text kept as text.
Uses matplotlib.figure.Figure directly (no pyplot), so it is safe in a multi-user web app.
"""
import io
import re
from dataclasses import asdict, dataclass, replace
from typing import Optional

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, NullFormatter

STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
         'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
         'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.default': 'regular'}

# font choice -> (family, fallbacks). Metric-compatible free fonts stand in where the named font is not installed.
FONTS = {
    'Arial': ('sans-serif', ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans']),
    'Helvetica': ('sans-serif', ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']),
    'Calibri': ('sans-serif', ['Calibri', 'Carlito', 'Liberation Sans', 'DejaVu Sans']),
    'Times New Roman': ('serif', ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif']),
    'Cambria': ('serif', ['Cambria', 'Caladea', 'Liberation Serif', 'DejaVu Serif']),
    'DejaVu Sans': ('sans-serif', ['DejaVu Sans']),
}

PRESETS = {  # name: (width in, height in, font pt)
    'Single column': (3.5, 2.6, 8.0),
    '1.5 column': (5.5, 3.4, 8.0),
    'Double column': (7.0, 4.3, 8.0),
    'Slide 16:9': (10.0, 5.63, 14.0),
}
LIMITS = {'width': (2.0, 12.0), 'height': (1.5, 10.0), 'font': (5.0, 16.0), 'line': (0.1, 3.0), 'frame': (0.2, 2.0),
          'marker': (0.0, 5.0), 'diff_ratio': (0.05, 1.0), 'ticks_ratio': (0.05, 1.5)}


@dataclass(frozen=True)
class Layout:
    width: float = 7.0              # in
    height: float = 4.3             # in
    font: str = 'Arial'
    font_size: float = 8.0          # pt; every other text scales with it
    line_width: float = 0.6         # pt, calculated / raw curve; background, Δ and tick marks scale with it
    frame_width: float = 0.6        # pt, axes and ticks
    marker_size: float = 1.3        # pt, observed circles (0 = line only)
    diff_ratio: float = 0.17        # Δ panel height / main panel height
    ticks_ratio: Optional[float] = None   # tick-row panel height / main panel height; None = room for 2 text lines per row
    side_text: bool = True          # legend, Rwp and phase names at the right

    def clipped(self):
        c = lambda v, k: float(np.clip(v, *LIMITS[k]))
        return replace(self, width=c(self.width, 'width'), height=c(self.height, 'height'), font_size=c(self.font_size, 'font'),
                       line_width=c(self.line_width, 'line'), frame_width=c(self.frame_width, 'frame'),
                       marker_size=c(self.marker_size, 'marker'), diff_ratio=c(self.diff_ratio, 'diff_ratio'),
                       ticks_ratio=None if self.ticks_ratio is None else c(self.ticks_ratio, 'ticks_ratio'))

    def as_dict(self):
        return asdict(self)


def _rc(L):
    fam, names = FONTS.get(L.font, FONTS['Arial'])
    fw = L.frame_width
    return {**STYLE, 'font.size': L.font_size, 'font.family': fam, f'font.{fam}': names,
            'axes.linewidth': fw, 'xtick.major.width': fw, 'ytick.major.width': fw, 'xtick.minor.width': 0.67 * fw,
            'ytick.minor.width': 0.67 * fw, 'xtick.major.size': 3 * L.font_size / 8, 'ytick.major.size': 3 * L.font_size / 8,
            'xtick.minor.size': 1.6 * L.font_size / 8, 'ytick.minor.size': 1.6 * L.font_size / 8}


def _tf(scale):
    if scale == 'sqrt':
        return lambda v: np.sqrt(np.maximum(v, 0))
    return lambda v: np.asarray(v, dtype=float)


def _sci(v, _pos=None):
    """Tick label in scientific notation: 0, 5×10³, 1.5×10⁴."""
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v))))
    m = round(v / 10 ** e, 2)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    ms = f'{m:g}'
    return f'$10^{{{e}}}$' if ms == '1' else (f'$-10^{{{e}}}$' if ms == '-1' else f'${ms}\\times10^{{{e}}}$')


def _yformatter(scale):
    if scale == 'linear':
        return mpl.ticker.FuncFormatter(_sci)
    return mpl.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}' if abs(v) >= 100 or v == 0 else f'{v:g}')


def _ylabel(scale, unit):
    return f'√I ({unit}$^{{1/2}}$)' if scale == 'sqrt' else f'Intensity ({unit})'


def _yrange(x, ys, xr, scale):
    m = (x >= xr[0]) & (x <= xr[1])
    vals = np.concatenate([np.asarray(_tf(scale)(y))[m] for y in ys if y is not None])
    if scale == 'log':
        vals = vals[vals > 0]
        if not len(vals):
            return None
        return vals.min() * 0.93, vals.max() * 1.35
    lo, hi = vals.min(), vals.max()
    sp = (hi - lo) or 1.0
    return lo - 0.04 * sp, hi + 0.18 * sp


def to_bytes(fig, fmt, dpi=None):
    """Figure -> file bytes. PNG defaults to 600 dpi."""
    buf = io.BytesIO()
    with mpl.rc_context(getattr(fig, '_xrd_rc', STYLE)):
        fig.savefig(buf, format=fmt, dpi=dpi or (600 if fmt == 'png' else None))
    return buf.getvalue()


def raw_figure(pat, width=7.0, height=4.3, font=8.0, scale='linear', xrange=None, points=False, layout=None):
    """The data exactly as read, nothing subtracted. layout (a Layout) overrides width, height and font."""
    L = (layout or Layout(width=width, height=height, font_size=font, line_width=0.7)).clipped()
    rc = _rc(L)
    with mpl.rc_context(rc):
        fig = Figure(figsize=(L.width, L.height))
        fig._xrd_rc = rc
        fi = L.font_size / 72  # inch per pt
        lm, rm, bm, tm = 5.6 * fi, 1.5 * fi, 3.6 * fi, 0.8 * fi   # margins scale with the font
        fig.subplots_adjust(left=min(lm / L.width, 0.4), right=1 - min(rm / L.width, 0.2), bottom=min(bm / L.height, 0.45),
                            top=1 - min(tm / L.height, 0.2))
        ax = fig.add_subplot(111)
        x, y = pat.x, pat.y
        xr = list(xrange) if xrange else [x.min(), x.max()]
        f = _tf(scale)
        ax.plot(x, f(y), '-', lw=L.line_width, color='black', zorder=2)
        if points and L.marker_size > 0:
            ax.plot(x, f(y), 'o', ms=L.marker_size, mfc='none', mec='#c0392b', mew=0.58 * L.line_width, zorder=3)
        if scale == 'log':
            ax.set_yscale('log')
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(*xr)
        yr = _yrange(x, [y], xr, scale)
        if yr:
            ax.set_ylim(*yr)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlabel('2θ (°, Cu Kα)')
        ax.set_ylabel(_ylabel(scale, pat.unit))
        if scale != 'log':
            ax.yaxis.set_major_formatter(_yformatter(scale))
    return fig


def _spread(L, gap):
    for _ in range(80):
        moved = False
        for k in range(1, len(L)):
            if L[k][1] - L[k - 1][1] < gap:
                m = (L[k][1] + L[k - 1][1]) / 2
                L[k - 1][1] = m - gap / 2 - 1e-6
                L[k][1] = m + gap / 2 + 1e-6
                moved = True
        if not moved:
            break
    return L


def indexed_figure(res, scale='log', xrange=None, visible=None, labels=True, calc=True, bkg=True, diff=True, layout=None):
    """Pattern with labelled reflections, Δ panel and tick rows (named at the right when layout.side_text)."""
    L = (layout or Layout()).clipped()
    S = res.stage1
    x, yo, yc, yb = S.x, S.yobs, S.ycal, S.bkg
    rows = [t for t in res.ticks if visible is None or t['name'] in visible]
    ns = max(len(rows), 1)
    xr = list(xrange) if xrange else [x.min(), x.max()]
    f = _tf(scale)
    k = L.font_size / 8                     # text scale relative to the 8 pt design
    lw = L.line_width
    rc = _rc(L)
    with mpl.rc_context(rc):
        fig = Figure(figsize=(L.width, L.height))
        fig._xrd_rc = rc
        left, bottom, top = 0.63 * k, 0.43 * k, 0.13 * k
        fb, ft = min(bottom / L.height, 0.4), 1 - min(top / L.height, 0.2)
        dr = L.diff_ratio if diff else 0.0
        if L.ticks_ratio is not None:
            tr = L.ticks_ratio
        else:                               # auto: room for two text lines per row at the current font size
            avail = (ft - fb) * L.height
            T = min(ns * 0.22 * k + 0.06, 0.55 * avail)
            tr = T / ((avail - T) / (1 + dr))
        ratios = [1.0] + ([L.diff_ratio] if diff else []) + [tr]
        right = 1.64 * k if L.side_text else 0.12 * k
        gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.0, left=min(left / L.width, 0.4),
                              right=1 - min(right / L.width, 0.5), top=ft, bottom=fb)
        a = fig.add_subplot(gs[0])
        ad = fig.add_subplot(gs[1], sharex=a) if diff else None
        at = fig.add_subplot(gs[-1], sharex=a)
        if L.marker_size > 0:
            a.plot(x, f(yo), 'o', ms=L.marker_size, mfc='none', mec='#c0392b', mew=0.58 * lw, label='Observed', zorder=2)
        else:
            a.plot(x, f(yo), '-', lw=lw, color='#c0392b', label='Observed', zorder=2)
        if calc:
            a.plot(x, f(yc), '-', lw=lw, color='black', label='Calculated', zorder=3)
        if bkg:
            a.plot(x, f(yb), '-', lw=0.85 * lw, color='#7f8a99', label='Background', zorder=1)
        if scale == 'log':
            a.set_yscale('log')
            a.yaxis.set_minor_formatter(NullFormatter())
        else:
            a.yaxis.set_minor_locator(AutoMinorLocator())
            a.yaxis.set_major_formatter(_yformatter(scale))
        a.set_xlim(*xr)
        yr = _yrange(x, [yo] + ([yc] if calc else []) + ([yb] if bkg else []), xr, scale)
        if yr and labels:   # room above the strongest peak for its rotated label
            yr = (yr[0], yr[1] * 3.2) if scale == 'log' else (yr[0], yr[1] + 0.22 * (yr[1] - yr[0]))
        if yr:
            a.set_ylim(*yr)
        a.set_ylabel(_ylabel(scale, 'counts'))
        if labels:
            vis_cols = {t['color'] for t in rows}
            Lb = []
            for t0, lab, c, _I in sorted(res.labels):
                if not (xr[0] < t0 < xr[1]) or (c not in vis_cols and lab != '*'):
                    continue
                if lab == '*' and not any(t['name'] == 'Unindexed' for t in rows):
                    continue
                i = int(np.argmin(abs(x - t0)))
                Lb.append([t0, t0, float(yo[max(0, i - 6):i + 7].max()), lab, c])
            Lb = _spread(Lb, 0.0135 * (xr[1] - xr[0]) * k * 7.0 / L.width)
            ylo, yhi = a.get_ylim()
            for t0, tl, yy, lab, c in Lb:
                if scale == 'log':
                    y0, y1 = yy * 1.06, yy * 1.22
                else:
                    sp = yhi - ylo
                    y0, y1 = f(yy) + 0.012 * sp, min(f(yy) + 0.05 * sp, yhi - 0.12 * sp)
                if abs(tl - t0) > 0.0006 * (xr[1] - xr[0]):
                    a.plot([t0, tl], [y0, y1], lw=0.6 * lw, color=c)
                a.text(tl, y1, lab, rotation=90, ha='center', va='bottom', fontsize=5.6 * k, color=c, clip_on=True)
        if L.side_text:
            a.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), fontsize=7 * k, handlelength=1.6, borderaxespad=0,
                     markerscale=2.5 if L.marker_size > 0 else 1)
            a.text(1.005, 0.30, f'$R_{{wp}}$ = {S.rwp_step2:.2f}%', transform=a.transAxes, fontsize=6.5 * k)
            if labels:
                a.text(1.005, 0.02, '* unindexed   β Cu Kβ', transform=a.transAxes, fontsize=6 * k, color='#444444')
        if diff:
            d = yo - yc
            m = (x >= xr[0]) & (x <= xr[1])
            ad.plot(x, d, '-', lw=0.75 * lw, color='#2b2b2b')
            ad.axhline(0, lw=0.65 * lw, color='#9aa0a8')
            lim = float(np.max(np.abs(d[m]))) * 1.1 if m.any() else 1.0
            lim = lim or 1.0
            ad.set_ylim(-lim, lim)
            ad.set_ylabel('Δ', labelpad=8)
            e = 10 ** np.floor(np.log10(0.6 * lim))          # ±one round value well inside the panel, clear of its edges
            tk = np.floor(0.6 * lim / e) * e
            ad.set_yticks([-tk, 0, tk])
            ad.yaxis.set_major_formatter(_yformatter('linear'))
            ad.tick_params(labelsize=6 * k)
        for kk, t in enumerate(rows):
            yrow = ns - 1 - kk
            pos = [r[0] for r in t['rows']]
            at.vlines(pos, yrow - 0.32, yrow + 0.32, color=t['color'], lw=1.15 * lw)
            if L.side_text:
                nm = t['label']
                val = t.get('val') or ''
                if len(val) > 40 and ', ' in val:
                    parts = val.split(', ')
                    val = ', '.join(parts[:len(parts) // 2]) + ',\n' + ', '.join(parts[len(parts) // 2:])
                at.text(1.006, (yrow + 0.5) / ns, nm + ('\n' + val if val else ''), transform=at.transAxes, fontsize=5.8 * k,
                        color=t['color'], va='center', ha='left', linespacing=1.05)
        at.set_ylim(-0.6, ns - 0.4)
        at.set_yticks([])
        at.tick_params(axis='y', which='both', left=False, right=False)
        at.set_xlabel('2θ (°, Cu Kα$_1$)')
        at.xaxis.set_minor_locator(AutoMinorLocator(5))
        for axx in [a] + ([ad] if ad is not None else []):
            axx.tick_params(axis='x', which='both', labelbottom=False)
    return fig


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or 'pattern'
