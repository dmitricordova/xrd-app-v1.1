"""Interactive Plotly figures for the app: the raw pattern, and the indexed pattern with Δ and tick-row panels.

One x axis shared by all panels (stacked y domains), so zoom and the hover spike line act on every panel.
Screen colours read on light and dark themes; "publication style" switches to white, black mirrored axes and
inside ticks, the look of the exported figure.
"""
import numpy as np
import plotly.graph_objects as go

from .util import sub_digits

SCREEN = dict(obs='#d6453a', calc='#2f6fdb', bkg='#2a9d5c', diff='#7a8494', grid='rgba(128,128,128,0.18)', axis='rgba(128,128,128,0.6)')
PUB = dict(obs='#c0392b', calc='#000000', bkg='#7f8a99', diff='#2b2b2b', grid=None, axis='#000000')


def _tf(scale):
    if scale == 'sqrt':
        return lambda v: np.sqrt(np.maximum(np.asarray(v, dtype=float), 0))
    return lambda v: np.asarray(v, dtype=float)


def _axis(pub, C):
    if pub:
        return dict(showline=True, linecolor='#000', linewidth=1, mirror='allticks', ticks='inside', ticklen=5, tickcolor='#000',
                    showgrid=False, zeroline=False, color='#000', minor=dict(ticks='inside', ticklen=3, tickcolor='#000'))
    return dict(showline=True, linecolor=C['axis'], ticks='outside', ticklen=4, tickcolor=C['axis'], showgrid=True, gridcolor=C['grid'], zeroline=False)


def _yrange(x, ys, xr, scale):
    m = (x >= xr[0]) & (x <= xr[1])
    vals = np.concatenate([_tf(scale)(y)[m] for y in ys])
    if scale == 'log':
        vals = vals[vals > 0]
        return [float(np.log10(vals.min() * 0.93)), float(np.log10(vals.max() * 1.35))] if len(vals) else None
    lo, hi = float(vals.min()), float(vals.max())
    sp = (hi - lo) or 1.0
    return [lo - 0.04 * sp, hi + 0.18 * sp]


SCI = dict(exponentformat='power', showexponent='all', minexponent=0)   # every label as m×10ⁿ


def _yfmt(scale):
    """Log: powers of ten; linear: scientific notation on every label; √: plain numbers."""
    if scale == 'log':
        return dict(exponentformat='power')
    if scale == 'linear':
        return SCI
    return dict(exponentformat='none', separatethousands=True)


def _spike(fig, pub):
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='dot',
                     spikecolor='#000' if pub else 'rgba(128,128,128,0.9)')


def raw_view(pat, scale='linear', points=False, xrange=None, pub=False, height=460, revision=''):
    C = PUB if pub else SCREEN
    f = _tf(scale)
    x, y = pat.x, pat.y
    xr = list(xrange) if xrange else [float(x.min()), float(x.max())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(y), mode='lines+markers' if points else 'lines', name='Observed',
                             line=dict(color='#000' if pub else C['obs'], width=1),
                             marker=dict(symbol='circle-open', size=4, color=C['obs']) if points else None,
                             customdata=y, hovertemplate='2θ %{x:.3f}°<br>%{customdata:,.0f} ' + pat.unit + '<extra></extra>'))
    ax = _axis(pub, C)
    unit = pat.unit
    fig.update_layout(height=height, margin=dict(l=70, r=16, t=10, b=48), showlegend=False, hovermode='closest', uirevision=revision,
                      xaxis=dict(ax, title='2θ (°, Cu Kα)', range=xr),
                      yaxis=dict(ax, type='log' if scale == 'log' else 'linear', range=_yrange(x, [y], xr, scale),
                                 title='√I (' + unit + '½)' if scale == 'sqrt' else f'Intensity ({unit})',
                                 **_yfmt(scale)))
    if pub:
        fig.update_layout(paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(family='Arial, Helvetica, sans-serif', color='#000'))
    _spike(fig, pub)
    return fig


def indexed_view(res, scale='log', xrange=None, visible=None, labels=True, calc=True, bkg=True, diff=True, pub=False,
                 highlight=None, revision='', narrow=False):
    S = res.stage1
    C = PUB if pub else SCREEN
    f = _tf(scale)
    x = S.x
    xr = list(xrange) if xrange else [float(x.min()), float(x.max())]
    rows = [t for t in res.ticks if visible is None or t['name'] in visible]
    ns = max(len(rows), 1)
    row_h = 18 if narrow else 20
    hT, hD, hI, gap, mt, mb = ns * row_h + 8, (60 if diff else 0), (300 if narrow else 400), 8, 10, 48
    tot = hT + hD + hI + gap * (2 if diff else 1)
    dT = [0, hT / tot]
    dD = [(hT + gap) / tot, (hT + gap + hD) / tot]
    dI = [(hT + gap + (hD + gap if diff else 0)) / tot, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(S.yobs), mode='markers', name='Observed', customdata=S.yobs,
                             marker=dict(symbol='circle-open', size=3.5, color=C['obs'], line=dict(width=0.8, color=C['obs'])),
                             hovertemplate='2θ %{x:.3f}°<br>yobs %{customdata:,.0f}<extra></extra>'))
    if calc:
        fig.add_trace(go.Scatter(x=x, y=f(S.ycal), mode='lines', name='Calculated', line=dict(color=C['calc'], width=1.1), hoverinfo='skip'))
    if bkg:
        fig.add_trace(go.Scatter(x=x, y=f(S.bkg), mode='lines', name=f'Background ({getattr(S, "bkg_label", "step-2")})', line=dict(color=C['bkg'], width=1.1), hoverinfo='skip'))
    if diff:
        fig.add_trace(go.Scatter(x=x, y=S.yobs - S.ycal, yaxis='y2', mode='lines', name='Δ', showlegend=False,
                                 line=dict(color=C['diff'], width=0.9), hovertemplate='Δ %{y:,.0f}<extra></extra>'))
    ann = []
    for k, t in enumerate(rows):
        yy = ns - 1 - k
        if t['rows']:
            fig.add_trace(go.Scatter(x=[r[0] for r in t['rows']], y=[yy] * len(t['rows']), yaxis='y3', mode='markers', showlegend=False,
                                     name=t['label'], customdata=[[r[1], r[2]] for r in t['rows']],
                                     marker=dict(symbol='line-ns-open', size=13, color=t['color'], line=dict(width=1.6, color=t['color'])),
                                     hovertemplate=t['label'] + '<br>%{customdata[0]}   d %{customdata[1]} Å<br>2θ %{x:.3f}°<extra></extra>'))
        ann.append(dict(xref='paper', x=0, xanchor='left', xshift=4, yref='y3', y=yy, showarrow=False, text=t['label'],
                        font=dict(size=10, color=t['color']), bgcolor='rgba(255,255,255,0.75)' if pub else 'rgba(0,0,0,0)'))
    if labels:
        vis_cols = {t['color'] for t in rows}
        show_un = any(t['name'] == 'Unindexed' for t in rows)
        for t0, lab, c, _I in res.labels:
            if not (xr[0] <= t0 <= xr[1]):
                continue
            if lab == '*' and not show_un:
                continue
            if lab != '*' and c not in vis_cols:
                continue
            i = int(np.argmin(abs(x - t0)))
            ym = float(S.yobs[max(0, i - 6):i + 7].max())
            yv = f(ym)
            ann.append(dict(x=t0, y=float(np.log10(max(yv, 1))) if scale == 'log' else float(yv), xref='x', yref='y', text=lab, textangle=-90,
                            showarrow=False, yanchor='bottom', yshift=4, font=dict(size=10, color=c)))
    shapes = []
    if highlight is not None:
        shapes.append(dict(type='rect', xref='x', yref='paper', x0=highlight - 0.07, x1=highlight + 0.07, y0=0, y1=1,
                           fillcolor='rgba(47,111,219,0.15)', line=dict(width=0), layer='below'))
    ax = _axis(pub, C)
    ys = [S.yobs] + ([S.ycal] if calc else []) + ([S.bkg] if bkg else [])
    yr = _yrange(x, ys, xr, scale)
    if yr and labels:   # room above the strongest peak for its rotated label
        yr = [yr[0], yr[1] + 0.45] if scale == 'log' else [yr[0], yr[1] + 0.2 * (yr[1] - yr[0])]
    layout = dict(height=tot + mt + mb, margin=dict(l=72, r=16, t=mt, b=mb), hovermode='closest', uirevision=revision,
                  legend=dict(x=1, xanchor='right', y=1, yanchor='top', bgcolor='rgba(0,0,0,0)'), annotations=ann, shapes=shapes,
                  xaxis=dict(ax, anchor='y3', range=xr, title='2θ (°, Cu Kα₁)'),
                  yaxis=dict(ax, domain=dI, type='log' if scale == 'log' else 'linear', range=yr, **_yfmt(scale),
                             title='√I (counts½)' if scale == 'sqrt' else 'Intensity (counts)'),
                  yaxis2=dict(ax, domain=dD if diff else [0, 0.001], visible=diff, title='Δ', nticks=3, anchor='x', **SCI),
                  yaxis3=dict(ax, domain=dT, range=[-0.55, ns - 0.45], showticklabels=False, ticks='', showgrid=False, fixedrange=True, anchor='x'))
    if pub:
        layout.update(paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(family='Arial, Helvetica, sans-serif', color='#000'))
    fig.update_layout(**layout)
    _spike(fig, pub)
    return fig


def tick_label(name):
    return sub_digits(name)
