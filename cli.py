#!/usr/bin/env python3
"""Command-line peak indexing with the same code as the app (replaces xrd_index_local.py).

    python3 cli.py examples/SnSeVSe22_27Aug26.csv
    python3 cli.py data.xy --phases phases.json --correction zero --out results/mysample
    python3 cli.py data.csv --test 30.38 26.52                 # fixed-position line tests
    python3 cli.py --write-phases phases.json                  # write the example phase set to edit
    python3 cli.py data.csv --raw-figure 3.5 2.6 8             # also write the raw-pattern figure at this size
    python3 cli.py data.csv --mp-search "Sn V Se O"            # screen experimentally made Materials Project entries
                                                               # (key from MP_API_KEY or ~/.config/mp/key)

Writes <out>_bkg24.csv, <out>_peak_indexing.csv, <out>_refined_parameters.csv, <out>_report.txt,
<out>_indexed.pdf/.png (600 dpi) and, with --raw-figure, <out>_pattern_figure.pdf/.png.
"""
import argparse
import os
import sys

from xrdcore import figures
from xrdcore.indexing import BkgSettings, IndexingError, background_and_peaks, index_pattern
from xrdcore.io import PatternError, load_pattern
from xrdcore.phases import EXAMPLE_PHASES, from_json, to_json, validate


def main(argv=None):
    ap = argparse.ArgumentParser(description='Peak indexing (and optionally a Le Bail fit) of a lab Cu Kα powder pattern.')
    ap.add_argument('pattern', nargs='?', help='refinement CSV or .xy/.xye/.txt/.dat file')
    ap.add_argument('--phases', help='phases JSON (default: the VSe2 / SnSe2 / V2O3 example set)')
    ap.add_argument('--correction', choices=['displacement', 'zero', 'none'], default='displacement')
    ap.add_argument('--radius', type=float, default=240.0, help='goniometer radius in mm (for the height error)')
    ap.add_argument('--out', help='output prefix (default: next to the input file)')
    ap.add_argument('--test', type=float, nargs='*', default=[], help='2θ positions (observed scale) for line tests')
    ap.add_argument('--scale', choices=['log', 'linear', 'sqrt'], default='log', help='intensity scale of the indexed figure')
    ap.add_argument('--raw-figure', type=float, nargs=3, metavar=('WIDTH_IN', 'HEIGHT_IN', 'FONT_PT'))
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--write-phases', metavar='FILE')
    ap.add_argument('--mp-search', metavar='ELEMENTS', help='e.g. "Sn V Se O": screen experimentally made Materials Project entries')
    ap.add_argument('--background', choices=['refit', 'file', 'file+poly'], default='refit',
                    help='refit: Chebyshev (+ decays) fitted to the data; file: the bkg column as it is; file+poly: that column + a correction')
    ap.add_argument('--bkg-order', type=int, help='Chebyshev terms (default 24 for refit, 8 for the file+poly correction)')
    ap.add_argument('--no-decays', action='store_true', help='leave out the four low-angle exponential decays')
    ap.add_argument('--refine-bkg', action='store_true', help='refine the background polynomial and the peaks together before indexing')
    ap.add_argument('--cif', nargs='+', metavar='FILE', help='screen these CIF structures against the pattern (needs pymatgen)')
    ap.add_argument('--lebail', choices=['fast', 'thorough'], help='also run the rietx Le Bail fit (needs requirements-lebail.txt)')
    a = ap.parse_args(argv)

    if a.write_phases:
        with open(a.write_phases, 'w', encoding='utf-8') as fh:
            fh.write(to_json(validate(EXAMPLE_PHASES)[0]))
        print('wrote', a.write_phases)
        return 0
    if not a.pattern:
        ap.error('give a pattern file')
    if a.phases:
        phases, errs = from_json(open(a.phases, encoding='utf-8').read())
    else:
        phases, errs = validate(EXAMPLE_PHASES)
    if errs:
        print('\n'.join(errs), file=sys.stderr)
        return 2
    try:
        pat = load_pattern(open(a.pattern, 'rb').read(), os.path.basename(a.pattern))
        mode = {'refit': 'step2'}.get(a.background, a.background)
        B = BkgSettings(mode=mode, order=a.bkg_order or (24 if mode == 'step2' else 8), decays=not a.no_decays, refine=a.refine_bkg)
        res = index_pattern(background_and_peaks(pat, bkg=B), phases, a.correction, a.radius)
    except (PatternError, IndexingError) as e:
        print(e, file=sys.stderr)
        return 1
    out = a.out or os.path.splitext(a.pattern)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    report = res.report
    if a.test:
        report += '\nLine tests:\n' + res.test_positions(a.test).to_string(index=False) + '\n'
    print(report)
    with open(out + '_bkg24.csv', 'w') as fh:
        fh.write(res.stage1.pattern_csv())
    res.table.to_csv(out + '_peak_indexing.csv', index=False)
    res.params.to_csv(out + '_refined_parameters.csv', index=False)
    with open(out + '_report.txt', 'w', encoding='utf-8') as fh:
        fh.write(report)
    written = ['_bkg24.csv', '_peak_indexing.csv', '_refined_parameters.csv', '_report.txt']
    if not a.no_figure:
        fig = figures.indexed_figure(res, scale=a.scale)
        for fmt in ('pdf', 'png'):
            with open(f'{out}_indexed.{fmt}', 'wb') as fh:
                fh.write(figures.to_bytes(fig, fmt))
        written.append('_indexed.pdf/.png')
    if a.raw_figure:
        fig = figures.raw_figure(pat, *a.raw_figure)
        for fmt in ('pdf', 'png'):
            with open(f'{out}_pattern_figure.{fmt}', 'wb') as fh:
                fh.write(figures.to_bytes(fig, fmt))
        written.append('_pattern_figure.pdf/.png')
    if a.mp_search:
        from xrdcore import mpsearch
        ok, msg = mpsearch.available()
        els, errs = mpsearch.parse_elements(a.mp_search) if ok else ([], [msg])
        key = os.environ.get('MP_API_KEY', '')
        kf = os.path.expanduser('~/.config/mp/key')
        if not key and os.path.exists(kf):
            key = open(kf).read().strip()
        if not key:
            errs.append('Set MP_API_KEY (or put the key in ~/.config/mp/key) for --mp-search.')
        if errs:
            print('\n'.join(errs), file=sys.stderr)
            return 2
        try:
            cands = mpsearch.fetch_candidates(els, key)
            print(f'\nMaterials Project: {len(cands)} experimentally made entries in the {", ".join(els)} systems; screening…')
            D = mpsearch.screen(res, cands)
        except mpsearch.MPSearchError as e:
            print(e, file=sys.stderr)
            return 1
        T = mpsearch.display_table(D)
        T.to_csv(out + '_mp_candidates.csv', index=False)
        show = T[T.verdict != 'not supported']
        print(show[['verdict', 'formula', 'space group', 'mp-id', 'reason']].to_string(index=False) if len(show) else 'No supported candidates.')
        written.append('_mp_candidates.csv')
    if a.cif:
        from xrdcore import cifphase, mpsearch
        ok, msg = cifphase.available()
        if not ok:
            print(msg, file=sys.stderr)
            return 2
        cands, errs = cifphase.read_cifs([(f, open(f, 'rb').read()) for f in a.cif])
        for e in errs:
            print(e, file=sys.stderr)
        if cands:
            try:
                D = mpsearch.screen(res, cands)
            except mpsearch.MPSearchError as e:
                print(e, file=sys.stderr)
                return 1
            T = mpsearch.display_table(D).rename(columns={'mp-id': 'file'}).drop(columns=['link', 'E above hull (eV/atom)'])
            print('\nCIF candidates:')
            print(T[['verdict', 'formula', 'space group', 'file', 'reason']].to_string(index=False) if len(T) else 'nothing to test in range')
            T.to_csv(out + '_cif_candidates.csv', index=False)
            written.append('_cif_candidates.csv')
    if a.lebail:
        from xrdcore import lebail
        try:
            lb = lebail.run(res, quality=a.lebail, progress=lambda t: print('  Le Bail', t, flush=True))
        except lebail.LeBailError as e:
            print(e, file=sys.stderr)
            return 1
        s = lb.stats
        print(f"\nLe Bail ({a.lebail}, {lb.elapsed:.0f} s): Rwp {s['rwp']:.2f} %, Rexp {s['rexp']:.2f} %, GoF {s['gof']:.2f}, "
              f"DW {s['dw']:.2f}, esds × {s['berar']:.2f} (Bérar–Lelann)")
        for n_ in lb.notes:
            print('  ' + n_)
        T = lb.table_vs_indexing(res)
        print(T.to_string(index=False))
        with open(out + '_lebail_fit.csv', 'w') as fh:
            fh.write(lb.fit_csv())
        lb.params.to_csv(out + '_lebail_parameters.csv', index=False)
        T.to_csv(out + '_lebail_vs_peaks.csv', index=False)
        written += ['_lebail_fit.csv', '_lebail_parameters.csv', '_lebail_vs_peaks.csv']
        if not a.no_figure:
            fig = figures.indexed_figure(lb.as_view(res), scale=a.scale, labels=False)
            for fmt in ('pdf', 'png'):
                with open(f'{out}_lebail.{fmt}', 'wb') as fh:
                    fh.write(figures.to_bytes(fig, fmt))
            written.append('_lebail.pdf/.png')
    print('wrote ' + out + ', '.join(written))
    return 0


if __name__ == '__main__':
    sys.exit(main())
