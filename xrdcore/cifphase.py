"""Candidate phases from CIF files (needs pymatgen).

    cands, errors = cifphase.read_cifs([(filename, bytes), ...])
    table = mpsearch.screen(index_result, cands, modes)          # the same screen as the Materials Project candidates
    phase = mpsearch.candidate_phase(row, names, i, x_range=...)  # cell refined to the peaks, or start_cell=True for the CIF cell
    phase = cifphase.direct_phase(cand, names, i, x_range=...)    # without screening: the CIF cell, lines >= 2 %

Each data block of each file becomes one candidate: the structure is read with its symmetry (and partial occupancies),
reduced to the conventional standard cell (symprec 0.1, as for Materials Project entries) and its reflections and
intensities are calculated for Cu Kα.
"""
import io
import os
import re
import warnings
from types import SimpleNamespace


class CIFError(ValueError):
    pass


def available():
    try:
        import pymatgen.core  # noqa: F401
        from pymatgen.io.cif import CifParser  # noqa: F401
        return True, ''
    except Exception as e:  # pragma: no cover
        return False, f'Reading CIF files needs pymatgen: pip install -r requirements-mp.txt ({e.__class__.__name__}: {e})'


def _blocks(text):
    return re.findall(r'^\s*data_(\S*)', text, flags=re.M)


def read_cifs(files, max_atoms=500):
    """files: [(name, bytes)]. Returns (candidates, errors). Candidate keys as mpsearch.fetch_candidates, plus source='cif', file."""
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    out, errors = [], []
    for name, data in files:
        text = data.decode('utf-8', errors='replace') if isinstance(data, (bytes, bytearray)) else str(data)
        stem = os.path.splitext(os.path.basename(name))[0]
        if not _blocks(text):
            errors.append(f'{name}: not a CIF file (no data_ block).')
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                parser = CifParser(io.StringIO(text), occupancy_tolerance=1.05)
                structs = parser.parse_structures(primitive=False)
        except Exception as e:
            msg = str(e).split('\n')[0][:200]
            errors.append(f'{name}: could not be read as a crystal structure ({msg}). '
                          'It needs the cell (_cell_length_a … _cell_angle_gamma), the space group or symmetry operators, and atom sites.')
            continue
        if not structs:
            errors.append(f'{name}: no structure with atom sites was found.')
            continue
        blocks = _blocks(text)
        for k, st in enumerate(structs):
            label = stem if len(structs) == 1 else f'{stem} [{blocks[k] if k < len(blocks) else k + 1}]'
            try:
                sga = SpacegroupAnalyzer(st, symprec=0.1)
                conv = sga.get_conventional_standard_structure()
                system, sg = sga.get_crystal_system(), sga.get_space_group_symbol()
            except Exception as e:
                errors.append(f'{label}: the symmetry could not be found ({e.__class__.__name__}).')
                continue
            if len(conv) > max_atoms:
                errors.append(f'{label}: {len(conv)} atoms in the conventional cell; more than {max_atoms} is too slow to screen.')
                continue
            formula = conv.composition.reduced_formula
            out.append(dict(mpid=f'cif:{label}', file=label, formula=formula, chemsys='-'.join(sorted(e.symbol for e in conv.composition.elements)),
                            sg=sg, system=system, e_hull=0.0, icsd='', structure=conv, source='cif'))
    return out, errors


def direct_phase(cand, existing_names=(), color_index=0, x_range=(5.0, 90.0), series='all', min_intensity=2.0):
    """A phase from the CIF as it is (no screening): CIF cell, its reflections of >= min_intensity % in the scan."""
    import numpy as np

    from . import mpsearch as M
    st = cand['structure']
    lat = st.lattice
    p0 = np.array([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma], float)
    H, F = M._reflections(st, 'powder')
    row = SimpleNamespace(p=p0, p0=p0, H=H, F=F, system=cand['system'], sg=cand['sg'], formula=cand['formula'], mpid=cand['mpid'],
                          mode_key='powder', source='cif', file=cand['file'])
    ph = M.candidate_phase(row, existing_names, color_index, min_intensity=min_intensity, x_range=x_range, series=series, start_cell=True)
    if not ph['hkl']:
        raise CIFError(f"{cand['file']}: no reflection of {min_intensity:g} % or more falls in {x_range[0]:.1f}–{x_range[1]:.1f}° 2θ.")
    return ph
