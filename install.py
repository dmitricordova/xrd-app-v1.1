#!/usr/bin/env python3
"""Set up the MaXXXLab Jr XRD Work-up App in its own virtual environment (.venv next to this file).

    python3 install.py                 # core app + Materials Project/CIF + Le Bail (each optional part skipped if it fails)
    python3 install.py --core-only     # only the packages the indexing and figures need
    python3 install.py --dev           # also pytest, then runs the test suite
    python3 install.py --recreate      # delete .venv and start again

Uses only the Python standard library. Run it with the Python you want the app to use (3.11 or newer;
3.12 or newer for the Materials Project search). setup.sh / setup.bat find a suitable Python and call this.
"""
import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / '.venv'
MIN = (3, 11)
GROUPS = [  # (name, requirements file, minimum Python, import check, what it enables)
    ('core', 'requirements.txt', (3, 11), 'numpy, scipy, pandas, matplotlib, plotly, streamlit', 'the app: raw pattern, indexing, figures'),
    ('mp', 'requirements-mp.txt', (3, 12), 'pymatgen.core, mp_api.client', 'Materials Project search and CIF upload'),
    ('lebail', 'requirements-lebail.txt', (3, 11), 'rietx', 'the Le Bail tab'),
    ('dev', 'requirements-dev.txt', (3, 11), 'pytest', 'the test suite'),
]


def say(msg='', kind=''):
    mark = {'ok': '[ok]   ', 'warn': '[warn] ', 'err': '[error]', 'step': '==>    '}.get(kind, '       ')
    print(f'{mark} {msg}' if msg else '', flush=True)


def venv_python():
    return VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run(cmd, quiet=False):
    r = subprocess.run([str(c) for c in cmd], cwd=HERE, capture_output=quiet, text=True)
    if r.returncode != 0 and quiet:
        tail = '\n'.join(((r.stdout or '') + (r.stderr or '')).strip().splitlines()[-15:])
        say('last lines of the output:\n' + tail)
    return r.returncode == 0


def write_launchers():
    """run.sh / run.command (macOS double-click) / run.bat start the app with the venv's Python."""
    sh = ('#!/bin/sh\n# Start the MaXXXLab Jr XRD Work-up App (created by install.py)\ncd "$(dirname "$0")"\n'
          'if [ ! -x .venv/bin/python ]; then echo "Run ./setup.sh first."; exit 1; fi\n'
          'exec .venv/bin/python -m streamlit run app.py "$@"\n')
    for name in ('run.sh', 'run.command'):
        p = HERE / name
        p.write_text(sh, newline='\n')
        p.chmod(0o755)
    (HERE / 'run.bat').write_text(
        '@echo off\r\nrem Start the MaXXXLab Jr XRD Work-up App (created by install.py)\r\ncd /d "%~dp0"\r\n'
        'if not exist ".venv\\Scripts\\python.exe" (echo Run setup.bat first. & pause & exit /b 1)\r\n'
        '".venv\\Scripts\\python.exe" -m streamlit run app.py %*\r\nif errorlevel 1 pause\r\n', newline='')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--core-only', action='store_true', help='skip the optional Materials Project and Le Bail packages')
    ap.add_argument('--no-mp', action='store_true', help='skip the Materials Project / CIF packages (mp-api, pymatgen)')
    ap.add_argument('--no-lebail', action='store_true', help='skip rietx (the Le Bail tab)')
    ap.add_argument('--dev', action='store_true', help='also install pytest and run the tests')
    ap.add_argument('--recreate', action='store_true', help='delete .venv first')
    a = ap.parse_args()

    say(f'MaXXXLab Jr XRD Work-up App setup in {HERE}', 'step')
    v = sys.version_info[:2]
    say(f'Python {sys.version.split()[0]} at {sys.executable}')
    if v < MIN:
        say(f'Python {MIN[0]}.{MIN[1]} or newer is needed (streamlit 1.66 requires it). Install Python 3.12 or 3.13 from '
            'python.org (Windows/macOS) or your package manager, then run this again with it.', 'err')
        return 1

    if a.recreate and VENV.exists():
        say('deleting the old .venv', 'step')
        shutil.rmtree(VENV)
    if not venv_python().exists():
        say('creating the virtual environment .venv', 'step')
        try:
            venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(VENV)
        except Exception as e:  # e.g. Debian/Ubuntu without python3-venv
            say(f'could not create .venv ({e.__class__.__name__}: {e}).', 'err')
            say('On Debian/Ubuntu install the venv module first:  sudo apt install python3-venv  (or python3.12-venv)')
            return 1
    else:
        say('using the existing .venv (add --recreate to start fresh)', 'ok')
    py = venv_python()
    vpy = subprocess.run([str(py), '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], capture_output=True, text=True).stdout.strip()
    vv = tuple(int(x) for x in vpy.split('.')) if vpy else v

    say('updating pip', 'step')
    run([py, '-m', 'pip', 'install', '--upgrade', 'pip', '--disable-pip-version-check', '-q'], quiet=True)

    wanted = ['core'] + ([] if a.core_only or a.no_mp else ['mp']) + ([] if a.core_only or a.no_lebail else ['lebail']) + (['dev'] if a.dev else [])
    results = {}
    for name, req, need, mods, what in GROUPS:
        if name not in wanted:
            results[name] = 'skipped (not requested)'
            continue
        if vv < need:
            if name == 'mp':     # mp-api needs 3.12+, but pymatgen alone (CIF upload) works on 3.11
                say('installing pymatgen for the CIF upload (the Materials Project client needs Python 3.12+)', 'step')
                ok = run([py, '-m', 'pip', 'install', 'pymatgen>=2024.1.1', '--disable-pip-version-check', '-q'], quiet=True) and \
                    run([py, '-c', 'import pymatgen.core'], quiet=True)
                results[name] = ('CIF upload installed; ' if ok else 'FAILED; ') + f'Materials Project search needs Python 3.12+ (venv: {vpy})'
                say(f'{what}: {results[name]}', 'warn')
                continue
            results[name] = f'skipped: needs Python {need[0]}.{need[1]}+, the venv has {vpy}'
            say(f'{what}: {results[name]}', 'warn')
            continue
        say(f'installing {req}  ({what}; this can take a few minutes)', 'step')
        ok = run([py, '-m', 'pip', 'install', '-r', HERE / req, '--disable-pip-version-check', '-q'], quiet=True)
        if ok:
            ok = run([py, '-c', f'import {mods}'], quiet=True)
        results[name] = 'installed' if ok else 'FAILED'
        say(f'{what}: {results[name]}', 'ok' if ok else ('err' if name == 'core' else 'warn'))
        if name == 'core' and not ok:
            say('The core packages could not be installed. Check the internet connection (or proxy) and run this again.', 'err')
            return 1

    write_launchers()
    say('wrote run.sh, run.command and run.bat', 'ok')
    if a.dev and results.get('dev') == 'installed':
        say('running the tests', 'step')
        run([py, '-m', 'pytest', '-q'])

    say()
    say('Summary', 'step')
    for name, _, _, _, what in GROUPS:
        say(f'{what:45s} {results.get(name, "skipped")}')
    say()
    start = 'run.bat (double-click, or type run.bat)' if os.name == 'nt' else './run.sh   (on macOS you can also double-click run.command)'
    say(f'Start the app with:  {start}', 'step')
    say('A browser tab opens at http://localhost:8501. Stop the app with Ctrl+C in this window.')
    if any(r == 'FAILED' for r in results.values()):
        say('Some optional parts failed; the app runs without them and says what is missing. Re-run install.py to retry.', 'warn')
    return 0


if __name__ == '__main__':
    sys.exit(main())
