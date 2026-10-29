@echo off
rem Set up the MaXXXLab Jr XRD Work-up App: finds Python 3.11+, creates .venv and installs everything.
rem   setup.bat                 core + Materials Project/CIF + Le Bail
rem   setup.bat --core-only     core only        (other options: --no-mp --no-lebail --dev --recreate)
cd /d "%~dp0"
set "PY="
for %%V in (3.13 3.12 3.11) do (
  if not defined PY (
    py -%%V -c "import venv" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys, venv; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [error] No Python 3.11 or newer found.
  echo         Install Python 3.12 or 3.13 from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
  echo         then run setup.bat again.
  pause
  exit /b 1
)
%PY% install.py %*
pause
