@echo off
rem Start the MaXXXLab Jr XRD Work-up App (created by install.py)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
".venv\Scripts\python.exe" -m streamlit run app.py %*
if errorlevel 1 pause
