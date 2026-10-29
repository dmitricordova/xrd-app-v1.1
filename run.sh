#!/bin/sh
# Start the MaXXXLab Jr XRD Work-up App (created by install.py)
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run ./setup.sh first."; exit 1; fi
exec .venv/bin/python -m streamlit run app.py "$@"
