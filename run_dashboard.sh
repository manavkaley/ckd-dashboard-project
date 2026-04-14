#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python ]; then
  echo "Virtual environment not found. Create it with: python -m venv .venv"
  exit 1
fi

. .venv/bin/activate
python -m streamlit run ckd_dashboard.py --server.address=0.0.0.0 --server.port=8501
