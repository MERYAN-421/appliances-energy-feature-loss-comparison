#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${EXPLORER_PYTHON:-.venv/bin/python}"
if [ ! -f data/energydata_complete.csv ]; then
  downloaded_csv=$(mktemp)
  trap 'rm -f "$downloaded_csv"' EXIT
  curl -fL --retry 2 https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv -o "$downloaded_csv"
  echo "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d  $downloaded_csv" | sha256sum -c -
  mv "$downloaded_csv" data/energydata_complete.csv
fi
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" test_reproducibility.py
test ! -e /tmp/energydata_complete_backup.csv
"$PYTHON_BIN" test_streamlit_pages.py
