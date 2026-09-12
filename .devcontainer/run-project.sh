#!/usr/bin/env bash
# Runs Cadence end to end. Python 3.11+ and NumPy (one pip install) are the
# only requirements.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=============================================================="
echo " Cadence - soft-computing traffic signal control."
echo " Python $(python --version 2>&1 | awk '{print $2}'), NumPy only."
echo "=============================================================="

for suite in trafficsim fuzzy optimisers softcomputing; do
  echo
  echo "[test] $suite"
  python tests/test_$suite.py | tail -2
done

echo
echo "[experiments] writing reports/*.json"
python run.py

echo
echo "=============================================================="
echo " Every number in the README is read from reports/*.json:"
echo " fuzzy vs best fixed, the GA-vs-grid-search offset comparison,"
echo " Hopfield capacity, and the XOR wall."
echo "=============================================================="
