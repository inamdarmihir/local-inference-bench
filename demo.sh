#!/usr/bin/env bash
# demo.sh
#
# One-command, clone-and-run demo: venv -> pip install -> embed the in-repo
# sample_corpus/ with FastEmbed -> price it against cited API pricing.
#
# This produces numbers from sample_corpus/ (6 short demo files), NOT the
# published results_local.json / results_local_warm.json numbers quoted in
# README.md, which came from a different, uncommitted 38-chunk corpus. See
# README.md "Inspect the published numbers" for those. This script never
# touches results_local.json or results_local_warm.json.
#
# Usage:
#   ./demo.sh
#
# Safe to re-run; it reuses the venv and cached model on subsequent runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="${VENV_DIR:-.venv}"
OUT="${OUT:-results_local_sample.json}"

echo "==> Using venv at ${VENV_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Installing pinned dependencies (requirements.txt)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo
echo "==> Running FastEmbed locally against sample_corpus/ (demo data, not the published corpus)"
python3 run_local_benchmark.py --corpus-dir sample_corpus --out "${OUT}"

echo
echo "==> Pricing that run against cited external pricing (offline, no API key used)"
python3 cost_comparison.py --results "${OUT}"

echo
echo "============================================================"
echo "Demo complete. Wrote ${OUT}."
echo "These numbers are from the small demo corpus in sample_corpus/,"
echo "not the published 38-chunk numbers in README.md - see README.md"
echo "\"Inspect the published numbers\" to inspect those instead."
echo "============================================================"
