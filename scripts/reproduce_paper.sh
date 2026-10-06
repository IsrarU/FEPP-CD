#!/usr/bin/env bash
# Regenerate every table, number and figure of the paper from the released
# prediction files (CPU only, ~3-4 minutes; no datasets or GPU needed).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/evaluate.py
python scripts/make_figures.py
python scripts/verify_paper.py
