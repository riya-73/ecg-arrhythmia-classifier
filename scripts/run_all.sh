#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ecg_classifier.download
python -m ecg_classifier.train
pytest -q
python -m ecg_classifier.report
