#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo 'Python 3.11+ is required. Set PYTHON_BIN to its executable.' >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
if [ ! -x .venv/bin/python ]; then "$PYTHON_BIN" -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
python -m pip install -r requirements.txt
python -m pip check
npm --prefix frontend ci
npm --prefix frontend run build
echo 'Setup complete. Run: source .venv/bin/activate'
echo 'Then: python -m ml.prepare_data && python -m ml.train_lr'
echo 'For GPU BERT training see scripts/train_v100.sh and README.md.'
