#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then source .venv/bin/activate; fi
if [ ! -f frontend/dist/index.html ]; then
  echo 'Build React first: npm --prefix frontend ci && npm --prefix frontend run build' >&2
  exit 1
fi
exec python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}" --no-access-log
