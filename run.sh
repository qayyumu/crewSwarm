#!/usr/bin/env bash
# crewSwarm — one-command launcher
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r backend/requirements.txt
fi

echo "crewSwarm on http://127.0.0.1:8000  (Ctrl-C to stop)"
exec .venv/bin/uvicorn app.main:app --app-dir backend --port "${PORT:-8000}"
