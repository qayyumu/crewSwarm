#!/usr/bin/env bash
# crewSwarm — one-command launcher
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  # crewai's pinned tiktoken has no Python 3.14 wheels, so prefer <=3.13
  PYBIN="${PYTHON:-}"
  if [ -z "$PYBIN" ]; then
    for cand in python3.13 python3.12 /opt/anaconda3/envs/agents/bin/python; do
      if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
        PYBIN="$cand"; break
      fi
    done
  fi
  [ -z "$PYBIN" ] && PYBIN=python3
  echo "creating venv with $PYBIN"
  "$PYBIN" -m venv .venv
  .venv/bin/pip install -r backend/requirements.txt
  echo "optional LLM mode: .venv/bin/pip install crewai && put OPENAI_API_KEY in .env"
fi

echo "crewSwarm on http://127.0.0.1:8000  (Ctrl-C to stop)"
exec .venv/bin/uvicorn app.main:app --app-dir backend --port "${PORT:-8000}"
