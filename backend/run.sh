#!/usr/bin/env bash
# Backend launcher (Linux/macOS/Git Bash).
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a

if [ ! -d .venv ]; then
  echo "Creating virtual env..."
  python3 -m venv .venv
fi
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "Starting API on http://127.0.0.1:8000 (H3_MOCK=${H3_MOCK:-unset})"
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
