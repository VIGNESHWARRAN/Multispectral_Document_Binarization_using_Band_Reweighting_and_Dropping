#!/usr/bin/env bash
# scripts/start.sh
# ─────────────────
# Starts both the FastAPI backend and the Streamlit frontend.
# Run from the repo root: bash scripts/start.sh

set -e

# ── Detect OS (Windows Git Bash vs Unix) ──────────────────────────────────────
IS_WINDOWS=false
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || -n "$WINDIR" ]]; then
    IS_WINDOWS=true
fi

# ── Activate venv if present ──────────────────────────────────────────────────
if [ "$IS_WINDOWS" = true ]; then
    if [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
        echo "✅ Virtual environment activated (Windows)"
    fi
else
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

# ── Install / upgrade dependencies ────────────────────────────────────────────
echo "📦 Installing dependencies from requirements.txt ..."
python -m pip install -q -r requirements.txt
echo "✅ Dependencies ready."

# ── Copy .env.example to .env if .env doesn't exist ──────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  Created .env from .env.example — please fill in your Auth0 credentials."
fi

# ── Start backend in background ───────────────────────────────────────────────
echo ""
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
python -m uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# ── Give backend a moment to start ────────────────────────────────────────────
sleep 2

# ── Start Streamlit frontend (foreground) ─────────────────────────────────────
echo "🚀 Starting Streamlit frontend on http://localhost:8501 ..."
echo ""
python -m streamlit run frontend/app.py --server.port 8501

# ── Cleanup on exit ───────────────────────────────────────────────────────────
kill $BACKEND_PID 2>/dev/null
echo "✅ AMDES stopped."
