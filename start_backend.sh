#!/usr/bin/env bash
# Start the HP+ Priority Checker backend.
# Run from the hp-priority-checker/ directory.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"
REPO_ROOT="$SCRIPT_DIR/../review-priority-predictor"
VENV="$SCRIPT_DIR/.venv"

# ── Create venv if needed ──────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment …"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── Install dependencies ───────────────────────────────────────────────────
pip install -q -r "$BACKEND/requirements.txt"

# ── Seed author profiles (first run only) ──────────────────────────────────
DB="$BACKEND/hp_checker.db"
if [ ! -f "$DB" ]; then
    echo "Seeding author profiles from dataset …"
    python "$BACKEND/seed_author_profiles.py"
fi

# ── Start Flask ────────────────────────────────────────────────────────────
export HP_CHECKER_PORT=${HP_CHECKER_PORT:-5487}
echo "Starting HP+ backend on http://127.0.0.1:${HP_CHECKER_PORT}"
cd "$BACKEND"
python app.py
