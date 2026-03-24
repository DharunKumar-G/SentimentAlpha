#!/bin/bash
# SentimentAlpha — Quick Launch Script
# Usage: ./run.sh [--dashboard | --full | --ingest | --prices | --sentiment | --factors | --backtest | --ml | --briefs | --schedule]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the working venv
VENV="$SCRIPT_DIR/venv_new"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Run: python3 -m venv venv_new && source venv_new/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$VENV/bin/activate"

# Default to dashboard if no argument given
if [ $# -eq 0 ]; then
    echo "Launching SentimentAlpha Dashboard..."
    streamlit run dashboard.py --server.port=8501
else
    python3 main.py "$@"
fi
