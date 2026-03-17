#!/bin/bash
cd "$(dirname "$0")"
echo "Starting AiCut Application..."
echo

PYTHON_EXE=""

if [ -f "/opt/anaconda3/bin/python" ]; then
    PYTHON_EXE="/opt/anaconda3/bin/python"
elif [ -f "$HOME/anaconda3/bin/python" ]; then
    PYTHON_EXE="$HOME/anaconda3/bin/python"
elif [ -f "$HOME/miniconda3/bin/python" ]; then
    PYTHON_EXE="$HOME/miniconda3/bin/python"
elif command -v python &> /dev/null; then
    PYTHON_EXE="python"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo

$PYTHON_EXE run_with_log.py

if [ $? -ne 0 ]; then
    echo
    echo "Error: Application exited with error code $?"
fi
