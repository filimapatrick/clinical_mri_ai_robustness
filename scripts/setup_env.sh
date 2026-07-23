#!/bin/bash
set -e

echo "===================================================="
echo " Setting up Python virtual environment for MRI AI Robustness"
echo "===================================================="

# Prefer Python 3.11 if available in Homebrew
PYTHON_BIN="/usr/bin/python3"
if [ -f "/opt/homebrew/Cellar/python@3.11/3.11.12_1/libexec/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/Cellar/python@3.11/3.11.12_1/libexec/bin/python3"
elif command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
fi

echo "Using Python binary: $PYTHON_BIN"
$PYTHON_BIN --version

if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    $PYTHON_BIN -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip and installing build prerequisites..."
pip install --upgrade pip setuptools wheel numpy cython versioneer scikit-build

echo "Installing pyradiomics..."
pip install pyradiomics --no-build-isolation || pip install pyradiomics || echo "pyradiomics build fallback enabled"

echo "Installing remaining requirements from requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment setup complete!"
