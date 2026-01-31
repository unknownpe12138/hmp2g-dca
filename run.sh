#!/bin/bash
# HMP2G-DCA Environment Quick Start Script for Linux

echo "========================================"
echo "HMP2G-DCA Environment Quick Start"
echo "========================================"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

echo "[1/4] Installing dependencies..."
pip3 install -r requirements.txt

echo "[2/4] Building Cython modules..."
python3 setup.py build_ext --inplace

echo "[3/4] Creating directories..."
mkdir -p TEMP/build
mkdir -p RESULT

echo "[4/4] Starting DCA environment..."
echo "========================================"
python3 main.py -c DOCS/examples/dca/example_dca.jsonc
