@echo off
REM HMP2G-DCA Environment Quick Start Script for Windows

echo ========================================
echo HMP2G-DCA Environment Quick Start
echo ========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt

echo [2/4] Building Cython modules...
python setup.py build_ext --inplace

echo [3/4] Creating directories...
if not exist "TEMP" mkdir TEMP
if not exist "TEMP\build" mkdir TEMP\build
if not exist "RESULT" mkdir RESULT

echo [4/4] Starting DCA environment...
echo ========================================
python main.py -c DOCS/examples/dca/example_dca.jsonc

pause
