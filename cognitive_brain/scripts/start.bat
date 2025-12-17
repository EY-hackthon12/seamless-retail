@echo off
echo ========================================
echo    Cognitive Brain - Quick Start
echo ========================================

cd /d "%~dp0"
cd ..

echo.
echo [1/3] Checking Python environment...
python --version

echo.
echo [2/3] Installing dependencies...
pip install -q -r requirements.txt

echo.
echo [3/3] Starting Brain API...
echo.
echo Server: http://localhost:8000
echo Docs:   http://localhost:8000/docs
echo.

python -m uvicorn cognitive_brain.api:app --host 0.0.0.0 --port 8000
