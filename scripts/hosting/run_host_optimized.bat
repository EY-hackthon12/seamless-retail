@echo off
echo ===================================================
echo   Running OPTIMIZED Choice Agent Host
echo   (Supports vLLM if on Linux, Fallback on Windows)
echo ===================================================

set BASE_MODEL=bigcode/starcoder2-3b
set ADAPTER_PATH=trained_models/code_agent_proto/final_adapter
set QUANTIZATION=4bit

:: Activate venv if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Run the optimized server
python scripts/hosting/serve_optimized.py --port 8000

pause
