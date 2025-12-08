@echo off
echo Starting Code Agent Host...
echo Model: StarCoder2-3B (Prototype)
echo Adapter: trained_models/code_agent_proto/final_adapter
echo.
echo Server will run at http://localhost:8000
echo Swagger UI at http://localhost:8000/docs
echo.

set BASE_MODEL=bigcode/starcoder2-3b
set ADAPTER_PATH=trained_models/code_agent_proto/final_adapter

python scripts/hosting/serve_code_agent.py --port 8000

pause
