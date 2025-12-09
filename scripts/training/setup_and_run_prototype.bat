@echo off
echo Checking CUDA availability...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %errorlevel% neq 0 (
    echo CUDA not found. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo Failed to install PyTorch. Exiting.
        exit /b %errorlevel%
    )
) else (
    echo CUDA is available.
)

echo.
echo Starting Training with Qwen2.5-Coder-0.5B (fast, ~1GB model)...
python scripts/training/train_code_agent.py ^
    --model_name "Qwen/Qwen2.5-Coder-0.5B" ^
    --batch_size 4 ^
    --grad_accum 4 ^
    --output_dir "trained_models/code_agent_proto"

echo.
echo Training complete!
pause
