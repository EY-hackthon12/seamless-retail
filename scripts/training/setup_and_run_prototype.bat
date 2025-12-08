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

echo Starting Training...
call scripts\training\run_prototype_3b.bat
