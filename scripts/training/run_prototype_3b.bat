@echo off
echo Starting PROTOTYPE Training with Qwen2.5-Coder-0.5B...
echo This will use dummy data and run for a few steps to verify the pipeline.
echo Using smaller model for faster download and training.

python scripts/training/train_code_agent.py ^
    --model_name "Qwen/Qwen2.5-Coder-0.5B" ^
    --batch_size 4 ^
    --grad_accum 4 ^
    --output_dir "trained_models/code_agent_proto"

echo Prototype run finished.
pause
