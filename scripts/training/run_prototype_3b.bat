@echo off
echo Starting PROTOTYPE Training with StarCoder2-3B...
echo This will use dummy data and run for a few steps to verify the pipeline.

python scripts/training/train_code_agent.py ^
    --model_name "bigcode/starcoder2-3b" ^
    --batch_size 2 ^
    --grad_accum 8 ^
    --output_dir "trained_models/code_agent_proto"

echo Prototype run finished.
pause
