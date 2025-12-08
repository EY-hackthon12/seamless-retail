@echo off
echo Starting PRODUCTION Training with StarCoder2-15B...
echo WARNING: This requires significant VRAM (12GB+ with 4-bit quantization).

:: In production, point dataset_name to your actual jsonl file, e.g., --dataset_name "data/code_corpus.jsonl"

python scripts/training/train_code_agent.py ^
    --model_name "bigcode/starcoder2-15b" ^
    --learning_rate 1.0e-4 ^
    --batch_size 16 ^
    --grad_accum 4 ^
    --epochs 3 ^
    --lora_r 32 ^
    --lora_alpha 64 ^
    --max_seq_length 8192 ^
    --output_dir "trained_models/code_agent_prod"

echo Production run finished.
pause
