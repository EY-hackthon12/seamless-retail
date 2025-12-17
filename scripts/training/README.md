# Code Agent Training Scripts

This directory contains the fine-tuning pipelines for the Code Agent.

## Directory Structure
- `scripts/training/`: Training scripts and configs.
- `scripts/inference/`: Scripts to test/run trained models.
- `trained_models/`: **Automatic destination** for all trained checkpoints.

## Scripts

### 1. Prototype (Dry Run)
**Script**: `run_prototype_3b.bat` or `setup_and_run_prototype.bat`
**Model**: `Qwen/Qwen2.5-Coder-0.5B` (~1GB, fast download)
**Output**: `trained_models/code_agent_proto`
**Purpose**:
- Validates the entire training loop (Data -> Tokenizer -> Model -> LoRA -> Trainer -> Save).
- Uses generated dummy data (if no dataset provided).
- **Recommended for first-time setup** to verify everything works.

### 2. Production (Full Training)
**Script**: `run_production_15b.bat`
**Model**: `bigcode/starcoder2-15b`
**Output**: `trained_models/code_agent_prod`
**Purpose**:
- The main event. Trains the 15B model.
- **Requirements**: 24GB+ VRAM (or use 4-bit quantization with 12GB+).

### 3. Inference (Testing)
**Script**: `scripts/inference/test_code_agent.py`
**Usage**:
```bash
python scripts/inference/test_code_agent.py --adapter_path trained_models/code_agent_proto/final_adapter --prompt "def hello_world():"
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers peft bitsandbytes datasets trl accelerate
   ```

2. **Run Prototype** (recommended first):
   ```cmd
   scripts\training\setup_and_run_prototype.bat
   ```
   *This checks CUDA, installs PyTorch if needed, and runs training.*
   *Adapter will be saved to `trained_models/code_agent_proto/final_adapter`.*

3. **Test the Model**:
   ```cmd
   python scripts\inference\test_code_agent.py
   ```

## Model Options

| Model | Size | VRAM Required | Download Time |
|-------|------|---------------|---------------|
| Qwen2.5-Coder-0.5B | ~1GB | ~4GB | ~2 min |
| Qwen2.5-Coder-1.5B | ~3GB | ~6GB | ~5 min |
| StarCoder2-3B | ~6GB | ~8GB | ~15 min |
| StarCoder2-15B | ~30GB | ~24GB+ | ~1 hour |

## Troubleshooting

### GPU not being used?
1. Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. If False, reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Check `nvidia-smi` to see GPU processes.
