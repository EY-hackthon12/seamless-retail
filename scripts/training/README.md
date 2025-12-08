# Code Agent Training Scripts

This directory contains the "Gold Medal" fine-tuning pipelines for the Code Agent.

## Directory Structure
- `scripts/training/`: Training scripts and configs.
- `scripts/inference/`: Scripts to test/run trained models.
- `trained_models/`: **Automatic destination** for all trained checkpoints.

## Scripts

### 1. Prototype (Dry Run)
**Script**: `run_prototype_3b.bat`
**Model**: `bigcode/starcoder2-3b`
**Output**: `trained_models/code_agent_proto`
**Purpose**:
- Validates the entire training loop (Data -> Tokenizer -> Model -> LoRA -> Trainer -> Save).
- Uses generated dummy data (if no dataset provided).

### 2. Production (Full Training)
**Script**: `run_production_15b.bat`
**Model**: `bigcode/starcoder2-15b`
**Output**: `trained_models/code_agent_prod`
**Purpose**:
- The main event. Trains the 15B model.
- **Requirements**: 12GB+ VRAM.

### 3. Inference (Testing)
**Script**: `scripts/inference/test_code_agent.py`
**Usage**:
```bash
python scripts/inference/test_code_agent.py --adapter_path trained_models/code_agent_proto/final_adapter --prompt "def hello_world():"
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch transformers peft bitsandbytes datasets trl accelerate
   ```

2. **Run Prototype**:
   ```cmd
   scripts\training\run_prototype_3b.bat
   ```
   *This will save the adapter to `trained_models/code_agent_proto/final_adapter`.*

3. **Test the Model**:
   ```cmd
   python scripts\inference\test_code_agent.py
   ```
