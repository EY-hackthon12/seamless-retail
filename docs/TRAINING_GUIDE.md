# Training Guide

> Complete guide for fine-tuning models in the Cognitive Retail Brain

---

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements for Training](#hardware-requirements-for-training)
- [Language & Reasoning Agent](#language--reasoning-agent)
- [Code Agent](#code-agent)
- [Translation Agent](#translation-agent)
- [CLaRa Training](#clara-training)
- [Knowledge Retrieval Agent](#knowledge-retrieval-agent)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Cognitive Retail Brain uses multiple specialized agents, each fine-tuned for specific tasks. This guide covers training all agent types with reproducible configurations.

### Agent Overview

| Agent | Base Model | Method | VRAM | Training Time |
|-------|------------|--------|------|---------------|
| Language & Reasoning | Mistral-7B | QLoRA | 12 GB | 2-4 hours |
| Code Generation | StarCoder2-3B | QLoRA | 8 GB | 1-2 hours |
| Translation | NLLB-200 | Full | 8 GB | 4-8 hours |
| RAG Compression | Mistral-7B | CLaRa 3-Stage | 16 GB | 12-24 hours |
| Knowledge Retrieval | Instructor-XL | Contrastive | 16 GB | 2-4 hours |

---

## Hardware Requirements for Training

### Minimum Requirements

```yaml
Training Minimum:
  GPU: RTX 4060 8GB
  RAM: 32 GB
  Storage: 100 GB SSD
  
Supports:
  - 3B parameter models with QLoRA
  - 7B parameter models with aggressive gradient accumulation
  - Small batch sizes (1-4)
```

### Recommended (7B Models)

```yaml
Training Recommended:
  GPU: RTX 4090 24GB (or A100 40GB)
  RAM: 64 GB
  Storage: 500 GB NVMe
  
Supports:
  - 7B parameter models at batch size 8
  - 15B parameter models with QLoRA
  - Full CLaRa training pipeline
```

---

## Language & Reasoning Agent

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | Mistral-7B-Instruct-v0.3 | State-of-the-art instruction model |
| **Method** | QLoRA (4-bit) | Memory-efficient fine-tuning |
| **Learning Rate** | 2e-4 | Standard for QLoRA |
| **Scheduler** | Cosine + Warmup | Smooth convergence |
| **Warmup Ratio** | 0.03 | Gradual LR ramp |
| **Epochs** | 2-3 | Prevent overfitting |
| **Batch Size** | 32 (effective) | Via gradient accumulation |
| **Context Length** | 8192 | Extended reasoning |
| **LoRA Rank** | 64 | High capacity |
| **LoRA Alpha** | 128 | 2× rank (standard) |
| **Target Modules** | All Linear | Maximum adaptation |

### Data Format (ChatML)

```python
# training_data.jsonl
{
    "messages": [
        {"role": "system", "content": "You are a helpful retail assistant..."},
        {"role": "user", "content": "Do you have navy suits in stock?"},
        {"role": "assistant", "content": "Yes! We have navy suits in sizes 38-46..."}
    ]
}
```

### Training Script

```python
# scripts/training/train_language_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA config - target all linear layers
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Training config
training_args = SFTConfig(
    output_dir="trained_models/language_agent",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch: 32
    max_length=8192,
    fp16=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch"
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args
)
trainer.train()
```

### Quick Start

```bash
python scripts/training/train_language_agent.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset_path data/retail_conversations.jsonl \
    --output_dir trained_models/language_agent \
    --epochs 3
```

---

## Code Agent

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | StarCoder2-3B | Code-specialized LLM |
| **Method** | QLoRA (4-bit) | Memory-efficient |
| **Learning Rate** | 1e-4 | Conservative for code |
| **Context Length** | 8192 | Full file context |
| **LoRA Rank** | 32 | Moderate capacity |
| **LoRA Alpha** | 64 | 2× rank |
| **Target Modules** | all-linear | StarCoder2 architecture |

### Data Format

```python
# code_training.jsonl
{
    "text": "def calculate_discount(price, discount_percent):\n    \"\"\"Calculate discounted price.\"\"\"\n    return price * (1 - discount_percent / 100)"
}
```

### Training Command

```bash
python scripts/training/train_code_agent.py \
    --model_name bigcode/starcoder2-3b \
    --dataset_name data/retail_code.jsonl \
    --output_dir trained_models/code_agent \
    --max_seq_length 8192 \
    --learning_rate 1.0e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3 \
    --lora_r 32 \
    --lora_alpha 64
```

### Using the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapter
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
model = PeftModel.from_pretrained(model, "trained_models/code_agent/final_adapter")

# Generate code
prompt = "def validate_credit_card(card_number):"
output = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
```

---

## Translation Agent

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | NLLB-200-Distilled-600M | 200-language support |
| **Method** | Full Fine-tune or LoRA | Based on data size |
| **Learning Rate** | 2e-5 | Conservative for seq2seq |
| **Context Length** | 512 | Translation pairs |
| **Languages** | 25+ supported | See language codes |

### Supported Languages

```python
SUPPORTED_LANGUAGES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "ori_Orya": "Odia",
    "urd_Arab": "Urdu",
    "asm_Beng": "Assamese",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "deu_Latn": "German",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "arb_Arab": "Arabic",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "ita_Latn": "Italian",
    "nld_Latn": "Dutch"
}
```

### Data Format

```python
# translation_pairs.jsonl
{
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "source_text": "Do you have this shirt in size medium?",
    "target_text": "क्या आपके पास यह शर्ट मीडियम साइज़ में है?"
}
```

### Training Command

```bash
python scripts/training/train_nllb_translation.py \
    --model_name facebook/nllb-200-distilled-600M \
    --dataset_path data/retail_translations.jsonl \
    --output_dir trained_models/translation_agent \
    --source_lang eng_Latn \
    --target_langs "hin_Deva,tam_Taml,tel_Telu" \
    --batch_size 16 \
    --epochs 3
```

---

## CLaRa Training

CLaRa uses a three-stage training process for document compression and RAG.

### Stage 1: Compression Pretraining

Train the compressor to learn salient document representations.

```bash
cd ml-clara-main

# Configure environment
export PYTHONPATH=$PWD:$PYTHONPATH

# Run Stage 1
bash scripts/train_pretraining.sh
```

**Key Parameters:**
```bash
--stage stage1
--compress_rate 32
--doc_max_length 256
--mse_loss
--qa_loss
--learning_rate 1e-4
--max_epochs 10
```

### Stage 2: Instruction Tuning

Fine-tune on downstream QA tasks.

```bash
bash scripts/train_instruction_tuning.sh \
    --pretrain_checkpoint stage1/model \
    --stage stage1_2 \
    --generation_top_k 5
```

### Stage 3: End-to-End

Joint reranker-generator training.

```bash
bash scripts/train_stage_end_to_end.sh \
    --pretrain_checkpoint stage2/model \
    --stage stage2 \
    --learning_rate 5e-6
```

### Full Pipeline Script

```bash
#!/bin/bash
# scripts/train_clara_full.sh

set -e

# Stage 1: ~4 hours on A100
echo "=== Stage 1: Compression Pretraining ==="
bash scripts/train_pretraining.sh

# Stage 2: ~4 hours on A100
echo "=== Stage 2: Instruction Tuning ==="
bash scripts/train_instruction_tuning.sh

# Stage 3: ~4 hours on A100
echo "=== Stage 3: End-to-End ==="
bash scripts/train_stage_end_to_end.sh

echo "=== CLaRa Training Complete ==="
```

---

## Knowledge Retrieval Agent

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | Instructor-XL (1.5B) | Instruction-aware embeddings |
| **Method** | Contrastive Fine-tuning | Semantic search optimization |
| **Learning Rate** | 2e-5 | Low LR for encoders |
| **Loss** | MultipleNegativesRankingLoss | In-batch negatives |
| **Batch Size** | 64+ | More negatives = better |
| **Epochs** | 1-2 | Embedding spaces distort easily |
| **Max Seq Length** | 512 | Standard for BERT-based |

### Data Format

```python
# retrieval_pairs.jsonl
{
    "instruction": "Represent the retail product description for retrieval:",
    "query": "comfortable running shoes",
    "positive": "Nike Air Zoom Pegasus - lightweight running shoe with responsive cushioning",
    "negative": "Leather formal Oxford shoes for business meetings"
}
```

### Training with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load model
model = SentenceTransformer('hkunlp/instructor-xl')

# Prepare data
train_examples = [
    InputExample(texts=[query, positive, negative])
    for query, positive, negative in data
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path='trained_models/retrieval_agent'
)
```

---

## Best Practices

### Memory Optimization

```python
# Gradient checkpointing for large models
model.gradient_checkpointing_enable()

# 8-bit optimizer
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)

# Disable caching during training
model.config.use_cache = False
```

### Data Quality

1. **Clean your data**: Remove duplicates, fix encoding issues
2. **Balance classes**: Avoid model bias toward common patterns
3. **Diverse examples**: Cover edge cases and rare scenarios
4. **Quality over quantity**: 1000 high-quality examples > 10000 noisy ones

### Hyperparameter Tuning

```python
# Suggested search space
learning_rates = [1e-5, 5e-5, 1e-4, 2e-4]
lora_ranks = [16, 32, 64]
batch_sizes = [4, 8, 16]

# Start with defaults, then tune:
# 1. Learning rate (most impact)
# 2. Batch size (memory vs convergence trade-off)
# 3. LoRA rank (capacity vs overfitting)
```

### Monitoring Training

```python
# Use Weights & Biases
import wandb
wandb.init(project="cognitive-brain-training")

training_args = SFTConfig(
    report_to="wandb",
    logging_steps=10,
    ...
)
```

---

## Troubleshooting

### Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `per_device_train_batch_size=1`
2. Increase gradient accumulation: `gradient_accumulation_steps=16`
3. Use gradient checkpointing: `model.gradient_checkpointing_enable()`
4. Reduce max length: `max_length=2048`
5. Use 4-bit QLoRA instead of 8-bit

### Loss Not Decreasing

**Solutions:**
1. Lower learning rate (try 1e-5)
2. Check data quality
3. Increase warmup steps
4. Verify labels are correct
5. Check for NaN in inputs

### Model Not Following Instructions

**Solutions:**
1. Use proper prompt template (ChatML for Mistral)
2. Ensure training data uses same format as inference
3. Add more diverse instruction examples
4. Train for more epochs

### Adapter Not Merging

```python
# Ensure compatibility
from peft import PeftModel

# Load and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # Creates merged model
model.save_pretrained("merged_model")
```

---

## Quick Reference

### Training Commands

```bash
# Language Agent
python scripts/training/train_language_agent.py --epochs 3

# Code Agent
python scripts/training/train_code_agent.py --epochs 3

# Translation Agent
python scripts/training/train_nllb_translation.py --epochs 3

# CLaRa (Full)
bash ml-clara-main/scripts/train_all_stages.sh
```

### Verify Training Output

```bash
# Check adapter files
ls -la trained_models/*/final_adapter/

# Test inference
python scripts/verify_training.py --model_path trained_models/code_agent
```

---

## Next Steps

- [LLM Hosting Guide](LLM_HOSTING_GUIDE.md) - Deploy trained models
- [Hardware Requirements](HARDWARE_REQUIREMENTS.md) - Training hardware
- [CLaRa Integration](CLARA_INTEGRATION.md) - Advanced RAG training
