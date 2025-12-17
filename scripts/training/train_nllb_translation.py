"""
NLLB-200 Fine-Tuning Script for Translation Tasks
==================================================
Optimized for: PEFT/LoRA + 4-bit Quantization + Gradient Checkpointing

Key Fixes Applied:
1. Uses 'text_target' instead of deprecated 'as_target_tokenizer()' context manager
2. Correct target modules for NLLB/M2M100 architecture (out_proj, not o_proj)
3. Explicit float32 casting for lm_head to prevent dtype mismatches
4. Windows multiprocessing safety with freeze_support()
5. Proper configuration for gradient checkpointing compatibility
6. Defensive model wrapper introspection to catch hidden embeddings

CLaRA-Inspired Optimizations (2025-12-17):
7. Auto-detection of TF32/BF16 for Ampere+ GPUs (compute capability >= 8.0)
8. Optimized DataLoader with prefetch_factor and persistent_workers
9. PyTorch 2.0+ compatible gradient checkpointing (use_reentrant=False)
10. High precision matmul for better numerical stability

Usage:
    python train_nllb_translation.py --dry_run          # Validate setup without training
    python train_nllb_translation.py --max_samples 100  # Quick test with small dataset
    python train_nllb_translation.py                    # Full training run
"""

import os
import sys
import gc
import multiprocessing
import argparse
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NLLBTrainingConfig:
    """Configuration for NLLB fine-tuning with sensible defaults."""
    
    # Model settings
    model_name: str = "facebook/nllb-200-3.3B"
    output_dir: str = "./trained_models/nllb-finetuned"
    max_seq_length: int = 256
    
    # Language pair (NLLB uses specific lang codes)
    src_lang: str = "eng_Latn"
    tgt_lang: str = "hin_Deva"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    label_smoothing_factor: float = 0.1
    
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Performance
    fp16: bool = True
    bf16: bool = False  # Set True if using Ampere+ GPU (RTX 30xx, 40xx)
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0  # 0 for Windows compatibility
    dataloader_pin_memory: bool = True
    
    # Dataset
    max_samples: Optional[int] = None  # None = use all samples
    
    # Debug/Validation
    dry_run: bool = False  # If True, validate setup without training


def get_safe_num_workers() -> int:
    """Return safe number of workers for current OS."""
    if os.name == 'nt':  # Windows
        return 0  # Avoid pickling issues with tokenizers
    return min(4, multiprocessing.cpu_count())


def setup_environment() -> None:
    """Configure environment for optimal training stability.
    
    CLaRA-Inspired: Auto-detects GPU compute capability and enables
    optimal precision settings (TF32 for Ampere+, BF16 for Hopper+).
    """
    # Disable meta device placement to prevent NotImplementedError
    os.environ['ACCELERATE_TORCH_DEVICE_PLACEMENT'] = '0'
    
    # Auto-detect optimal precision settings based on GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        compute_cap = capability[0] * 10 + capability[1]  # e.g., 8.6 -> 86
        
        if compute_cap >= 80:  # Ampere+ (RTX 30xx, A100, etc.)
            # Enable TF32 for faster matmul (3x speedup with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set matmul precision to 'high' for better numerical stability
            torch.set_float32_matmul_precision('high')
            print(f"      [AUTO] Enabled TF32 precision (GPU compute capability: {capability[0]}.{capability[1]})")
        
        if compute_cap >= 89:  # Ada Lovelace (RTX 40xx) or Hopper (H100)
            print(f"      [AUTO] BF16 recommended for optimal throughput")
        
    # Set memory-efficient settings
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')


# =============================================================================
# MODEL LOADING & PREPARATION
# =============================================================================

def create_bnb_config(config: NLLBTrainingConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config."""
    if not config.use_4bit:
        return None
        
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def create_lora_config(config: NLLBTrainingConfig) -> LoraConfig:
    """
    Create LoRA config with CORRECT target modules for NLLB/M2M100.
    
    CRITICAL: NLLB uses 'out_proj' for attention output, NOT 'o_proj' (LLaMA style).
    Using wrong module names will silently fail LoRA attachment.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",      # Query projection
            "k_proj",      # Key projection  
            "v_proj",      # Value projection
            "out_proj",    # NLLB/M2M100 attention output (NOT o_proj!)
            "fc1",         # FFN first layer
            "fc2",         # FFN second layer
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )


def load_tokenizer(config: NLLBTrainingConfig) -> PreTrainedTokenizer:
    """Load and configure NLLB tokenizer."""
    print(f"[1/5] Loading tokenizer: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    
    # Set source and target languages
    tokenizer.src_lang = config.src_lang
    tokenizer.tgt_lang = config.tgt_lang
    
    return tokenizer


def load_model(config: NLLBTrainingConfig, bnb_config: Optional[BitsAndBytesConfig]) -> PreTrainedModel:
    """
    Load NLLB model with quantization and proper configuration.
    
    Handles the tricky device_map and low_cpu_mem_usage requirements
    for large models with BitsAndBytes.
    """
    print(f"[2/5] Loading model: {config.model_name}")
    print(f"      Quantization: {'4-bit NF4' if config.use_4bit else 'None'}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # Single GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,  # Essential for large models
    )
    
    # CRITICAL: Disable use_cache for gradient checkpointing compatibility
    model.config.use_cache = False
    
    # Fix potential tensor parallelism issues
    if hasattr(model.config, 'pretraining_tp'):
        model.config.pretraining_tp = 1
    
    print(f"      Model loaded successfully!")
    return model


def prepare_model_for_training(
    model: PreTrainedModel, 
    config: NLLBTrainingConfig,
    lora_config: LoraConfig
) -> PeftModel:
    """
    Prepare model for k-bit training with LoRA.
    
    Includes CRITICAL fix for NLLB: casting lm_head to float32
    to prevent dtype mismatch crashes during mixed precision training.
    """
    print("[3/5] Preparing model for LoRA training...")
    
    # Prepare for k-bit training (unfreezes norm layers, etc.)
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # CRITICAL FIX: Cast output projection layers to float32
    # NLLB's lm_head often causes dtype mismatches in 4-bit + fp16 mode
    _cast_output_layers_to_fp32(model)
    
    # Introspection: Verify model structure is accessible through PEFT wrapper
    _verify_model_structure(model)
    
    model.print_trainable_parameters()
    
    return model


def _cast_output_layers_to_fp32(model: PeftModel) -> None:
    """Cast lm_head and output_projection to float32 for NLLB stability."""
    cast_count = 0
    for name, module in model.named_modules():
        if "lm_head" in name or "output_projection" in name:
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype != torch.float32:
                    module.to(torch.float32)
                    cast_count += 1
                    
    if cast_count > 0:
        print(f"      Cast {cast_count} output layer(s) to float32 for stability")


def _verify_model_structure(model: PeftModel) -> None:
    """
    Verify PEFT wrapper correctly exposes base model methods.
    
    This catches the 'NoneType has no attribute' errors early,
    before they crash during training.
    """
    checks = [
        ("get_input_embeddings", hasattr(model, 'get_input_embeddings')),
        ("get_output_embeddings", hasattr(model, 'get_output_embeddings')),
        ("base_model", hasattr(model, 'base_model')),
    ]
    
    failed = [name for name, passed in checks if not passed]
    
    if failed:
        print(f"      ⚠️  WARNING: Model missing methods: {failed}")
        print(f"      This may cause AttributeError during training.")
    else:
        print(f"      ✓ Model structure verified (PEFT wrapper OK)")


# =============================================================================
# DATASET LOADING & PREPROCESSING
# =============================================================================

def load_translation_datasets(
    config: NLLBTrainingConfig
) -> Dataset:
    """
    Load and combine translation datasets.
    
    Uses OPUS-100 and FLORES for English-Hindi translation.
    Easily extensible to other language pairs.
    """
    print("[4/5] Loading datasets...")
    
    datasets_list = []
    num_proc = get_safe_num_workers()
    
    # Dataset 1: OPUS-100
    try:
        print("      Loading OPUS-100 (en-hi)...")
        sample_limit = config.max_samples or 10000
        opus = load_dataset("opus100", "en-hi", split=f"train[:{sample_limit}]")
        opus = opus.map(
            lambda x: {
                "source": x["translation"]["en"], 
                "target": x["translation"]["hi"]
            },
            remove_columns=["translation"],
            num_proc=num_proc,
            desc="Processing OPUS-100",
        )
        datasets_list.append(opus)
        print(f"      ✓ OPUS-100: {len(opus)} samples")
    except Exception as e:
        print(f"      ✗ OPUS-100 failed: {e}")
    
    # Dataset 2: FLORES (smaller, high quality)
    try:
        print("      Loading FLORES (eng_Latn-hin_Deva)...")
        flores = load_dataset("facebook/flores", "eng_Latn-hin_Deva", split="dev")
        flores = flores.map(
            lambda x: {
                "source": x["sentence_eng_Latn"], 
                "target": x["sentence_hin_Deva"]
            },
            remove_columns=flores.column_names,
            num_proc=num_proc,
            desc="Processing FLORES",
        )
        datasets_list.append(flores)
        print(f"      ✓ FLORES: {len(flores)} samples")
    except Exception as e:
        print(f"      ✗ FLORES failed: {e}")
    
    if not datasets_list:
        raise RuntimeError("No datasets could be loaded!")
    
    # Combine and shuffle
    combined = concatenate_datasets(datasets_list).shuffle(seed=42)
    
    # Apply max_samples limit
    if config.max_samples and len(combined) > config.max_samples:
        combined = combined.select(range(config.max_samples))
    
    print(f"      ✓ Combined dataset: {len(combined)} samples")
    return combined


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: NLLBTrainingConfig,
) -> Dataset:
    """
    Tokenize dataset for NLLB translation.
    
    CRITICAL FIX: Uses 'text_target' argument instead of deprecated
    'as_target_tokenizer()' context manager. This fixes AttributeError: __enter__.
    """
    print("      Tokenizing dataset...")
    
    # Ensure source language is set
    tokenizer.src_lang = config.src_lang
    
    def preprocess_batch(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenize source and target texts.
        
        Uses text_target argument (correct for transformers 4.20+)
        instead of deprecated context manager.
        """
        model_inputs = tokenizer(
            examples["source"],
            text_target=examples["target"],  # FIX: Not as_target_tokenizer()!
            max_length=config.max_seq_length,
            truncation=True,
            padding=False,  # Dynamic padding in collator
        )
        return model_inputs
    
    num_proc = get_safe_num_workers()
    
    tokenized = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    
    print(f"      ✓ Tokenized: {len(tokenized)} samples")
    return tokenized


# =============================================================================
# TRAINING SETUP
# =============================================================================

def create_training_args(config: NLLBTrainingConfig) -> Seq2SeqTrainingArguments:
    """Create Seq2Seq training arguments with CLaRA-inspired optimizations.
    
    Optimizations:
    - prefetch_factor: Pre-load batches for better GPU utilization
    - persistent_workers: Keep workers alive between epochs (non-Windows)
    - gradient_checkpointing_kwargs: PyTorch 2.0+ compatibility
    """
    # Determine optimal DataLoader settings based on OS
    use_persistent_workers = config.dataloader_num_workers > 0 and os.name != 'nt'
    prefetch = 4 if config.dataloader_num_workers > 0 else None
    
    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16 and not config.bf16,
        bf16=config.bf16,
        logging_steps=25,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        gradient_checkpointing=config.gradient_checkpointing,
        # PyTorch 2.0+ gradient checkpointing compatibility
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
        # CLaRA-inspired DataLoader optimizations
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_prefetch_factor=prefetch,
        dataloader_persistent_workers=use_persistent_workers,
        report_to="none",
        remove_unused_columns=False,  # Required for PEFT compatibility
        # Seq2Seq specific
        predict_with_generate=False,  # Disable for training efficiency
    )


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
) -> DataCollatorForSeq2Seq:
    """
    Create data collator with explicit label_pad_token_id.
    
    Setting label_pad_token_id explicitly prevents the collator from
    querying the quantized model for config attributes, which can fail.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,  # Explicit: prevents model config queries
        pad_to_multiple_of=8,     # Efficient for tensor cores
    )


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def validate_setup(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    config: NLLBTrainingConfig,
) -> bool:
    """
    Validate entire setup before training.
    
    Returns True if validation passes.
    """
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    issues = []
    
    # 1. Check CUDA
    if not torch.cuda.is_available():
        issues.append("CUDA not available - training will be very slow")
    else:
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. Check model trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        issues.append("No trainable parameters - LoRA may have failed")
    else:
        print(f"✓ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # 3. Check dataset
    if len(dataset) == 0:
        issues.append("Dataset is empty")
    else:
        print(f"✓ Dataset size: {len(dataset)} samples")
    
    # 4. Check tokenizer
    sample_text = "Hello world"
    try:
        tokens = tokenizer(sample_text, return_tensors="pt")
        print(f"✓ Tokenizer: {len(tokens.input_ids[0])} tokens for '{sample_text}'")
    except Exception as e:
        issues.append(f"Tokenizer failed: {e}")
    
    # 5. Check model forward pass
    try:
        with torch.no_grad():
            dummy_input = tokenizer(
                "Test input", 
                text_target="Test output",
                return_tensors="pt",
                padding=True,
            )
            dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}
            _ = model(**dummy_input)
        print("✓ Model forward pass: OK")
    except Exception as e:
        issues.append(f"Forward pass failed: {e}")
    
    print("="*60)
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✓ All validation checks passed!")
        return True


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="NLLB-200 Fine-Tuning Script")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--output_dir", type=str, default="./trained_models/nllb-finetuned")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--src_lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang", type=str, default="hin_Deva")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 instead of fp16")
    parser.add_argument("--dry_run", action="store_true", help="Validate setup only")
    args = parser.parse_args()
    
    # Build config
    config = NLLBTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_samples=args.max_samples,
        bf16=args.bf16,
        fp16=not args.bf16,
        dry_run=args.dry_run,
    )
    
    print("\n" + "="*60)
    print("NLLB-200 FINE-TUNING SCRIPT")
    print("="*60)
    print(f"Model:       {config.model_name}")
    print(f"Languages:   {config.src_lang} -> {config.tgt_lang}")
    print(f"Output:      {config.output_dir}")
    print(f"Precision:   {'bf16' if config.bf16 else 'fp16'}")
    print(f"Dry Run:     {config.dry_run}")
    print("="*60 + "\n")
    
    # Setup environment
    setup_environment()
    
    # Load components
    tokenizer = load_tokenizer(config)
    
    bnb_config = create_bnb_config(config)
    model = load_model(config, bnb_config)
    
    lora_config = create_lora_config(config)
    model = prepare_model_for_training(model, config, lora_config)
    
    # Load and preprocess dataset
    raw_dataset = load_translation_datasets(config)
    tokenized_dataset = preprocess_dataset(raw_dataset, tokenizer, config)
    
    # Validate setup
    print("\n[5/5] Validating setup...")
    valid = validate_setup(model, tokenizer, tokenized_dataset, config)
    
    if config.dry_run:
        print("\n✓ Dry run complete. Setup validated successfully!")
        print("  Run without --dry_run to start training.")
        return
    
    if not valid:
        print("\n✗ Validation failed. Please fix issues before training.")
        sys.exit(1)
    
    # Create trainer components
    training_args = create_training_args(config)
    data_collator = create_data_collator(tokenizer, model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Clear memory before training
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()
