
import os
import sys
import torch
import transformers
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
import argparse

def train():
    parser = argparse.ArgumentParser(description="Train Code Agent (StarCoder2)")
    parser.add_argument("--model_name", type=str, default="bigcode/starcoder2-3b", help="Model identifier")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name or path to jsonl")
    parser.add_argument("--output_dir", type=str, default="trained_models/code_agent", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Context length (spec says 8192, using 2048 for proto to save VRAM)")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args()

    print(f"--> Starting training with model: {args.model_name}")
    print(f"--> Config: {args}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # StarCoder2 might generally use eos as pad

    # 2. Load Model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False # distinct for training
    )
    
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config
    # Target modules: The spec mentions c_attn etc, but StarCoder2 is Llama-like. 
    # using 'all-linear' is safest and gold-standard for coverage.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear" 
    )

    # 4. Dataset
    if args.dataset_name and os.path.exists(args.dataset_name):
        data = load_dataset("json", data_files=args.dataset_name, split="train")
    elif args.dataset_name:
        data = load_dataset(args.dataset_name, split="train")
    else:
        print("--> No dataset provided, creating dummy code dataset for PROTOTYPE.")
        dummy_data = [
            {"text": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
            {"text": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"},
            {"text": "class Database:\n    def __init__(self):\n        self.data = {}\n    def set(self, key, value):\n        self.data[key] = value\n    def get(self, key):\n        return self.data.get(key)"}
        ] * 10
        data = Dataset.from_list(dummy_data)

    # 5. Training Arguments
    # 5. Training Arguments (SFTConfig for newer trl)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        max_steps=10 if not args.dataset_name else -1, # Run short if dummy
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        save_strategy="no", # For prototype, don't fill disk
        fp16=True,
        optim="paged_adamw_8bit",
        push_to_hub=False,
        report_to="none",
        # SFT specific args
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("--> Training...")
    trainer.train()

    print("--> Saving adapter...")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    print("--> Done!")

if __name__ == "__main__":
    train()
