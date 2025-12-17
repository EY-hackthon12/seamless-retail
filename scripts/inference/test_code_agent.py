
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import os

def test_inference():
    parser = argparse.ArgumentParser(description="Test Trained Code Agent")
    parser.add_argument("--base_model", type=str, default="bigcode/starcoder2-3b", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default="trained_models/code_agent_proto/final_adapter", help="Path to trained LoRA adapter")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):", help="Code prompt to complete")
    args = parser.parse_args()

    print(f"--> Loading base model: {args.base_model}")
    
    # Load 4-bit base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"--> Loading adapter from: {args.adapter_path}")
    if os.path.exists(args.adapter_path):
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"!! Adapter path {args.adapter_path} not found. Running with base model only.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    
    print(f"--> Generating for prompt: '{args.prompt}'")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.2, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("-" * 40)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 40)

if __name__ == "__main__":
    test_inference()
