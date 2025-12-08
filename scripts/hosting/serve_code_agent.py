
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn
import argparse

app = FastAPI(title="Code Agent Host", description="API to serve trained code agent models")

# Global model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2
    do_sample: bool = True

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # Defaults - can be overridden by env vars or similar if needed for advanced usage
    # For now, hardcoding based on expected usage or passing args if we were running as script
    # But since uvicorn runs this, we'll check ENV vars or default to prototype
    
    base_model_id = os.getenv("BASE_MODEL", "bigcode/starcoder2-3b")
    adapter_path = os.getenv("ADAPTER_PATH", "trained_models/code_agent_proto/final_adapter")
    
    print(f"--> Loading Service...")
    print(f"--> Base Model: {base_model_id}")
    print(f"--> Adapter: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    print("--> Loading Base Model (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        print(f"--> Loading Adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"!! Adapter not found at {adapter_path}. Using base model.")
        model = base_model
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    print("--> Service Ready!")

@app.post("/generate")
async def generate_code(request: GenerateRequest):
    global model, tokenizer
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Optional: strip prompt if desired, but user often wants continuation
    return {"generated_text": generated_text}

if __name__ == "__main__":
    # If run directly
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
