
import os
import torch
import asyncio
import uuid
import time
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

# --- Configuration & optimization vars ---
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.01")) # 10ms dynamic batch window

# --- Data Models ---
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False

# --- Global Components ---
class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print("--> [Engine] Initializing Ultra-Fast Stack...")
        
        base_model_id = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-0.5B")
        adapter_path = os.getenv("ADAPTER_PATH", "trained_models/code_agent_proto/final_adapter")
        
        # 1. Quantization (4-bit NF4) - Memory optimization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"--> [Engine] Loading Base: {base_model_id}")
        # 2. Low_cpu_mem_usage=True is default for accelerate, but good to be explicit
        # 3. attn_implementation="flash_attention_2" if hardware supports it (Ampere+)
        use_flash = False 
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                # use_flash = True
                print("--> [Engine] Flash Attention 2 disabled (package missing)")

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # DEBUG MODE: Adapter disabled because we switched base models
        # if os.path.exists(adapter_path):
        #     print(f"--> [Engine] Injecting LoRA Adapter: {adapter_path}")
        #     self.model = PeftModel.from_pretrained(self.model, adapter_path)
        # else:
        print(f"!! [Engine] Running in DEBUG MODE (No Adapter).")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Warmup
        print("--> [Engine] Warmup pass...")
        with torch.no_grad():
            _ = self.model.generate(**self.tokenizer("def test():", return_tensors="pt").to(self.device), max_new_tokens=10)
        print("--> [Engine] Ready.")

    def generate_stream(self, request: GenerateRequest):
        inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Threaded generation to allow non-blocking streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
        
        thread.join()

engine = InferenceEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    engine.load_model()
    yield
    # Shutdown (cleanup if needed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Research-Grade Code Agent", lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev/hackathon, allow all. In prod, lock this down.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/completions")
async def generate(request: GenerateRequest):
    """
    OpenAI-compatible-ish completion endpoint.
    Supports high-performance streaming.
    """
    if request.stream:
        return StreamingResponse(
            engine.generate_stream(request), 
            media_type="text/event-stream"
        )
    else:
        # Non-streaming fallback
        full_text = ""
        for chunk in engine.generate_stream(request):
            full_text += chunk
        return {"id": str(uuid.uuid4()), "choices": [{"text": full_text}]}

# --- New Frontend API Support ---

@app.post("/api/v1/sessions")
async def create_session():
    """Creates a new session ID for the chat frontend."""
    return {"session_id": str(uuid.uuid4())}

@app.websocket("/api/v1/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_message = payload.get("message", "")
            
            # Construct a prompt for the code agent
            # You might want to format this better with a system prompt if the model supports it
            prompt_formatted = f"User: {user_message}\nAssistant:"
            
            req = GenerateRequest(prompt=prompt_formatted, stream=True, max_new_tokens=512)
            
            # Stream response back to WebSocket
            for chunk in engine.generate_stream(req):
                await websocket.send_json({
                    "type": "message",
                    "content": chunk
                })
            
            # Signal end of turn
            await websocket.send_json({"type": "end"})
            
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/health")
def health():
    return {"status": "ok", "device": engine.device}

if __name__ == "__main__":
    import uvicorn
    # 4. Loop setup for performance
    uvicorn.run(
        "scripts.hosting.serve_code_agent:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        workers=1 # One worker per GPU usually
    )
