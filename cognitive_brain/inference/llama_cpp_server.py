"""
llama.cpp Server - CPU/Low-VRAM Inference
==========================================

Fallback server for consumer hardware using GGUF quantized models.
Works on CPU-only systems and GPUs with <8GB VRAM.

Features:
- GGUF model support (Q4_K_M, Q5_K_M, Q8_0)
- Partial GPU offloading
- OpenAI-compatible API
- Streaming responses
- Low memory footprint
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import hardware detector
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from cognitive_brain.core.hardware_detector import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cognitive Brain - llama.cpp Server",
    description="CPU/Low-VRAM LLM serving using GGUF models",
    version="1.0.0"
)

# Global state
llm = None
model_config = {}


class ChatMessage(BaseModel):
    """Chat message."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request."""
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    """Text completion request."""
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into ChatML prompt."""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def find_gguf_model(model_dir: str = "models") -> Optional[str]:
    """Find a GGUF model file in the models directory."""
    models_path = Path(model_dir)
    
    # Check environment variable first
    if os.getenv("GGUF_MODEL_PATH"):
        return os.getenv("GGUF_MODEL_PATH")
    
    # Search for GGUF files
    if models_path.exists():
        gguf_files = list(models_path.glob("**/*.gguf"))
        if gguf_files:
            # Prefer Q4_K_M models for balance of quality/speed
            for f in gguf_files:
                if "q4_k_m" in f.name.lower():
                    return str(f)
            return str(gguf_files[0])
    
    return None


@app.on_event("startup")
async def startup_event():
    """Initialize llama.cpp engine on startup."""
    global llm, model_config
    
    # Detect hardware
    detector = HardwareDetector(verbose=True)
    hw_config = detector.get_recommended_config()
    profile = detector.detect()
    
    # Find model
    model_path = find_gguf_model()
    
    if not model_path:
        logger.warning("No GGUF model found. Set GGUF_MODEL_PATH environment variable.")
        logger.warning("Download a model from: https://huggingface.co/TheBloke")
        return
    
    # Calculate GPU layers based on hardware
    n_gpu_layers = 0
    if profile.has_gpu:
        vram_gb = profile.best_gpu.total_vram_gb
        if vram_gb < 4:
            n_gpu_layers = 10  # Minimal offload
        elif vram_gb < 8:
            n_gpu_layers = 25  # Partial offload
        else:
            n_gpu_layers = -1  # Full offload
    
    # Context size based on available resources
    n_ctx = 4096
    if profile.has_gpu and profile.best_gpu.total_vram_gb >= 8:
        n_ctx = 8192
    
    n_threads = os.cpu_count() // 2 or 4
    
    logger.info("=" * 60)
    logger.info("llama.cpp Server Starting")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Hardware Tier: {hw_config.tier.name}")
    logger.info(f"GPU Layers: {n_gpu_layers}")
    logger.info(f"Context Size: {n_ctx}")
    logger.info(f"CPU Threads: {n_threads}")
    logger.info("=" * 60)
    
    model_config = {
        "model": model_path,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
    }
    
    try:
        from llama_cpp import Llama
        
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("Model loaded successfully!")
        
    except ImportError:
        logger.error("llama-cpp-python not installed.")
        logger.error("Install with: pip install llama-cpp-python")
        llm = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        llm = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if llm is not None else "unhealthy",
        "model_loaded": llm is not None,
        "config": model_config
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    import uuid
    
    if request.stream:
        async def generate_stream():
            for output in llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                echo=False,
                stream=True
            ):
                token = output['choices'][0]['text']
                data = {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": token, "finish_reason": None}]
                }
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    start_time = time.perf_counter()
    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        echo=False,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    generated_text = output['choices'][0]['text']
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "model": "gguf-local",
        "choices": [{
            "index": 0,
            "text": generated_text,
            "finish_reason": output['choices'][0].get('finish_reason', 'stop'),
        }],
        "usage": output.get('usage', {})
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    import uuid
    
    prompt = format_chat_prompt(request.messages)
    stop_tokens = request.stop or ["<|im_end|>"]
    
    if request.stream:
        async def generate_stream():
            for output in llm(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_tokens,
                echo=False,
                stream=True
            ):
                token = output['choices'][0]['text']
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    start_time = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop_tokens,
        echo=False,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    generated_text = output['choices'][0]['text'].strip()
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": "gguf-local",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": output['choices'][0].get('finish_reason', 'stop'),
        }],
        "usage": output.get('usage', {})
    }


@app.post("/generate")
async def generate_text(request: CompletionRequest):
    """Simple generation endpoint."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    
    start_time = time.perf_counter()
    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        echo=False,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    generated_text = output['choices'][0]['text']
    tokens_generated = output['usage']['completion_tokens']
    
    return {
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "latency_ms": round(latency_ms, 2),
        "tokens_per_second": round(tokens_generated / (latency_ms / 1000), 2) if latency_ms > 0 else 0,
    }


def main():
    """Run the llama.cpp server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="llama.cpp Server for Cognitive Brain")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
