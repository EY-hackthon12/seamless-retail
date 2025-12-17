"""
vLLM Server - High-Throughput LLM Serving
==========================================

Production-grade server for Mistral-7B-Retail using vLLM.
Features:
- PagedAttention for efficient KV-cache
- Continuous batching for high throughput
- AWQ/GPTQ quantization support
- OpenAI-compatible API
- Streaming responses
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import hardware detector for auto-configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from cognitive_brain.core.hardware_detector import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cognitive Brain - vLLM Server",
    description="High-throughput LLM serving for Mistral-7B-Retail",
    version="1.0.0"
)

# Global state
llm = None
tokenizer = None
model_config = {}


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="mistral-7b-retail")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    """Text completion request."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


class GenerationResponse(BaseModel):
    """Generation response."""
    id: str
    object: str = "text_completion"
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string using ChatML format."""
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


@app.on_event("startup")
async def startup_event():
    """Initialize vLLM engine on startup."""
    global llm, model_config
    
    # Detect hardware
    detector = HardwareDetector(verbose=True)
    hw_config = detector.get_recommended_config()
    
    # Get model path from environment or use default
    model_path = os.getenv("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
    
    # Determine quantization based on hardware
    quantization = None
    if hw_config.tier in [HardwareTier.CONSUMER, HardwareTier.LOW_VRAM]:
        quantization = "awq"
        # Use AWQ model if available
        if "awq" not in model_path.lower():
            model_path = os.getenv("MODEL_PATH_AWQ", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
    
    tensor_parallel_size = hw_config.tensor_parallel_size
    gpu_memory_utilization = hw_config.gpu_memory_utilization
    max_model_len = min(hw_config.max_context_length, 8192)  # Cap at 8k for safety
    
    logger.info("=" * 60)
    logger.info("vLLM Server Starting")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Hardware Tier: {hw_config.tier.name}")
    logger.info(f"Quantization: {quantization or 'none'}")
    logger.info(f"Tensor Parallel: {tensor_parallel_size}")
    logger.info(f"Max Context: {max_model_len}")
    logger.info("=" * 60)
    
    model_config = {
        "model": model_path,
        "quantization": quantization,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
    }
    
    try:
        from vllm import LLM
        
        llm = LLM(
            model=model_path,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully!")
        
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
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


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mistral-7b-retail",
                "object": "model",
                "owned_by": "cognitive-brain",
                "permission": []
            }
        ]
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion (OpenAI compatible)."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from vllm import SamplingParams
    import time
    import uuid
    
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    
    start_time = time.perf_counter()
    outputs = llm.generate([request.prompt], sampling_params)
    latency = time.perf_counter() - start_time
    
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    return GenerationResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        model="mistral-7b-retail",
        choices=[{
            "index": 0,
            "text": generated_text,
            "finish_reason": output.outputs[0].finish_reason,
        }],
        usage={
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        }
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion (OpenAI compatible)."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from vllm import SamplingParams
    import time
    import uuid
    
    # Format messages into prompt
    prompt = format_chat_messages(request.messages)
    
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or ["<|im_end|>"],
    )
    
    if request.stream:
        # Streaming response
        async def generate_stream():
            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # Simulate streaming by yielding chunks
            chunk_size = 10
            for i in range(0, len(generated_text), chunk_size):
                chunk = generated_text[i:i+chunk_size]
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "model": "mistral-7b-retail",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.01)
            
            # Final chunk
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    start_time = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    latency = time.perf_counter() - start_time
    
    output = outputs[0]
    generated_text = output.outputs[0].text.strip()
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": "mistral-7b-retail",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": output.outputs[0].finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        }
    }


@app.post("/generate")
async def generate_text(request: CompletionRequest):
    """Simple generation endpoint (non-OpenAI format)."""
    global llm
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from vllm import SamplingParams
    import time
    
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    
    start_time = time.perf_counter()
    outputs = llm.generate([request.prompt], sampling_params)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    output = outputs[0]
    generated_text = output.outputs[0].text
    tokens_generated = len(output.outputs[0].token_ids)
    
    return {
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "latency_ms": round(latency_ms, 2),
        "tokens_per_second": round(tokens_generated / (latency_ms / 1000), 2) if latency_ms > 0 else 0,
    }


def main():
    """Run the vLLM server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Server for Cognitive Brain")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
