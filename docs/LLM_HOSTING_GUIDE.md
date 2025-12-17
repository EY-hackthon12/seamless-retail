# LLM Hosting Guide

> Complete guide for hosting and serving LLMs in the Cognitive Retail Brain

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Backend Selection](#backend-selection)
- [Hardware Auto-Scaling](#hardware-auto-scaling)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Topologies](#deployment-topologies)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Cognitive Retail Brain uses an **Adaptive Inference Engine** that automatically detects hardware capabilities and configures the optimal LLM serving backend. This ensures the system runs efficiently across a wide range of hardware—from CPU-only laptops to multi-GPU datacenter servers.

### Key Features

| Feature | Description |
|---------|-------------|
| **Auto Hardware Detection** | Scans CUDA devices, VRAM, compute capability |
| **Dynamic Backend Selection** | Chooses vLLM, llama.cpp, or PyTorch based on hardware |
| **Quantization Optimization** | Automatic AWQ/GPTQ/GGUF selection |
| **Multi-GPU Support** | Tensor parallelism for distributed inference |
| **Streaming Responses** | Real-time token streaming for chat interfaces |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ADAPTIVE INFERENCE ENGINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Hardware Detector │───▶│ Inference Config │───▶│ Backend Selector │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
│           │                                               │              │
│           ▼                                               ▼              │
│  ┌──────────────────┐                          ┌──────────────────┐     │
│  │ GPU Enumeration  │                          │   Engine Init    │     │
│  │ • VRAM Detection │                          │   • vLLM         │     │
│  │ • Compute Cap    │                          │   • llama.cpp    │     │
│  │ • Multi-GPU      │                          │   • PyTorch      │     │
│  └──────────────────┘                          └──────────────────┘     │
│                                                         │               │
│                                                         ▼               │
│                                              ┌──────────────────┐       │
│                                              │  Unified API     │       │
│                                              │  • generate()    │       │
│                                              │  • stream()      │       │
│                                              └──────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Backend Selection

### Decision Matrix

| Condition | Backend | Quantization | Reason |
|-----------|---------|--------------|--------|
| No GPU detected | llama.cpp | Q4_K_M | CPU-optimized inference |
| VRAM < 8GB | llama.cpp | Q4_K_M | Partial GPU offload |
| VRAM 8-12GB | vLLM | AWQ 4-bit | High throughput with quantization |
| VRAM 12-24GB | vLLM | FP16 | Full precision, high context |
| VRAM > 24GB | vLLM | BF16 | Maximum quality |
| Multi-GPU | vLLM + TP | FP16 | Tensor parallelism |

### Backend Comparison

#### vLLM (Recommended for GPUs ≥8GB)

```python
# Automatically selected for consumer+ tier hardware
from cognitive_brain.inference.adaptive_engine import AdaptiveInferenceEngine

engine = AdaptiveInferenceEngine()
engine.load_model("mistral-7b-instruct-v0.3-awq")

result = engine.generate(
    prompt="What products do you recommend?",
    max_new_tokens=256,
    temperature=0.7
)
```

**Advantages:**
- PagedAttention for efficient KV-cache
- Continuous batching for high throughput
- AWQ/GPTQ quantization support
- Tensor parallelism for multi-GPU

#### llama.cpp (CPU or Low VRAM)

```python
# Automatically selected when VRAM < 8GB
engine = AdaptiveInferenceEngine()
engine.load_model("mistral-7b-instruct-v0.3.Q4_K_M.gguf")

# Streaming inference
async for token in engine.generate_stream(prompt, max_new_tokens=256):
    print(token, end="", flush=True)
```

**Advantages:**
- Runs on CPU-only systems
- Efficient GGUF quantization (Q4_K_M, Q5_K_M)
- Partial GPU offload for hybrid systems
- Low memory footprint

#### PyTorch Native (Custom Models)

```python
# For fine-tuned adapters or custom architectures
from cognitive_brain.inference.adaptive_engine import PyTorchEngine

engine = PyTorchEngine()
engine.load_model(
    "trained_models/code_agent/final_adapter",
    load_in_4bit=True
)
```

**Advantages:**
- Full compatibility with HuggingFace models
- LoRA/PEFT adapter support
- BitsAndBytes quantization
- `torch.compile()` optimization

---

## Hardware Auto-Scaling

### Detection Process

```python
from cognitive_brain.core.hardware_detector import HardwareDetector

detector = HardwareDetector()
profile = detector.detect()

print(f"GPUs: {profile.gpu_count}")
print(f"Total VRAM: {profile.total_vram_gb:.1f} GB")
print(f"Best GPU: {profile.best_gpu.name}")

config = detector.get_recommended_config()
print(f"Tier: {config.tier.name}")
print(f"Backend: {config.backend.value}")
print(f"Quantization: {config.quantization.value}")
```

### Hardware Tiers

#### CPU_ONLY
```yaml
VRAM: N/A
Backend: llama.cpp
Quantization: Q4_K_M (GGUF)
Max Batch: 1
Max Context: 2048
Models:
  - mistral-7b-instruct-v0.3.Q4_K_M.gguf
  - starcoder2-3b.Q4_K_M.gguf
```

#### LOW_VRAM (<8GB)
```yaml
VRAM: <8GB (e.g., RTX 3060 6GB)
Backend: llama.cpp
Quantization: Q4_K_M (GGUF)
Max Batch: 1
Max Context: 4096
Flash Attention: Enabled
GPU Layers: Partial offload (20 layers)
```

#### CONSUMER (8-12GB)
```yaml
VRAM: 8-12GB (e.g., RTX 4060 8GB, RTX 4070 12GB)
Backend: vLLM
Quantization: AWQ 4-bit
Max Batch: 8
Max Context: 8192
Flash Attention: Enabled
GPU Memory Utilization: 90%
Models:
  - mistral-7b-instruct-v0.3-awq
  - starcoder2-3b-awq
```

#### PROSUMER (12-24GB)
```yaml
VRAM: 12-24GB (e.g., RTX 4080, RTX 3090, RTX 4090)
Backend: vLLM
Quantization: FP16
Max Batch: 32
Max Context: 16384
Flash Attention: Enabled
GPU Memory Utilization: 90%
Models:
  - mistral-7b-instruct-v0.3
  - starcoder2-7b
```

#### DATACENTER (24GB+)
```yaml
VRAM: 24-80GB (e.g., A100 40/80GB, H100)
Backend: vLLM
Quantization: BF16 (or FP8 on Hopper)
Max Batch: 128
Max Context: 32768
Flash Attention: Enabled
GPU Memory Utilization: 95%
Models:
  - mistral-7b-instruct-v0.3
  - starcoder2-15b
```

#### MULTI_GPU
```yaml
GPUs: 2+
Backend: vLLM with Tensor Parallelism
Tensor Parallel Size: GPU count
Quantization: FP16
Max Batch: 128
Max Context: 32768
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Run with hardware auto-detection
docker-compose up cognitive-brain

# Or run standalone
docker run -d \
  --gpus all \
  -p 8001:8001 \
  -v ./models:/app/models \
  cognitive-brain:latest \
  python scripts/hosting/serve_optimized.py
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r cognitive_brain/requirements.txt

# Run hardware detection
python -m cognitive_brain.core.hardware_detector

# Start the server
python scripts/hosting/serve_optimized.py --port 8001
```

### Option 3: Batch Scripts (Windows)

```batch
REM Quick prototype mode (4-bit quantization)
scripts\hosting\run_host.bat

REM Optimized with batching
scripts\hosting\run_host_optimized.bat
```

---

## Configuration

### Environment Variables

```bash
# Model selection
export BASE_MODEL="bigcode/starcoder2-3b"
export ADAPTER_PATH="trained_models/code_agent/final_adapter"

# Quantization (4bit, 8bit, none)
export QUANTIZATION="4bit"

# Server config
export HOST="0.0.0.0"
export PORT="8001"

# Override backend (optional)
export FORCE_BACKEND="vllm"  # or "llama_cpp", "pytorch"
```

### Programmatic Configuration

```python
from cognitive_brain.inference.adaptive_engine import AdaptiveInferenceEngine
from cognitive_brain.core.hardware_detector import InferenceBackend

# Override auto-detection
engine = AdaptiveInferenceEngine(auto_detect=False)
engine.load_model(
    model_path="mistral-7b-instruct-v0.3-awq",
    backend=InferenceBackend.VLLM,
    force_cpu=False
)

# Custom generation config
result = engine.generate(
    prompt="Hello!",
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
)
```

---

## Deployment Topologies

### Single Node (Development)

```yaml
# docker-compose.yml
services:
  llm-server:
    build: .
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Multi-GPU (Production)

```yaml
services:
  llm-server:
    environment:
      - TENSOR_PARALLEL_SIZE=2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Kubernetes (Enterprise)

```yaml
# See infra/helm/cognitive-brain for full charts
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: llm-server
          image: cognitive-brain:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

---

## Performance Benchmarks

### Expected Performance by Tier

| Tier | Model | Tokens/sec | Latency (TTFT) | Batch Throughput |
|------|-------|------------|----------------|------------------|
| CPU Only | Mistral-7B Q4 | 5-10 | 500-1000ms | N/A |
| Low VRAM | Mistral-7B Q4 | 15-25 | 200-400ms | N/A |
| Consumer (RTX 4060) | Mistral-7B AWQ | 40-60 | 50-100ms | 100 tok/s |
| Prosumer (RTX 4090) | Mistral-7B FP16 | 80-120 | 20-50ms | 300 tok/s |
| Datacenter (A100) | Mistral-7B BF16 | 150-200 | 10-30ms | 1000 tok/s |

### Running Benchmarks

```bash
# Run benchmark suite
python scripts/hosting/benchmark.py

# Test streaming latency
python scripts/hosting/test_streaming.py
```

---

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Reduce `max_context_length`
2. Use stronger quantization (AWQ → GGUF Q4_K_M)
3. Lower `gpu_memory_utilization` to 0.80
4. Clear CUDA cache: `torch.cuda.empty_cache()`

#### vLLM Not Available

```
ImportError: vLLM not installed
```

**Solutions:**
1. Install vLLM: `pip install vllm`
2. On Windows: vLLM not supported, use llama.cpp or PyTorch fallback
3. Check CUDA version compatibility

#### Slow Inference on CPU

**Solutions:**
1. Use GGUF models with llama.cpp
2. Reduce context length
3. Enable all CPU threads: `n_threads = os.cpu_count()`
4. Use Q4_K_M quantization for best speed

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# View detailed hardware detection
from cognitive_brain.core.hardware_detector import HardwareDetector
detector = HardwareDetector(verbose=True)
detector.print_summary()
```

---

## API Reference

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Synchronous text generation |
| `/generate_stream` | POST | Streaming text generation |
| `/health` | GET | Server health check |
| `/models` | GET | List loaded models |

### Request Schema

```json
{
  "prompt": "What products do you recommend?",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "do_sample": true,
  "stream": false
}
```

### Response Schema

```json
{
  "generated_text": "Based on your preferences...",
  "token_count": 128,
  "computation_time": 1.23,
  "throughput": 104.1
}
```

---

## Next Steps

- [Hardware Requirements](HARDWARE_REQUIREMENTS.md) - Detailed hardware specifications
- [Training Guide](TRAINING_GUIDE.md) - Fine-tuning models
- [CLaRa Integration](CLARA_INTEGRATION.md) - RAG compression setup
- [Project Overview](project_overview.md) - Full system architecture
