# Hardware Requirements

> Scaling the Cognitive Retail Brain from CPU to Datacenter

---

## Table of Contents

- [Overview](#overview)
- [Hardware Tiers](#hardware-tiers)
- [VRAM Requirements by Model](#vram-requirements-by-model)
- [Minimum Requirements](#minimum-requirements)
- [Recommended Configurations](#recommended-configurations)
- [Multi-GPU Setups](#multi-gpu-setups)
- [Cloud Deployment](#cloud-deployment)
- [Optimization Strategies](#optimization-strategies)

---

## Overview

The Cognitive Retail Brain automatically adapts to available hardware through its **Hardware Detector** module. The system scales "brain power" based on GPU capabilities—more VRAM and compute enables larger models, higher batch sizes, and longer context windows.

### Brain Power Formula

```
Brain Power = (Total VRAM GB × 10) × (1 + (GPU_Count - 1) × 0.8)
```

| Score | Tier | Capability |
|-------|------|------------|
| < 100 | 🧠 Consumer | Single 7B model, basic inference |
| 100-300 | 🧠🧠 Prosumer | Multiple models, long context |
| 300-500 | 🧠🧠🧠 Professional | High throughput, parallel agents |
| > 500 | 🧠🧠🧠🧠 Datacenter | Maximum performance, multi-GPU |

---

## Hardware Tiers

### Tier Comparison

| Tier | VRAM | Example GPUs | Backend | Quantization | Max Batch | Max Context | Brain Score |
|------|------|--------------|---------|--------------|-----------|-------------|-------------|
| **CPU Only** | N/A | None | llama.cpp | Q4_K_M | 1 | 2,048 | 🧠 |
| **Low VRAM** | <8GB | RTX 3060 6GB | llama.cpp | Q4_K_M | 1 | 4,096 | 🧠 60 |
| **Consumer** | 8-12GB | RTX 4060 8GB, RTX 4070 12GB | vLLM | AWQ 4-bit | 8 | 8,192 | 🧠 80-120 |
| **Prosumer** | 12-24GB | RTX 4080 16GB, RTX 4090 24GB | vLLM | FP16 | 32 | 16,384 | 🧠🧠 120-240 |
| **Datacenter** | 24-80GB | A100, H100 | vLLM | BF16/FP8 | 128 | 32,768 | 🧠🧠🧠🧠 240-800 |
| **Multi-GPU** | 2×24GB+ | 2×RTX 4090, 4×A100 | vLLM + TP | FP16 | 128 | 32,768 | 🧠🧠🧠🧠 400+ |

---

## VRAM Requirements by Model

### Core Models

| Model | Purpose | VRAM (FP16) | VRAM (AWQ 4-bit) | VRAM (Q4_K_M) |
|-------|---------|-------------|------------------|---------------|
| **Mistral-7B-Instruct** | Language & Reasoning | 14 GB | 4.5 GB | 4.0 GB |
| **StarCoder2-3B** | Code Generation | 6 GB | 2.0 GB | 1.8 GB |
| **StarCoder2-7B** | Code (Advanced) | 14 GB | 4.5 GB | 4.0 GB |
| **StarCoder2-15B** | Code (Maximum) | 30 GB | 10 GB | 8.5 GB |
| **NLLB-200-Distilled** | Translation | 2.5 GB | N/A | N/A |
| **CLaRa-7B** | RAG Compression | 14 GB | 4.5 GB | 4.0 GB |

### Vision Models

| Model | Purpose | VRAM |
|-------|---------|------|
| **YOLOv9-S** | Object Detection | 0.5 GB |
| **YOLOv9-M** | Object Detection | 1.0 GB |
| **YOLOv9-E** | Object Detection | 2.0 GB |
| **CLIP ViT-L/14** | Image Embeddings | 1.5 GB |

### Embedding Models

| Model | Purpose | VRAM |
|-------|---------|------|
| **Instructor-XL** | Document Embedding | 3.0 GB |
| **BGE-Large** | Retrieval | 1.5 GB |
| **DistilBERT** | Classification | 0.3 GB |

---

## Minimum Requirements

### Development / Testing

```yaml
Minimum:
  CPU: 4 cores (8 threads recommended)
  RAM: 16 GB
  GPU: None (CPU inference via llama.cpp)
  Storage: 50 GB SSD
  
Performance:
  Inference Speed: 5-10 tokens/second
  Max Context: 2,048 tokens
  Batch Size: 1
  
Limitations:
  - Single request processing only
  - No GPU acceleration
  - Limited to Q4_K_M quantized models
```

### Production (Single User)

```yaml
Recommended:
  CPU: 8 cores
  RAM: 32 GB
  GPU: RTX 4060 8GB or equivalent
  Storage: 100 GB NVMe SSD
  
Performance:
  Inference Speed: 40-60 tokens/second
  Max Context: 8,192 tokens
  Batch Size: 8
  
Capabilities:
  - Multiple concurrent users
  - AWQ 4-bit quantization
  - vLLM with PagedAttention
```

---

## Recommended Configurations

### Entry-Level Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│  ENTRY-LEVEL (Single-User / Demo)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4060 8GB                                           │
│  CPU: Intel i5-12400 / AMD Ryzen 5 5600                            │
│  RAM: 32 GB DDR5                                                    │
│  Storage: 500 GB NVMe                                               │
├─────────────────────────────────────────────────────────────────────┤
│  Models:                                                            │
│  • Mistral-7B-Instruct-AWQ (4.5 GB)                                │
│  • StarCoder2-3B-AWQ (2.0 GB)                                      │
│  • Free VRAM: ~1.5 GB (overhead)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Performance:                                                       │
│  • 40-60 tokens/sec per user                                       │
│  • 8 concurrent requests                                            │
│  • 8K context window                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Mid-Range Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│  MID-RANGE (Small Team / SMB)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4090 24GB                                          │
│  CPU: Intel i9-13900K / AMD Ryzen 9 7950X                          │
│  RAM: 64 GB DDR5                                                    │
│  Storage: 2 TB NVMe                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Models:                                                            │
│  • Mistral-7B-Instruct-FP16 (14 GB)                                │
│  • StarCoder2-7B (4 GB AWQ)                                        │
│  • CLaRa-7B (4 GB AWQ)                                             │
│  • Free VRAM: ~2 GB                                                │
├─────────────────────────────────────────────────────────────────────┤
│  Performance:                                                       │
│  • 80-120 tokens/sec                                               │
│  • 32 concurrent requests                                           │
│  • 16K context window                                               │
│  • Full multi-agent orchestration                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Enterprise Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│  ENTERPRISE (High Traffic / Production)                             │
├─────────────────────────────────────────────────────────────────────┤
│  GPU: NVIDIA A100 80GB × 2 (or H100)                               │
│  CPU: AMD EPYC 7763 / Intel Xeon                                   │
│  RAM: 256 GB DDR5 ECC                                              │
│  Storage: 4 TB NVMe RAID                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Models (Full Precision):                                           │
│  • Mistral-7B-Instruct BF16 (14 GB)                                │
│  • StarCoder2-15B (30 GB)                                          │
│  • CLaRa-7B (14 GB)                                                │
│  • Vision Pipeline (5 GB)                                          │
│  • Free VRAM: ~100 GB for KV cache                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Performance:                                                       │
│  • 150-200 tokens/sec                                              │
│  • 128+ concurrent requests                                         │
│  • 32K context window                                               │
│  • Tensor Parallelism (TP=2)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Multi-GPU Setups

### NVLink / NVSwitch Configuration

For multi-GPU inference with tensor parallelism:

| Setup | GPUs | Total VRAM | TP Size | Max Model |
|-------|------|------------|---------|-----------|
| Dual Consumer | 2 × RTX 4090 | 48 GB | 2 | 34B (AWQ) |
| Quad A100 | 4 × A100 80GB | 320 GB | 4 | 70B (FP16) |
| Octo H100 | 8 × H100 80GB | 640 GB | 8 | 175B (FP16) |

### Configuration Example

```python
# Automatic multi-GPU detection
from cognitive_brain.core.hardware_detector import HardwareDetector

detector = HardwareDetector()
config = detector.get_recommended_config()

# With 2 GPUs:
# config.tensor_parallel_size = 2
# config.use_tensor_parallelism = True
```

### vLLM Tensor Parallelism

```bash
# Environment setup for 2-GPU TP
export CUDA_VISIBLE_DEVICES=0,1

# Start vLLM with TP=2
python -m vllm.entrypoints.api_server \
    --model mistral-7b-instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90
```

---

## Cloud Deployment

### AWS Instances

| Instance | GPU | VRAM | vCPU | RAM | $/hr | Use Case |
|----------|-----|------|------|-----|------|----------|
| g5.xlarge | A10G | 24 GB | 4 | 16 GB | $1.01 | Development |
| g5.2xlarge | A10G | 24 GB | 8 | 32 GB | $1.21 | Production |
| g5.4xlarge | A10G | 24 GB | 16 | 64 GB | $1.62 | Multi-model |
| p4d.24xlarge | 8×A100 | 320 GB | 96 | 1152 GB | $32.77 | Enterprise |

### GCP Instances

| Instance | GPU | VRAM | Use Case |
|----------|-----|------|----------|
| n1-standard-4 + T4 | T4 | 16 GB | Budget |
| n1-standard-8 + L4 | L4 | 24 GB | Production |
| a2-highgpu-1g | A100 | 40 GB | High performance |
| a2-megagpu-16g | 16×A100 | 640 GB | Enterprise |

### Azure Instances

| Instance | GPU | VRAM | Use Case |
|----------|-----|------|----------|
| NC6s_v3 | V100 | 16 GB | Development |
| NC24ads_A100_v4 | A100 | 80 GB | Production |
| ND96amsr_A100_v4 | 8×A100 | 640 GB | Enterprise |

---

## Optimization Strategies

### Memory Optimization

| Technique | VRAM Savings | Speed Impact |
|-----------|--------------|--------------|
| AWQ 4-bit | 75% | -5% |
| GPTQ 4-bit | 75% | -10% |
| GGUF Q4_K_M | 75% | -15% (CPU) |
| Flash Attention 2 | 20-40% | +10% |
| PagedAttention (vLLM) | 30-50% | +20% |
| Gradient Checkpointing | 30% (training) | -20% |

### Context Length vs VRAM

Memory grows approximately linearly with context length due to KV cache:

| Context | 7B FP16 | 7B AWQ | Notes |
|---------|---------|--------|-------|
| 2,048 | 14 GB | 4.5 GB | Baseline |
| 4,096 | 15 GB | 5.0 GB | +7% |
| 8,192 | 17 GB | 6.0 GB | +21% |
| 16,384 | 21 GB | 8.0 GB | +50% |
| 32,768 | 29 GB | 12 GB | +107% |

### Batch Size Recommendations

| VRAM | Recommended Batch | Max Context Trade-off |
|------|-------------------|----------------------|
| 8 GB | 4-8 | 4K context |
| 12 GB | 8-16 | 8K context |
| 24 GB | 16-32 | 16K context |
| 40 GB | 32-64 | 32K context |
| 80 GB | 64-128 | 32K context |

---

## Quick Reference Card

### Decision Tree

```
Start
  │
  ├─ No GPU? ──────────────▶ llama.cpp + Q4_K_M
  │
  ├─ VRAM < 8GB? ──────────▶ llama.cpp + Q4_K_M + GPU offload
  │
  ├─ VRAM 8-12GB? ─────────▶ vLLM + AWQ
  │
  ├─ VRAM 12-24GB? ────────▶ vLLM + FP16
  │
  ├─ VRAM > 24GB? ─────────▶ vLLM + BF16
  │
  └─ Multi-GPU? ───────────▶ vLLM + Tensor Parallelism
```

### One-Line Setup

```bash
# Auto-detect and show recommendations
python -m cognitive_brain.core.hardware_detector
```

---

## Next Steps

- [LLM Hosting Guide](LLM_HOSTING_GUIDE.md) - Detailed server setup
- [Training Guide](TRAINING_GUIDE.md) - Hardware for fine-tuning
- [CLaRa Integration](CLARA_INTEGRATION.md) - RAG-specific requirements
