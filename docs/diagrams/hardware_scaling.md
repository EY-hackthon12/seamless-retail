# Hardware Scaling

```mermaid
flowchart TB
    subgraph DETECT["Hardware Detection"]
        HD[Detect GPUs/VRAM]
    end

    subgraph TIER["Hardware Tiers"]
        CPU[CPU Only<br/>No GPU]
        LOW[Low VRAM<br/>&lt;8GB]
        CONSUMER[Consumer<br/>8-12GB]
        PRO[Prosumer<br/>12-24GB]
        DC[Datacenter<br/>&gt;24GB]
        MULTI[Multi-GPU]
    end

    subgraph BACKEND["Inference Backend"]
        LLAMA[llama.cpp<br/>GGUF Q4]
        VLLM_Q[vLLM<br/>AWQ 4-bit]
        VLLM_FP[vLLM<br/>FP16]
        VLLM_TP[vLLM<br/>Tensor Parallel]
    end

    subgraph CONFIG["Configuration"]
        BATCH[Batch Size]
        CTX[Context Length]
        QUANT[Quantization]
    end

    HD --> CPU
    HD --> LOW
    HD --> CONSUMER
    HD --> PRO
    HD --> DC
    HD --> MULTI
    
    CPU --> LLAMA
    LOW --> LLAMA
    CONSUMER --> VLLM_Q
    PRO --> VLLM_FP
    DC --> VLLM_FP
    MULTI --> VLLM_TP
    
    LLAMA --> BATCH
    VLLM_Q --> BATCH
    VLLM_FP --> BATCH
    VLLM_TP --> BATCH
    
    BATCH --> CTX
    CTX --> QUANT
```

## Tier Configuration

| Tier | GPUs | VRAM | Backend | Quantization | Batch | Context |
|------|------|------|---------|--------------|-------|---------|
| CPU Only | 0 | - | llama.cpp | Q4_K_M | 1 | 2048 |
| Low VRAM | 1 | <8GB | llama.cpp | Q4_K_M | 1 | 4096 |
| Consumer | 1 | 8-12GB | vLLM | AWQ | 8 | 8192 |
| Prosumer | 1 | 12-24GB | vLLM | FP16 | 32 | 16384 |
| Datacenter | 1 | >24GB | vLLM | BF16 | 128 | 32768 |
| Multi-GPU | 2+ | any | vLLM+TP | FP16 | 128 | 32768 |

## Brain Power Formula

```
score = total_vram × 10 × (1 + (gpu_count - 1) × 0.8)
```

More GPUs = exponentially more brain power.
