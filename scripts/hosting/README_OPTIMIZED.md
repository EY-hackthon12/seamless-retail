# ⚡ Optimized LLM Hosting Architecture

A research-grade, highly optimized hosting engine designed to maximize throughput and minimize latency across consumer and enterprise hardware.

## 🚀 Architectural Features

1.  **Hardware Autoscaling**:
    *   Automatically detects System RAM, GPU VRAM, and Compute Capability.
    *   Dynamically adjusts batch sizes (1-16) and context window limits (2k-8k).
    *   Optimizes quantization strategy (Int4/FP16) based on available memory headroom.

2.  **Adaptive Memory Management**:
    *   Utilizes `bitsandbytes` NF4 (Normal Float 4) quantization for >4x memory reduction.
    *   Implements intelligent Double Quantization to squeeze efficient storage.
    *   Supports dynamic CPU offloading for larger models on constrained hardware.

3.  **High-Performance Streaming**:
    *   Zero-latency token streaming architecture for "instant" user feedback.
    *   Asynchronous event loop handling for non-blocking concurrent request management.
    *   Dynamic micro-batching to saturate CUDA cores efficiently without OOMing.

## 🛠️ Usage

### Quick Start (Local)

The engine automatically adapts to your environment (Consumer GPU, Datacenter GPU, or CPU).

**Run the Host:**
```bash
python scripts/hosting/serve_optimized.py --model "bigcode/starcoder2-3b" --port 8000
```
*Note: Dependencies must be installed via `requirements.txt`*

### Integration

**API Endpoints:**
*   `POST /generate`: Standard full-text completion.
*   `POST /generate_stream`: Server-Sent Events (SSE) for real-time streaming.

**Example Request:**
```json
{
  "prompt": "def optimization_algorithm():",
  "max_new_tokens": 256,
  "temperature": 0.2
}
```

### Benchmarking

Validate the system performance on your specific hardware config:

```bash
python scripts/hosting/benchmark.py --stream --users 10
```
*Look for 'Throughput' and 'TTFT' (Time To First Token) metrics.*

## ⚙️ Configuration

The engine uses heuristics to tune itself, but you can override specific parameters via environment variables if needed for specialized research experiments.

*   `BASE_MODEL`: Default model path.
*   `QUANTIZATION`: Force `4bit` or `None`.
*   `ENABLE_COMPILE`: Set `1` to force `torch.compile` (Experimental on Windows).
