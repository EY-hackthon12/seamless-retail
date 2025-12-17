# Gemini API Fallback Architecture

## 1. Overview
High-performance, agent-aware fallback system that automatically re-routes LLM requests to the Gemini API when local models are unavailable.
See: `cognitive_brain/inference/gemini_fallback.py`

## 2. Smart Model Routing
The system dynamically selects the optimal Gemini model based on the agent's specific needs and complexity:

| Agent Ecosystem | Recommended Model | Use Case | Latency Target |
|-----------------|-------------------|----------|----------------|
| **Core Reasoning** (Empathy, Code, Sales) | `gemini-2.5-flash` | Complex reasoning, emotional intelligence | < 1.5s |
| **Context Heavy** (CLaRa, RAG) | `gemini-2.5-pro` | Massive context retrieval (up to 2M tokens) | < 3.0s |
| **High Velocity** (Router, Classifier) | `gemini-2.0-flash-lite` | Instant classification & routing | < 500ms |

## 3. Architecture

### 3.1 Adaptive Integration
Integrated directly into the `AdaptiveInferenceEngine` via `InferenceBackend.GEMINI_API`.
- **Automatic Failover**: `load_model()` attempts local load first, then seamlessly switches to Gemini if it fails.
- **Connection Pooling**: Uses `aiohttp` reuse for high-throughput scenarios.
- **Streaming**: Full support for token streaming responses.

### 3.2 Self-Context Management
Implemented via `ContextManager` class for each agent:
- **Token Estimation**: Fast character-based estimation (~4 chars/token).
- **Auto-Truncation**: Automatically prioritizes content to fit window:
  1. Truncates conversation history (keeping recent turns)
  2. Truncates RAG context (middle-out)
  3. Preserves System & User prompts (Critical)

## 4. Configuration

Configured via environment variables in `.env`:

```bash
# Enable fallback system
GEMINI_FALLBACK_ENABLED=true

# API Key (Required)
GOOGLE_API_KEY=AIzaSy...

# Default preference (overridden by smart routing)
GEMINI_PREFERRED_MODEL=gemini-2.5-flash
```

## 5. Usage Example

```python
from cognitive_brain.inference.gemini_fallback import GeminiEngine

# Initialize
engine = GeminiEngine()
engine.load_model()

# Agent-aware generation
result = await engine.generate_for_agent(
    prompt="Find me a blue dress",
    agent_name="empathy",  # Routes to gemini-2.5-flash
    max_new_tokens=200
)

print(f"Used: {result['model_used']}")
print(f"Response: {result['text']}")
```
