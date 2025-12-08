# Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant Gateway as API Gateway
    participant Router as Meta Router
    participant Cache as Redis
    participant Vector as ChromaDB
    participant LLM as Inference Engine
    participant Brain as Deep Learning

    User->>Gateway: POST /chat
    Gateway->>Cache: Get session context
    Cache-->>Gateway: Previous context
    Gateway->>Router: Classify intent
    Router-->>Gateway: intent + lobes
    
    par Parallel Execution
        Gateway->>Vector: Semantic search
        Gateway->>Brain: Predict demand
    end
    
    Vector-->>Gateway: Similar products
    Brain-->>Gateway: Demand score
    
    Gateway->>LLM: Generate response
    LLM-->>Gateway: Streaming tokens
    Gateway->>Cache: Update session
    Gateway-->>User: SSE stream
```

## Request Lifecycle

1. User sends chat message
2. Gateway retrieves session context from Redis
3. Meta Router classifies intent
4. Router selects cognitive lobes
5. Lobes execute in parallel:
   - Vector search for relevant products
   - Brain predicts demand signals
6. LLM generates response using context
7. Response streams back to user
8. Session state updated in cache
