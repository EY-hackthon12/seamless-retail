# AI Router + RAG Orchestrator — Hackathon Documentation

A production-style, multi-model “OpenAPI Router” that orchestrates ~15 LLMs (cloud + fine-tuned + local), backed by a RAG Master for grounded answers, with cost/latency-aware routing, safety, and robust observability—using dummy configs you can hyperlink in slides.

---

## Table of Contents
- [One-liner](#one-liner)
- [Key Highlights](#key-highlights)
- [Architecture](#architecture)
- [Request Lifecycle](#request-lifecycle)
- [Routing Policy](#routing-policy)
- [Model Registry (Dummy, 15 models)](#model-registry-dummy-15-models)
- [Cloud Router Adapters (Interface)](#cloud-router-adapters-interface)
- [RAG Master (Strategy + Config)](#rag-master-strategy--config)
- [OpenAPI Router (Endpoints)](#openapi-router-endpoints)
- [Prompt Templates (Dummy)](#prompt-templates-dummy)
- [Fine-tuned Models (Niche)](#fine-tuned-models-niche)
- [Cost, Latency, and Safety Controls](#cost-latency-and-safety-controls)
- [Failure Handling](#failure-handling)
- [Observability](#observability)
- [Security and Data Governance](#security-and-data-governance)
- [Deployment Topologies](#deployment-topologies)
- [Testing Strategy](#testing-strategy)
- [Runbook (Cheat Sheet)](#runbook-cheat-sheet)
- [Dummy Data and Quick Start](#dummy-data-and-quick-start)
- [Slides: What to Hyperlink](#slides-what-to-hyperlink)

---

## One-liner
A policy-driven, multi-provider LLM router with a RAG Master that delivers grounded answers, routes across ~15 models for cost/latency/quality, and includes safety + observability—using dummy data suitable for a slick hackathon demo.

---

## Key Highlights
- OpenAPI-first gateway for standardized access and easy slide hyperlinks
- Router core with intent detection, policy-based routing, budget controls
- RAG Master: hybrid retrieval (dense + BM25), reranking, chunk fusion, citations
- Cloud router adapters for ~15 models (OpenAI, Anthropic, Google, Azure, Cohere, Mistral, etc.)
- Cost/latency/safety-aware routing with failover and canary promotion
- Strong observability: metrics, traces, structured logs

---

## Architecture

High-level components:
- Clients: web app, CLI, notebooks
- API Gateway: OpenAPI, auth, rate limits
- Router Core: intent/policy engine, model registry, budgets, failover
- RAG Master: retrieval → rerank → fuse → citations → guardrails
- Cloud Routers: adapters to providers (~15 models)
- Shared services: Vector DB (dummy), Cache, Feature Store, Observability, Key mgmt

ASCII diagram:
```
[Client]
   |
[API Gateway] --(auth/limits)----------------------------------------------.
   |                                                                         \
[Router Core] --(intent/policy)--> [RAG Master] --(context/citations)--> [Plan]
   |                                                                         /
   '----> [Cloud Router Adapters] -> [Providers x ~15] <---(telemetry)-----'
                |                               \
             [Cache]                            [Observability]
                |                                   | \
             [Vector DB]                         [Metrics][Traces]
```

---

## Request Lifecycle
1) Ingest: normalize request, assign request_id, tenant, trace
2) Intent + constraints: classify (code/math/creative/qa/tool-use/vision), compute budgets (token, cost, latency), apply compliance constraints (region/PII)
3) RAG (if knowledge-grounded): hybrid retrieve → rerank → chunk fuse → inject citations; hallucination guard with re-ask when low confidence
4) Routing policy: score models by capability fit, live latency, cost, queue depth, historical win-rate, safety
5) Prompt build: system + style + tools schema + retrieved context (token budgeting)
6) Execute: primary model (streaming optional); on failure/timeout/safety trigger → fallback cascade
7) Post-process: unify tool outputs, verify citations, redact PII, safety filter, compress for cache
8) Return: stream or full; record spans, counters, costs

---

## Routing Policy
Inputs: intent, constraints, budgets, live telemetry per model (p50 latency, error rate, rpm/tpm usage)

Outputs: ordered plan (primary + fallbacks), per-call budgets, sampling params, canary fraction

Example policy weights by task:
- Creative: capability 0.35, safety 0.2, latency 0.1, cost 0.1, history 0.2, load 0.05
- Code: capability 0.4, safety 0.2, latency 0.1, cost 0.05, history 0.2, load 0.05
- Q&A (RAG): capability 0.3, safety 0.2, latency 0.15, cost 0.1, history 0.2, load 0.05
- Math/tool-use: capability 0.4, safety 0.2, latency 0.1, cost 0.05, history 0.2, load 0.05

Pseudocode:
```python
def route(request, models, telemetry, policy):
    intent = detect_intent(request)
    constraints = derive_constraints(request)
    candidates = [m for m in models if m.supports(intent) and m.meets(constraints)]
    scored = []
    for m in candidates:
        s_cap = policy.weights.capability * m.capability_score[intent]
        s_cost = policy.weights.cost * (1 - normalized_cost(m))
        s_lat  = policy.weights.latency * (1 - normalized_latency(telemetry[m]))
        s_saf  = policy.weights.safety * m.safety_grade
        s_hist = policy.weights.history * m.win_rate[intent]
        s_load = policy.weights.load * (1 - normalized_queue(telemetry[m]))
        score = s_cap + s_cost + s_lat + s_saf + s_hist + s_load
        scored.append((m, score))
    ordered = [m for m,_ in sorted(scored, key=lambda x: x[1], reverse=True)]
    primary, fallbacks = ordered[0], ordered[1:policy.max_fallbacks+1]
    return Plan(primary=primary, fallbacks=fallbacks, budgets=budget_for(request, primary, policy))
```

---

## Model Registry (Dummy, 15 models)
Use this JSON in slides or demos (hyperlink-friendly). Values are illustrative.

```json
{
  "models": [
    {"id": "openai:gpt-4o", "capability_score": {"creative": 0.95, "code": 0.9, "qa": 0.95, "math": 0.8}, "cost": 4.0, "max_tps": 50, "region": ["us","eu"], "safety_grade": 0.95},
    {"id": "anthropic:claude-3-5-sonnet", "capability_score": {"creative": 0.9, "code": 0.85, "qa": 0.95, "math": 0.85}, "cost": 3.8, "max_tps": 40, "region": ["us"], "safety_grade": 0.97},
    {"id": "google:gemini-1.5-pro", "capability_score": {"creative": 0.9, "code": 0.8, "qa": 0.92, "math": 0.85}, "cost": 3.2, "max_tps": 45, "region": ["us","apac"], "safety_grade": 0.94},
    {"id": "azure:gpt-4o", "capability_score": {"creative": 0.95, "code": 0.9, "qa": 0.95, "math": 0.8}, "cost": 4.1, "max_tps": 60, "region": ["eu"], "safety_grade": 0.95},
    {"id": "cohere:command-r+", "capability_score": {"creative": 0.82, "code": 0.75, "qa": 0.9, "math": 0.75}, "cost": 2.4, "max_tps": 70, "region": ["us"], "safety_grade": 0.93},
    {"id": "mistral:large", "capability_score": {"creative": 0.85, "code": 0.8, "qa": 0.88, "math": 0.8}, "cost": 2.1, "max_tps": 80, "region": ["eu"], "safety_grade": 0.92},
    {"id": "together:llama-3.1-70b", "capability_score": {"creative": 0.85, "code": 0.83, "qa": 0.87, "math": 0.78}, "cost": 1.9, "max_tps": 90, "region": ["us"], "safety_grade": 0.9},
    {"id": "bedrock:anthropic-sonnet", "capability_score": {"creative": 0.9, "code": 0.85, "qa": 0.93, "math": 0.85}, "cost": 3.9, "max_tps": 50, "region": ["us"], "safety_grade": 0.96},
    {"id": "xai:grok-2", "capability_score": {"creative": 0.9, "code": 0.82, "qa": 0.88, "math": 0.8}, "cost": 3.0, "max_tps": 30, "region": ["us"], "safety_grade": 0.9},
    {"id": "groq:llama-3.1-70b", "capability_score": {"creative": 0.83, "code": 0.82, "qa": 0.86, "math": 0.78}, "cost": 1.7, "max_tps": 150, "region": ["us"], "safety_grade": 0.9},
    {"id": "fireworks:mixtral-8x22b", "capability_score": {"creative": 0.85, "code": 0.8, "qa": 0.88, "math": 0.8}, "cost": 1.8, "max_tps": 80, "region": ["us"], "safety_grade": 0.9},
    {"id": "openrouter:assorted-best", "capability_score": {"creative": 0.88, "code": 0.8, "qa": 0.9, "math": 0.8}, "cost": 2.2, "max_tps": 100, "region": ["global"], "safety_grade": 0.9},
    {"id": "local:vllm-finetune-legal-13b", "capability_score": {"creative": 0.6, "code": 0.55, "qa": 0.8, "math": 0.6}, "cost": 0.2, "max_tps": 20, "region": ["onprem"], "safety_grade": 0.85},
    {"id": "local:vllm-finetune-med-13b", "capability_score": {"creative": 0.6, "code": 0.55, "qa": 0.82, "math": 0.62}, "cost": 0.2, "max_tps": 20, "region": ["onprem"], "safety_grade": 0.87},
    {"id": "azure:small-fast", "capability_score": {"creative": 0.7, "code": 0.65, "qa": 0.75, "math": 0.7}, "cost": 0.8, "max_tps": 200, "region": ["eu","us"], "safety_grade": 0.9}
  ]
}
```

---

## Cloud Router Adapters (Interface)
```ts
export interface LLMAdapter {
  id(): string
  supports(intent: string): boolean
  call(input: ChatRequest, opts: CallOptions): Promise<ChatResponse>
  stream?(input: ChatRequest, opts: CallOptions): AsyncIterable<ChatChunk>
  tools?(): ToolSchema[]
  health(): Promise<Health>
}
```

---

## RAG Master (Strategy + Config)
Strategy:
- Hybrid Retrieval: BM25 (sparse) + dense embeddings; union top-k; metadata filters (tenant, source, recency)
- Rerank: cross-encoder or lightweight LLM reranker; drop off-topic chunks
- Chunk fusion: combine overlaps; de-duplicate; token limit
- Grounding: inject citations; “answer only from context” for compliance domains
- Hallucination guard: verify claims via selective re-retrieval; if low score → re-ask
- Re-ask: expand query if empty retrieval

Config (dummy):
```yaml
retrievers:
  - name: dense
    index: vectordb://demo/index/legal
    embedder: "text-embed-large"
    top_k: 12
  - name: bm25
    index: "s3://dummy-corpus/bm25/legal"
    top_k: 25
fusion:
  max_context_tokens: 2000
  dedupe_jaccard: 0.8
  reranker: "cross-encoder-mini"
guardrails:
  allow_out_of_context: false
  min_rerank_score: 0.55
  reask_threshold: 0.5
```

---

## OpenAPI Router (Endpoints)
Minimal spec for slides/demos:
```yaml
openapi: 3.0.3
info:
  title: RAG Master OpenAPI Router
  version: "0.1.0"
servers: [{ url: https://api.demo-router.fake/v1 }]
paths:
  /chat:
    post:
      summary: Chat completion with optional RAG and routing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatRequest'
      responses:
        '200':
          description: Chat response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
  /stream:
    post:
      summary: Streaming chat
  /rag/query:
    post:
      summary: Retrieve + grounded answer only
  /tools:
    get:
      summary: List registered tools
  /metrics:
    get:
      summary: Router metrics snapshot
components:
  schemas:
    ChatRequest:
      type: object
      properties:
        messages: { type: array, items: { $ref: '#/components/schemas/Message' } }
        enable_rag: { type: boolean, default: true }
        preferred_models: { type: array, items: { type: string } }
        constraints:
          type: object
          properties:
            max_latency_ms: { type: integer }
            max_cost_usd: { type: number }
            region: { type: string }
            safety_level: { type: string, enum: [low, medium, high] }
    Message:
      type: object
      properties:
        role: { type: string, enum: [system, user, assistant, tool] }
        content: { type: string }
    ChatResponse:
      type: object
      properties:
        output: { type: string }
        model: { type: string }
        citations: { type: array, items: { $ref: '#/components/schemas/Citation' } }
        usage: { type: object, properties: { input_tokens: { type: integer }, output_tokens: { type: integer }, cost_usd: { type: number } } }
    Citation:
      type: object
      properties:
        id: { type: string }
        uri: { type: string }
        span: { type: string }
```

---

## Prompt Templates (Dummy)
```yaml
prompts:
  system_default: |
    You are a helpful assistant. Obey safety. Cite sources if provided in context.
  domain_legal: |
    You are a legal domain assistant. Answer strictly from provided context with citations.
styles:
  concise: "Max 6 sentences. Use bullets."
  verbose: "Explain reasoning briefly. Include section headers."
```

---

## Fine-tuned Models (Niche)
- Register fine-tune versions as separate model IDs with boosted niche scores
- Canary: route 5–10% of eligible traffic; compare win-rate and safety; promote or rollback
- Versioning: e.g., `local:vllm-finetune-legal-13b@2025-11-01`; keep last N versions hot
- Data flow: no training on user content by default; opt-in with redaction

---

## Cost, Latency, and Safety Controls
- Token budgets per request; early truncation; adaptive compression (semantic cache)
- Rate limiting: per-tenant RPM/TPM, burst, concurrency; provider quota guards; circuit breakers
- Safety: input PII redaction; output safety classifiers; high severity → safer model or refusal
- Domain policies (medical/legal): force RAG grounding and “no speculation” mode

---

## Failure Handling
- Fast-fail on provider 429/5xx with exponential backoff + jitter; then fallback
- Hedged requests for tail latency (optional): send backup after threshold if no first token
- Idempotency via request_id; retriable errors only once to avoid duplicates

---

## Observability
- Traces: one span per stage (intent, retrieve, rerank, route, call, post)
- Metrics (per model, per intent):
  - p50/p95 latency, success rate, tool-call rate
  - tokens in/out, cost USD, cache hit rate
  - safety violations, fallback rate
- Logs: structured, no PII by default; allowlist redaction
- Dashboards: “quality by intent”, “cost heatmap”, “fallbacks over time”

---

## Security and Data Governance
- Secrets via env vars (placeholders): `{{OPENAI_API_KEY}}`, `{{ANTHROPIC_API_KEY}}`; never log values
- Data retention (dummy): chat bodies 24h, traces 7d, metrics 30d; override per tenant
- Regionality: prefer models in user/tenant region; block cross-region if flagged

---

## Deployment Topologies
- Single-node demo: one process; in-memory cache; dummy vector DB URI
- Scaled: API (stateless) + worker pool for RAG + autoscaled adapters; shared Redis + real vector DB
- Edge option: read-only RAG at edge, centralized model calls

---

## Testing Strategy
- Golden tests: snapshot expected outputs with fixed seeds; assert citations present
- Shadow routing: send copy to candidate models; compute win-rate offline
- Load tests: tokens/s, p95 latency, 429 behavior
- Fault injection: simulate provider 5xx/latency spikes to validate fallbacks

---

## Runbook (Cheat Sheet)
- Traffic spike: increase small-fast weight; enable hedged requests; tighten budgets
- Provider outage: trip circuit-breaker; reroute to alternates; notify on-call
- Cost overrun: lower max_cost_usd per tenant; raise cache TTL; prefer cheaper models for low-risk intents
- Safety escalation: switch to high-safety preset; disable unsafe tools

---

## Dummy Data and Quick Start
Environment variables (placeholders; for demo slides):
```bash
setx OPENAI_API_KEY {{OPENAI_API_KEY}}
setx ANTHROPIC_API_KEY {{ANTHROPIC_API_KEY}}
setx VECTORDB_URI vectordb://demo
```

Sample knowledge base doc (dummy):
```json
{"id":"doc-42","uri":"https://example.com/dummy.pdf","text":"Dummy policy: Always cite sources. Use legal domain constraints.","tags":["legal","policy"],"updated":"2025-11-01"}
```

Example chat request (RAG on, preferred models):
```json
{
  "messages": [
    {"role":"user","content":"Summarize the key obligations from our dummy legal policy and cite sources."}
  ],
  "enable_rag": true,
  "preferred_models": ["anthropic:claude-3-5-sonnet","openai:gpt-4o"],
  "constraints": {"max_latency_ms": 4000, "max_cost_usd": 0.02, "region": "us", "safety_level": "high"}
}
```

Example response (dummy):
```json
{
  "output": "- Cite all sources\n- Answer strictly from context\n- Redact PII\n[Citations: 1]",
  "model": "anthropic:claude-3-5-sonnet",
  "citations": [{"id":"doc-42#p1","uri":"https://example.com/dummy.pdf","span":"lines 1-3"}],
  "usage": {"input_tokens": 1420, "output_tokens": 120, "cost_usd": 0.018}
}
```

Curl smoke (non-stream):
```bash
curl -X POST https://api.demo-router.fake/v1/chat \
  -H "Authorization: Bearer {{HACKATHON_TOKEN}}" \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hello from demo\"}],\"enable_rag\":false}"
```

---

## Slides: What to Hyperlink
- OpenAPI spec block (in this doc) or a placeholder file `openapi.yaml`
- Model registry JSON (in this doc) or a placeholder file `registry.json`
- Screenshot of dashboard panels (latency, cost, fallback rate)
- ASCII diagram + bullets for routing policy
- Dummy knowledge base sample + citations

---

Tip: You can copy blocks directly into separate files (e.g., `openapi.yaml`, `registry.json`, `rag.config.yaml`) if you want to hyperlink beyond this single doc.
