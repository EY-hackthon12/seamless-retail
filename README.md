# Agentic AI for Seamless Retail

A multi-agent, omnichannel retail skeleton.

Stack
- Backend: FastAPI, LangGraph/LangChain, Pydantic, PostgreSQL, ChromaDB, Pandas, NumPy
- Frontend: React + TypeScript + Tailwind CSS (Vite)
- Infra: Docker, docker-compose, Kubernetes manifests

Services (docker-compose)
- api: FastAPI backend on :8000
- frontend: React static site served by nginx on :5173
- postgres: persistent memory/state on :5432
- chroma: vector DB on :8001

Quick start
1) Copy .env.example to .env and fill values (OpenAI/HF keys optional for now)
2) docker compose up --build
3) Open http://localhost:5173

Project layout
- backend/ — FastAPI app, agents, memory, routers, schemas
- frontend/ — React+Tailwind app
- infra/
  - docker/ — Dockerfiles, compose
  - k8s/ — Kubernetes manifests
- scripts/ — helper scripts (DB init)

Notes
- All business logic is stubbed; replace with your implementations.
- Backend exposes /health and /api/v1/chat endpoints.
- Frontend calls /api/v1/chat and renders responses.
