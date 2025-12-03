# Agentic AI for Seamless Retail ("Cognitive Retail Brain")

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Tech](https://img.shields.io/badge/Stack-FastAPI%20|%20React%20|%20LangGraph%20|%20Docker-green)

## 📖 Overview
The **Cognitive Retail Brain** is an advanced multi-agent AI system designed to solve the "context gap" in omnichannel retail. It enables a seamless customer journey by maintaining conversation history and user preferences across different touchpoints (Mobile App -> In-Store Kiosk).

Powered by **LangGraph**, it orchestrates specialized agents (Sales, Inventory, Loyalty, Recommendation) to provide intelligent, context-aware assistance.

## ✨ Key Features
- **Multi-Agent Architecture**: Hub-and-spoke design with a GPT-4o Router and specialized sub-agents.
- **Seamless Context Handover**: Generate a QR code in the Mobile App and scan it at the Kiosk to instantly recall your session.
- **Local Model Hosting**: Built-in `ModelManager` to run fine-tuned local LLMs (Mistral, Llama 3) for privacy and cost efficiency.
- **Persistent Memory**: PostgreSQL database stores user sessions, cart items, and conversation history.
- **Modern UI**: Responsive Mobile Chat and Interactive Kiosk Dashboard built with React, TypeScript, and Tailwind CSS.
- **Dockerized**: Full-stack deployment (Backend + Frontend + DB) with a single command.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd seamless-retail-main
   ```

2. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=password
   POSTGRES_DB=retail_brain
   ```

3. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

4. **Access the App**:
   - **Mobile App**: [http://localhost:8000](http://localhost:8000) (or http://localhost:5173 for dev)
   - **Kiosk Interface**: Navigate to the Kiosk tab or `/kiosk`
   - **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📚 Documentation
Detailed documentation is available in the `docs/` directory:

- **[Project Overview & Architecture](docs/project_overview.md)**: High-level design, agent workflow, and decision logic.
- **[User Guide & Walkthrough](docs/user_guide.md)**: Step-by-step guide to running the demo and testing features.
- **[Model Training Specs](docs/training_specs/)**: Research and hyperparameters for fine-tuning specialized agents (Mistral, StarCoder2, etc.).

## 🛠️ Tech Stack
- **Backend**: FastAPI, LangGraph, LangChain, SQLAlchemy, AsyncPG.
- **Frontend**: React, Vite, TypeScript, Tailwind CSS, Framer Motion.
- **Database**: PostgreSQL.
- **Infrastructure**: Docker, Docker Compose.
- **AI/ML**: OpenAI GPT-4o (Router), Local LLMs (via `ModelManager`).

## 📂 Directory Structure
```
├── agents/             # LangGraph agent definitions (Sales, Inventory, etc.)
├── app/                # FastAPI application core
│   ├── api/            # API endpoints (Mobile, Kiosk, Models)
│   ├── core/           # Config and settings
│   ├── db/             # Database schema and connection
│   └── services/       # Business logic (ModelManager)
├── docker/             # Dockerfile
├── docs/               # Documentation and Training Specs
├── frontend/           # React application
├── models/             # Directory for local LLM weights
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing
Contributions are welcome! Please read the documentation and follow the code style.
