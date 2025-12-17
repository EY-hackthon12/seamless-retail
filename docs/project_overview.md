# Project Overview: Cognitive Retail Brain

## 1. Vision
The goal of this project is to create a "Cognitive Retail Brain" that unifies the customer experience across digital and physical channels. By leveraging a multi-agent AI system, we ensure that a customer's conversation history, preferences, and intent are preserved and utilized to provide personalized assistance, whether they are chatting on their phone or interacting with an in-store kiosk.

## 2. Architecture
The system follows a **Hub-and-Spoke Multi-Agent Architecture** orchestrated by **LangGraph**.

### Components
1.  **Sales Agent (Router)**: The central brain (GPT-4o). It analyzes user input and routes tasks to specialized workers.
2.  **Specialized Agents**:
    -   **Inventory Agent**: Checks stock levels (SQL/Mock).
    -   **Recommendation Agent**: Suggests products based on context (RAG/Fine-tuned Model).
    -   **Loyalty Agent**: Retrieves user tier and points.
3.  **Shared Memory**: A PostgreSQL database acts as the "Hippocampus," storing long-term session data.
4.  **Interfaces**:
    -   **Mobile App**: For remote engagement and QR generation.
    -   **Kiosk**: For in-store identification and context recall.

### Data Flow
1.  **Mobile**: User asks "Do you have a blue suit?" -> Sales Agent -> Inventory Agent -> Response.
2.  **Handover**: User walks into store -> Opens App -> Generates QR (Session ID).
3.  **Kiosk**: User scans QR -> Kiosk API retrieves Session Context -> Recommendation Agent suggests matching shoes -> Display.

## 3. Key Technologies
-   **LangGraph**: For stateful, cyclic agent workflows.
-   **FastAPI**: High-performance async API.
-   **PostgreSQL**: Robust relational database for persistence.
-   **React/Vite**: Modern, fast frontend framework.
-   **Docker**: Containerization for consistent deployment.

## 4. Local Model Strategy
To reduce costs and ensure privacy, the system supports a hybrid model approach:
-   **Router**: High-intelligence model (GPT-4o) for planning.
-   **Workers**: Smaller, fine-tuned local models (Mistral 7B, Llama 3) hosted via the internal `ModelManager`.
-   **Training**: See `docs/training_specs/` for details on how we fine-tune these models.
