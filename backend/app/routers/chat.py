from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.agents.graph import compiled_graph

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    route: str
    trace: list[str]

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Run one pass through the compiled graph for demo
    state = {"user_id": req.user_id, "message": req.message, "trace": []}
    out = compiled_graph.invoke(state)
    return ChatResponse(route=out.get("route") or "end", trace=out.get("trace", []))
