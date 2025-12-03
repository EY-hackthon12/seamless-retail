from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.graph import app_graph
from langchain_core.messages import HumanMessage
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def mobile_chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Invoke the LangGraph agent
    inputs = {"messages": [HumanMessage(content=request.message)], "session_id": session_id}
    
    # We need to handle the async generator or invoke properly
    # For simplicity in this demo, we'll assume invoke returns the final state
    # Note: app_graph.ainvoke is the async method
    
    try:
        result = await app_graph.ainvoke(inputs)
        last_message = result["messages"][-1].content
        return ChatResponse(response=last_message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
