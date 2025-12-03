from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.graph import app_graph
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

class ScanRequest(BaseModel):
    session_id: str

class ScanResponse(BaseModel):
    welcome_message: str
    recommendations: str

@router.post("/scan", response_model=ScanResponse)
async def kiosk_scan(request: ScanRequest):
    # Simulate the "Welcome Back" flow
    # In a real app, we would fetch the session context from DB.
    # Here, we'll ask the agent to generate a welcome message based on the session ID (which implies context).
    
    # We inject a system-like message to prompt the agent to act as the Kiosk
    prompt = "User has just scanned their QR code at the Kiosk. Welcome them back and offer recommendations based on their previous mobile chat context."
    
    inputs = {
        "messages": [HumanMessage(content=prompt)], 
        "session_id": request.session_id
    }
    
    try:
        result = await app_graph.ainvoke(inputs)
        response_text = result["messages"][-1].content
        
        # Simple parsing for the demo, assuming the agent returns a combined message
        return ScanResponse(
            welcome_message="Welcome back to UrbanVogue!",
            recommendations=response_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
