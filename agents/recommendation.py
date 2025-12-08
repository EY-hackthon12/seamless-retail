from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

from app.services.model_manager import model_manager

class RecommendationInput(BaseModel):
    context: str = Field(description="Current user context or items in cart")
    use_local_model: bool = Field(default=False, description="Whether to use the local fine-tuned model")

@tool("get_recommendations", args_schema=RecommendationInput)
async def get_recommendations(context: str, use_local_model: bool = False) -> str:
    """Provides product recommendations based on the current context."""
    
    if use_local_model:
        # Example of using the local model manager
        # In a real scenario, we would pass a specific model name like "fashion-mistral-7b"
        return await model_manager.predict("optimized-code-agent", context)

    if "blue suit" in context.lower():
        return "Recommended: Brown Leather Oxfords. They pair perfectly with the Blue Suit."
    return "Recommended: Classic White Shirt."
