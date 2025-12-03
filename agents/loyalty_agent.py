from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

class LoyaltyCheckInput(BaseModel):
    user_id: int = Field(description="ID of the user to check loyalty status for")

@tool("check_loyalty_status", args_schema=LoyaltyCheckInput)
def check_loyalty_status(user_id: int) -> str:
    """Checks the loyalty tier and points balance for a user."""
    # Mock logic for the demo
    return "User is Gold Tier. Balance: 1250 points. Eligible for 15% discount."
