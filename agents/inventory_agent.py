from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

class InventoryCheckInput(BaseModel):
    product_name: str = Field(description="Name of the product to check inventory for")
    color: Optional[str] = Field(description="Color of the product")
    size: Optional[str] = Field(description="Size of the product")

@tool("check_inventory", args_schema=InventoryCheckInput)
def check_inventory(product_name: str, color: str = None, size: str = None) -> str:
    """Checks the inventory database for the availability of a specific product."""
    # In a real implementation, this would query the 'cart_items' or a separate 'inventory' table.
    # For the demo, we mock the specific "Blue Suit" and "Brown Oxfords" scenario.
    
    query = f"{product_name} {color if color else ''} {size if size else ''}".strip().lower()
    
    if "blue suit" in query:
        return "In Stock: Blue Suit (Size 40R, 42R). 5 units available at this location."
    elif "brown oxfords" in query or "shoes" in query:
        return "In Stock: Brown Leather Oxfords (Size 9, 10, 11). 3 pairs available."
    else:
        return f"Checking system... {product_name} is currently out of stock at this location."
