from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Simple shared state for the graph
class ConversationState(BaseModel):
    user_id: str | None = None
    message: str
    route: Literal["recommendation","inventory","loyalty","payment","fulfillment","postpurchase","end"] | None = None
    trace: list[str] = Field(default_factory=list)


def sales_router(state: ConversationState) -> str:
    msg = state.message.lower()
    if any(k in msg for k in ["belt","recommend", "suggest"]):
        return "recommendation"
    if "stock" in msg or "available" in msg:
        return "inventory"
    if "redeem" in msg or "points" in msg:
        return "loyalty"
    if any(k in msg for k in ["pay","purchase","checkout"]):
        return "payment"
    if any(k in msg for k in ["pickup","ship","deliver"]):
        return "fulfillment"
    if any(k in msg for k in ["receipt","care","feedback"]):
        return "postpurchase"
    return "recommendation"


def node_stub(name: str):
    def fn(state: ConversationState) -> ConversationState:
        state.trace.append(f"{name} handled")
        state.route = name if name != "end" else "end"
        return state
    return fn


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("recommendation", node_stub("recommendation"))
    graph.add_node("inventory", node_stub("inventory"))
    graph.add_node("loyalty", node_stub("loyalty"))
    graph.add_node("payment", node_stub("payment"))
    graph.add_node("fulfillment", node_stub("fulfillment"))
    graph.add_node("postpurchase", node_stub("postpurchase"))

    graph.set_entry_point("recommendation")
    graph.add_conditional_edges("recommendation", lambda s: sales_router(s), {
        "recommendation": "inventory",
        "inventory": "inventory",
        "loyalty": "loyalty",
        "payment": "payment",
        "fulfillment": "fulfillment",
        "postpurchase": "postpurchase",
    })
    graph.add_edge("inventory", "loyalty")
    graph.add_edge("loyalty", "payment")
    graph.add_edge("payment", "fulfillment")
    graph.add_edge("fulfillment", "postpurchase")
    graph.add_edge("postpurchase", END)

    return graph.compile()

compiled_graph = build_graph()
