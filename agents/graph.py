from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from agents.sales_agent import sales_agent_executor

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str

# Define the nodes
async def call_sales_agent(state: AgentState):
    messages = state['messages']
    response = await sales_agent_executor.ainvoke({"input": messages[-1].content, "chat_history": messages[:-1]})
    return {"messages": [response['output']]}

# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("sales_agent", call_sales_agent)

workflow.set_entry_point("sales_agent")
workflow.add_edge("sales_agent", END)

# Compile the graph
app_graph = workflow.compile()
