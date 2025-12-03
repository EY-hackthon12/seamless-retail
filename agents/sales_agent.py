from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import SystemMessage
from app.core.config import settings
from agents.tools import check_inventory, check_loyalty_status, get_recommendations

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)

# Define the tools available to the Sales Agent
tools = [check_inventory, check_loyalty_status, get_recommendations]

# System prompt for the Sales Agent
system_prompt = """You are the 'Sales Agent', the primary interface for the 'Cognitive Retail Brain'.
Your goal is to assist customers seamlessly across mobile and kiosk channels.
You have access to specialized tools:
- check_inventory: Use this when a user asks about product availability.
- check_loyalty_status: Use this to check points or tiers.
- get_recommendations: Use this to suggest items based on what the user is looking for.

Always be polite, professional, and helpful.
If you don't know the answer, use a tool.
If the user mentions a specific context (like "wedding" or "blue suit"), remember it.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
sales_agent = create_openai_functions_agent(llm, tools, prompt)

# Create the executor
sales_agent_executor = AgentExecutor(agent=sales_agent, tools=tools, verbose=True)
