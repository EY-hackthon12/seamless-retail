"""
Advanced LangGraph Orchestration
=================================

Async parallel execution of cognitive lobes with state management.
Implements the Cortex Grid pattern for multi-agent coordination.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from dataclasses import dataclass
import operator

from langgraph.graph import StateGraph, END

# Import cognitive lobes and router
from cognitive_brain.orchestration.cognitive_lobes import (
    CognitiveLobe, LobeInput, LobeOutput,
    InventoryLobe, EmpathyLobe, VisualLobe, CodeLobe, RecommendationLobe,
    get_all_lobes
)
from cognitive_brain.orchestration.meta_router import MetaRouter, get_router

logger = logging.getLogger(__name__)


# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class AgentState(TypedDict):
    """State for the cognitive brain graph."""
    # Input
    query: str
    user_id: Optional[str]
    session_id: Optional[str]
    context: Dict[str, Any]
    
    # Routing
    primary_lobe: str
    parallel_lobes: List[str]
    routing_confidence: float
    
    # Processing
    lobe_responses: Annotated[Dict[str, LobeOutput], operator.or_]
    
    # Output
    final_response: str
    response_metadata: Dict[str, Any]


# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================

async def route_query(state: AgentState) -> AgentState:
    """
    Route the incoming query to appropriate lobes.
    
    Uses the Meta-Router to classify intent and determine
    which lobes should process the query.
    """
    router = get_router()
    
    # Route the query
    result = router.route(state["query"], state.get("context"))
    parallel = router.route_parallel(state["query"])
    
    logger.info(f"Routed to {result.lobe_name} (confidence: {result.confidence:.2f})")
    
    return {
        **state,
        "primary_lobe": result.lobe_name,
        "parallel_lobes": [lobe for lobe, _ in parallel],
        "routing_confidence": result.confidence,
    }


async def execute_lobes_parallel(state: AgentState) -> AgentState:
    """
    Execute cognitive lobes in parallel.
    
    Runs all selected lobes concurrently for maximum throughput.
    """
    lobes = get_all_lobes()
    lobe_names = state["parallel_lobes"]
    
    if not lobe_names:
        lobe_names = [state["primary_lobe"]]
    
    # Create input
    input = LobeInput(
        query=state["query"],
        context=state.get("context"),
        user_id=state.get("user_id"),
        session_id=state.get("session_id")
    )
    
    # Execute lobes in parallel
    async def run_lobe(name: str) -> tuple:
        lobe = lobes.get(name)
        if lobe:
            try:
                output = await lobe.process(input)
                return name, output
            except Exception as e:
                logger.error(f"Lobe {name} failed: {e}")
                return name, LobeOutput(
                    response=f"Error processing request: {str(e)}",
                    confidence=0.0
                )
        return name, None
    
    # Run all lobes concurrently
    tasks = [run_lobe(name) for name in lobe_names]
    results = await asyncio.gather(*tasks)
    
    # Collect responses
    lobe_responses = {}
    for name, output in results:
        if output:
            lobe_responses[name] = output
    
    return {
        **state,
        "lobe_responses": lobe_responses,
    }


async def synthesize_response(state: AgentState) -> AgentState:
    """
    Synthesize final response from lobe outputs.
    
    Combines responses based on confidence and relevance.
    """
    responses = state.get("lobe_responses", {})
    primary_lobe = state["primary_lobe"]
    
    if not responses:
        return {
            **state,
            "final_response": "I apologize, but I'm having trouble processing your request. Please try again.",
            "response_metadata": {"error": True}
        }
    
    # Get primary response
    primary_response = responses.get(primary_lobe)
    
    if primary_response:
        final_response = primary_response.response
        metadata = primary_response.metadata or {}
    else:
        # Use the highest confidence response
        best = max(responses.items(), key=lambda x: x[1].confidence)
        final_response = best[1].response
        metadata = best[1].metadata or {}
    
    # Add supplementary info from other lobes if relevant
    supplementary = []
    for lobe_name, output in responses.items():
        if lobe_name != primary_lobe and output.confidence > 0.7:
            supplementary.append(f"[{lobe_name}]: {output.response[:100]}...")
    
    if supplementary and len(supplementary) <= 2:
        metadata["supplementary"] = supplementary
    
    # Calculate total latency
    total_latency = sum(r.latency_ms for r in responses.values())
    metadata["total_latency_ms"] = total_latency
    metadata["lobes_used"] = list(responses.keys())
    
    return {
        **state,
        "final_response": final_response,
        "response_metadata": metadata,
    }


def should_continue(state: AgentState) -> str:
    """Determine if more processing is needed."""
    # Check if we have valid responses
    if state.get("lobe_responses"):
        return "synthesize"
    return "end"


# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================

def build_cognitive_graph() -> StateGraph:
    """
    Build the Cognitive Brain LangGraph.
    
    Flow:
    1. Route query to appropriate lobes
    2. Execute lobes in parallel
    3. Synthesize final response
    """
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("execute", execute_lobes_parallel)
    workflow.add_node("synthesize", synthesize_response)
    
    # Set entry point
    workflow.set_entry_point("route")
    
    # Add edges
    workflow.add_edge("route", "execute")
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "synthesize": "synthesize",
            "end": END
        }
    )
    workflow.add_edge("synthesize", END)
    
    return workflow


# Compile the graph
cognitive_graph = build_cognitive_graph().compile()


# ==============================================================================
# PUBLIC API
# ==============================================================================

async def process_query(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a user query through the cognitive brain.
    
    Args:
        query: User's input query
        user_id: Optional user identifier
        session_id: Optional session identifier
        context: Optional context dict (e.g., has_image, preferences)
        
    Returns:
        Dict with response, metadata, and processing info
    """
    initial_state = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "context": context or {},
        "primary_lobe": "",
        "parallel_lobes": [],
        "routing_confidence": 0.0,
        "lobe_responses": {},
        "final_response": "",
        "response_metadata": {},
    }
    
    # Run the graph
    final_state = await cognitive_graph.ainvoke(initial_state)
    
    return {
        "response": final_state["final_response"],
        "metadata": final_state["response_metadata"],
        "routing": {
            "primary_lobe": final_state["primary_lobe"],
            "parallel_lobes": final_state["parallel_lobes"],
            "confidence": final_state["routing_confidence"],
        }
    }


class CognitiveBrain:
    """
    High-level API for the Cognitive Brain.
    
    Provides a simple interface for interacting with the
    multi-agent orchestration system.
    """
    
    def __init__(self):
        self._graph = build_cognitive_graph().compile()
        self._router = get_router()
        self._lobes = get_all_lobes()
    
    async def chat(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple chat interface.
        
        Args:
            message: User message
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            Response string
        """
        result = await process_query(
            query=message,
            user_id=user_id,
            session_id=session_id,
            context=kwargs
        )
        return result["response"]
    
    async def process(
        self,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full processing with metadata.
        
        Returns complete response with routing info and metadata.
        """
        return await process_query(query=message, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get brain status."""
        return {
            "router_ready": self._router.is_ready(),
            "lobes": {
                name: lobe.is_ready() 
                for name, lobe in self._lobes.items()
            }
        }


# Global brain instance
_brain: Optional[CognitiveBrain] = None


def get_brain() -> CognitiveBrain:
    """Get the global cognitive brain instance."""
    global _brain
    if _brain is None:
        _brain = CognitiveBrain()
    return _brain


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    async def test_brain():
        print("=" * 60)
        print("Testing Cognitive Brain LangGraph")
        print("=" * 60)
        
        brain = CognitiveBrain()
        
        test_queries = [
            "Hello! I'm looking for a gift for my friend's wedding",
            "Do you have the blue summer dress in stock?",
            "What's trending this week?",
            "Write a Python function to calculate discounts",
        ]
        
        for query in test_queries:
            print(f"\n[QUERY] {query}")
            result = await brain.process(query)
            print(f"[ROUTING] {result['routing']['primary_lobe']} "
                  f"(confidence: {result['routing']['confidence']:.2f})")
            print(f"[RESPONSE] {result['response'][:200]}...")
            print(f"[METADATA] {result['metadata']}")
        
        print("\n" + "=" * 60)
        print("Brain test complete!")
        print("=" * 60)
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_brain())
