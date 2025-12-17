"""
Quick test for Gemini Fallback Engine
"""
import asyncio
import sys
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBLvWDV8hhCo6rAGQ6ATsrBm0Afto9AGBU"

sys.path.insert(0, '.')

from cognitive_brain.inference.gemini_fallback import (
    GeminiEngine, 
    AGENT_MODEL_ASSIGNMENT,
    get_context_manager,
)


async def test():
    print("=" * 60)
    print("GEMINI FALLBACK ENGINE TEST")
    print("=" * 60)
    print()
    
    # Show assignments
    print("Agent Model Assignments:")
    for a, m in AGENT_MODEL_ASSIGNMENT.items():
        print(f"  {a:20} -> {m.value}")
    print()
    
    # Test context manager
    print("-" * 60)
    print("Testing Context Manager")
    print("-" * 60)
    ctx_mgr = get_context_manager("empathy")
    usage = ctx_mgr.calculate_context_usage(
        system_prompt="You are a helpful assistant.",
        user_prompt="What products do you recommend?",
        context="Previous purchases: Blue dress, running shoes.",
    )
    print(f"Model: {usage['model']}")
    print(f"Total tokens: {usage['total_input_tokens']}")
    print(f"Utilization: {usage['utilization']*100:.2f}%")
    print(f"Fits in context: {usage['fits_in_context']}")
    print()
    
    # Initialize engine
    print("-" * 60)
    print("Testing Generation")
    print("-" * 60)
    engine = GeminiEngine()
    engine.load_model()
    print("Engine loaded!")
    print()
    
    # Test empathy agent (should use gemini-2.5-flash)
    print("[EMPATHY AGENT - gemini-2.5-flash]")
    result = await engine.generate_for_agent(
        prompt="Hello! I need help finding a gift for my mom.",
        agent_name="empathy",
        max_new_tokens=100,
    )
    print(f"Model: {result['model_used']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"Response: {result['text'][:200]}...")
    print()
    
    # Test router (should use gemini-2.0-flash-lite for speed)
    print("[ROUTER - gemini-2.0-flash-lite]")
    result = await engine.generate_for_agent(
        prompt="inventory check for blue shoes",
        agent_name="router",
        max_new_tokens=50,
    )
    print(f"Model: {result['model_used']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"Response: {result['text'][:100]}")
    print()
    
    # Test code agent
    print("[CODE AGENT - gemini-2.5-flash]")
    result = await engine.generate_for_agent(
        prompt="Write a Python function to calculate discount percentage",
        agent_name="code",
        max_new_tokens=150,
    )
    print(f"Model: {result['model_used']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"Response: {result['text'][:300]}...")
    
    # Cleanup - just mark as done, let Python handle cleanup
    engine._loaded = False
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
