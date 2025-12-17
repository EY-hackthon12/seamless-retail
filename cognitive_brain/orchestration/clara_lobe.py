"""
CLaRa-Enhanced Cognitive Lobe
==============================
Integrates CLaRa's compressed latent retrieval into the cognitive brain.

This lobe uses memory tokens that can be directly injected into the LLM,
achieving up to 32x context compression while preserving semantic meaning.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch

# Import CLaRa RAG
from scripts.brain.clara_rag import (
    CLaRaRAGMemory,
    CLaRaMemoryConfig,
    get_clara_memory,
)

# Import base lobe structure
try:
    from cognitive_brain.orchestration.cognitive_lobes import (
        CognitiveLobe,
        LobeInput,
        LobeOutput,
    )
    HAS_COGNITIVE_BRAIN = True
except ImportError:
    HAS_COGNITIVE_BRAIN = False
    
    # Fallback definitions
    @dataclass
    class LobeInput:
        query: str
        context: Optional[Dict[str, Any]] = None
        user_id: Optional[str] = None
        session_id: Optional[str] = None
    
    @dataclass
    class LobeOutput:
        response: str
        confidence: float = 0.0
        latency_ms: float = 0.0
        metadata: Optional[Dict[str, Any]] = None
    
    class CognitiveLobe:
        def __init__(self, name: str):
            self.name = name
            self._ready = False
        
        async def process(self, input: LobeInput) -> LobeOutput:
            raise NotImplementedError
        
        def is_ready(self) -> bool:
            return self._ready

logger = logging.getLogger(__name__)


class CLaRaContextLobe(CognitiveLobe):
    """
    CLaRa-Enhanced Context Retrieval Lobe.
    
    This lobe provides compressed context to other lobes using CLaRa's
    latent compression. It can:
    
    1. Store user interactions as compressed memory tokens
    2. Retrieve relevant context using differentiable top-k
    3. Return compressed vectors ready for LLM injection
    
    Usage in cognitive brain:
        lobe = CLaRaContextLobe()
        result = await lobe.process(LobeInput(query="What did I buy last time?"))
        # result.metadata["compressed_context"] contains memory tokens
    """
    
    def __init__(
        self,
        name: str = "clara_context",
        compress_rate: int = 32,
        top_k: int = 5,
    ):
        super().__init__(name)
        
        self.config = CLaRaMemoryConfig(
            compress_rate=compress_rate,
            top_k=top_k,
        )
        
        self.memory: Optional[CLaRaRAGMemory] = None
        self._init_memory()
    
    def _init_memory(self):
        """Initialize CLaRa memory."""
        try:
            self.memory = get_clara_memory(self.config)
            self._ready = True
            logger.info(f"CLaRa Context Lobe initialized (compression: {self.config.compress_rate}x)")
        except Exception as e:
            logger.error(f"Failed to initialize CLaRa memory: {e}")
            self._ready = False
    
    async def process(self, input: LobeInput) -> LobeOutput:
        """
        Process a query and return relevant compressed context.
        
        The compressed context is returned in metadata["compressed_context"]
        and can be directly injected into an LLM's embedding space.
        """
        start_time = time.time()
        
        if self.memory is None:
            return LobeOutput(
                response="CLaRa memory not initialized",
                confidence=0.0,
                latency_ms=0.0,
                metadata={"error": True}
            )
        
        try:
            # Search for relevant context
            result = self.memory.search(
                query_text=input.query,
                user_id=input.user_id,
                k=self.config.top_k,
                return_compressed=True,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Build response from retrieved context
            if result["results"]:
                context_summary = self._summarize_context(result["results"])
                confidence = max(r["score"] for r in result["results"])
            else:
                context_summary = "No relevant context found."
                confidence = 0.0
            
            # Return with compressed context in metadata
            return LobeOutput(
                response=context_summary,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata={
                    "compressed_context": result["compressed_context"],
                    "num_documents": len(result["results"]),
                    "compression_rate": self.config.compress_rate,
                    "results": result["results"],
                }
            )
            
        except Exception as e:
            logger.error(f"CLaRa retrieval error: {e}")
            return LobeOutput(
                response=f"Error retrieving context: {str(e)}",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": True}
            )
    
    def _summarize_context(self, results: List[Dict[str, Any]]) -> str:
        """Create a text summary of retrieved context."""
        if not results:
            return "No relevant context available."
        
        summaries = []
        for i, r in enumerate(results[:3]):  # Top 3
            text = r["text"][:200]
            score = r["score"]
            summaries.append(f"[{score:.2f}] {text}...")
        
        return "Relevant context:\n" + "\n".join(summaries)
    
    async def store_interaction(
        self,
        user_id: str,
        text: str,
        interaction_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store a user interaction in CLaRa memory.
        
        The interaction will be compressed and stored for future retrieval.
        """
        if self.memory is None:
            logger.warning("Cannot store interaction: CLaRa memory not initialized")
            return
        
        self.memory.add_memory(user_id, text, interaction_id, metadata)
        logger.debug(f"Stored interaction {interaction_id} for user {user_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.memory is None:
            return {"error": "Memory not initialized"}
        return self.memory.stats()


class CLaRaEnhancedEmpathyLobe(CognitiveLobe):
    """
    Empathy Lobe enhanced with CLaRa compressed context.
    
    This lobe handles customer chat with personalized context retrieved
    using CLaRa's compressed latent vectors.
    """
    
    def __init__(
        self,
        name: str = "clara_empathy",
        llm_backend: str = "vllm",  # or "local", "api"
    ):
        super().__init__(name)
        
        self.llm_backend = llm_backend
        self.context_lobe = CLaRaContextLobe()
        self._ready = self.context_lobe.is_ready()
    
    async def process(self, input: LobeInput) -> LobeOutput:
        """
        Process a customer query with CLaRa-compressed context.
        """
        start_time = time.time()
        
        # First, get compressed context
        context_result = await self.context_lobe.process(input)
        
        # Build prompt with context
        if context_result.metadata.get("results"):
            context_text = context_result.response
        else:
            context_text = "No previous context available."
        
        prompt = self._build_prompt(input.query, context_text, input.context)
        
        # Generate response (simulated for now)
        response = await self._generate_response(prompt)
        
        # Store this interaction for future retrieval
        if input.user_id:
            interaction_id = int(time.time() * 1000)  # Use timestamp as ID
            full_interaction = f"User: {input.query}\nAssistant: {response}"
            await self.context_lobe.store_interaction(
                input.user_id,
                full_interaction,
                interaction_id,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LobeOutput(
            response=response,
            confidence=0.85,  # Would come from LLM logprobs in production
            latency_ms=latency_ms,
            metadata={
                "context_used": bool(context_result.metadata.get("results")),
                "compression_rate": context_result.metadata.get("compression_rate"),
                "llm_backend": self.llm_backend,
            }
        )
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the prompt for the LLM."""
        system_prompt = """You are a helpful retail assistant for Seamless Retail.
You have access to the customer's previous interactions and preferences.
Be friendly, personalized, and helpful.

Previous context:
{context}

Answer the customer's question based on this context when relevant."""
        
        return system_prompt.format(context=context) + f"\n\nCustomer: {query}\n\nAssistant:"
    
    async def _generate_response(self, prompt: str) -> str:
        """
        Generate a response using the configured LLM backend.
        
        In production, this would call vLLM, llama.cpp, or an API.
        For now, returns a simulated response.
        """
        # TODO: Integrate with actual LLM backends
        # - vLLM: self._call_vllm(prompt)
        # - llama.cpp: self._call_llamacpp(prompt)
        # - API: self._call_api(prompt)
        
        await asyncio.sleep(0.05)  # Simulate inference time
        
        return (
            "Based on your preferences, I'd recommend checking out our new "
            "summer collection! We have some beautiful floral dresses that "
            "I think you'd love. Would you like me to show you some options?"
        )


# =============================================================================
# LOBE REGISTRY
# =============================================================================

CLARA_LOBES = {
    "clara_context": CLaRaContextLobe,
    "clara_empathy": CLaRaEnhancedEmpathyLobe,
}


def get_clara_lobes() -> Dict[str, CognitiveLobe]:
    """Get all CLaRa-enhanced cognitive lobes."""
    return {name: cls() for name, cls in CLARA_LOBES.items()}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test_clara_lobe():
        print("=" * 60)
        print("CLaRa Context Lobe Test")
        print("=" * 60)
        
        # Initialize lobe
        lobe = CLaRaContextLobe()
        print(f"\nLobe ready: {lobe.is_ready()}")
        print(f"Stats: {lobe.get_stats()}")
        
        # Store some interactions
        interactions = [
            ("user1", "I bought a blue dress last week and loved it!", 1),
            ("user1", "I prefer cotton fabrics over synthetic", 2),
            ("user1", "My budget is usually around $100", 3),
            ("user2", "I'm looking for formal business wear", 4),
        ]
        
        print("\nStoring interactions...")
        for user_id, text, iid in interactions:
            await lobe.store_interaction(user_id, text, iid)
            print(f"  Stored: {text[:50]}...")
        
        # Test retrieval
        print("\n" + "-" * 60)
        print("Testing Retrieval")
        print("-" * 60)
        
        test_queries = [
            ("user1", "What did I buy recently?"),
            ("user1", "What's my preferred fabric?"),
            ("user2", "What style am I looking for?"),
        ]
        
        for user_id, query in test_queries:
            print(f"\n[QUERY] ({user_id}) {query}")
            
            input = LobeInput(query=query, user_id=user_id)
            result = await lobe.process(input)
            
            print(f"[RESPONSE] {result.response[:100]}...")
            print(f"[CONFIDENCE] {result.confidence:.3f}")
            print(f"[LATENCY] {result.latency_ms:.2f}ms")
            
            if result.metadata.get("compressed_context") is not None:
                ctx = result.metadata["compressed_context"]
                print(f"[COMPRESSED] Shape: {ctx.shape}")
        
        # Test enhanced empathy lobe
        print("\n" + "=" * 60)
        print("CLaRa Enhanced Empathy Lobe Test")
        print("=" * 60)
        
        empathy = CLaRaEnhancedEmpathyLobe()
        
        input = LobeInput(
            query="Can you recommend something for me?",
            user_id="user1",
        )
        
        result = await empathy.process(input)
        print(f"\n[QUERY] {input.query}")
        print(f"[RESPONSE] {result.response}")
        print(f"[CONTEXT USED] {result.metadata.get('context_used')}")
        print(f"[LATENCY] {result.latency_ms:.2f}ms")
        
        print("\n" + "=" * 60)
        print("✓ CLaRa Lobe tests complete!")
        print("=" * 60)
    
    asyncio.run(test_clara_lobe())
