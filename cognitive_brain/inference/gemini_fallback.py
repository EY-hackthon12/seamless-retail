"""
Gemini API Fallback Engine
===========================

Provides a robust fallback mechanism using Google's Gemini API when no local
LLM models are available or when local inference fails.

Features:
- Multi-model fallback chain (cycles through all Gemini models)
- Connection pooling with aiohttp for efficiency
- Exponential backoff for rate limit handling
- Streaming response support
- Token usage tracking
- Unified interface matching InferenceEngine ABC

Models (in fallback order):
1. gemini-2.0-flash-exp  - Newest, fastest multimodal
2. gemini-1.5-pro        - Most capable, 2M context
3. gemini-1.5-flash      - Balanced speed/capability
4. gemini-1.5-flash-8b   - Lightweight, fast fallback
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncIterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


# =============================================================================
# GEMINI MODEL REGISTRY - SMART ASSIGNMENT BASED ON AGENT REQUIREMENTS
# =============================================================================

class GeminiModel(Enum):
    """
    Available Gemini models with smart assignment.
    
    Default: gemini-2.0-flash-exp for all complex agents
    Lower models only for simple/fast tasks (e.g., intent classification)
    """
    
    # PRIMARY - Use for all main agents (Empathy, Code, Recommendation)
    GEMINI_2_FLASH_EXP = "gemini-2.0-flash-exp"
    
    # SECONDARY - Only if 2.0 is unavailable or for specific use cases
    GEMINI_15_PRO = "gemini-1.5-pro"  # 2M context for massive docs
    
    # LIGHTWEIGHT - Only for simple/fast tasks (intent routing, classification)
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_15_FLASH_8B = "gemini-1.5-flash-8b"  # Fastest, simple tasks only
    
    @classmethod
    def get_default(cls) -> "GeminiModel":
        """Get the default model (Gemini 2.0 Flash)."""
        return cls.GEMINI_2_FLASH_EXP
    
    @classmethod
    def get_fallback_chain(cls) -> List["GeminiModel"]:
        """Get fallback chain starting with best model."""
        return [
            cls.GEMINI_2_FLASH_EXP,  # Primary - newest, fastest
            cls.GEMINI_15_PRO,        # Fallback - most capable
            cls.GEMINI_15_FLASH,      # Fallback 2
        ]


class AgentComplexity(Enum):
    """Agent complexity levels for model assignment."""
    HIGH = "high"           # Complex reasoning, code, recommendations
    MEDIUM = "medium"       # Standard chat, Q&A
    LOW = "low"             # Simple classification, routing


# Smart model assignment based on agent type/complexity
AGENT_MODEL_ASSIGNMENT = {
    # Complex agents - ALWAYS use Gemini 2.0 Flash
    "empathy": GeminiModel.GEMINI_2_FLASH_EXP,
    "clara_empathy": GeminiModel.GEMINI_2_FLASH_EXP,
    "code": GeminiModel.GEMINI_2_FLASH_EXP,
    "recommendation": GeminiModel.GEMINI_2_FLASH_EXP,
    "sales": GeminiModel.GEMINI_2_FLASH_EXP,
    
    # Medium complexity - Gemini 2.0 Flash (can fallback to 1.5 Flash)
    "inventory": GeminiModel.GEMINI_2_FLASH_EXP,
    "loyalty": GeminiModel.GEMINI_2_FLASH_EXP,
    "visual": GeminiModel.GEMINI_2_FLASH_EXP,
    
    # Simple/fast tasks - Use lightweight model
    "router": GeminiModel.GEMINI_15_FLASH_8B,        # Intent routing - speed critical
    "classifier": GeminiModel.GEMINI_15_FLASH_8B,   # Classification tasks
    "intent": GeminiModel.GEMINI_15_FLASH_8B,       # Intent detection
    
    # RAG with massive context - Use 1.5 Pro for 2M context
    "clara_context": GeminiModel.GEMINI_15_PRO,     # May need massive context
    "rag": GeminiModel.GEMINI_15_PRO,               # Document retrieval
}


def get_model_for_agent(agent_name: str) -> GeminiModel:
    """
    Get the appropriate Gemini model for a specific agent.
    
    Strategy:
    - Complex reasoning agents → Gemini 2.0 Flash (best performance)
    - Simple routing/classification → Gemini 1.5 Flash 8B (fastest)
    - Massive context RAG → Gemini 1.5 Pro (2M context)
    - Default → Gemini 2.0 Flash
    """
    agent_key = agent_name.lower().replace("lobe", "").replace("agent", "").strip()
    return AGENT_MODEL_ASSIGNMENT.get(agent_key, GeminiModel.GEMINI_2_FLASH_EXP)


# Model context limits and characteristics
GEMINI_MODEL_INFO = {
    GeminiModel.GEMINI_2_FLASH_EXP: {
        "max_context": 1_000_000,
        "max_output": 8192,
        "rpm_limit": 10,  # Requests per minute (free tier)
        "tier": "primary",
        "description": "Newest, fastest - USE FOR ALL MAIN AGENTS",
        "use_for": ["empathy", "code", "recommendation", "sales", "chat"],
    },
    GeminiModel.GEMINI_15_PRO: {
        "max_context": 2_000_000,  # 2M context!
        "max_output": 8192,
        "rpm_limit": 2,
        "tier": "context-heavy",
        "description": "2M context - for massive document RAG only",
        "use_for": ["rag", "clara_context", "document_analysis"],
    },
    GeminiModel.GEMINI_15_FLASH: {
        "max_context": 1_000_000,
        "max_output": 8192,
        "rpm_limit": 15,
        "tier": "fallback",
        "description": "Fallback if 2.0 unavailable",
        "use_for": ["fallback"],
    },
    GeminiModel.GEMINI_15_FLASH_8B: {
        "max_context": 1_000_000,
        "max_output": 8192,
        "rpm_limit": 15,
        "tier": "lightweight",
        "description": "Fastest - ONLY for simple routing/classification",
        "use_for": ["router", "classifier", "intent"],
    },
}


# =============================================================================
# SELF-CONTEXT CALCULATION - Each agent manages its own context window
# =============================================================================

class ContextManager:
    """
    Self-context calculation for agents.
    
    Each agent can:
    - Estimate token count before sending
    - Track context window usage
    - Auto-truncate to fit model limits
    - Reserve tokens for output
    """
    
    # Approximate tokens per character (for estimation)
    CHARS_PER_TOKEN = 4.0
    
    def __init__(self, model: GeminiModel):
        self.model = model
        self.model_info = GEMINI_MODEL_INFO[model]
        self.max_context = self.model_info["max_context"]
        self.max_output = self.model_info["max_output"]
        
        # Reserve 20% of context for safety margin + output
        self.usable_context = int(self.max_context * 0.8) - self.max_output
        
        # Track usage
        self._current_tokens = 0
        self._history: List[Dict[str, int]] = []
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        
        Uses character-based estimation (fast, ~90% accurate for English).
        For exact counts, use a tokenizer.
        """
        return int(len(text) / self.CHARS_PER_TOKEN)
    
    def calculate_context_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate context usage for a request.
        
        Returns detailed breakdown of token usage and remaining capacity.
        """
        # Calculate each component
        system_tokens = self.estimate_tokens(system_prompt) if system_prompt else 0
        user_tokens = self.estimate_tokens(user_prompt)
        context_tokens = self.estimate_tokens(context) if context else 0
        
        history_tokens = 0
        if conversation_history:
            for msg in conversation_history:
                history_tokens += self.estimate_tokens(msg.get("content", ""))
        
        total_tokens = system_tokens + user_tokens + context_tokens + history_tokens
        remaining = self.usable_context - total_tokens
        utilization = total_tokens / self.usable_context
        
        return {
            "model": self.model.value,
            "max_context": self.max_context,
            "usable_context": self.usable_context,
            "breakdown": {
                "system": system_tokens,
                "user": user_tokens,
                "context": context_tokens,
                "history": history_tokens,
            },
            "total_input_tokens": total_tokens,
            "remaining_tokens": max(0, remaining),
            "utilization": min(1.0, utilization),
            "fits_in_context": remaining > 0,
            "needs_truncation": remaining < 0,
        }
    
    def truncate_to_fit(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_start: bool = True,
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed (default: usable_context)
            preserve_start: If True, keep beginning; else keep end
            
        Returns:
            Truncated text with indicator
        """
        max_tokens = max_tokens or self.usable_context
        estimated = self.estimate_tokens(text)
        
        if estimated <= max_tokens:
            return text
        
        # Calculate how many characters to keep
        target_chars = int(max_tokens * self.CHARS_PER_TOKEN)
        
        if preserve_start:
            truncated = text[:target_chars]
            return truncated + "\n\n[... truncated for context limit ...]"
        else:
            truncated = text[-target_chars:]
            return "[... truncated for context limit ...]\n\n" + truncated
    
    def build_optimized_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Build an optimized prompt that fits within context limits.
        
        Automatically truncates components in priority order:
        1. Conversation history (oldest first)
        2. Context (from middle)
        3. User prompt (never truncate system)
        
        Returns:
            Dict with optimized prompt components and usage stats
        """
        # Calculate initial usage
        usage = self.calculate_context_usage(
            system_prompt, user_prompt, context, conversation_history
        )
        
        # If it fits, return as-is
        if usage["fits_in_context"]:
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context": context,
                "conversation_history": conversation_history,
                "usage": usage,
                "was_truncated": False,
            }
        
        # Need to truncate - prioritize keeping recent context
        optimized_history = conversation_history or []
        optimized_context = context or ""
        
        # Step 1: Limit history to most recent turns
        if optimized_history and len(optimized_history) > max_history_turns:
            optimized_history = optimized_history[-max_history_turns:]
        
        # Step 2: Recalculate and truncate context if needed
        usage = self.calculate_context_usage(
            system_prompt, user_prompt, optimized_context, optimized_history
        )
        
        if not usage["fits_in_context"] and optimized_context:
            # Calculate how much context we can keep
            available_for_context = usage["remaining_tokens"] + usage["breakdown"]["context"]
            if available_for_context > 100:  # Keep at least 100 tokens
                optimized_context = self.truncate_to_fit(
                    optimized_context,
                    max_tokens=max(100, available_for_context),
                    preserve_start=True
                )
        
        # Step 3: Final check
        usage = self.calculate_context_usage(
            system_prompt, user_prompt, optimized_context, optimized_history
        )
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": optimized_context if optimized_context else None,
            "conversation_history": optimized_history if optimized_history else None,
            "usage": usage,
            "was_truncated": True,
        }
    
    def get_model_recommendation(self, estimated_tokens: int) -> GeminiModel:
        """
        Recommend the best model based on token requirements.
        
        Helps with dynamic model selection for varying context sizes.
        """
        if estimated_tokens > 1_000_000:
            return GeminiModel.GEMINI_15_PRO  # 2M context
        elif estimated_tokens > 500_000:
            return GeminiModel.GEMINI_15_FLASH  # 1M context
        elif estimated_tokens < 1000:
            return GeminiModel.GEMINI_15_FLASH_8B  # Fast for small prompts
        else:
            return GeminiModel.GEMINI_2_FLASH_EXP  # Default best


def get_context_manager(agent_name: str) -> ContextManager:
    """Get a context manager configured for a specific agent."""
    model = get_model_for_agent(agent_name)
    return ContextManager(model)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeminiConfig:
    """Configuration for Gemini API fallback."""
    
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")))
    preferred_model: GeminiModel = GeminiModel.GEMINI_2_FLASH_EXP
    enable_fallback_chain: bool = True
    max_retries: int = 3
    base_retry_delay: float = 1.0  # Exponential backoff base
    timeout_seconds: float = 60.0
    
    # Generation defaults
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    
    # System prompt for retail context
    system_instruction: str = """You are a helpful AI assistant for Seamless Retail, 
an advanced retail AI system. You provide accurate, friendly, and personalized 
responses to customer queries about products, inventory, recommendations, and 
general retail assistance. Be concise but thorough."""

    def validate(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.api_key)


@dataclass
class GeminiGenerationConfig:
    """Generation configuration for Gemini API."""
    
    max_output_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    
    text: str
    model_used: GeminiModel
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "STOP"
    fallback_attempts: int = 0
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.latency_ms > 0:
            return self.completion_tokens / (self.latency_ms / 1000)
        return 0.0


# =============================================================================
# GEMINI API CLIENT
# =============================================================================

class GeminiAPIClient:
    """
    Async client for Google Gemini API with connection pooling and retry logic.
    
    Uses aiohttp for efficient async HTTP requests with connection reuse.
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self._session = None
        self._request_count = 0
        self._total_tokens = 0
    
    async def _get_session(self):
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            import aiohttp
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                ttl_dns_cache=300,  # DNS cache TTL
                keepalive_timeout=30,
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, model: GeminiModel, stream: bool = False) -> str:
        """Build the API URL for a specific model."""
        action = "streamGenerateContent" if stream else "generateContent"
        return f"{self.BASE_URL}/{model.value}:{action}?key={self.config.api_key}"
    
    def _build_payload(
        self,
        prompt: str,
        gen_config: GeminiGenerationConfig,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the request payload."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": gen_config.max_output_tokens,
                "temperature": gen_config.temperature,
                "topP": gen_config.top_p,
                "topK": gen_config.top_k,
            }
        }
        
        # Add stop sequences if provided
        if gen_config.stop_sequences:
            payload["generationConfig"]["stopSequences"] = gen_config.stop_sequences
        
        # Add system instruction
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        return payload
    
    async def generate(
        self,
        prompt: str,
        model: Optional[GeminiModel] = None,
        gen_config: Optional[GeminiGenerationConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> GeminiResponse:
        """
        Generate a response from Gemini API.
        
        Implements fallback chain if enabled and primary model fails.
        """
        model = model or self.config.preferred_model
        gen_config = gen_config or GeminiGenerationConfig(
            max_output_tokens=self.config.default_max_tokens,
            temperature=self.config.default_temperature,
            top_p=self.config.default_top_p,
            top_k=self.config.default_top_k,
        )
        system_instruction = system_instruction or self.config.system_instruction
        
        # Get fallback chain
        if self.config.enable_fallback_chain:
            models_to_try = GeminiModel.get_fallback_chain()
            # Move preferred model to front
            if model in models_to_try:
                models_to_try.remove(model)
            models_to_try.insert(0, model)
        else:
            models_to_try = [model]
        
        last_error = None
        fallback_attempts = 0
        
        for current_model in models_to_try:
            try:
                response = await self._try_generate(
                    prompt=prompt,
                    model=current_model,
                    gen_config=gen_config,
                    system_instruction=system_instruction,
                )
                response.fallback_attempts = fallback_attempts
                return response
                
            except Exception as e:
                last_error = e
                fallback_attempts += 1
                logger.warning(
                    f"Gemini {current_model.value} failed: {e}. "
                    f"Trying next model... ({fallback_attempts}/{len(models_to_try)})"
                )
                # Small delay before trying next model
                await asyncio.sleep(0.5)
        
        # All models failed
        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_error}"
        )
    
    async def _try_generate(
        self,
        prompt: str,
        model: GeminiModel,
        gen_config: GeminiGenerationConfig,
        system_instruction: str,
    ) -> GeminiResponse:
        """Try to generate with a specific model, with retry logic."""
        session = await self._get_session()
        url = self._build_url(model, stream=False)
        payload = self._build_payload(prompt, gen_config, system_instruction)
        
        for attempt in range(self.config.max_retries):
            start_time = time.perf_counter()
            
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_response(data, model, latency_ms)
                    
                    elif resp.status == 429:  # Rate limited
                        retry_after = float(resp.headers.get("Retry-After", 5))
                        delay = min(retry_after, self.config.base_retry_delay * (2 ** attempt))
                        logger.warning(f"Rate limited. Waiting {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    
                    elif resp.status == 503:  # Service unavailable
                        delay = self.config.base_retry_delay * (2 ** attempt)
                        logger.warning(f"Service unavailable. Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Gemini API error {resp.status}: {error_text[:500]}"
                        )
                        
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise
        
        raise RuntimeError(f"Max retries exceeded for {model.value}")
    
    def _parse_response(
        self,
        data: Dict[str, Any],
        model: GeminiModel,
        latency_ms: float,
    ) -> GeminiResponse:
        """Parse Gemini API response."""
        # Extract text content
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            
            finish_reason = candidates[0].get("finishReason", "STOP")
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse Gemini response: {e}")
        
        # Extract token usage
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)
        
        # Track usage
        self._request_count += 1
        self._total_tokens += total_tokens
        
        return GeminiResponse(
            text=text,
            model_used=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
    
    async def generate_stream(
        self,
        prompt: str,
        model: Optional[GeminiModel] = None,
        gen_config: Optional[GeminiGenerationConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response from Gemini API.
        
        Yields text chunks as they arrive.
        """
        model = model or self.config.preferred_model
        gen_config = gen_config or GeminiGenerationConfig()
        system_instruction = system_instruction or self.config.system_instruction
        
        session = await self._get_session()
        url = self._build_url(model, stream=True)
        payload = self._build_payload(prompt, gen_config, system_instruction)
        
        try:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Gemini API error {resp.status}: {error_text[:500]}")
                
                # Parse streaming response (NDJSON format)
                async for line in resp.content:
                    if not line:
                        continue
                    
                    line_text = line.decode("utf-8").strip()
                    if not line_text or line_text == "[" or line_text == "]":
                        continue
                    
                    # Remove leading comma for array elements
                    if line_text.startswith(","):
                        line_text = line_text[1:]
                    
                    try:
                        chunk = json.loads(line_text)
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield part["text"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            # If streaming fails, fall back to non-streaming
            logger.warning(f"Streaming failed: {e}. Falling back to non-streaming.")
            response = await self.generate(prompt, model, gen_config, system_instruction)
            yield response.text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_count,
            "total_tokens": self._total_tokens,
            "api_key_set": bool(self.config.api_key),
        }
    
    async def generate_for_agent(
        self,
        prompt: str,
        agent_name: str,
        gen_config: Optional[GeminiGenerationConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> GeminiResponse:
        """
        Generate a response using the optimal model for a specific agent.
        
        This is the RECOMMENDED method for multi-agent systems.
        Automatically selects the right Gemini model based on agent complexity:
        - Complex agents (empathy, code, recommendation) → Gemini 2.0 Flash
        - Simple tasks (router, classifier) → Gemini 1.5 Flash 8B (fastest)
        - RAG with massive context → Gemini 1.5 Pro (2M context)
        
        Args:
            prompt: The input prompt
            agent_name: Name of the agent (e.g., "empathy", "code", "router")
            gen_config: Optional generation config
            system_instruction: Optional system instruction
            
        Returns:
            GeminiResponse with the generated text
        """
        # Get the appropriate model for this agent
        model = get_model_for_agent(agent_name)
        
        logger.info(f"[{agent_name}] Using {model.value} for generation")
        
        return await self.generate(
            prompt=prompt,
            model=model,
            gen_config=gen_config,
            system_instruction=system_instruction,
        )


# =============================================================================
# GEMINI INFERENCE ENGINE (matches InferenceEngine ABC)
# =============================================================================

class GeminiEngine:
    """
    Gemini-based inference engine that matches the InferenceEngine interface.
    
    This allows seamless integration with the AdaptiveInferenceEngine as a
    fallback when local models are unavailable.
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self._client: Optional[GeminiAPIClient] = None
        self._loaded = False
    
    def load_model(self, model_name: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the Gemini client.
        
        Args:
            model_name: Optional preferred model name (e.g., "gemini-1.5-flash")
        """
        if not self.config.validate():
            raise ValueError(
                "Gemini API key not set. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        
        # Parse model name if provided
        if model_name:
            for model in GeminiModel:
                if model.value == model_name or model.name == model_name:
                    self.config.preferred_model = model
                    break
        
        self._client = GeminiAPIClient(self.config)
        self._loaded = True
        logger.info(f"Gemini engine loaded. Preferred model: {self.config.preferred_model.value}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous generation (runs async in event loop).
        
        Returns dict matching the GenerationResult structure.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.generate_async(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                **kwargs,
            )
        )
    
    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async generation."""
        if not self._loaded or not self._client:
            raise RuntimeError("Gemini engine not loaded. Call load_model() first.")
        
        gen_config = GeminiGenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences or [],
        )
        
        response = await self._client.generate(prompt, gen_config=gen_config)
        
        return {
            "text": response.text,
            "tokens_generated": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "tokens_per_second": response.tokens_per_second,
            "finish_reason": response.finish_reason,
            "model_used": response.model_used.value,
            "fallback_attempts": response.fallback_attempts,
        }
    
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation."""
        if not self._loaded or not self._client:
            raise RuntimeError("Gemini engine not loaded. Call load_model() first.")
        
        gen_config = GeminiGenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        async for chunk in self._client.generate_stream(prompt, gen_config=gen_config):
            yield chunk
    
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self._loaded and self._client is not None
    
    def unload(self) -> None:
        """Unload the engine."""
        if self._client:
            asyncio.get_event_loop().run_until_complete(self._client.close())
            self._client = None
        self._loaded = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        status = {
            "loaded": self._loaded,
            "backend": "gemini_api",
            "preferred_model": self.config.preferred_model.value,
            "fallback_enabled": self.config.enable_fallback_chain,
            "api_key_set": bool(self.config.api_key),
        }
        
        if self._client:
            status.update(self._client.get_stats())
        
        return status
    
    async def generate_for_agent(
        self,
        prompt: str,
        agent_name: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response using the optimal model for a specific agent.
        
        This is the RECOMMENDED method for multi-agent systems.
        Uses smart model selection based on agent complexity.
        
        Args:
            prompt: The input prompt
            agent_name: Name of the agent (e.g., "empathy", "code", "router")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_instruction: Optional system instruction
            
        Returns:
            Dict with text, model_used, latency, etc.
        """
        if not self._loaded or not self._client:
            raise RuntimeError("Gemini engine not loaded. Call load_model() first.")
        
        gen_config = GeminiGenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        response = await self._client.generate_for_agent(
            prompt=prompt,
            agent_name=agent_name,
            gen_config=gen_config,
            system_instruction=system_instruction,
        )
        
        return {
            "text": response.text,
            "tokens_generated": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "tokens_per_second": response.tokens_per_second,
            "finish_reason": response.finish_reason,
            "model_used": response.model_used.value,
            "agent": agent_name,
            "fallback_attempts": response.fallback_attempts,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_engine: Optional[GeminiEngine] = None


def get_gemini_engine() -> GeminiEngine:
    """Get the global Gemini engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = GeminiEngine()
    return _global_engine


async def quick_generate(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Quick one-off generation with Gemini.
    
    Handles engine initialization automatically.
    """
    engine = get_gemini_engine()
    if not engine.is_loaded():
        engine.load_model()
    
    result = await engine.generate_async(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return result["text"]


def is_gemini_available() -> bool:
    """Check if Gemini API is available (API key is set)."""
    config = GeminiConfig()
    return config.validate()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_gemini():
        print("=" * 60)
        print("Gemini Fallback Engine Test")
        print("=" * 60)
        
        # Check if API key is available
        if not is_gemini_available():
            print("\n❌ GOOGLE_API_KEY not set. Cannot test.")
            print("   Set the environment variable and try again.")
            return False
        
        print("\n✓ API key found")
        
        # Initialize engine
        engine = GeminiEngine()
        engine.load_model()
        print(f"✓ Engine loaded. Preferred: {engine.config.preferred_model.value}")
        
        # Test generation
        print("\n" + "-" * 60)
        print("Testing Generation")
        print("-" * 60)
        
        test_prompts = [
            "Explain what you are in one sentence.",
            "What's 2+2?",
            "Recommend a product for someone looking for a gift.",
        ]
        
        for prompt in test_prompts:
            print(f"\n[PROMPT] {prompt}")
            result = await engine.generate_async(prompt, max_new_tokens=100)
            print(f"[RESPONSE] {result['text'][:200]}...")
            print(f"[MODEL] {result['model_used']}")
            print(f"[LATENCY] {result['latency_ms']:.0f}ms")
            print(f"[TOKENS] {result['tokens_generated']} @ {result['tokens_per_second']:.1f} tok/s")
        
        # Test streaming
        print("\n" + "-" * 60)
        print("Testing Streaming")
        print("-" * 60)
        
        print("\n[PROMPT] Tell me a short joke.")
        print("[STREAM] ", end="", flush=True)
        async for chunk in engine.generate_stream("Tell me a short joke.", max_new_tokens=100):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Get stats
        print("-" * 60)
        print("Engine Status")
        print("-" * 60)
        status = engine.get_status()
        for k, v in status.items():
            print(f"  {k}: {v}")
        
        # Cleanup
        engine.unload()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
    
    # Run test
    if "--test" in sys.argv:
        success = asyncio.run(test_gemini())
        sys.exit(0 if success else 1)
    else:
        print("Usage: python gemini_fallback.py --test")
        print("\nMake sure GOOGLE_API_KEY is set.")
