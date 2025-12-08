"""
Meta-Router - Intent Classification and Lobe Routing
======================================================

The Meta-Router is the "conscious" decision maker that routes
incoming requests to the appropriate cognitive lobe.

Uses a lightweight DistilBERT-based classifier for fast intent detection.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of intent routing."""
    lobe_name: str
    intent: str
    confidence: float
    all_scores: Dict[str, float]


class MetaRouter:
    """
    Meta-Router for Intent Classification.
    
    Routes incoming queries to the appropriate cognitive lobe based on
    intent classification. Uses DistilBERT for efficient inference.
    
    Intents:
    - product_inquiry -> EmpathyLobe + InventoryLobe
    - inventory_check -> InventoryLobe
    - recommendation -> RecommendationLobe
    - visual_search -> VisualLobe
    - code_request -> CodeLobe
    - general_chat -> EmpathyLobe
    """
    
    # Intent to lobe mapping
    INTENT_TO_LOBE = {
        "product_inquiry": "empathy",
        "inventory_check": "inventory",
        "recommendation": "recommendation",
        "visual_search": "visual",
        "code_request": "code",
        "store_info": "empathy",
        "order_status": "inventory",
        "general_chat": "empathy",
    }
    
    # Keyword patterns for rule-based fallback
    KEYWORD_PATTERNS = {
        "inventory_check": ["stock", "available", "inventory", "in store", "warehouse"],
        "recommendation": ["recommend", "suggest", "similar", "like this", "trending", "popular", "gift"],
        "visual_search": ["image", "picture", "photo", "looks like", "find similar", "visual"],
        "code_request": ["code", "script", "function", "debug", "programming", "api"],
        "product_inquiry": ["price", "product", "item", "buy", "purchase", "cost"],
        "store_info": ["store", "location", "hours", "address", "near me"],
        "order_status": ["order", "delivery", "shipping", "tracking", "return"],
    }
    
    def __init__(self, use_model: bool = True):
        self._model = None
        self._tokenizer = None
        self.use_model = use_model
        self._ready = False
        
        self.intent_labels = list(self.INTENT_TO_LOBE.keys())
    
    def load_model(self, model_name: str = "distilbert-base-uncased"):
        """Load the classification model."""
        if not self.use_model:
            logger.info("MetaRouter: Using rule-based routing (no model)")
            self._ready = True
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            from cognitive_brain.core.neural_architectures import IntentClassifier
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._encoder = AutoModel.from_pretrained(model_name)
            
            # Intent classifier head
            hidden_size = self._encoder.config.hidden_size
            self._classifier = IntentClassifier(
                hidden_dim=hidden_size,
                num_intents=len(self.intent_labels)
            )
            
            if torch.cuda.is_available():
                self._encoder = self._encoder.cuda()
                self._classifier = self._classifier.cuda()
            
            self._encoder.eval()
            self._classifier.eval()
            self._ready = True
            
            logger.info("MetaRouter: Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"MetaRouter: Failed to load model: {e}")
            logger.info("MetaRouter: Falling back to rule-based routing")
            self.use_model = False
            self._ready = True
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """
        Route a query to the appropriate cognitive lobe.
        
        Args:
            query: User query text
            context: Optional context (e.g., has_image, user_preferences)
            
        Returns:
            RoutingResult with lobe name, intent, and confidence
        """
        # Check context for forced routing
        if context:
            if context.get("has_image"):
                return RoutingResult(
                    lobe_name="visual",
                    intent="visual_search",
                    confidence=1.0,
                    all_scores={"visual_search": 1.0}
                )
            
            if context.get("force_lobe"):
                lobe = context["force_lobe"]
                intent = next(
                    (k for k, v in self.INTENT_TO_LOBE.items() if v == lobe),
                    "general_chat"
                )
                return RoutingResult(
                    lobe_name=lobe,
                    intent=intent,
                    confidence=1.0,
                    all_scores={intent: 1.0}
                )
        
        # Try model-based routing
        if self.use_model and self._model is not None:
            return self._route_with_model(query)
        
        # Fall back to rule-based routing
        return self._route_with_rules(query)
    
    def _route_with_model(self, query: str) -> RoutingResult:
        """Route using the trained classifier."""
        with torch.no_grad():
            # Tokenize
            inputs = self._tokenizer(
                query,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Encode
            outputs = self._encoder(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Classify
            logits, _ = self._classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            
            # Get top intent
            top_idx = probs.argmax(dim=-1).item()
            top_prob = probs[0, top_idx].item()
            intent = self.intent_labels[top_idx]
            
            # Build all scores
            all_scores = {
                self.intent_labels[i]: probs[0, i].item()
                for i in range(len(self.intent_labels))
            }
        
        return RoutingResult(
            lobe_name=self.INTENT_TO_LOBE[intent],
            intent=intent,
            confidence=top_prob,
            all_scores=all_scores
        )
    
    def _route_with_rules(self, query: str) -> RoutingResult:
        """Route using keyword rules."""
        query_lower = query.lower()
        scores = {}
        
        # Score each intent based on keyword matches
        for intent, keywords in self.KEYWORD_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score / len(keywords)
        
        # Add base scores for common intents
        scores["general_chat"] = scores.get("general_chat", 0) + 0.1
        scores["product_inquiry"] = scores.get("product_inquiry", 0) + 0.1
        
        # Find best match
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] * 2, 1.0)  # Scale up
        else:
            best_intent = "general_chat"
            confidence = 0.5
        
        # Normalize scores
        total = sum(scores.values()) or 1.0
        all_scores = {k: v / total for k, v in scores.items()}
        
        return RoutingResult(
            lobe_name=self.INTENT_TO_LOBE[best_intent],
            intent=best_intent,
            confidence=confidence,
            all_scores=all_scores
        )
    
    def route_parallel(self, query: str) -> List[Tuple[str, float]]:
        """
        Determine which lobes should be queried in parallel.
        
        Returns lobes with confidence > 0.3 for parallel execution.
        """
        result = self.route(query)
        
        parallel_lobes = []
        for intent, score in result.all_scores.items():
            if score > 0.3:
                lobe = self.INTENT_TO_LOBE[intent]
                if (lobe, score) not in parallel_lobes:
                    parallel_lobes.append((lobe, score))
        
        # Sort by confidence
        parallel_lobes.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 3
        return parallel_lobes[:3]
    
    def is_ready(self) -> bool:
        return self._ready


# Singleton instance
_router: Optional[MetaRouter] = None


def get_router() -> MetaRouter:
    """Get the global meta-router instance."""
    global _router
    if _router is None:
        _router = MetaRouter(use_model=False)  # Start with rules
        _router.load_model()
    return _router


if __name__ == "__main__":
    # Test the router
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Meta-Router")
    print("=" * 60)
    
    router = MetaRouter(use_model=False)
    router.load_model()
    
    test_queries = [
        "Is this product in stock?",
        "Can you recommend something for a summer wedding?",
        "I'm looking for something that looks like this image",
        "Write a Python function to process orders",
        "Hello, I need help with my order",
        "What's trending this week?",
        "Where is your nearest store?",
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"\n  Query: {query}")
        print(f"  → Lobe: {result.lobe_name} | Intent: {result.intent} | Confidence: {result.confidence:.2f}")
        
        parallel = router.route_parallel(query)
        print(f"  → Parallel: {parallel}")
    
    print("\n" + "=" * 60)
    print("Router test complete!")
    print("=" * 60)
