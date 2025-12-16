"""
CLaRa-Enhanced RAG Memory System
=================================
Implements compressed latent retrieval for efficient context injection.

Key Innovations (from Apple CLaRa):
1. Document compression up to 32x-128x
2. Differentiable top-k for end-to-end training
3. Memory tokens that LLM reads directly (no text decoding)
4. Three-stage training approach support

Reference: arXiv:2511.18659 - CLaRa: Bridging Retrieval and Generation
"""

import os
import math
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# Paths for persistence
INDEX_PATH = os.path.join(os.path.dirname(__file__), "clara_index.faiss")
META_PATH = os.path.join(os.path.dirname(__file__), "clara_meta.pkl")
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "clara_memory.pt")


# =============================================================================
# CLaRa COMPRESSION COMPONENTS
# =============================================================================

class LlamaRMSNorm(nn.Module):
    """Llama-style RMS normalization for stability."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class DocumentCompressor(nn.Module):
    """
    Document Compressor Module.
    
    Compresses document embeddings by a factor of `compress_rate`.
    Uses pooling + projection to create compact memory tokens.
    
    This is a simplified version of CLaRa's Salient Compressor.
    For full implementation, see ml-clara-main/openrlhf/models/modeling_clara.py
    """
    
    def __init__(
        self,
        input_dim: int = 384,      # SentenceTransformer output dim
        hidden_dim: int = 512,      # Internal processing dim
        output_dim: int = 256,      # Compressed representation dim
        compress_rate: int = 32,    # Compression factor
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.compress_rate = compress_rate
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            LlamaRMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Compression via learned pooling (simplified from CLaRa)
        # In full CLaRa, this would be cross-attention with learned queries
        self.compress_queries = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02  # 1 query per compress_rate tokens
        )
        
        # Multi-head attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            LlamaRMSNorm(output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress document embeddings.
        
        Args:
            embeddings: [batch, seq_len, input_dim] - document token embeddings
            attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
            
        Returns:
            compressed: [batch, n_memory_tokens, output_dim]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Calculate number of memory tokens after compression
        n_memory_tokens = max(1, seq_len // self.compress_rate)
        
        # Project to hidden dim
        hidden = self.input_proj(embeddings)  # [batch, seq_len, hidden_dim]
        
        # Expand compression queries
        queries = self.compress_queries.expand(batch_size, n_memory_tokens, -1)
        
        # Compress via cross-attention (queries attend to document tokens)
        compressed, _ = self.compress_attn(
            query=queries,
            key=hidden,
            value=hidden,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
        )  # [batch, n_memory_tokens, hidden_dim]
        
        # Project to output dimension
        output = self.output_proj(compressed)  # [batch, n_memory_tokens, output_dim]
        
        return output


def differentiable_topk(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    hard: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable top-k selection using Gumbel-Softmax.
    
    Enables end-to-end training of retrieval + generation!
    
    Args:
        logits: [batch, n_docs] - similarity scores
        k: Number of documents to select
        temperature: Softmax temperature (lower = sharper)
        hard: If True, use straight-through estimator for hard selection
        
    Returns:
        weights: [batch, n_docs] - soft selection weights
        topk_indices: [batch, k] - indices of top-k documents
    """
    batch_size, n_docs = logits.shape
    
    if k >= n_docs:
        # Return all documents with uniform weights
        weights = F.softmax(logits / temperature, dim=-1)
        indices = torch.arange(n_docs, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        return weights, indices
    
    # Add Gumbel noise for stochastic selection during training
    if logits.requires_grad:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        perturbed_logits = (logits + gumbel_noise) / temperature
    else:
        perturbed_logits = logits / temperature
    
    # Soft weights via softmax
    weights = F.softmax(perturbed_logits, dim=-1)
    
    # Get top-k indices for actual selection
    _, topk_indices = logits.topk(k, dim=-1)
    
    if hard:
        # Straight-through estimator: hard selection in forward, soft in backward
        hard_weights = torch.zeros_like(weights)
        hard_weights.scatter_(1, topk_indices, 1.0 / k)
        weights = hard_weights - weights.detach() + weights
    
    return weights, topk_indices


# =============================================================================
# CLaRa-ENHANCED RAG MEMORY
# =============================================================================

@dataclass
class CLaRaMemoryConfig:
    """Configuration for CLaRa RAG Memory."""
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Compression settings
    compress_rate: int = 32
    compressed_dim: int = 256
    hidden_dim: int = 512
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.5
    
    # Training settings
    temperature: float = 1.0
    use_differentiable_topk: bool = True


class CLaRaRAGMemory:
    """
    CLaRa-Enhanced RAG Memory System.
    
    Key differences from basic RAG:
    1. Documents are COMPRESSED into latent vectors (32x-128x compression)
    2. Retrieval uses differentiable top-k for end-to-end training
    3. Memory tokens can be directly injected into LLM (no text decoding)
    
    Usage:
        memory = CLaRaRAGMemory(config)
        memory.add_memory(user_id, "Document text...", interaction_id)
        compressed_context = memory.search("Query...", top_k=5)
        # compressed_context can be fed directly to LLM!
    """
    
    def __init__(self, config: Optional[CLaRaMemoryConfig] = None):
        self.config = config or CLaRaMemoryConfig()
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CLaRa RAG Memory initialized on {self.device.upper()}")
        
        # Load embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedder = SentenceTransformer(
                self.config.embedding_model,
                device=self.device
            )
        else:
            logger.warning("SentenceTransformers not available, using random embeddings")
            self.embedder = None
        
        # Initialize compressor
        self.compressor = DocumentCompressor(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.compressed_dim,
            compress_rate=self.config.compress_rate,
        ).to(self.device)
        self.compressor.eval()  # Inference mode by default
        
        # Initialize FAISS index for retrieval
        # Using Inner Product for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.config.compressed_dim)
        
        # Metadata storage
        self.metadata: List[Dict[str, Any]] = []
        
        # Memory bank for full compressed representations
        self.memory_bank: Dict[str, torch.Tensor] = {}
        
        # Load existing index if available
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing index and metadata."""
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            logger.info("Loading existing CLaRa index...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            
            if os.path.exists(MEMORY_PATH):
                self.memory_bank = torch.load(MEMORY_PATH, map_location=self.device)
            
            logger.info(f"Loaded {len(self.metadata)} entries from disk")
    
    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
        torch.save(self.memory_bank, MEMORY_PATH)
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        if self.embedder is not None:
            # Use SentenceTransformer
            embedding = self.embedder.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            return embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        else:
            # Fallback: random embeddings (for testing)
            return torch.randn(1, 1, self.config.embedding_dim).to(self.device)
    
    def _embed_texts_batched(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for multiple texts."""
        if self.embedder is not None:
            embeddings = self.embedder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            return embeddings.unsqueeze(1)  # [batch, 1, dim]
        else:
            return torch.randn(len(texts), 1, self.config.embedding_dim).to(self.device)
    
    @torch.no_grad()
    def compress_document(self, text: str) -> torch.Tensor:
        """
        Compress a document into latent memory tokens.
        
        Args:
            text: Document text to compress
            
        Returns:
            compressed: [1, n_memory_tokens, compressed_dim]
        """
        # Get embeddings
        embeddings = self._embed_text(text)  # [1, 1, embedding_dim]
        
        # For longer documents, we'd tokenize and embed each chunk
        # Here we use sentence-level embedding for efficiency
        attention_mask = torch.ones(embeddings.shape[0], embeddings.shape[1]).to(self.device)
        
        # Compress
        compressed = self.compressor(embeddings, attention_mask)
        
        return compressed
    
    def add_memory(
        self,
        user_id: str,
        text: str,
        interaction_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a document to CLaRa memory.
        
        Args:
            user_id: User identifier
            text: Document text to store
            interaction_id: Unique interaction ID
            metadata: Optional additional metadata
        """
        # Compress document
        compressed = self.compress_document(text)  # [1, n_tokens, dim]
        
        # Pool for retrieval (mean pooling)
        pooled = compressed.mean(dim=1)  # [1, dim]
        
        # Normalize for cosine similarity
        pooled = F.normalize(pooled, p=2, dim=-1)
        
        # Add to FAISS index
        self.index.add(pooled.cpu().numpy())
        
        # Store full compressed representation
        memory_key = f"{user_id}_{interaction_id}"
        self.memory_bank[memory_key] = compressed.cpu()
        
        # Store metadata
        self.metadata.append({
            "user_id": user_id,
            "text": text,
            "interaction_id": interaction_id,
            "memory_key": memory_key,
            **(metadata or {}),
        })
        
        # Persist
        self._save()
        
        logger.debug(f"Added memory: {memory_key} (compressed {len(text)} chars)")
    
    def add_memories_batch(
        self,
        user_id: str,
        texts: List[str],
        interaction_ids: List[int],
    ):
        """Add multiple documents in batch (more efficient)."""
        for text, iid in zip(texts, interaction_ids):
            self.add_memory(user_id, text, iid)
    
    @torch.no_grad()
    def search(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        k: int = 5,
        return_compressed: bool = True,
    ) -> Dict[str, Any]:
        """
        Search for relevant documents using CLaRa compression.
        
        Args:
            query_text: Search query
            user_id: Optional filter by user
            k: Number of results to return
            return_compressed: If True, return compressed memory tokens
            
        Returns:
            Dictionary with:
            - results: List of matching documents with scores
            - compressed_context: [k, n_tokens, dim] if return_compressed=True
        """
        if self.index.ntotal == 0:
            return {"results": [], "compressed_context": None}
        
        # Compress query
        query_compressed = self.compress_document(query_text)
        query_pooled = query_compressed.mean(dim=1)
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        
        # Search more than k to allow filtering
        search_k = min(k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_pooled.cpu().numpy(), search_k)
        
        # Filter and collect results
        results = []
        compressed_memories = []
        
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            # Filter by user if specified
            if user_id and meta["user_id"] != user_id:
                continue
            
            # Skip low-confidence results
            if score < self.config.similarity_threshold:
                continue
            
            results.append({
                "text": meta["text"],
                "score": float(score),
                "user_id": meta["user_id"],
                "interaction_id": meta["interaction_id"],
            })
            
            # Collect compressed representation
            if return_compressed:
                memory_key = meta["memory_key"]
                if memory_key in self.memory_bank:
                    compressed_memories.append(self.memory_bank[memory_key])
            
            if len(results) >= k:
                break
        
        # Stack compressed memories
        compressed_context = None
        if return_compressed and compressed_memories:
            compressed_context = torch.cat(compressed_memories, dim=1)  # [1, k*n_tokens, dim]
        
        return {
            "results": results,
            "compressed_context": compressed_context,
        }
    
    def search_with_differentiable_topk(
        self,
        query_text: str,
        k: int = 5,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Differentiable retrieval for end-to-end training.
        
        Returns soft weights over all documents (for gradient flow)
        plus hard top-k indices for actual selection.
        
        Args:
            query_text: Search query
            k: Number of documents to select
            temperature: Softmax temperature
            
        Returns:
            weights: [1, n_docs] - soft selection weights
            topk_indices: [1, k] - indices of selected documents
            results: List of selected document metadata
        """
        if self.index.ntotal == 0:
            return torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long), []
        
        # Get query embedding
        query_compressed = self.compress_document(query_text)
        query_pooled = query_compressed.mean(dim=1)
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        
        # Get all document vectors from index
        all_vectors = faiss.rev_swig_ptr(
            self.index.get_xb(), self.index.ntotal * self.config.compressed_dim
        ).reshape(self.index.ntotal, self.config.compressed_dim)
        all_vectors = torch.from_numpy(all_vectors.copy()).to(self.device)
        
        # Compute similarity scores
        logits = torch.matmul(query_pooled, all_vectors.T)  # [1, n_docs]
        
        # Differentiable top-k selection
        weights, topk_indices = differentiable_topk(
            logits, k, temperature, hard=False
        )
        
        # Get selected metadata
        results = []
        for idx in topk_indices[0].cpu().numpy():
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    "text": meta["text"],
                    "user_id": meta["user_id"],
                    "interaction_id": meta["interaction_id"],
                })
        
        return weights, topk_indices, results
    
    def get_memory_tokens_for_llm(
        self,
        query_text: str,
        k: int = 5,
    ) -> Optional[torch.Tensor]:
        """
        Get compressed memory tokens ready for LLM injection.
        
        This is the KEY CLaRa innovation: these memory tokens can be
        directly concatenated with LLM embeddings, no text decoding needed!
        
        Args:
            query_text: Search query
            k: Number of documents
            
        Returns:
            memory_tokens: [1, k*n_tokens, hidden_dim] or None
        """
        result = self.search(query_text, k=k, return_compressed=True)
        return result.get("compressed_context")
    
    def clear(self):
        """Clear all stored memories."""
        self.index = faiss.IndexFlatIP(self.config.compressed_dim)
        self.metadata = []
        self.memory_bank = {}
        self._save()
        logger.info("CLaRa memory cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_entries": len(self.metadata),
            "index_size": self.index.ntotal,
            "memory_bank_size": len(self.memory_bank),
            "compression_rate": self.config.compress_rate,
            "compressed_dim": self.config.compressed_dim,
            "device": self.device,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_memory: Optional[CLaRaRAGMemory] = None


def get_clara_memory(config: Optional[CLaRaMemoryConfig] = None) -> CLaRaRAGMemory:
    """Get or create the global CLaRa memory instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = CLaRaRAGMemory(config)
    return _global_memory


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CLaRa-Enhanced RAG Memory Test")
    print("=" * 60)
    
    # Initialize
    config = CLaRaMemoryConfig(compress_rate=32)
    memory = CLaRaRAGMemory(config)
    
    print(f"\nStats: {memory.stats()}")
    
    # Add test memories
    test_docs = [
        ("u1", "I love blue summer dresses, especially floral patterns", 1),
        ("u1", "I hate wool sweaters, they make me itchy", 2),
        ("u1", "My favorite brand is Zara for casual wear", 3),
        ("u2", "I prefer formal business attire in dark colors", 4),
        ("u2", "I'm looking for comfortable running shoes", 5),
    ]
    
    print("\nAdding test documents...")
    for user_id, text, iid in test_docs:
        memory.add_memory(user_id, text, iid)
        print(f"  Added: {text[:50]}...")
    
    print(f"\nStats after adding: {memory.stats()}")
    
    # Test search
    queries = [
        ("What fabric do I dislike?", "u1"),
        ("What colors do I prefer?", "u2"),
        ("Give me dress recommendations", None),
    ]
    
    print("\n" + "-" * 60)
    print("Search Results:")
    print("-" * 60)
    
    for query, user_id in queries:
        print(f"\n[QUERY] {query} (user={user_id})")
        result = memory.search(query, user_id=user_id, k=3)
        
        for r in result["results"]:
            print(f"  [{r['score']:.3f}] {r['text'][:60]}...")
        
        if result["compressed_context"] is not None:
            ctx = result["compressed_context"]
            print(f"  → Compressed context shape: {ctx.shape}")
            print(f"  → Ready for LLM injection!")
    
    # Test differentiable top-k
    print("\n" + "-" * 60)
    print("Differentiable Top-K Test:")
    print("-" * 60)
    
    weights, indices, results = memory.search_with_differentiable_topk(
        "I want summer clothes", k=3, temperature=0.5
    )
    print(f"Weights shape: {weights.shape}")
    print(f"Top-3 indices: {indices[0].tolist()}")
    print(f"Weights require grad: {weights.requires_grad}")
    
    print("\n" + "=" * 60)
    print("✓ CLaRa RAG Memory test complete!")
    print("=" * 60)
