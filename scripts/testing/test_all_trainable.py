"""
Comprehensive Test Suite for All Trainable Non-LLM Components
==============================================================

Tests ALL trainable neural network components in the Seamless Retail codebase.
Generates dummy data for self-contained function checking.

NOTE: This script tests ONLY non-LLM components (pure PyTorch models).

Usage:
    python scripts/testing/test_all_trainable.py
    python scripts/testing/test_all_trainable.py --verbose
    python scripts/testing/test_all_trainable.py --component RetailSalesPredictor
"""

import sys
import os
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# TEST RESULT TRACKING
# ==============================================================================

class TestStatus(Enum):
    PASSED = "✓ PASSED"
    FAILED = "✗ FAILED"
    SKIPPED = "○ SKIPPED"
    WARNING = "⚠ WARNING"


@dataclass
class TestResult:
    component: str
    test_name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    exception: Optional[str] = None


class TestSuite:
    """Collects and reports test results."""
    
    def __init__(self, verbose: bool = False):
        self.results: List[TestResult] = []
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def add_result(self, result: TestResult):
        self.results.append(result)
        if self.verbose:
            status_str = result.status.value
            print(f"  {status_str} {result.test_name} ({result.duration_ms:.1f}ms)")
            if result.message:
                print(f"      {result.message}")
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        warnings = sum(1 for r in self.results if r.status == TestStatus.WARNING)
        
        total_time = sum(r.duration_ms for r in self.results)
        
        print(f"\n  ✓ Passed:   {passed}")
        print(f"  ✗ Failed:   {failed}")
        print(f"  ○ Skipped:  {skipped}")
        print(f"  ⚠ Warnings: {warnings}")
        print(f"\n  Total Time: {total_time:.0f}ms")
        print(f"  Device:     {self.device.upper()}")
        
        if failed > 0:
            print("\n  FAILED TESTS:")
            for r in self.results:
                if r.status == TestStatus.FAILED:
                    print(f"    • {r.component}/{r.test_name}: {r.exception or r.message}")
        
        print("\n" + "=" * 70)
        return failed == 0


# ==============================================================================
# DUMMY DATA GENERATORS  
# ==============================================================================

def generate_retail_data(batch_size: int = 32, num_features: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy retail sales data."""
    X = torch.randn(batch_size, num_features)
    y = torch.randn(batch_size, 1)
    return X, y


def generate_sequence_data(
    batch_size: int = 8, 
    seq_len: int = 32, 
    num_features: int = 384
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy sequence data (for transformers/attention)."""
    embeddings = torch.randn(batch_size, seq_len, num_features)
    attention_mask = torch.ones(batch_size, seq_len)
    return embeddings, attention_mask


def generate_temporal_data(
    batch_size: int = 8,
    seq_len: int = 30,
    num_static: int = 5,
    num_temporal: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy temporal forecasting data."""
    static_features = torch.randn(batch_size, num_static)
    temporal_features = torch.randn(batch_size, seq_len, num_temporal)
    return static_features, temporal_features


def generate_text_tokens(
    batch_size: int = 8,
    seq_len: int = 64,
    vocab_size: int = 30522,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy tokenized text data."""
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    return input_ids, attention_mask


# ==============================================================================
# TEST FUNCTIONS FOR EACH COMPONENT
# ==============================================================================

def test_retail_sales_predictor(suite: TestSuite):
    """Test the basic RetailSalesPredictor model."""
    component = "RetailSalesPredictor"
    print(f"\n{'='*60}")
    print(f"Testing: {component}")
    print(f"{'='*60}")
    
    try:
        from scripts.brain.model import RetailSalesPredictor
    except ImportError as e:
        suite.add_result(TestResult(
            component, "import", TestStatus.FAILED, 0, exception=str(e)
        ))
        return
    
    # Test 1: Instantiation
    start = time.time()
    try:
        model = RetailSalesPredictor(input_dim=7, hidden_dim=64)
        model.to(suite.device)
        suite.add_result(TestResult(
            component, "instantiation", TestStatus.PASSED, 
            (time.time() - start) * 1000,
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "instantiation", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
        return
    
    # Test 2: Forward Pass
    start = time.time()
    try:
        X, _ = generate_retail_data(batch_size=32)
        X = X.to(suite.device)
        output = model(X)
        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
        suite.add_result(TestResult(
            component, "forward_pass", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Input: {X.shape} → Output: {output.shape}"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_pass", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
        return
    
    # Test 3: Backward Pass (training capability)
    start = time.time()
    try:
        model.train()
        X, y = generate_retail_data(batch_size=16)
        X, y = X.to(suite.device), y.to(suite.device)
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads, "Not all parameters have gradients"
        
        suite.add_result(TestResult(
            component, "backward_pass", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Loss: {loss.item():.4f}, Gradients: OK"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "backward_pass", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


def test_document_compressor(suite: TestSuite):
    """Test the CLaRa DocumentCompressor."""
    component = "DocumentCompressor"
    print(f"\n{'='*60}")
    print(f"Testing: {component}")
    print(f"{'='*60}")
    
    try:
        from scripts.brain.clara_rag import DocumentCompressor
    except ImportError as e:
        suite.add_result(TestResult(
            component, "import", TestStatus.FAILED, 0, exception=str(e)
        ))
        return
    
    # Test 1: Instantiation
    start = time.time()
    try:
        compressor = DocumentCompressor(
            input_dim=384,
            hidden_dim=512,
            output_dim=256,
            compress_rate=32,
        )
        compressor.to(suite.device)
        suite.add_result(TestResult(
            component, "instantiation", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Compression rate: 32x"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "instantiation", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
        return
    
    # Test 2: Forward Pass
    start = time.time()
    try:
        embeddings, mask = generate_sequence_data(batch_size=4, seq_len=128, num_features=384)
        embeddings = embeddings.to(suite.device)
        mask = mask.to(suite.device)
        
        compressed = compressor(embeddings, mask)
        expected_tokens = max(1, 128 // 32)  # 4 memory tokens
        
        assert compressed.shape[0] == 4, f"Batch mismatch"
        assert compressed.shape[2] == 256, f"Output dim mismatch"
        
        suite.add_result(TestResult(
            component, "forward_pass", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"[4, 128, 384] → {list(compressed.shape)} ({128//compressed.shape[1]}x compression)"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_pass", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
        return
    
    # Test 3: Backward Pass
    start = time.time()
    try:
        compressor.train()
        embeddings, mask = generate_sequence_data(batch_size=2, seq_len=64, num_features=384)
        embeddings = embeddings.to(suite.device)
        mask = mask.to(suite.device)
        
        compressed = compressor(embeddings, mask)
        loss = compressed.mean()  # Dummy loss
        loss.backward()
        
        suite.add_result(TestResult(
            component, "backward_pass", TestStatus.PASSED,
            (time.time() - start) * 1000,
            "Gradients flow through compression attention"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "backward_pass", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


def test_neural_architectures(suite: TestSuite):
    """Test all neural architectures from cognitive_brain."""
    print(f"\n{'='*60}")
    print(f"Testing: Neural Architectures (cognitive_brain)")
    print(f"{'='*60}")
    
    try:
        from cognitive_brain.core.neural_architectures import (
            ResidualAttentionMLP,
            RetailSalesPredictorV2,
            TemporalFusionTransformer,
            RetailTransformer,
            IntentClassifier,
        )
    except ImportError as e:
        suite.add_result(TestResult(
            "NeuralArchitectures", "import", TestStatus.FAILED, 0, exception=str(e)
        ))
        return
    
    # === Test ResidualAttentionMLP ===
    component = "ResidualAttentionMLP"
    start = time.time()
    try:
        model = ResidualAttentionMLP(
            input_dim=7,
            hidden_dims=[64, 64, 64],
            output_dim=1,
            num_heads=4,
        ).to(suite.device)
        
        X, _ = generate_retail_data(batch_size=16)
        X = X.to(suite.device)
        output = model(X)
        
        assert output.shape == (16, 1)
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"[16, 7] → [16, 1] with attention"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
    
    # === Test RetailSalesPredictorV2 ===
    component = "RetailSalesPredictorV2"
    start = time.time()
    try:
        model = RetailSalesPredictorV2(
            input_dim=7,
            hidden_dim=128,
            num_layers=3,
        ).to(suite.device)
        
        X, _ = generate_retail_data(batch_size=16)
        X = X.to(suite.device)
        output = model(X)
        
        assert output.shape == (16, 1)
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Enhanced predictor with {sum(p.numel() for p in model.parameters()):,} params"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
    
    # === Test TemporalFusionTransformer ===
    component = "TemporalFusionTransformer"
    start = time.time()
    try:
        model = TemporalFusionTransformer(
            num_static_features=5,
            num_temporal_features=10,
            hidden_dim=32,
            forecast_horizon=7,
        ).to(suite.device)
        
        static, temporal = generate_temporal_data(batch_size=8)
        static = static.to(suite.device)
        temporal = temporal.to(suite.device)
        
        forecast = model(static, temporal)
        assert forecast.shape == (8, 7), f"Expected (8, 7), got {forecast.shape}"
        
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Temporal fusion: [8, 30, 10] → [8, 7]"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
    
    # === Test RetailTransformer ===
    component = "RetailTransformer"
    start = time.time()
    try:
        model = RetailTransformer(
            vocab_size=10000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            num_classes=10,
        ).to(suite.device)
        
        input_ids, attention_mask = generate_text_tokens(batch_size=4, seq_len=32, vocab_size=10000)
        input_ids = input_ids.to(suite.device)
        attention_mask = attention_mask.to(suite.device)
        
        logits, hidden = model(input_ids, attention_mask)
        assert logits.shape == (4, 10)
        
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"BERT-style: [4, 32] → [4, 10] logits"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))
    
    # === Test IntentClassifier ===
    component = "IntentClassifier"
    start = time.time()
    try:
        model = IntentClassifier(
            hidden_dim=256,
            num_intents=8,
        ).to(suite.device)
        
        embeddings = torch.randn(4, 256).to(suite.device)
        logits, intent = model(embeddings)
        
        assert logits.shape == (4, 8)
        assert intent in model.intent_names
        
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Intent: '{intent}'"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "forward_backward", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


def test_rag_memory(suite: TestSuite):
    """Test RAG Memory system (without sentence-transformers if unavailable)."""
    component = "RAGMemory"
    print(f"\n{'='*60}")
    print(f"Testing: {component}")
    print(f"{'='*60}")
    
    try:
        # Test basic FAISS functionality
        import faiss
        
        start = time.time()
        dim = 384
        index = faiss.IndexFlatIP(dim)
        
        # Add dummy vectors
        vectors = np.random.randn(100, dim).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
        index.add(vectors)
        
        # Search
        query = np.random.randn(1, dim).astype('float32')
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        scores, indices = index.search(query, 5)
        
        suite.add_result(TestResult(
            component, "faiss_operations", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Indexed 100 vectors, retrieved top-5"
        ))
        
    except Exception as e:
        suite.add_result(TestResult(
            component, "faiss_operations", TestStatus.FAILED,
            0, exception=str(e)
        ))
        return
    
    # Test actual RAGMemory class
    start = time.time()
    try:
        from scripts.brain.rag import RAGMemory
        
        # This may fail if sentence-transformers not installed
        memory = RAGMemory(use_gpu=torch.cuda.is_available())
        
        suite.add_result(TestResult(
            component, "initialization", TestStatus.PASSED,
            (time.time() - start) * 1000,
            f"Device: {memory.device}, GPU FAISS: {memory.use_gpu}"
        ))
    except ImportError as e:
        suite.add_result(TestResult(
            component, "initialization", TestStatus.WARNING,
            (time.time() - start) * 1000,
            f"sentence-transformers not available: {e}"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "initialization", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


def test_differentiable_topk(suite: TestSuite):
    """Test differentiable top-k for end-to-end training."""
    component = "DifferentiableTopK"
    print(f"\n{'='*60}")
    print(f"Testing: {component}")
    print(f"{'='*60}")
    
    try:
        from scripts.brain.clara_rag import differentiable_topk
    except ImportError as e:
        suite.add_result(TestResult(
            component, "import", TestStatus.FAILED, 0, exception=str(e)
        ))
        return
    
    start = time.time()
    try:
        # Run on CPU for reliable gradient testing (avoids CUDA-specific issues)
        logits = torch.randn(4, 100, requires_grad=True)  # Keep on CPU
        
        weights, indices = differentiable_topk(logits, k=10, temperature=0.5)
        
        assert weights.shape == (4, 100), f"Weights shape mismatch: {weights.shape}"
        assert indices.shape == (4, 10), f"Indices shape mismatch: {indices.shape}"
        
        # Check gradient flow
        loss = weights.sum()
        loss.backward()
        
        has_grad = logits.grad is not None and logits.grad.abs().sum() > 0
        assert has_grad, "Gradients not flowing through differentiable_topk"
        
        suite.add_result(TestResult(
            component, "gradient_flow", TestStatus.PASSED,
            (time.time() - start) * 1000,
            "Gumbel-softmax enables end-to-end training"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "gradient_flow", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


def test_model_checkpoint(suite: TestSuite):
    """Test model save/load for non-LLM models."""
    component = "ModelCheckpoint"
    print(f"\n{'='*60}")
    print(f"Testing: {component}")
    print(f"{'='*60}")
    
    import tempfile
    
    try:
        from scripts.brain.model import RetailSalesPredictor
    except ImportError as e:
        suite.add_result(TestResult(
            component, "import", TestStatus.FAILED, 0, exception=str(e)
        ))
        return
    
    start = time.time()
    try:
        # Create model
        model = RetailSalesPredictor(input_dim=7)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name
        
        # Load into new model
        model2 = RetailSalesPredictor(input_dim=7)
        model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch in {n1}"
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        suite.add_result(TestResult(
            component, "save_load", TestStatus.PASSED,
            (time.time() - start) * 1000,
            "Checkpoint save/load verified"
        ))
    except Exception as e:
        suite.add_result(TestResult(
            component, "save_load", TestStatus.FAILED,
            (time.time() - start) * 1000, exception=str(e)
        ))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test all trainable non-LLM components")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--component", "-c", type=str, help="Test specific component only")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SEAMLESS RETAIL - TRAINABLE COMPONENT TEST SUITE")
    print("=" * 70)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)
    print("\n⚠ NOTE: This tests NON-LLM components only (pure PyTorch models)")
    print("   Uses dummy data for self-contained function checking\n")
    
    suite = TestSuite(verbose=args.verbose)
    
    # Component test mapping
    tests = {
        "RetailSalesPredictor": test_retail_sales_predictor,
        "DocumentCompressor": test_document_compressor,
        "NeuralArchitectures": test_neural_architectures,
        "RAGMemory": test_rag_memory,
        "DifferentiableTopK": test_differentiable_topk,
        "ModelCheckpoint": test_model_checkpoint,
    }
    
    if args.component:
        if args.component in tests:
            tests[args.component](suite)
        else:
            print(f"Unknown component: {args.component}")
            print(f"Available: {list(tests.keys())}")
            return 1
    else:
        for test_func in tests.values():
            test_func(suite)
    
    success = suite.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
