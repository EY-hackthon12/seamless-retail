#!/usr/bin/env python
"""
LLM Hosting Verification Script
================================

Comprehensive verification of the Cognitive Brain's LLM hosting infrastructure.
Run this script to validate your setup before deployment.

Usage:
    python scripts/verify_llm_hosting.py [--full]
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a verification test."""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class VerificationSuite:
    """Suite of verification tests for LLM hosting."""
    
    def __init__(self, full_test: bool = False):
        self.full_test = full_test
        self.results: List[TestResult] = []
    
    def run_all(self) -> bool:
        """Run all verification tests."""
        print("=" * 70)
        print("  COGNITIVE BRAIN - LLM HOSTING VERIFICATION")
        print("=" * 70)
        print()
        
        # Core tests
        self._test_python_environment()
        self._test_pytorch()
        self._test_cuda()
        self._test_hardware_detector()
        self._test_dependencies()
        
        if self.full_test:
            self._test_inference_engine()
            self._test_model_loading()
        
        # Print summary
        self._print_summary()
        
        return all(r.passed for r in self.results)
    
    def _add_result(self, name: str, passed: bool, message: str, duration_ms: float = 0.0):
        """Add a test result."""
        self.results.append(TestResult(name, passed, message, duration_ms))
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            print(f"          └─ {message}")
    
    def _test_python_environment(self):
        """Test Python version and environment."""
        import platform
        
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 9
        
        self._add_result(
            "Python Version",
            is_valid,
            f"Python {version.major}.{version.minor}.{version.micro} "
            f"({'OK' if is_valid else 'Requires 3.9+'})"
        )
    
    def _test_pytorch(self):
        """Test PyTorch installation."""
        try:
            import torch
            self._add_result(
                "PyTorch",
                True,
                f"Version {torch.__version__}"
            )
        except ImportError as e:
            self._add_result("PyTorch", False, str(e))
    
    def _test_cuda(self):
        """Test CUDA availability."""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                cuda_version = torch.version.cuda
                
                self._add_result(
                    "CUDA",
                    True,
                    f"CUDA {cuda_version} | {device_count} GPU(s) | {device_name}"
                )
            else:
                self._add_result(
                    "CUDA",
                    True,  # Not a failure, just info
                    "Not available (CPU mode enabled)"
                )
        except Exception as e:
            self._add_result("CUDA", False, str(e))
    
    def _test_hardware_detector(self):
        """Test hardware detection module."""
        try:
            # Add project root to path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from cognitive_brain.core.hardware_detector import (
                HardwareDetector, 
                get_inference_config
            )
            
            start = time.perf_counter()
            detector = HardwareDetector(verbose=False)
            profile = detector.detect()
            config = detector.get_recommended_config()
            duration = (time.perf_counter() - start) * 1000
            
            self._add_result(
                "Hardware Detector",
                True,
                f"Tier: {config.tier.name} | Backend: {config.backend.value}",
                duration
            )
        except ImportError as e:
            self._add_result(
                "Hardware Detector",
                False,
                f"Module not found: {e}"
            )
        except Exception as e:
            self._add_result("Hardware Detector", False, str(e))
    
    def _test_dependencies(self):
        """Test optional dependencies."""
        deps = {
            "vllm": "vLLM (high-throughput inference)",
            "llama_cpp": "llama-cpp-python (CPU inference)",
            "peft": "PEFT (LoRA adapters)",
            "bitsandbytes": "bitsandbytes (quantization)",
            "transformers": "Transformers (HuggingFace)",
            "accelerate": "Accelerate (distributed)",
            "deepspeed": "DeepSpeed (training)",
        }
        
        available = []
        missing = []
        
        for module, name in deps.items():
            try:
                __import__(module)
                available.append(name.split()[0])
            except ImportError:
                missing.append(name.split()[0])
        
        # Core deps are required
        core_deps = ["transformers", "peft", "accelerate"]
        core_missing = [d for d in core_deps if d not in [a.lower() for a in available]]
        
        if core_missing:
            self._add_result(
                "Dependencies",
                False,
                f"Missing core: {', '.join(core_missing)}"
            )
        else:
            self._add_result(
                "Dependencies",
                True,
                f"Available: {', '.join(available)}"
            )
            
        if missing:
            print(f"          └─ Optional missing: {', '.join(missing)}")
    
    def _test_inference_engine(self):
        """Test adaptive inference engine initialization."""
        try:
            from cognitive_brain.inference.adaptive_engine import (
                AdaptiveInferenceEngine,
                get_engine
            )
            
            start = time.perf_counter()
            engine = AdaptiveInferenceEngine()
            duration = (time.perf_counter() - start) * 1000
            
            status = engine.get_status()
            
            self._add_result(
                "Inference Engine",
                True,
                f"Initialized | Backend: {status.get('backend', 'N/A')}",
                duration
            )
        except Exception as e:
            self._add_result("Inference Engine", False, str(e))
    
    def _test_model_loading(self):
        """Test model loading (smoke test with small model)."""
        print("  [....] Model Loading (this may take a moment)")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Use smallest possible model for quick test
            model_id = "gpt2"  # Tiny model for verification
            
            start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Quick inference test
            inputs = tokenizer("Hello, I am", return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            duration = (time.perf_counter() - start) * 1000
            
            # Clear to not affect subsequent tests
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move cursor up to overwrite the loading message
            print("\033[F\033[K", end="")
            self._add_result(
                "Model Loading",
                True,
                f"GPT-2 loaded and generated successfully",
                duration
            )
        except Exception as e:
            print("\033[F\033[K", end="")
            self._add_result("Model Loading", False, str(e))
    
    def _print_summary(self):
        """Print test summary."""
        print()
        print("-" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        if passed == total:
            print(f"  ✓ All {total} tests passed!")
            print()
            print("  Your system is ready for LLM hosting.")
        else:
            failed = total - passed
            print(f"  {passed}/{total} tests passed ({failed} failed)")
            print()
            print("  Please fix the failed tests before deployment.")
        
        print()
        print("  Next Steps:")
        print("  1. Run hardware detection:  python -m cognitive_brain.core.hardware_detector")
        print("  2. Start LLM server:        python scripts/hosting/serve_optimized.py")
        print("  3. Test inference:          curl http://localhost:8001/health")
        print()
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify LLM hosting setup")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite including model loading"
    )
    args = parser.parse_args()
    
    suite = VerificationSuite(full_test=args.full)
    success = suite.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
