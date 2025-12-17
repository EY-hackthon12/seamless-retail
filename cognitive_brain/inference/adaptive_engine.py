"""
Adaptive Inference Engine
==========================

Unified inference API that auto-selects the optimal backend based on hardware.

Backends:
- vLLM: High-throughput serving for medium/high VRAM GPUs
- llama.cpp: CPU/low VRAM fallback with GGUF quantization
- PyTorch: Direct inference for custom models
- Triton: NVIDIA Triton for vision models

Features:
- Automatic hardware detection and backend selection
- Streaming response support
- Batch inference optimization
- Multi-GPU tensor parallelism
"""

from __future__ import annotations

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    Optional, List, Dict, Any, AsyncIterator, Iterator, Union
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

# Import hardware detector
from cognitive_brain.core.hardware_detector import (
    HardwareDetector, 
    InferenceConfig, 
    InferenceBackend,
    QuantizationLevel,
    HardwareTier
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False


@dataclass 
class GenerationResult:
    """Result from text generation."""
    text: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "complete"
    

class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load a model for inference."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the current model."""
        pass


class VLLMEngine(InferenceEngine):
    """
    vLLM Inference Engine for high-throughput LLM serving.
    
    Features:
    - PagedAttention for efficient KV-cache
    - Continuous batching
    - Tensor parallelism for multi-GPU
    - AWQ/GPTQ quantization support
    """
    
    def __init__(self, hardware_config: Optional[InferenceConfig] = None):
        self._llm = None
        self._model_path = None
        self._hardware_config = hardware_config or HardwareDetector().get_recommended_config()
        
    def load_model(
        self, 
        model_path: str,
        quantization: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Load a model using vLLM.
        
        Args:
            model_path: HuggingFace model ID or local path
            quantization: "awq", "gptq", or None for FP16
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        # Use hardware config defaults if not specified
        if quantization is None:
            quant = self._hardware_config.quantization
            if quant == QuantizationLevel.INT4_AWQ:
                quantization = "awq"
            elif quant == QuantizationLevel.INT4_GPTQ:
                quantization = "gptq"
        
        if tensor_parallel_size is None:
            tensor_parallel_size = self._hardware_config.tensor_parallel_size
        
        if max_model_len is None:
            max_model_len = self._hardware_config.max_context_length
        
        logger.info(f"Loading model with vLLM: {model_path}")
        logger.info(f"  Quantization: {quantization or 'none'}")
        logger.info(f"  Tensor Parallel: {tensor_parallel_size}")
        logger.info(f"  Max Context: {max_model_len}")
        
        self._llm = LLM(
            model=model_path,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            **kwargs
        )
        self._model_path = model_path
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using vLLM."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        from vllm import SamplingParams
        import time
        
        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
        )
        
        start_time = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling_params)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0,
            finish_reason=output.outputs[0].finish_reason or "complete"
        )
    
    async def generate_stream(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Stream generation with vLLM."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        # vLLM streaming is handled differently
        # For now, yield complete response
        result = self.generate(prompt, config)
        yield result.text
    
    def is_loaded(self) -> bool:
        return self._llm is not None
    
    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            torch.cuda.empty_cache()


class LlamaCppEngine(InferenceEngine):
    """
    llama.cpp Inference Engine for CPU and low-VRAM GPUs.
    
    Uses GGUF quantized models for efficient inference on consumer hardware.
    Perfect for RTX 4060 8GB and below.
    """
    
    def __init__(self):
        self._llm = None
        self._model_path = None
    
    def load_model(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_threads: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Load a GGUF model using llama-cpp-python.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_threads: CPU threads to use (None = auto)
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python"
            )
        
        if n_threads is None:
            n_threads = os.cpu_count() // 2 or 4
        
        logger.info(f"Loading GGUF model: {model_path}")
        logger.info(f"  Context: {n_ctx}, GPU Layers: {n_gpu_layers}, Threads: {n_threads}")
        
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
            **kwargs
        )
        self._model_path = model_path
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using llama.cpp."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        import time
        
        start_time = time.perf_counter()
        output = self._llm(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
            echo=False
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        generated_text = output['choices'][0]['text']
        tokens_generated = output['usage']['completion_tokens']
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0,
            finish_reason=output['choices'][0].get('finish_reason', 'complete')
        )
    
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Stream generation with llama.cpp."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        for output in self._llm(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
            echo=False,
            stream=True
        ):
            token = output['choices'][0]['text']
            yield token
            await asyncio.sleep(0)  # Yield control
    
    def is_loaded(self) -> bool:
        return self._llm is not None
    
    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None


class PyTorchEngine(InferenceEngine):
    """
    Native PyTorch Inference Engine.
    
    For custom models and when other backends aren't available.
    Uses HuggingFace Transformers with optional quantization.
    """
    
    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(
        self,
        model_path: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ) -> None:
        """
        Load a model using HuggingFace Transformers.
        
        Args:
            model_path: HuggingFace model ID or local path
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model with PyTorch: {model_path}")
        
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                )
                logger.info(f"  Using {'4-bit' if load_in_4bit else '8-bit'} quantization")
            except ImportError:
                logger.warning("bitsandbytes not installed, loading without quantization")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            **kwargs
        )
        
        if self._device == "cuda" and quantization_config is None:
            self._model = self._model.cuda()
        
        self._model.eval()
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using PyTorch."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        import time
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 0,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Decode only new tokens
        input_len = inputs['input_ids'].shape[1]
        generated_text = self._tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )
        tokens_generated = outputs.shape[1] - input_len
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        )
    
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Streaming generation with PyTorch (uses TextIteratorStreamer)."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        
        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            # Fallback to non-streaming
            result = self.generate(prompt, config)
            yield result.text
            return
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            do_sample=config.do_sample,
            streamer=streamer,
        )
        
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
            await asyncio.sleep(0)
        
        thread.join()
    
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None
    
    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()


class AdaptiveInferenceEngine:
    """
    Adaptive Inference Engine - The Brain's Inference Core.
    
    Automatically selects the optimal inference backend based on
    detected hardware capabilities. Provides a unified API for
    text generation regardless of backend.
    
    Scaling Logic:
    - CPU Only: llama.cpp with Q4_K_M
    - Low VRAM (<8GB): llama.cpp with GPU offload
    - Consumer (8-12GB): vLLM with AWQ
    - Prosumer (12-24GB): vLLM with FP16
    - Datacenter (>24GB): vLLM with tensor parallelism
    """
    
    def __init__(self, auto_detect: bool = True):
        self._detector = HardwareDetector()
        self._config: Optional[InferenceConfig] = None
        self._engine: Optional[InferenceEngine] = None
        self._gemini_engine = None  # Gemini API fallback
        self._using_gemini: bool = False
        self._model_path: Optional[str] = None
        
        if auto_detect:
            self._config = self._detector.get_recommended_config()
            logger.info(f"Hardware detected: {self._config.tier.name}")
            logger.info(f"Recommended backend: {self._config.backend.value}")
    
    @property
    def hardware_config(self) -> InferenceConfig:
        """Get the current hardware configuration."""
        if self._config is None:
            self._config = self._detector.get_recommended_config()
        return self._config
    
    @property
    def backend(self) -> Optional[InferenceBackend]:
        """Get the active backend."""
        if self._engine is None:
            return None
        if isinstance(self._engine, VLLMEngine):
            return InferenceBackend.VLLM
        elif isinstance(self._engine, LlamaCppEngine):
            return InferenceBackend.LLAMA_CPP
        elif isinstance(self._engine, PyTorchEngine):
            return InferenceBackend.PYTORCH
        return None
    
    def load_model(
        self,
        model_path: str,
        backend: Optional[InferenceBackend] = None,
        force_cpu: bool = False,
        use_gemini_fallback: bool = True,
        **kwargs
    ) -> None:
        """
        Load a model with automatic backend selection.
        
        Args:
            model_path: Model identifier (HF ID, local path, or GGUF path)
            backend: Force specific backend (None = auto-detect)
            force_cpu: Force CPU-only inference
            use_gemini_fallback: If True, fall back to Gemini API if local load fails
            **kwargs: Backend-specific arguments
        """
        # Check for Gemini API backend request
        if backend == InferenceBackend.GEMINI_API:
            self._load_gemini_fallback(**kwargs)
            return
        
        # Determine backend
        if backend is None:
            if force_cpu:
                backend = InferenceBackend.LLAMA_CPP
            elif model_path.endswith('.gguf'):
                backend = InferenceBackend.LLAMA_CPP
            else:
                backend = self.hardware_config.backend
        
        # Try to load local model
        try:
            self._load_local_model(model_path, backend, force_cpu, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            
            if use_gemini_fallback:
                logger.info("Falling back to Gemini API...")
                try:
                    self._load_gemini_fallback(**kwargs)
                    return
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Both local model and Gemini fallback failed. "
                        f"Local: {e}, Gemini: {fallback_error}"
                    )
            else:
                raise
    
    def _load_local_model(
        self,
        model_path: str,
        backend: InferenceBackend,
        force_cpu: bool,
        **kwargs
    ) -> None:
        """Load a local model with the specified backend."""
        logger.info(f"Initializing {backend.value} backend...")
        
        if backend == InferenceBackend.VLLM:
            self._engine = VLLMEngine(self.hardware_config)
            self._engine.load_model(model_path, **kwargs)
            
        elif backend == InferenceBackend.LLAMA_CPP:
            self._engine = LlamaCppEngine()
            # Determine GPU layers based on hardware
            n_gpu_layers = 0 if force_cpu else -1  # -1 = all on GPU
            if self.hardware_config.tier == HardwareTier.LOW_VRAM:
                n_gpu_layers = 20  # Partial offload
            self._engine.load_model(
                model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=kwargs.get('n_ctx', self.hardware_config.max_context_length),
                **kwargs
            )
            
        elif backend == InferenceBackend.PYTORCH:
            self._engine = PyTorchEngine()
            load_in_4bit = self.hardware_config.quantization in [
                QuantizationLevel.INT4_AWQ,
                QuantizationLevel.INT4_GPTQ
            ]
            self._engine.load_model(
                model_path,
                load_in_4bit=load_in_4bit,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self._model_path = model_path
    
    def _load_gemini_fallback(self, **kwargs) -> None:
        """Load Gemini API as fallback backend."""
        try:
            from cognitive_brain.inference.gemini_fallback import (
                GeminiEngine,
                GeminiConfig,
                is_gemini_available,
            )
        except ImportError:
            raise RuntimeError("Gemini fallback module not available")
        
        if not is_gemini_available():
            raise RuntimeError(
                "Gemini API key not set. Set GOOGLE_API_KEY environment variable."
            )
        
        logger.info("Initializing Gemini API backend...")
        
        config = GeminiConfig()
        self._gemini_engine = GeminiEngine(config)
        self._gemini_engine.load_model(**kwargs)
        self._model_path = "gemini-api-fallback"
        self._using_gemini = True
        
        logger.info(f"Gemini fallback active. Model: {config.preferred_model.value}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with text and metrics
        """
        if self._engine is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return self._engine.generate(prompt, config)
    
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream text generation.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Generated text tokens
        """
        if self._engine is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        async for token in self._engine.generate_stream(prompt, config):
            yield token
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._engine is not None and self._engine.is_loaded()
    
    def unload(self) -> None:
        """Unload the current model and free resources."""
        if self._engine is not None:
            self._engine.unload()
            self._engine = None
            self._model_path = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "loaded": self.is_loaded(),
            "model": self._model_path,
            "backend": self.backend.value if self.backend else None,
            "hardware_tier": self.hardware_config.tier.name,
            "quantization": self.hardware_config.quantization.value,
            "max_context": self.hardware_config.max_context_length,
        }


# Singleton instance
_engine: Optional[AdaptiveInferenceEngine] = None


def get_engine() -> AdaptiveInferenceEngine:
    """Get the global adaptive inference engine."""
    global _engine
    if _engine is None:
        _engine = AdaptiveInferenceEngine()
    return _engine


if __name__ == "__main__":
    # Test hardware detection and engine initialization
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Adaptive Inference Engine Test")
    print("=" * 60)
    
    engine = AdaptiveInferenceEngine()
    config = engine.hardware_config
    
    print(f"\nHardware Tier: {config.tier.name}")
    print(f"Recommended Backend: {config.backend.value}")
    print(f"Quantization: {config.quantization.value}")
    print(f"Max Batch Size: {config.max_batch_size}")
    print(f"Max Context: {config.max_context_length}")
    
    print("\n" + "=" * 60)
    print("Engine ready for model loading")
    print("=" * 60)
