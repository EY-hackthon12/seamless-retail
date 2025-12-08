"""
Hardware Detector - Adaptive GPU/System Profiler
=================================================

Auto-detects hardware capabilities and recommends optimal inference strategy.
Scales from consumer GPUs (RTX 3060/4060) to datacenter (A100/H100).

Gold Standard Implementation:
- Full CUDA device enumeration
- VRAM detection with overhead calculation
- Compute capability profiling
- Multi-GPU topology detection
- Automatic quantization level selection
"""

from __future__ import annotations

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Beartype for runtime type checking (Gold Standard)
try:
    from beartype import beartype
    from beartype.typing import Annotated
except ImportError:
    def beartype(func):
        return func
    Annotated = None

# PyTorch for GPU detection
import torch

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """Hardware capability tiers for inference optimization."""
    CPU_ONLY = auto()           # No GPU - CPU inference only
    LOW_VRAM = auto()           # <8GB VRAM - Heavy quantization (Q4_K_M)
    CONSUMER = auto()           # 8-12GB VRAM - AWQ/GPTQ 4-bit
    PROSUMER = auto()           # 12-24GB VRAM - Mixed precision
    DATACENTER = auto()         # 24-48GB VRAM - BF16 with tensor parallelism
    MULTI_GPU = auto()          # Multiple GPUs - Distributed inference


class QuantizationLevel(Enum):
    """Quantization levels for model optimization."""
    NONE = "none"               # Full precision (FP32/BF16)
    FP16 = "fp16"               # Half precision
    INT8 = "int8"               # 8-bit quantization
    INT4_AWQ = "awq"            # 4-bit AWQ
    INT4_GPTQ = "gptq"          # 4-bit GPTQ
    GGUF_Q4_K_M = "q4_k_m"      # llama.cpp 4-bit
    GGUF_Q5_K_M = "q5_k_m"      # llama.cpp 5-bit
    GGUF_Q8_0 = "q8_0"          # llama.cpp 8-bit


class InferenceBackend(Enum):
    """Available inference backends."""
    PYTORCH = "pytorch"         # Native PyTorch
    VLLM = "vllm"               # vLLM for high-throughput
    LLAMA_CPP = "llama_cpp"     # llama.cpp for CPU/low VRAM
    TRITON = "triton"           # NVIDIA Triton
    ONNX = "onnx"               # ONNX Runtime
    TENSORRT = "tensorrt"       # TensorRT (NVIDIA optimized)


@dataclass
class GPUInfo:
    """Detailed GPU information."""
    device_id: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: Tuple[int, int]
    is_available: bool = True
    cuda_cores: int = 0
    tensor_cores: int = 0
    
    @property
    def compute_capability_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
    
    @property
    def supports_bf16(self) -> bool:
        """BF16 requires Ampere (8.0) or newer."""
        return self.compute_capability >= (8, 0)
    
    @property
    def supports_fp8(self) -> bool:
        """FP8 requires Hopper (9.0) or newer."""
        return self.compute_capability >= (9, 0)


@dataclass
class SystemProfile:
    """Complete system hardware profile."""
    gpus: List[GPUInfo] = field(default_factory=list)
    total_system_ram_gb: float = 0.0
    available_system_ram_gb: float = 0.0
    cpu_cores: int = 0
    cpu_threads: int = 0
    platform: str = ""
    cuda_version: str = ""
    pytorch_version: str = ""
    
    @property
    def total_vram_gb(self) -> float:
        return sum(gpu.total_vram_gb for gpu in self.gpus)
    
    @property
    def gpu_count(self) -> int:
        return len(self.gpus)
    
    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0
    
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.total_vram_gb)


@dataclass
class InferenceConfig:
    """Recommended inference configuration based on hardware."""
    tier: HardwareTier
    backend: InferenceBackend
    quantization: QuantizationLevel
    max_batch_size: int
    max_context_length: int
    use_flash_attention: bool
    use_tensor_parallelism: bool
    tensor_parallel_size: int
    gpu_memory_utilization: float
    recommended_models: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.name,
            "backend": self.backend.value,
            "quantization": self.quantization.value,
            "max_batch_size": self.max_batch_size,
            "max_context_length": self.max_context_length,
            "use_flash_attention": self.use_flash_attention,
            "use_tensor_parallelism": self.use_tensor_parallelism,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "recommended_models": self.recommended_models,
        }


class HardwareDetector:
    """
    Gold-Standard Hardware Detection and Profiling System.
    
    Automatically detects available hardware and recommends optimal
    inference configuration for the Cognitive Retail Brain.
    
    Features:
    - Multi-GPU detection and topology analysis
    - VRAM profiling with overhead estimation
    - Automatic quantization level selection
    - Backend recommendation (vLLM, llama.cpp, Triton)
    - Dynamic scaling based on available resources
    """
    
    # VRAM requirements for different model sizes (in GB)
    MODEL_VRAM_REQUIREMENTS = {
        "mistral-7b-fp16": 14.0,
        "mistral-7b-awq": 4.5,
        "mistral-7b-gptq": 4.5,
        "mistral-7b-gguf-q4": 4.0,
        "llama-3-8b-fp16": 16.0,
        "llama-3-8b-awq": 5.0,
        "starcoder2-3b-fp16": 6.0,
        "starcoder2-3b-awq": 2.0,
        "yolov9-onnx": 0.5,
        "clip-vit-large": 1.5,
        "distilbert-base": 0.3,
    }
    
    # Overhead for CUDA context, KV cache, etc.
    VRAM_OVERHEAD_GB = 1.5
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._profile: Optional[SystemProfile] = None
        self._config: Optional[InferenceConfig] = None
    
    @beartype
    def detect(self) -> SystemProfile:
        """
        Perform comprehensive hardware detection.
        
        Returns:
            SystemProfile with detailed hardware information
        """
        if self._profile is not None:
            return self._profile
        
        profile = SystemProfile(
            platform=sys.platform,
            pytorch_version=torch.__version__,
        )
        
        # CPU Detection
        profile.cpu_cores = os.cpu_count() or 1
        profile.cpu_threads = profile.cpu_cores  # Logical cores
        
        # RAM Detection
        try:
            import psutil
            mem = psutil.virtual_memory()
            profile.total_system_ram_gb = mem.total / (1024**3)
            profile.available_system_ram_gb = mem.available / (1024**3)
        except ImportError:
            # Fallback if psutil not available
            profile.total_system_ram_gb = 16.0  # Assume 16GB
            profile.available_system_ram_gb = 8.0
        
        # CUDA Detection
        if torch.cuda.is_available():
            profile.cuda_version = torch.version.cuda or "unknown"
            
            for device_id in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_id)
                
                # Get memory info
                total_mem = props.total_memory / (1024**3)
                try:
                    torch.cuda.set_device(device_id)
                    free_mem = torch.cuda.mem_get_info(device_id)[0] / (1024**3)
                except Exception:
                    free_mem = total_mem * 0.9  # Estimate 90% free
                
                gpu = GPUInfo(
                    device_id=device_id,
                    name=props.name,
                    total_vram_gb=total_mem,
                    free_vram_gb=free_mem,
                    compute_capability=(props.major, props.minor),
                    cuda_cores=props.multi_processor_count * 128,  # Approx
                )
                
                profile.gpus.append(gpu)
                
                if self.verbose:
                    logger.info(f"Detected GPU {device_id}: {gpu.name} "
                               f"({gpu.total_vram_gb:.1f}GB VRAM, SM {gpu.compute_capability_str})")
        else:
            if self.verbose:
                logger.warning("No CUDA-capable GPU detected. Using CPU-only mode.")
        
        self._profile = profile
        return profile
    
    @beartype
    def get_tier(self, profile: Optional[SystemProfile] = None) -> HardwareTier:
        """
        Determine hardware tier based on detected capabilities.
        
        Args:
            profile: Optional pre-computed profile
            
        Returns:
            HardwareTier enum value
        """
        if profile is None:
            profile = self.detect()
        
        if not profile.has_gpu:
            return HardwareTier.CPU_ONLY
        
        if profile.gpu_count > 1:
            return HardwareTier.MULTI_GPU
        
        vram = profile.best_gpu.total_vram_gb
        
        if vram < 8:
            return HardwareTier.LOW_VRAM
        elif vram < 12:
            return HardwareTier.CONSUMER
        elif vram < 24:
            return HardwareTier.PROSUMER
        else:
            return HardwareTier.DATACENTER
    
    @beartype
    def get_recommended_config(self, profile: Optional[SystemProfile] = None) -> InferenceConfig:
        """
        Generate optimal inference configuration based on hardware.
        
        This is the core "brain scaling" logic - more GPU power = more capability.
        
        Args:
            profile: Optional pre-computed profile
            
        Returns:
            InferenceConfig with recommended settings
        """
        if self._config is not None:
            return self._config
        
        if profile is None:
            profile = self.detect()
        
        tier = self.get_tier(profile)
        
        # Default configuration
        config = InferenceConfig(
            tier=tier,
            backend=InferenceBackend.PYTORCH,
            quantization=QuantizationLevel.FP16,
            max_batch_size=1,
            max_context_length=2048,
            use_flash_attention=False,
            use_tensor_parallelism=False,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
        )
        
        # Tier-specific optimizations
        if tier == HardwareTier.CPU_ONLY:
            config.backend = InferenceBackend.LLAMA_CPP
            config.quantization = QuantizationLevel.GGUF_Q4_K_M
            config.max_batch_size = 1
            config.max_context_length = 2048
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
                "code": "starcoder2-3b.Q4_K_M.gguf",
            }
            
        elif tier == HardwareTier.LOW_VRAM:
            # < 8GB VRAM (e.g., RTX 3060 6GB)
            config.backend = InferenceBackend.LLAMA_CPP
            config.quantization = QuantizationLevel.GGUF_Q4_K_M
            config.max_batch_size = 1
            config.max_context_length = 4096
            config.use_flash_attention = True
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
                "code": "starcoder2-3b.Q4_K_M.gguf",
            }
            
        elif tier == HardwareTier.CONSUMER:
            # 8-12GB VRAM (RTX 4060/4070, RTX 3070)
            config.backend = InferenceBackend.VLLM
            config.quantization = QuantizationLevel.INT4_AWQ
            config.max_batch_size = 8
            config.max_context_length = 8192
            config.use_flash_attention = True
            config.gpu_memory_utilization = 0.90
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3-awq",
                "code": "starcoder2-3b-awq",
                "vision": "yolov9s.onnx",
                "clip": "clip-vit-large-patch14",
            }
            
        elif tier == HardwareTier.PROSUMER:
            # 12-24GB VRAM (RTX 4080, RTX 3090, RTX 4090)
            config.backend = InferenceBackend.VLLM
            config.quantization = QuantizationLevel.FP16
            config.max_batch_size = 32
            config.max_context_length = 16384
            config.use_flash_attention = True
            config.gpu_memory_utilization = 0.90
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3",
                "code": "starcoder2-7b",
                "vision": "yolov9m.onnx",
                "clip": "clip-vit-large-patch14",
            }
            
        elif tier == HardwareTier.DATACENTER:
            # 24-48GB VRAM (A100, H100)
            config.backend = InferenceBackend.VLLM
            config.quantization = QuantizationLevel.NONE  # BF16
            config.max_batch_size = 128
            config.max_context_length = 32768
            config.use_flash_attention = True
            config.gpu_memory_utilization = 0.95
            best_gpu = profile.best_gpu
            if best_gpu and best_gpu.supports_bf16:
                config.quantization = QuantizationLevel.NONE
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3",
                "code": "starcoder2-15b",
                "vision": "yolov9e.onnx",
                "clip": "clip-vit-large-patch14",
            }
            
        elif tier == HardwareTier.MULTI_GPU:
            # Multiple GPUs - Enable tensor parallelism
            config.backend = InferenceBackend.VLLM
            config.quantization = QuantizationLevel.FP16
            config.max_batch_size = 128
            config.max_context_length = 32768
            config.use_flash_attention = True
            config.use_tensor_parallelism = True
            config.tensor_parallel_size = profile.gpu_count
            config.gpu_memory_utilization = 0.90
            config.recommended_models = {
                "llm": "mistral-7b-instruct-v0.3",
                "code": "starcoder2-15b",
                "vision": "yolov9e.onnx",
                "clip": "clip-vit-large-patch14",
            }
        
        self._config = config
        return config
    
    @beartype
    def can_load_model(self, model_key: str, profile: Optional[SystemProfile] = None) -> bool:
        """
        Check if a specific model can be loaded given hardware constraints.
        
        Args:
            model_key: Key from MODEL_VRAM_REQUIREMENTS
            profile: Optional pre-computed profile
            
        Returns:
            True if model can be loaded
        """
        if profile is None:
            profile = self.detect()
        
        if model_key not in self.MODEL_VRAM_REQUIREMENTS:
            logger.warning(f"Unknown model: {model_key}")
            return True  # Assume it can be loaded
        
        required_vram = self.MODEL_VRAM_REQUIREMENTS[model_key] + self.VRAM_OVERHEAD_GB
        
        if not profile.has_gpu:
            # CPU-only: Check system RAM instead
            return required_vram < profile.available_system_ram_gb
        
        return required_vram < profile.best_gpu.free_vram_gb
    
    @beartype
    def print_summary(self) -> str:
        """
        Generate human-readable hardware summary.
        
        Returns:
            Formatted summary string
        """
        profile = self.detect()
        config = self.get_recommended_config(profile)
        
        lines = [
            "=" * 60,
            "  COGNITIVE BRAIN - HARDWARE DETECTION REPORT",
            "=" * 60,
            "",
            f"  Platform: {profile.platform}",
            f"  PyTorch: {profile.pytorch_version}",
            f"  CUDA: {profile.cuda_version or 'N/A'}",
            "",
            f"  System RAM: {profile.total_system_ram_gb:.1f} GB",
            f"  CPU Cores: {profile.cpu_cores}",
            "",
        ]
        
        if profile.has_gpu:
            lines.append("  GPUs Detected:")
            for gpu in profile.gpus:
                lines.append(f"    [{gpu.device_id}] {gpu.name}")
                lines.append(f"        VRAM: {gpu.total_vram_gb:.1f} GB "
                           f"(Free: {gpu.free_vram_gb:.1f} GB)")
                lines.append(f"        Compute: SM {gpu.compute_capability_str}")
                lines.append(f"        BF16: {'✓' if gpu.supports_bf16 else '✗'} | "
                           f"FP8: {'✓' if gpu.supports_fp8 else '✗'}")
            lines.append("")
            lines.append(f"  Total VRAM: {profile.total_vram_gb:.1f} GB")
        else:
            lines.append("  GPUs Detected: None (CPU-only mode)")
        
        lines.extend([
            "",
            "-" * 60,
            "  RECOMMENDED CONFIGURATION",
            "-" * 60,
            "",
            f"  Hardware Tier: {config.tier.name}",
            f"  Backend: {config.backend.value}",
            f"  Quantization: {config.quantization.value}",
            f"  Max Batch Size: {config.max_batch_size}",
            f"  Max Context: {config.max_context_length}",
            f"  Flash Attention: {'✓' if config.use_flash_attention else '✗'}",
            f"  Tensor Parallelism: {'✓' if config.use_tensor_parallelism else '✗'}"
            f" (TP={config.tensor_parallel_size})",
            "",
            "  Recommended Models:",
        ])
        
        for model_type, model_name in config.recommended_models.items():
            lines.append(f"    {model_type}: {model_name}")
        
        lines.extend([
            "",
            "=" * 60,
            f"  BRAIN POWER SCORE: {self._calculate_brain_power(profile)}",
            "=" * 60,
        ])
        
        summary = "\n".join(lines)
        print(summary)
        return summary
    
    def _calculate_brain_power(self, profile: SystemProfile) -> str:
        """Calculate a 'brain power' score based on hardware."""
        if not profile.has_gpu:
            return "🧠 (CPU Mode)"
        
        total_vram = profile.total_vram_gb
        gpu_count = profile.gpu_count
        
        # Score based on VRAM and GPU count
        # More GPUs = exponential brain power increase
        base_score = total_vram * 10
        multiplier = 1 + (gpu_count - 1) * 0.8  # 80% efficiency per additional GPU
        
        score = int(base_score * multiplier)
        
        # Brain emoji scale
        if score < 100:
            return f"🧠 {score} (Consumer)"
        elif score < 300:
            return f"🧠🧠 {score} (Prosumer)"
        elif score < 500:
            return f"🧠🧠🧠 {score} (Professional)"
        else:
            return f"🧠🧠🧠🧠 {score} (Datacenter)"
    
    def to_json(self) -> str:
        """Export configuration as JSON."""
        profile = self.detect()
        config = self.get_recommended_config(profile)
        
        data = {
            "profile": {
                "platform": profile.platform,
                "cuda_version": profile.cuda_version,
                "gpu_count": profile.gpu_count,
                "total_vram_gb": profile.total_vram_gb,
                "gpus": [
                    {
                        "id": gpu.device_id,
                        "name": gpu.name,
                        "vram_gb": gpu.total_vram_gb,
                        "compute_capability": gpu.compute_capability_str,
                    }
                    for gpu in profile.gpus
                ],
                "ram_gb": profile.total_system_ram_gb,
            },
            "config": config.to_dict(),
        }
        
        return json.dumps(data, indent=2)


# Singleton instance for easy access
_detector: Optional[HardwareDetector] = None


def get_detector() -> HardwareDetector:
    """Get the global hardware detector instance."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> SystemProfile:
    """Convenience function to detect hardware."""
    return get_detector().detect()


def get_inference_config() -> InferenceConfig:
    """Convenience function to get recommended inference config."""
    return get_detector().get_recommended_config()


if __name__ == "__main__":
    # Run hardware detection when executed directly
    logging.basicConfig(level=logging.INFO)
    detector = HardwareDetector()
    detector.print_summary()
    
    # Export config
    config_path = Path(__file__).parent.parent / "config" / "hardware_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(detector.to_json())
    print(f"\nConfig exported to: {config_path}")
