"""
🔥 POWER TOOLS - SUPER LAUNCHER 🔥
====================================

Launch the entire Seamless Retail application WITHOUT requiring 
downloaded models. Provides graceful degradation with stub responses.

This is "super script level shit" - it just works, no matter what.

Features:
- ✓ Hardware auto-detection
- ✓ Model-free operation with stub responses  
- ✓ Graceful degradation for all components
- ✓ Service health checks
- ✓ One-command app launch

Usage:
    python power_tools/super_launcher.py
    python power_tools/super_launcher.py --dry-run
    python power_tools/super_launcher.py --no-frontend
"""

import sys
import os
import argparse
import subprocess
import time
import threading
import signal
import socket
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ServiceStatus(Enum):
    STOPPED = "⏹ STOPPED"
    STARTING = "🔄 STARTING"
    RUNNING = "✅ RUNNING"
    DEGRADED = "⚠ DEGRADED"
    FAILED = "❌ FAILED"


@dataclass
class ServiceConfig:
    name: str
    port: int
    command: List[str]
    health_endpoint: str = "/health"
    required: bool = True
    cwd: Optional[str] = None


# Service definitions
SERVICES = {
    "brain_api": ServiceConfig(
        name="Cognitive Brain API",
        port=8001,
        command=["python", "-m", "uvicorn", "cognitive_brain.api:app", "--host", "0.0.0.0", "--port", "8001"],
        health_endpoint="/health",
        required=True,
    ),
    "brain_server": ServiceConfig(
        name="Brain Prediction Server",
        port=8002,
        command=["python", "scripts/brain/serve_brain.py"],
        health_endpoint="/health",
        required=False,
    ),
}


# ==============================================================================
# HARDWARE DETECTION
# ==============================================================================

class HardwareInfo:
    """Detect and report available hardware."""
    
    def __init__(self):
        self.has_cuda = False
        self.cuda_device_count = 0
        self.cuda_memory_gb = 0.0
        self.cuda_device_name = "N/A"
        self.cpu_count = os.cpu_count() or 1
        self.system_memory_gb = 0.0
        
        self._detect()
    
    def _detect(self):
        """Detect available hardware."""
        # CUDA detection
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self.cuda_device_count = torch.cuda.device_count()
                self.cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.cuda_device_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # System memory (rough estimate)
        try:
            import psutil
            self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
    
    def print_summary(self):
        """Print hardware summary."""
        print("\n" + "=" * 60)
        print("🖥️  HARDWARE DETECTION")
        print("=" * 60)
        print(f"  CPU Cores:     {self.cpu_count}")
        print(f"  System RAM:    {self.system_memory_gb:.1f} GB" if self.system_memory_gb else "  System RAM:    Unknown")
        
        if self.has_cuda:
            print(f"  GPU:           {self.cuda_device_name}")
            print(f"  VRAM:          {self.cuda_memory_gb:.1f} GB")
            print(f"  GPU Count:     {self.cuda_device_count}")
        else:
            print("  GPU:           None (CPU mode)")
        
        print("=" * 60 + "\n")
    
    def get_tier(self) -> str:
        """Get hardware tier classification."""
        if not self.has_cuda:
            return "CPU_ONLY"
        elif self.cuda_memory_gb < 8:
            return "LOW_VRAM"
        elif self.cuda_memory_gb < 12:
            return "CONSUMER"
        elif self.cuda_memory_gb < 24:
            return "PROSUMER"
        else:
            return "DATACENTER"


# ==============================================================================
# MODEL STUBS (for graceful degradation)
# ==============================================================================

class ModelStubs:
    """Stub responses when models are not available."""
    
    @staticmethod
    def sales_prediction(features: Dict) -> Dict:
        """Stub for sales prediction."""
        import random
        return {
            "predicted_sales": round(random.uniform(1000, 5000), 2),
            "confidence": 0.0,
            "warning": "Running in STUB mode - model not loaded",
            "is_stub": True,
        }
    
    @staticmethod
    def llm_response(query: str) -> Dict:
        """Stub for LLM responses."""
        return {
            "response": f"[STUB MODE] I received your query: '{query[:50]}...'. "
                       "The LLM models are not currently loaded. Please download models to enable full functionality.",
            "model": "stub",
            "tokens": 0,
            "is_stub": True,
        }
    
    @staticmethod
    def rag_search(query: str) -> Dict:
        """Stub for RAG search."""
        return {
            "results": [],
            "warning": "RAG memory not initialized - returning empty results",
            "is_stub": True,
        }
    
    @staticmethod
    def intent_classification(query: str) -> Dict:
        """Stub for intent classification."""
        return {
            "intent": "general_chat",
            "confidence": 0.0,
            "warning": "Intent classifier not loaded - defaulting to general_chat",
            "is_stub": True,
        }


# ==============================================================================
# SERVICE LAUNCHER
# ==============================================================================

class ServiceLauncher:
    """Launch and manage services with graceful degradation."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status: Dict[str, ServiceStatus] = {}
        self.hardware = HardwareInfo()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\n🛑 Shutting down services...")
        self.stop_all()
        sys.exit(0)
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False
    
    def _wait_for_service(self, port: int, timeout: int = 30) -> bool:
        """Wait for a service to become available."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(("127.0.0.1", port))
                    return True
            except (OSError, socket.timeout):
                time.sleep(0.5)
        return False
    
    def check_models(self) -> Dict[str, bool]:
        """Check which models are available."""
        models = {
            "sales_brain": os.path.exists(os.path.join(PROJECT_ROOT, "scripts/brain/sales_brain.pth")),
            "rag_index": os.path.exists(os.path.join(PROJECT_ROOT, "scripts/brain/rag_index.faiss")),
            "clara_index": os.path.exists(os.path.join(PROJECT_ROOT, "scripts/brain/clara_index.faiss")),
        }
        
        # Check for LLM models
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if os.path.exists(models_dir):
            models["llm_available"] = len(os.listdir(models_dir)) > 0
        else:
            models["llm_available"] = False
        
        # Check trained_models
        trained_dir = os.path.join(PROJECT_ROOT, "trained_models")
        if os.path.exists(trained_dir):
            models["trained_adapters"] = len(os.listdir(trained_dir)) > 0
        else:
            models["trained_adapters"] = False
        
        return models
    
    def print_model_status(self):
        """Print model availability status."""
        models = self.check_models()
        
        print("\n" + "=" * 60)
        print("📦 MODEL STATUS")
        print("=" * 60)
        
        for name, available in models.items():
            status = "✅ Available" if available else "❌ Missing (stub mode)"
            print(f"  {name:20s} {status}")
        
        if not any(models.values()):
            print("\n  ⚠️  No models found - app will run in FULL STUB MODE")
            print("      All responses will be placeholders.")
        
        print("=" * 60 + "\n")
    
    def start_service(self, service_id: str, config: ServiceConfig) -> bool:
        """Start a single service."""
        print(f"  🔄 Starting {config.name} on port {config.port}...")
        
        if self.dry_run:
            print(f"     [DRY-RUN] Would run: {' '.join(config.command)}")
            self.status[service_id] = ServiceStatus.RUNNING
            return True
        
        # Check port availability
        if not self._is_port_available(config.port):
            print(f"     ⚠️  Port {config.port} already in use - service may already be running")
            self.status[service_id] = ServiceStatus.RUNNING
            return True
        
        try:
            # Start process
            cwd = config.cwd or PROJECT_ROOT
            process = subprocess.Popen(
                config.command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            
            self.processes[service_id] = process
            self.status[service_id] = ServiceStatus.STARTING
            
            # Wait for service to start
            if self._wait_for_service(config.port, timeout=15):
                self.status[service_id] = ServiceStatus.RUNNING
                print(f"     ✅ {config.name} started successfully")
                return True
            else:
                self.status[service_id] = ServiceStatus.DEGRADED
                print(f"     ⚠️  {config.name} started but not responding (degraded mode)")
                return not config.required
                
        except Exception as e:
            self.status[service_id] = ServiceStatus.FAILED
            print(f"     ❌ Failed to start {config.name}: {e}")
            return not config.required
    
    def start_all(self) -> bool:
        """Start all services."""
        print("\n" + "=" * 60)
        print("🚀 STARTING SERVICES")
        print("=" * 60)
        
        all_success = True
        for service_id, config in SERVICES.items():
            success = self.start_service(service_id, config)
            if not success and config.required:
                all_success = False
        
        return all_success
    
    def stop_all(self):
        """Stop all services."""
        for service_id, process in self.processes.items():
            if process.poll() is None:  # Still running
                print(f"  Stopping {service_id}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        self.processes.clear()
        self.status.clear()
    
    def print_status(self):
        """Print service status."""
        print("\n" + "=" * 60)
        print("📊 SERVICE STATUS")
        print("=" * 60)
        
        for service_id, config in SERVICES.items():
            status = self.status.get(service_id, ServiceStatus.STOPPED)
            print(f"  {config.name:25s} {status.value}")
        
        print("=" * 60 + "\n")
    
    def launch(self):
        """Main launch sequence."""
        print("\n" + "=" * 60)
        print("🔥 POWER TOOLS - SUPER LAUNCHER 🔥")
        print("=" * 60)
        print("  Seamless Retail Application Launcher")
        print("  Runs even without downloaded models!")
        print("=" * 60)
        
        # Hardware detection
        self.hardware.print_summary()
        
        # Model status
        self.print_model_status()
        
        # Start services
        if self.dry_run:
            print("🧪 DRY-RUN MODE - No services will actually start\n")
        
        success = self.start_all()
        
        # Print final status
        self.print_status()
        
        if success:
            print("✅ Application launched successfully!")
            print("\n📍 Access points:")
            for service_id, config in SERVICES.items():
                if self.status.get(service_id) == ServiceStatus.RUNNING:
                    print(f"   • {config.name}: http://localhost:{config.port}")
            
            print("\n⌨️  Press Ctrl+C to stop all services\n")
            
            if not self.dry_run:
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        else:
            print("❌ Some required services failed to start")
            self.stop_all()
            return 1
        
        return 0


# ==============================================================================
# QUICK TESTS (run without models)
# ==============================================================================

def run_quick_tests():
    """Run quick tests that work without models."""
    print("\n" + "=" * 60)
    print("🧪 QUICK TESTS (No Models Required)")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: PyTorch import
    tests_total += 1
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
    
    # Test 2: FAISS import
    tests_total += 1
    try:
        import faiss
        print(f"  ✅ FAISS available")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ FAISS import failed: {e}")
    
    # Test 3: Neural architectures import
    tests_total += 1
    try:
        from cognitive_brain.core.neural_architectures import ResidualAttentionMLP
        print(f"  ✅ Neural architectures importable")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ Neural architectures import failed: {e}")
    
    # Test 4: Model stubs work
    tests_total += 1
    try:
        result = ModelStubs.sales_prediction({"test": 1})
        assert "predicted_sales" in result
        assert result["is_stub"] == True
        print(f"  ✅ Model stubs functional")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Model stubs failed: {e}")
    
    # Test 5: Hardware detection
    tests_total += 1
    try:
        hw = HardwareInfo()
        assert hw.cpu_count > 0
        print(f"  ✅ Hardware detection: {hw.get_tier()}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Hardware detection failed: {e}")
    
    print("=" * 60)
    print(f"  Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60 + "\n")
    
    return tests_passed == tests_total


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Super Launcher - Launch app without models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python power_tools/super_launcher.py              # Full launch
  python power_tools/super_launcher.py --dry-run    # Test mode
  python power_tools/super_launcher.py --test       # Quick tests only
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't actually start services")
    parser.add_argument("--test", action="store_true", help="Run quick tests only")
    parser.add_argument("--no-brain", action="store_true", help="Skip brain API")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_tests()
        return 0 if success else 1
    
    # Modify services based on args
    if args.no_brain:
        SERVICES.pop("brain_api", None)
    
    launcher = ServiceLauncher(dry_run=args.dry_run)
    return launcher.launch()


if __name__ == "__main__":
    sys.exit(main())
