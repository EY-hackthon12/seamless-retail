#!/usr/bin/env python3
"""
Seamless Retail AI - Demo Script
================================

This script provides a quick demonstration of the Seamless Retail AI system.
It showcases the core capabilities without requiring full model downloads.

Usage:
    python scripts/demo.py [--mode {quick,full,interactive}]

Author: Seamless Retail Team
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DemoRunner:
    """Orchestrates the demo experience."""
    
    def __init__(self, mode: str = "quick"):
        self.mode = mode
        self.start_time = datetime.now()
        
    def print_banner(self):
        """Print the demo banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗███████╗ █████╗ ███╗   ███╗██╗     ███████╗███████╗███████╗        ║
║   ██╔════╝██╔════╝██╔══██╗████╗ ████║██║     ██╔════╝██╔════╝██╔════╝        ║
║   ███████╗█████╗  ███████║██╔████╔██║██║     █████╗  ███████╗███████╗        ║
║   ╚════██║██╔══╝  ██╔══██║██║╚██╔╝██║██║     ██╔══╝  ╚════██║╚════██║        ║
║   ███████║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████╗███████║███████║        ║
║   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝        ║
║                                                                               ║
║                    🛒  RETAIL AI  -  Demo Mode  🤖                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"  🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  📍 Mode: {self.mode.upper()}")
        print("─" * 80)
        print()
    
    def check_environment(self) -> dict:
        """Check environment and dependencies."""
        print("🔍 Checking Environment...")
        
        checks = {
            "Python Version": sys.version.split()[0],
            "Project Root": str(PROJECT_ROOT),
            "GPU Available": self._check_gpu(),
            "Cognitive Brain": self._check_module("cognitive_brain"),
            "Agents": self._check_module("agents"),
            "Backend": self._check_module("backend"),
        }
        
        for check, status in checks.items():
            icon = "✅" if status and status != "Not Found" else "⚠️"
            print(f"   {icon} {check}: {status}")
        
        print()
        return checks
    
    def _check_gpu(self) -> str:
        """Check if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            return "CPU Only"
        except ImportError:
            return "PyTorch Not Installed"
    
    def _check_module(self, module_name: str) -> str:
        """Check if a module directory exists."""
        module_path = PROJECT_ROOT / module_name
        if module_path.exists():
            count = len(list(module_path.glob("**/*.py")))
            return f"Found ({count} files)"
        return "Not Found"
    
    def demo_agents(self):
        """Demonstrate agent capabilities."""
        print("🤖 Agent System Demo")
        print("─" * 40)
        
        agents = [
            ("Customer Agent", "Handles customer queries and interactions"),
            ("Product Agent", "Manages product recommendations"),
            ("Inventory Agent", "Tracks stock and inventory levels"),
            ("Analytics Agent", "Provides business insights"),
        ]
        
        for name, description in agents:
            print(f"   • {name}")
            print(f"     └─ {description}")
        
        print()
    
    def demo_cognitive_brain(self):
        """Demonstrate cognitive brain capabilities."""
        print("🧠 Cognitive Brain Demo")
        print("─" * 40)
        
        lobes = [
            ("Language Lobe", "NLP processing & understanding"),
            ("Vision Lobe", "Image & product recognition"),
            ("Memory Lobe", "Context retention & recall"),
            ("Decision Lobe", "Strategic planning & actions"),
        ]
        
        for name, capability in lobes:
            print(f"   • {name}")
            print(f"     └─ {capability}")
        
        print()
    
    def demo_sample_interaction(self):
        """Show a sample interaction."""
        print("💬 Sample Interaction")
        print("─" * 40)
        
        conversation = [
            ("Customer", "I'm looking for a gift for my mom's birthday."),
            ("AI", "I'd love to help! Could you tell me a bit about her interests?"),
            ("Customer", "She loves gardening and cooking."),
            ("AI", "Perfect! Based on that, I'd recommend:\n"
                   "     1. 🌱 Premium Herb Garden Kit - $34.99\n"
                   "     2. 📚 Italian Cookbook Collection - $29.99\n"
                   "     3. 🌿 Indoor Smart Planter - $49.99\n"
                   "     All available for same-day delivery!"),
        ]
        
        for speaker, message in conversation:
            prefix = "👤" if speaker == "Customer" else "🤖"
            print(f"   {prefix} {speaker}: {message}")
            print()
        
        print()
    
    def run_quick_demo(self):
        """Run a quick demonstration."""
        self.print_banner()
        self.check_environment()
        self.demo_agents()
        self.demo_cognitive_brain()
        self.demo_sample_interaction()
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print("─" * 80)
        print(f"✨ Demo completed in {duration:.2f} seconds")
        print("   Run 'python scripts/demo.py --mode full' for extended demo")
        print()
    
    def run_full_demo(self):
        """Run the full demonstration suite."""
        self.print_banner()
        self.check_environment()
        self.demo_agents()
        self.demo_cognitive_brain()
        self.demo_sample_interaction()
        
        # Additional full demo features
        print("📊 System Capabilities Overview")
        print("─" * 40)
        
        capabilities = [
            "Multi-language support (10+ languages)",
            "Real-time inventory tracking",
            "Personalized recommendations",
            "Sentiment analysis",
            "Voice interface ready",
            "Multi-channel integration",
        ]
        
        for cap in capabilities:
            print(f"   ✓ {cap}")
        
        print()
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print("─" * 80)
        print(f"✨ Full demo completed in {duration:.2f} seconds")
        print()
    
    def run_interactive_demo(self):
        """Run an interactive demonstration."""
        self.print_banner()
        self.check_environment()
        
        print("🎮 Interactive Mode")
        print("─" * 40)
        print("   Type 'help' for commands, 'exit' to quit")
        print()
        
        commands = {
            "help": "Show available commands",
            "agents": "Show agent information",
            "brain": "Show cognitive brain info",
            "sample": "Show sample interaction",
            "status": "Show system status",
            "exit": "Exit demo",
        }
        
        while True:
            try:
                user_input = input("🔹 demo> ").strip().lower()
                
                if user_input == "exit":
                    print("👋 Goodbye!")
                    break
                elif user_input == "help":
                    print("\n   Available commands:")
                    for cmd, desc in commands.items():
                        print(f"     • {cmd}: {desc}")
                    print()
                elif user_input == "agents":
                    self.demo_agents()
                elif user_input == "brain":
                    self.demo_cognitive_brain()
                elif user_input == "sample":
                    self.demo_sample_interaction()
                elif user_input == "status":
                    self.check_environment()
                elif user_input:
                    print(f"   Unknown command: '{user_input}'. Type 'help' for options.\n")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                break
    
    def run(self):
        """Run the demo based on mode."""
        if self.mode == "quick":
            self.run_quick_demo()
        elif self.mode == "full":
            self.run_full_demo()
        elif self.mode == "interactive":
            self.run_interactive_demo()
        else:
            print(f"Unknown mode: {self.mode}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Seamless Retail AI Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo.py                    # Quick demo
  python scripts/demo.py --mode full        # Full demo
  python scripts/demo.py --mode interactive # Interactive mode
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["quick", "full", "interactive"],
        default="quick",
        help="Demo mode (default: quick)"
    )
    
    args = parser.parse_args()
    
    runner = DemoRunner(mode=args.mode)
    runner.run()


if __name__ == "__main__":
    main()
