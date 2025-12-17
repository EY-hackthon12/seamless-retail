import os
import sys
import subprocess
import time
import signal
import platform
import webbrowser
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

def print_header():
    print("=" * 60)
    print("   🛒 SEAMLESS RETAIL - FULL SYSTEM DEMO LAUNCHER 🚀")
    print("=" * 60)

def check_requirements():
    print("[1/4] Checking requirements...")
    
    # Check Node.js
    try:
        subprocess.run(["node", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("   ✅ Node.js found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Node.js not found! Please install Node.js to run the frontend.")
        return False

    # Check Python (obviously running, but checking environment)
    print(f"   ✅ Python {sys.version.split()[0]} found")
    return True

def start_backend():
    print("[2/4] Starting Backend (FastAPI)...")
    
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    
    # Windows-specific creation flags to open in new window if possible, or just run in background
    kwargs = {}
    if platform.system() == "Windows":
        # CREATE_NEW_CONSOLE = 0x00000010
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            **kwargs
        )
        print(f"   🚀 Backend started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"   ❌ Failed to start backend: {e}")
        return None

def start_frontend():
    print("[3/4] Starting Frontend (Next.js)...")
    
    cmd = ["npm", "run", "dev"]
    if platform.system() == "Windows":
        cmd = ["npm.cmd", "run", "dev"]
        
    kwargs = {}
    if platform.system() == "Windows":
        # CREATE_NEW_CONSOLE = 0x00000010
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(FRONTEND_DIR),
            **kwargs
        )
        print(f"   🚀 Frontend started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"   ❌ Failed to start frontend: {e}")
        return None

def main():
    print_header()
    
    if not check_requirements():
        input("\nPress Enter to exit...")
        return

    backend_process = start_backend()
    time.sleep(2) # Give backend a moment
    
    frontend_process = start_frontend()
    time.sleep(2) # Give frontend a moment
    
    print("\n[4/4] System is running!")
    print("   👉 Backend: http://localhost:8000/docs")
    print("   👉 Frontend: http://localhost:3000")
    print("\n   Opening frontend in browser...")
    webbrowser.open("http://localhost:3000")
    
    print("\n" + "=" * 60)
    print("   PRESS CTRL+C TO STOP ALL SERVICES")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("   ⚠️ Backend process stopped unexpectedly!")
                break
            if frontend_process.poll() is not None:
                print("   ⚠️ Frontend process stopped unexpectedly!")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
    finally:
        if backend_process:
            backend_process.terminate()
            print("   ✓ Backend stopped")
        if frontend_process:
            frontend_process.terminate() # npm run dev spawns children, this might not kill everything on Windows perfectly without a tree kill, but good enough for demo
            print("   ✓ Frontend stopped")
        print("👋 Demo closed.")

if __name__ == "__main__":
    main()
