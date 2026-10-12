import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

def find_python_executable():
    # Prefer virtual environment python if it exists
    venv_python_win = ROOT_DIR / "venv" / "Scripts" / "python.exe"
    venv_python_unix = ROOT_DIR / "venv" / "bin" / "python"
    
    if venv_python_win.exists():
        return str(venv_python_win)
    elif venv_python_unix.exists():
        return str(venv_python_unix)
    return sys.executable

def main():
    py_exec = find_python_executable()
    print("=" * 60)
    print("🚀 Starting Multi-Source RAG Application")
    print("=" * 60)
    print(f"🐍 Using Python: {py_exec}")
    print(f"📁 Root: {ROOT_DIR}")
    print("-" * 60)

    # 1. Start Backend (FastAPI via uvicorn)
    print("▶️  Starting Backend (FastAPI on http://localhost:8000)...")
    backend_proc = subprocess.Popen(
        [py_exec, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=str(ROOT_DIR),
        shell=False
    )

    # Give backend a moment to initialize
    time.sleep(1.5)

    # 2. Start Frontend (Vite)
    print("▶️  Starting Frontend (React Vite on http://localhost:5173)...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        shell=True if os.name == "nt" else False
    )

    # 3. Open Browser
    time.sleep(2)
    print("\n🌐 Opening application in your browser...")
    webbrowser.open("http://localhost:5173")

    print("\n" + "=" * 60)
    print("✅ All services are RUNNING!")
    print("   • Frontend UI: http://localhost:5173")
    print("   • Backend API: http://localhost:8000")
    print("   • API Docs:    http://localhost:8000/docs")
    print("   (Press Ctrl+C in this terminal to stop both servers)")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        frontend_proc.terminate()
        backend_proc.terminate()
        print("✅ Servers stopped.")

if __name__ == "__main__":
    main()
