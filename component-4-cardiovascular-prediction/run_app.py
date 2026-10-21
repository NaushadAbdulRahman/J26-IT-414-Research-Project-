"""
CardioLiver-AI Launcher
Launches both the FastAPI backend (Port 8000) and the Vite + React frontend (Port 5173).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
FRONTEND_DIR = ROOT / "frontend"


def main():
    print("=" * 70)
    print(" CardioLiver-AI: MASLD Cardiovascular Risk Prediction & Decision Support")
    print(" Decoupled Architecture: FastAPI (Backend) + Vite React (Frontend)")
    print("=" * 70)
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    print("      - Interactive Swagger Docs: http://127.0.0.1:8000/docs")

    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--reload"],
        cwd=str(ROOT),
    )

    time.sleep(2)

    print("\n[2/2] Starting Vite React Frontend on http://127.0.0.1:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=str(FRONTEND_DIR),
    )

    print("\n" + "=" * 70)
    print(" System Ready!")
    print(" Open in Browser: http://127.0.0.1:5173/")
    print(" FastAPI Swagger: http://127.0.0.1:8000/docs")
    print(" Press Ctrl+C in this terminal to stop both servers.")
    print("=" * 70 + "\n")

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Servers stopped cleanly.")


if __name__ == "__main__":
    main()
