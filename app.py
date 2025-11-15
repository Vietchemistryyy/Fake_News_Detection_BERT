#!/usr/bin/env python3
"""
Fake News Detection System Orchestrator
Manages backend (FastAPI) and frontend (Next.js) services
"""

import os
import sys
import subprocess
import threading
import time
import argparse
import json
from pathlib import Path

class SystemOrchestrator:
    def __init__(self):
        self.workspace_root = Path(__file__).parent.absolute()
        self.api_dir = self.workspace_root / "api"
        self.fe_dir = self.workspace_root / "fe"
        self.model_dir = self.workspace_root / "models" / "BERT"
        self.processes = []
        self.setup_complete = False
    
    def log(self, message: str, level: str = "INFO"):
        """Print colored log messages."""
        colors = {
            "INFO": "\033[94m",
            "SUCCESS": "\033[92m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "RESET": "\033[0m"
        }
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        print(f"{color}[{level}]{reset} {message}")
    
    def check_python_packages(self) -> bool:
        """Check if required Python packages are installed."""
        self.log("Checking Python packages...", "INFO")
        required_packages = [
            "fastapi",
            "uvicorn",
            "torch",
            "transformers",
            "openai",
            "numpy",
            "sklearn",
            "dotenv"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package if package != "sklearn" else "sklearn")
            except ImportError:
                missing.append(package)
        
        if missing:
            self.log(f"Missing packages: {', '.join(missing)}", "WARNING")
            self.log("Installing packages...", "INFO")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(self.api_dir / "requirements.txt")],
                    check=True
                )
                self.log("✓ Python packages installed", "SUCCESS")
                return True
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to install packages: {e}", "ERROR")
                return False
        
        self.log("✓ All Python packages available", "SUCCESS")
        return True
    
    def check_nodejs(self) -> bool:
        """Check if Node.js and npm are installed."""
        self.log("Checking Node.js...", "INFO")
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            self.log("✓ Node.js is available", "SUCCESS")
            return True
        except FileNotFoundError:
            self.log("Node.js or npm not found", "WARNING")
            self.log("Node.js is required for frontend. Download from: https://nodejs.org/", "INFO")
            self.log("For backend-only mode, use: python app.py --backend-only", "INFO")
            return False
    
    def check_model(self) -> bool:
        """Check if model exists."""
        self.log("Checking BERT model...", "INFO")
        if self.model_dir.exists():
            self.log(f"✓ Model found at {self.model_dir}", "SUCCESS")
            return True
        else:
            self.log(f"Warning: Model not found at {self.model_dir}", "WARNING")
            self.log("The API will fallback to downloading roberta-base from HuggingFace", "INFO")
            return False
    
    def setup_env_files(self) -> bool:
        """Create or update .env files."""
        self.log("Setting up environment files...", "INFO")
        
        # API .env
        api_env_path = self.api_dir / ".env"
        if not api_env_path.exists():
            self.log("Creating api/.env", "INFO")
            # File already created by create_file
        
        # Frontend .env.local
        fe_env_path = self.fe_dir / ".env.local"
        if not fe_env_path.exists():
            self.log("Creating fe/.env.local", "INFO")
            # File already created by create_file
        
        self.log("✓ Environment files ready", "SUCCESS")
        return True
    
    def install_frontend_deps(self) -> bool:
        """Install Node.js dependencies for frontend."""
        self.log("Installing frontend dependencies...", "INFO")
        try:
            subprocess.run(
                ["npm.cmd", "install"],
                cwd=self.fe_dir,
                check=True
            )
            self.log("✓ Frontend dependencies installed", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install frontend deps: {e}", "ERROR")
            return False
    
    def run_backend(self) -> bool:
        """Start FastAPI backend."""
        self.log("Starting FastAPI backend...", "INFO")
        try:
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.api_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(("Backend", process))
            self.log("✓ Backend starting on http://0.0.0.0:8000", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to start backend: {e}", "ERROR")
            return False
    
    def run_frontend(self) -> bool:
        """Start Next.js frontend."""
        self.log("Starting Next.js frontend...", "INFO")
        try:
            cmd = ["npm.cmd", "run", "dev"]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.fe_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(("Frontend", process))
            self.log("✓ Frontend starting on http://localhost:3000", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to start frontend: {e}", "ERROR")
            return False
    
    def setup(self) -> bool:
        """Run initial setup."""
        self.log("=" * 60, "INFO")
        self.log("Fake News Detection System - Setup", "INFO")
        self.log("=" * 60, "INFO")
        
        checks = [
            ("Python packages", self.check_python_packages),
            ("Node.js", self.check_nodejs),
            ("BERT model", self.check_model),
            ("Environment files", self.setup_env_files),
            ("Frontend dependencies", self.install_frontend_deps),
        ]
        
        for check_name, check_func in checks:
            if not check_func():
                # Skip frontend checks if Node.js not available
                if check_name == "Node.js":
                    self.log("⚠ Node.js not found - frontend will not be available", "WARNING")
                    self.log("Use --backend-only to run only API server", "INFO")
                    continue
                elif check_name in ["Frontend dependencies"]:
                    self.log("⚠ Skipping frontend dependency check", "WARNING")
                    continue
                elif check_name not in ["BERT model"]:  # Model is optional
                    self.log(f"Setup incomplete: {check_name} check failed", "ERROR")
                    return False
        
        self.setup_complete = True
        self.log("=" * 60, "SUCCESS")
        self.log("✓ Setup complete!", "SUCCESS")
        self.log("=" * 60, "SUCCESS")
        return True
    
    def start_services(self, backend_only: bool = False, frontend_only: bool = False) -> bool:
        """Start backend and/or frontend services."""
        if not backend_only and not frontend_only:
            backend_only = False
            frontend_only = False
        
        self.log("=" * 60, "INFO")
        self.log("Starting Services", "INFO")
        self.log("=" * 60, "INFO")
        
        if not backend_only:
            if not self.run_backend():
                return False
            time.sleep(2)  # Wait for backend to start
        
        if not frontend_only:
            # Check if frontend is available
            npm_available = True
            try:
                subprocess.run(["npm.cmd", "--version"], capture_output=True, check=True)
            except FileNotFoundError:
                npm_available = False
                self.log("⚠ Frontend not available (Node.js not installed)", "WARNING")
            
            if npm_available:
                if not self.run_frontend():
                    return False
        
        self.log("=" * 60, "SUCCESS")
        self.log("✓ Services started!", "SUCCESS")
        self.log("=" * 60, "SUCCESS")
        self.log("\nAPI Documentation: http://localhost:8000/docs", "INFO")
        if npm_available:
            self.log("Web Interface: http://localhost:3000", "INFO")
        self.log("\nPress Ctrl+C to stop services", "INFO")
        
        return True
    
    def cleanup(self):
        """Terminate all child processes."""
        self.log("\nShutting down services...", "INFO")
        for name, process in self.processes:
            if process.poll() is None:
                self.log(f"Stopping {name}...", "INFO")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.log("✓ All services stopped", "SUCCESS")
    
    def run(self, backend_only: bool = False, frontend_only: bool = False, 
            setup_only: bool = False, skip_setup: bool = False):
        """Main orchestration logic."""
        try:
            # Setup
            if not skip_setup:
                if not self.setup():
                    sys.exit(1)
            
            if setup_only:
                self.log("Setup complete. Exiting.", "INFO")
                return
            
            # Start services
            if not self.start_services(backend_only=backend_only, frontend_only=frontend_only):
                sys.exit(1)
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.log("", "INFO")
            self.cleanup()
            self.log("Bye!", "SUCCESS")
            sys.exit(0)
        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            self.cleanup()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detection System Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Run full setup and start both services
  python app.py --backend-only     # Start only FastAPI backend
  python app.py --frontend-only    # Start only Next.js frontend
  python app.py --setup-only       # Run setup only, don't start services
  python app.py --skip-setup       # Skip setup and start services
        """
    )
    
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only FastAPI backend"
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Start only Next.js frontend"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run setup only, don't start services"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setup and directly start services"
    )
    
    args = parser.parse_args()
    
    orchestrator = SystemOrchestrator()
    orchestrator.run(
        backend_only=args.backend_only,
        frontend_only=args.frontend_only,
        setup_only=args.setup_only,
        skip_setup=args.skip_setup
    )

if __name__ == "__main__":
    main()
