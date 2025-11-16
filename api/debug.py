#!/usr/bin/env python3
"""
Comprehensive Debug Tool for Fake News Detection System
Run this to diagnose ALL issues
"""

import sys
import os
import subprocess
import requests
import time
from pathlib import Path

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

# ============================================================================
# CHECK 1: Python Environment
# ============================================================================

def check_python_version():
    """Check Python version"""
    print_section("CHECK 1: Python Environment")
    
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible (>= 3.8)")
        return True
    else:
        print_error("Python version too old! Need >= 3.8")
        return False

# ============================================================================
# CHECK 2: Required Packages
# ============================================================================

def check_packages():
    """Check if required packages are installed"""
    print_section("CHECK 2: Required Packages")
    
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'torch': 'torch',
        'transformers': 'transformers',
        'openai': 'openai',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
    }
    
    all_installed = True
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            print_info(f"   Install: pip install {package_name}")
            all_installed = False
    
    return all_installed

# ============================================================================
# CHECK 3: File Structure
# ============================================================================

def check_file_structure():
    """Check if all required files exist"""
    print_section("CHECK 3: File Structure")
    
    root = Path.cwd()
    
    required_files = [
        'api/main.py',
        'api/config.py',
        'api/model_loader.py',
        'api/openai_verifier.py',
        'api/utils.py',
        'api/requirements.txt',
        'fe/package.json',
        'fe/src/pages/index.js',
        'fe/src/pages/detector.js',
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

# ============================================================================
# CHECK 4: Configuration Files
# ============================================================================

def check_config_files():
    """Check configuration files"""
    print_section("CHECK 4: Configuration Files")
    
    # Check api/config.py
    try:
        sys.path.insert(0, 'api')
        import config
        
        print_success("api/config.py loaded successfully")
        
        # Check important settings
        print_info(f"   MODEL_NAME: {config.MODEL_NAME}")
        print_info(f"   MODEL_PATH: {config.MODEL_PATH}")
        print_info(f"   API_HOST: {config.HOST}")
        print_info(f"   API_PORT: {config.PORT}")
        print_info(f"   CORS_ORIGINS: {config.CORS_ORIGINS}")
        
        # Check OpenAI
        if config.OPENAI_API_KEY:
            masked_key = config.OPENAI_API_KEY[:8] + "..." + config.OPENAI_API_KEY[-4:]
            print_success(f"   OPENAI_API_KEY: {masked_key}")
        else:
            print_warning("   OPENAI_API_KEY: Not set (OpenAI features disabled)")
        
        print_info(f"   ENABLE_OPENAI: {config.ENABLE_OPENAI}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return False

# ============================================================================
# CHECK 5: Model Files
# ============================================================================

def check_model_files():
    """Check if BERT model exists"""
    print_section("CHECK 5: BERT Model Files")
    
    try:
        import config
        model_path = Path(config.MODEL_PATH)
        
        if model_path.exists():
            print_success(f"Model directory exists: {model_path}")
            
            # Check for model files
            required_model_files = ['config.json', 'pytorch_model.bin']
            
            for file in required_model_files:
                file_path = model_path / file
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print_success(f"   {file} ({size_mb:.1f} MB)")
                else:
                    print_warning(f"   {file} not found (will download from HuggingFace)")
            
            return True
        else:
            print_warning(f"Model directory not found: {model_path}")
            print_info("   Model will be downloaded from HuggingFace on first run")
            return True
            
    except Exception as e:
        print_error(f"Error checking model: {e}")
        return False

# ============================================================================
# CHECK 6: Backend API Status
# ============================================================================

def check_backend_status():
    """Check if backend API is running"""
    print_section("CHECK 6: Backend API Status")
    
    api_url = "http://localhost:8000"
    
    # Test /health endpoint
    print_info("Testing /health endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            print_success("Backend API is running!")
            
            data = response.json()
            print_info(f"   Status: {data.get('status')}")
            print_info(f"   Model loaded: {data.get('model_loaded')}")
            print_info(f"   OpenAI available: {data.get('openai_available')}")
            
            return True
        else:
            print_error(f"Backend returned status code: {response.status_code}")
            return False
            
    except requests.ConnectionError:
        print_error("Backend API is NOT running!")
        print_info("   Start with: python app.py --backend-only")
        print_info("   Or: cd api && uvicorn main:app --reload")
        return False
    except Exception as e:
        print_error(f"Error connecting to backend: {e}")
        return False

# ============================================================================
# CHECK 7: Test Prediction
# ============================================================================

def test_prediction():
    """Test actual prediction"""
    print_section("CHECK 7: Test Prediction")
    
    api_url = "http://localhost:8000/predict"
    
    test_text = "Scientists discover new species of dolphin in the Pacific Ocean. The marine mammals were found near remote islands."
    
    print_info("Sending test prediction request...")
    
    try:
        response = requests.post(
            api_url,
            json={
                "text": test_text,
                "verify_with_openai": False,
                "mc_dropout": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print_success("Prediction successful!")
            
            data = response.json()
            print_info(f"   Label: {data.get('label')}")
            print_info(f"   Confidence: {data.get('confidence', 0):.2%}")
            print_info(f"   Real: {data.get('probabilities', {}).get('real', 0):.2%}")
            print_info(f"   Fake: {data.get('probabilities', {}).get('fake', 0):.2%}")
            
            return True
        else:
            print_error(f"Prediction failed with status: {response.status_code}")
            print_error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Prediction error: {e}")
        return False

# ============================================================================
# CHECK 8: Frontend Status
# ============================================================================

def check_frontend_status():
    """Check if frontend is running"""
    print_section("CHECK 8: Frontend Status")
    
    frontend_url = "http://localhost:3000"
    
    print_info("Testing frontend...")
    try:
        response = requests.get(frontend_url, timeout=5)
        
        if response.status_code == 200:
            print_success("Frontend is running!")
            return True
        else:
            print_error(f"Frontend returned status: {response.status_code}")
            return False
            
    except requests.ConnectionError:
        print_warning("Frontend is NOT running")
        print_info("   Start with: cd fe && npm run dev")
        return False
    except Exception as e:
        print_error(f"Error connecting to frontend: {e}")
        return False

# ============================================================================
# CHECK 9: Frontend Environment
# ============================================================================

def check_frontend_env():
    """Check frontend environment variables"""
    print_section("CHECK 9: Frontend Environment")
    
    env_file = Path("fe/.env.local")
    
    if env_file.exists():
        print_success(".env.local exists")
        
        with open(env_file, 'r') as f:
            content = f.read()
            print_info("   Contents:")
            for line in content.strip().split('\n'):
                print_info(f"      {line}")
        
        if "NEXT_PUBLIC_API_URL" in content:
            print_success("   NEXT_PUBLIC_API_URL is configured")
        else:
            print_warning("   NEXT_PUBLIC_API_URL not found")
        
        return True
    else:
        print_error(".env.local NOT FOUND")
        print_info("   Creating .env.local...")
        
        try:
            env_file.parent.mkdir(parents=True, exist_ok=True)
            with open(env_file, 'w') as f:
                f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
            print_success("   Created .env.local")
            return True
        except Exception as e:
            print_error(f"   Failed to create .env.local: {e}")
            return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all checks"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("█" * 80)
    print("FAKE NEWS DETECTION SYSTEM - COMPREHENSIVE DEBUG TOOL".center(80))
    print("█" * 80)
    print(f"{Colors.RESET}\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("File Structure", check_file_structure),
        ("Configuration", check_config_files),
        ("Model Files", check_model_files),
        ("Backend API", check_backend_status),
        ("Test Prediction", test_prediction),
        ("Frontend", check_frontend_status),
        ("Frontend Config", check_frontend_env),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Check '{name}' failed with exception: {e}")
            results[name] = False
    
    # Summary
    print_section("SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for name, result in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if result else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"{status}  {name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.RESET}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL CHECKS PASSED! System is healthy!{Colors.RESET}\n")
    elif failed <= 2:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some issues found, but system may still work{Colors.RESET}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Multiple issues found. System likely won't work.{Colors.RESET}\n")
        print(f"{Colors.YELLOW}Recommended actions:{Colors.RESET}")
        
        if not results.get("Required Packages"):
            print(f"   1. Install packages: pip install -r api/requirements.txt")
        
        if not results.get("Backend API"):
            print(f"   2. Start backend: python app.py --backend-only")
        
        if not results.get("Frontend"):
            print(f"   3. Start frontend: cd fe && npm install && npm run dev")
        
        print()

if __name__ == "__main__":
    main()