#!/usr/bin/env python3
"""Complete system test - Backend + Model"""

import sys
import os
sys.path.insert(0, 'api')

def test_imports():
    """Test all required imports"""
    print("\n[1] Testing imports...")
    try:
        import fastapi
        import uvicorn
        import torch
        import transformers
        from model_loader import ModelLoader
        from openai_verifier import OpenAIVerifier
        import config
        print(" All imports successful")
        return True
    except ImportError as e:
        print(f" Import failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n[2] Testing configuration...")
    import config
    print(f"  - Model path: {config.MODEL_PATH}")
    print(f"  - Model exists: {os.path.exists(config.MODEL_PATH)}")
    print(f"  - API Host: {config.HOST}:{config.PORT}")
    print(f"  - CORS Origins: {config.CORS_ORIGINS}")
    print(f"  - Max length: {config.MAX_LENGTH}")
    print(" Configuration loaded")
    return True

def test_model_files():
    """Test model files exist"""
    print("\n[3] Testing model files...")
    import config
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt",
        "tokenizer.json"
    ]
    
    all_exist = True
    for file in required_files:
        path = os.path.join(config.MODEL_PATH, file)
        exists = os.path.exists(path)
        status = "" if exists else ""
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print(" All model files present")
    return all_exist

def test_model_loading():
    """Test model loading"""
    print("\n[4] Testing model loading...")
    from model_loader import ModelLoader
    
    ml = ModelLoader()
    success = ml.load_model()
    
    if success:
        print(f" Model loaded on {ml.device}")
        print(f"  - Model type: {type(ml.model).__name__}")
        print(f"  - Num labels: {ml.model.config.num_labels}")
        return True
    else:
        print(" Model loading failed")
        return False

def test_prediction():
    """Test prediction"""
    print("\n[5] Testing prediction...")
    from model_loader import ModelLoader
    
    ml = ModelLoader()
    ml.load_model()
    
    test_cases = [
        "Scientists announce breakthrough in renewable energy technology.",
        "Aliens landed in New York City yesterday, government confirms.",
        "Stock market reaches new high amid economic recovery."
    ]
    
    for i, text in enumerate(test_cases, 1):
        try:
            result = ml.predict(text)
            print(f"  Test {i}: {result['label']} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"   Test {i} failed: {e}")
            return False
    
    print(" All predictions successful")
    return True

def test_api_structure():
    """Test API structure"""
    print("\n[6] Testing API structure...")
    from main import app
    
    routes = [route.path for route in app.routes]
    expected_routes = ["/health", "/predict", "/predict-batch", "/models/info"]
    
    for route in expected_routes:
        if route in routes:
            print(f"   {route}")
        else:
            print(f"   {route} missing")
            return False
    
    print(" All API routes present")
    return True

def main():
    print("=" * 70)
    print("FAKE NEWS DETECTION SYSTEM - COMPLETE TEST")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Files", test_model_files),
        ("Model Loading", test_model_loading),
        ("Predictions", test_prediction),
        ("API Structure", test_api_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = " PASS" if success else " FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n ALL TESTS PASSED! System is ready to run.")
        print("\nTo start the system:")
        print("  Backend only:  python app.py --backend-only")
        print("  Full system:   python app.py")
        print("\nOr manually:")
        print("  Backend:  cd api && uvicorn main:app --reload")
        print("  Frontend: cd fe && npm run dev")
        return True
    else:
        print("\n  Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
