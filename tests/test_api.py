#!/usr/bin/env python3
"""Test API endpoints"""

import sys
import requests
import time

API_URL = "http://localhost:8000"

def wait_for_api(timeout=30):
    """Wait for API to be ready"""
    print("\n[0] Waiting for API to be ready...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print(" API is ready")
                return True
        except:
            time.sleep(1)
    
    print(" API not responding")
    return False

def test_health_endpoint():
    """Test health check endpoint"""
    print("\n[1] Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f" Health check passed")
            print(f"  - Status: {data.get('status')}")
            print(f"  - Models loaded: {data.get('models_loaded')}")
            print(f"  - Database connected: {data.get('database_connected')}")
            return True
        else:
            print(f" Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Health check error: {e}")
        return False

def test_register_endpoint():
    """Test user registration"""
    print("\n[2] Testing /auth/register endpoint...")
    try:
        # Generate unique username
        import random
        username = f"test_user_{random.randint(1000, 9999)}"
        
        response = requests.post(
            f"{API_URL}/auth/register",
            json={
                "username": username,
                "email": f"{username}@test.com",
                "password": "test123456"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" Registration successful")
            print(f"  - Username: {data['user']['username']}")
            print(f"  - Token received: {data['access_token'][:20]}...")
            return True, data['access_token'], username
        else:
            print(f" Registration failed: {response.status_code}")
            print(f"  - Error: {response.text}")
            return False, None, None
            
    except Exception as e:
        print(f" Registration error: {e}")
        return False, None, None

def test_login_endpoint(username):
    """Test user login"""
    print("\n[3] Testing /auth/login endpoint...")
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            json={
                "username": username,
                "password": "test123456"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" Login successful")
            print(f"  - Token received: {data['access_token'][:20]}...")
            return True, data['access_token']
        else:
            print(f" Login failed: {response.status_code}")
            print(f"  - Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f" Login error: {e}")
        return False, None

def test_predict_endpoint(token):
    """Test prediction endpoint"""
    print("\n[4] Testing /predict endpoint...")
    try:
        # Test English
        response = requests.post(
            f"{API_URL}/predict",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "text": "Scientists announce breakthrough in renewable energy technology that could change the world.",
                "language": "en",
                "verify_with_ai": False,
                "mc_dropout": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" English prediction successful")
            print(f"  - Label: {data['label']}")
            print(f"  - Confidence: {data['confidence']:.1%}")
            print(f"  - Language: {data['language']}")
        else:
            print(f" English prediction failed: {response.status_code}")
            return False
        
        # Test Vietnamese
        response = requests.post(
            f"{API_URL}/predict",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "text": "Các nhà khoa học công bố đột phá trong công nghệ năng lượng tái tạo có thể thay đổi thế giới.",
                "language": "vi",
                "verify_with_ai": False,
                "mc_dropout": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" Vietnamese prediction successful")
            print(f"  - Label: {data['label']}")
            print(f"  - Confidence: {data['confidence']:.1%}")
            print(f"  - Language: {data['language']}")
            return True
        else:
            print(f" Vietnamese prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Prediction error: {e}")
        return False

def test_history_endpoint(token):
    """Test history endpoint"""
    print("\n[5] Testing /history endpoint...")
    try:
        response = requests.get(
            f"{API_URL}/history",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" History retrieved successfully")
            print(f"  - Total queries: {data['total']}")
            print(f"  - Queries in response: {len(data['queries'])}")
            return True
        else:
            print(f" History retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" History error: {e}")
        return False

def test_stats_endpoint(token):
    """Test stats endpoint"""
    print("\n[6] Testing /history/stats endpoint...")
    try:
        response = requests.get(
            f"{API_URL}/history/stats",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f" Stats retrieved successfully")
            print(f"  - Total queries: {data['total_queries']}")
            print(f"  - By language: {data['by_language']}")
            print(f"  - By prediction: {data['by_prediction']}")
            return True
        else:
            print(f" Stats retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Stats error: {e}")
        return False

def main():
    print("=" * 70)
    print("API ENDPOINTS TEST")
    print("=" * 70)
    print("\nMake sure the API is running:")
    print("  cd api && python main.py")
    print("  or: uvicorn api.main:app --reload")
    
    # Wait for API
    if not wait_for_api():
        print("\n API is not running. Please start it first.")
        return False
    
    results = []
    token = None
    username = None
    
    # Test health
    success = test_health_endpoint()
    results.append(("Health Check", success))
    
    if not success:
        print("\n  API health check failed. Stopping tests.")
        return False
    
    # Test register
    success, token, username = test_register_endpoint()
    results.append(("Register", success))
    
    if not success or not token:
        print("\n  Registration failed. Stopping tests.")
        return False
    
    # Test login
    success, login_token = test_login_endpoint(username)
    results.append(("Login", success))
    
    # Test predict
    success = test_predict_endpoint(token)
    results.append(("Predict", success))
    
    # Test history
    success = test_history_endpoint(token)
    results.append(("History", success))
    
    # Test stats
    success = test_stats_endpoint(token)
    results.append(("Stats", success))
    
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
        print("\n ALL TESTS PASSED! API is working correctly.")
        return True
    else:
        print("\n  Some tests failed.")
        print("\nTroubleshooting:")
        print("1. Make sure API is running: cd api && python main.py")
        print("2. Check MongoDB is running: sc query MongoDB")
        print("3. Check API logs for errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
