#!/usr/bin/env python3
"""Test MongoDB connection and operations"""

import sys
import os
sys.path.insert(0, 'api')

def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\n[1] Testing MongoDB connection...")
    try:
        from database import db
        
        # Try to connect
        db.connect()
        
        if db.connected:
            print(" MongoDB connected successfully")
            print(f"  - Database: {db.db.name}")
            print(f"  - Collections: {db.db.list_collection_names()}")
            return True
        else:
            print(" MongoDB connection failed")
            return False
            
    except Exception as e:
        print(f" MongoDB connection error: {e}")
        return False

def test_user_operations():
    """Test user CRUD operations"""
    print("\n[2] Testing user operations...")
    try:
        from database import db
        import auth
        
        if not db.connected:
            db.connect()
        
        # Test create user
        test_username = "test_user_temp"
        test_email = "test@temp.com"
        test_password = "test123"
        
        # Delete if exists
        existing = db.get_user_by_username(test_username)
        if existing:
            db.users.delete_one({"username": test_username})
            print("  - Cleaned up existing test user")
        
        # Create user
        password_hash = auth.hash_password(test_password)
        user_id = db.create_user(test_username, test_email, password_hash)
        
        if user_id:
            print(f" User created: {user_id}")
        else:
            print(" Failed to create user")
            return False
        
        # Get user
        user = db.get_user_by_username(test_username)
        if user:
            print(f" User retrieved: {user['username']}")
        else:
            print(" Failed to retrieve user")
            return False
        
        # Verify password
        if auth.verify_password(test_password, user["password_hash"]):
            print(" Password verification works")
        else:
            print(" Password verification failed")
            return False
        
        # Cleanup
        db.users.delete_one({"username": test_username})
        print(" Test user cleaned up")
        
        return True
        
    except Exception as e:
        print(f" User operations error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_operations():
    """Test query CRUD operations"""
    print("\n[3] Testing query operations...")
    try:
        from database import db
        
        if not db.connected:
            db.connect()
        
        # Create test user first
        test_username = "test_query_user"
        test_email = "query@test.com"
        
        # Cleanup
        db.users.delete_one({"username": test_username})
        
        # Create user
        import auth
        password_hash = auth.hash_password("test123")
        user_id = db.create_user(test_username, test_email, password_hash)
        
        # Save query
        query_id = db.save_query(
            user_id=user_id,
            text="Test news article",
            language="en",
            prediction={
                "label": "fake",
                "confidence": 0.85,
                "probabilities": {"real": 0.15, "fake": 0.85}
            }
        )
        
        if query_id:
            print(f" Query saved: {query_id}")
        else:
            print(" Failed to save query")
            return False
        
        # Get queries
        queries = db.get_user_queries(user_id, limit=10)
        if queries and len(queries) > 0:
            print(f" Retrieved {len(queries)} queries")
        else:
            print(" Failed to retrieve queries")
            return False
        
        # Get stats
        stats = db.get_query_stats(user_id)
        if stats and stats.get("total_queries", 0) > 0:
            print(f" Stats retrieved: {stats['total_queries']} total queries")
        else:
            print(" Failed to retrieve stats")
            return False
        
        # Cleanup
        db.queries.delete_many({"user_id": user_id})
        db.users.delete_one({"username": test_username})
        print(" Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f" Query operations error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("MONGODB CONNECTION & OPERATIONS TEST")
    print("=" * 70)
    
    tests = [
        ("MongoDB Connection", test_mongodb_connection),
        ("User Operations", test_user_operations),
        ("Query Operations", test_query_operations),
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
        print("\n ALL TESTS PASSED! MongoDB is working correctly.")
        return True
    else:
        print("\n  Some tests failed.")
        print("\nTroubleshooting:")
        print("1. Check if MongoDB is running:")
        print("   Windows: sc query MongoDB")
        print("   macOS/Linux: sudo systemctl status mongod")
        print("\n2. Start MongoDB:")
        print("   Windows: net start MongoDB (as Administrator)")
        print("   macOS: brew services start mongodb-community")
        print("   Linux: sudo systemctl start mongod")
        print("\n3. Check connection string in api/.env:")
        print("   MONGODB_URL=mongodb://localhost:27017/")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
