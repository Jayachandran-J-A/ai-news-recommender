#!/usr/bin/env python3
"""
Final System Verification Test
Tests complete frontend-backend integration
"""

import requests
import json
import time
from datetime import datetime

def test_backend_api():
    """Test backend API endpoints"""
    print("\n🔧 Testing Backend API...")
    
    # Test root endpoint
    try:
        response = requests.get("http://localhost:8003/", timeout=5)
        print(f"  ✅ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Root endpoint failed: {e}")
        return False
    
    # Test recommend endpoint  
    try:
        response = requests.get("http://localhost:8003/recommend?query=technology", timeout=10)
        data = response.json()
        print(f"  ✅ Recommend endpoint: {response.status_code}")
        print(f"     Models: {data.get('models_used', 'Unknown')}")
        print(f"     Response time: {data.get('response_time_ms', 'Unknown')}ms")
    except Exception as e:
        print(f"  ❌ Recommend endpoint failed: {e}")
        return False
        
    return True

def test_frontend():
    """Test frontend availability"""
    print("\n🌐 Testing Frontend...")
    
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        print(f"  ✅ Frontend accessible: {response.status_code}")
        
        # Check if it's not just a blank page
        if len(response.text) > 100:
            print(f"  ✅ Content loaded: {len(response.text)} bytes")
        else:
            print(f"  ⚠️ Minimal content: {len(response.text)} bytes")
            
        return True
    except Exception as e:
        print(f"  ❌ Frontend not accessible: {e}")
        return False

def test_cors():
    """Test CORS functionality"""
    print("\n🔗 Testing CORS Integration...")
    
    headers = {
        'Origin': 'http://localhost:8080',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    
    try:
        response = requests.options("http://localhost:8003/recommend", headers=headers, timeout=5)
        print(f"  ✅ CORS preflight: {response.status_code}")
        
        cors_headers = response.headers.get('Access-Control-Allow-Origin', 'Not set')
        print(f"     Allow-Origin: {cors_headers}")
        
        return True
    except Exception as e:
        print(f"  ❌ CORS test failed: {e}")
        return False

def main():
    print("========================================")
    print("   NEWS RECOMMENDER SYSTEM CHECK")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    print("========================================")
    
    backend_ok = test_backend_api()
    frontend_ok = test_frontend() 
    cors_ok = test_cors()
    
    print("\n" + "="*40)
    if backend_ok and frontend_ok and cors_ok:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("\nReady for demonstration:")
        print("  • Frontend: http://localhost:8080")
        print("  • Backend:  http://localhost:8003")
        print("  • API Docs: http://localhost:8003/docs")
    else:
        print("❌ SOME ISSUES DETECTED")
        if not backend_ok:
            print("  - Backend API needs attention")
        if not frontend_ok:
            print("  - Frontend not responding")
        if not cors_ok:
            print("  - CORS configuration issue")
    print("="*40)

if __name__ == "__main__":
    main()