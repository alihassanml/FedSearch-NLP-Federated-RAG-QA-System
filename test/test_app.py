import requests
import json
import time
from typing import Dict

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_response(response: requests.Response):
    """Pretty print API response"""
    try:
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")

def test_root():
    """Test root endpoint"""
    print_section("1. Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print_response(response)
    return response.status_code == 200

def test_health():
    """Test health check"""
    print_section("2. Testing Health Check")
    response = requests.get(f"{BASE_URL}/api/health")
    print_response(response)
    return response.status_code == 200

def test_index_documents():
    """Test document indexing"""
    print_section("3. Indexing Documents")
    print("⏳ This may take 2-3 minutes (downloading models)...")
    
    response = requests.post(
        f"{BASE_URL}/api/index",
        json={"reindex": False}
    )
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Indexed {data['documents_indexed']} documents")
        return True
    return False

def test_query(question: str, expected_keywords: list = None):
    """Test a query"""
    print(f"\n📝 Question: {question}")
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "question": question,
            "top_k": 3
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Answer: {data['answer']}")
        print(f"📊 Confidence: {data['confidence']}")
        print(f"📚 Retrieved {len(data['retrieved_documents'])} documents")
        
        # Show sources
        for i, doc in enumerate(data['retrieved_documents'], 1):
            print(f"   Source {i}: {doc['source']} (score: {doc['score']:.3f})")
        
        # Check if answer contains expected keywords
        if expected_keywords:
            answer_lower = data['answer'].lower()
            found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            if found:
                print(f"✓ Found keywords: {found}")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        print_response(response)
        return False

def test_document_stats():
    """Test document statistics"""
    print_section("Document Statistics")
    response = requests.get(f"{BASE_URL}/api/documents/stats")
    print_response(response)
    return response.status_code == 200

def run_all_tests():
    """Run complete test suite"""
    print("🚀 FedSearch-NLP API Test Suite")
    print(f"Testing server at: {BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Server is not running!")
        print("Please start the server first:")
        print("  python -m uvicorn app.main:app --reload")
        return
    
    # Test basic endpoints
    results = []
    results.append(("Root Endpoint", test_root()))
    results.append(("Health Check", test_health()))
    
    # Index documents
    results.append(("Document Indexing", test_index_documents()))
    
    # Wait a moment for indexing to complete
    time.sleep(2)
    
    # Test queries
    print_section("4. Testing Queries")
    
    queries = [
        ("How many days of annual leave do employees get?", ["20", "days", "annual"]),
        ("What is the password policy?", ["12", "characters", "password"]),
        ("What is the data retention policy?", ["7 years", "retention"]),
        ("What is the pricing for CloudSync Pro?", ["15", "month", "CloudSync"]),
        ("How often are backups performed?", ["daily", "backup"]),
        ("What compliance certifications does the company have?", ["ISO", "SOC"]),
    ]
    
    for question, keywords in queries:
        success = test_query(question, keywords)
        results.append((f"Query: {question[:40]}...", success))
        time.sleep(1)  # Rate limiting
    
    # Test stats
    results.append(("Document Stats", test_document_stats()))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed\n")
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n📚 Your API is ready to use!")
        print(f"   Docs: {BASE_URL}/docs")
        print(f"   ReDoc: {BASE_URL}/redoc")
    else:
        print(f"\n⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")