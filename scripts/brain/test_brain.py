import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_sales_prediction():
    print("\n--- Testing Sales Prediction ---")
    url = f"{BASE_URL}/predict_sales"
    payload = {
        "day_of_week": 5, "is_weekend": 1, "is_holiday": 1, 
        "promo": 1, "rainfall": 0.0, "footfall": 800, "inventory": 500
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Failed: {e}")

def test_user_profile():
    print("\n--- Testing User Profile ---")
    user_id = "test_user_001"
    
    # 1. Update/Create
    url = f"{BASE_URL}/user/profile"
    payload = {
        "user_id": user_id,
        "name": "Alice Retail",
        "email": "alice@example.com",
        "preferences": {"category": "shoes", "budget": "high"}
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Update Status: {resp.status_code}")
        print(f"Update Response: {resp.json()}")
    except Exception as e:
        print(f"Update Failed: {e}")
        return

    # 2. Get
    url = f"{BASE_URL}/user/profile/{user_id}"
    try:
        resp = requests.get(url)
        print(f"Get Status: {resp.status_code}")
        print(f"Get Response: {resp.json()}")
    except Exception as e:
        print(f"Get Failed: {e}")

def test_rag_memory():
    print("\n--- Testing RAG Memory ---")
    user_id = "test_user_001"
    
    # 1. Add Memory
    url = f"{BASE_URL}/memory/add"
    memories = [
        "Customer prefers red sneakers.",
        "Customer asked about return policy for sale items.",
        "Customer has a budget of around $200."
    ]
    
    for mem in memories:
        try:
            resp = requests.post(url, json={"user_id": user_id, "text": mem})
            print(f"Added Memory: '{mem}' -> {resp.json()}")
        except Exception as e:
            print(f"Add Failed: {e}")
            
    # 2. Search
    print("\n... Searching Memory ...")
    url = f"{BASE_URL}/memory/search"
    query = "What kind of shoes does the customer like?"
    try:
        resp = requests.post(url, json={"user_id": user_id, "query": query, "k": 2})
        print(f"Query: '{query}'")
        results = resp.json().get("results", [])
        for r in results:
            print(f" - Found: {r['text']} (Score: {r['score']:.4f})")
    except Exception as e:
        print(f"Search Failed: {e}")

if __name__ == "__main__":
    # Wait for server to potentially start if run immediately
    time.sleep(2) 
    
    test_sales_prediction()
    test_user_profile()
    test_rag_memory()
