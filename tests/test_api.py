import requests
import json

# Test the recommend endpoint
print("Testing /recommend endpoint...")
response = requests.get("http://localhost:8003/recommend?query=news&k=5")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*50 + "\n")

# Test the debug endpoint
print("Testing /debug/info endpoint...")
response = requests.get("http://localhost:8003/debug/info")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
