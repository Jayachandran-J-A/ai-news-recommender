import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommend import recommend_for_user
import json

print("Testing recommend_for_user function directly...")
try:
    results = recommend_for_user(query="news", k=5)
    print(f"\nReturned {len(results)} results")
    print(f"\nType: {type(results)}")
    print(f"\nFirst result structure:")
    if results:
        print(json.dumps(results[0], indent=2, default=str))
    else:
        print("Empty results!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
