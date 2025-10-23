import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import faiss
from src.recommend import META_CSV, INDEX_FP

print("Checking data sync...")
meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")

index = faiss.read_index(INDEX_FP)
print(f"FAISS index vectors: {index.ntotal}")

if len(meta) != index.ntotal:
    print(f"\n❌ MISMATCH! Meta has {len(meta)} rows but index has {index.ntotal} vectors")
else:
    print(f"\n✅ Sizes match!")
