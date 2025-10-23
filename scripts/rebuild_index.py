"""Rebuild FAISS index from existing metadata"""
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import faiss
import numpy as np
from fastembed import TextEmbedding
from src.recommend import META_CSV, INDEX_FP, DATA_DIR, MODEL_NAME

print("Loading metadata...")
meta = pd.read_csv(META_CSV)
print(f"Found {len(meta)} articles in metadata")

print("\nLoading embedding model...")
cache_dir = os.path.join(DATA_DIR, "fastembed_cache")
model = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)

print("\nEmbedding all articles...")
texts = (meta["title"].fillna("") + " [SEP] " + meta["summary"].fillna("")).tolist()

# Embed in batches
all_vecs = []
batch_size = 64
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    vecs = list(model.embed(batch, batch_size=len(batch), parallel=None))
    all_vecs.extend(vecs)
    print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")

arr = np.vstack(all_vecs).astype("float32")
faiss.normalize_L2(arr)

print(f"\nCreating new FAISS index with {len(arr)} vectors...")
dim = arr.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(arr)

print(f"\nSaving index to {INDEX_FP}...")
faiss.write_index(index, INDEX_FP)

print(f"\n✅ Done! Index rebuilt with {index.ntotal} vectors matching {len(meta)} metadata rows")
