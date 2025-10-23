import os
import re
import feedparser
import pandas as pd
import numpy as np
from tqdm import tqdm
from fastembed import TextEmbedding
import faiss
import json
from .topics import build_topic_vectors, assign_categories_to_embeddings

from .config_feeds import FEEDS
from .utils import sha16, parse_pub_date

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
META_CSV = os.path.join(DATA_DIR, "meta.csv")
INDEX_FP = os.path.join(DATA_DIR, "index.faiss")
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # via fastembed


def clean_html(text):
    """Remove HTML tags and decode entities from RSS feed content"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = text.replace('&mdash;', '—').replace('&ndash;', '–')
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_meta():
    if os.path.exists(META_CSV):
        return pd.read_csv(META_CSV)
    return pd.DataFrame(columns=["doc_id", "url", "title", "summary", "published", "source", "categories"]) 


def save_meta(df: pd.DataFrame):
    df.to_csv(META_CSV, index=False)


def load_index(dim: int):
    if os.path.exists(INDEX_FP):
        return faiss.read_index(INDEX_FP)
    return faiss.IndexFlatIP(dim)


def save_index(index):
    faiss.write_index(index, INDEX_FP)


def fetch_feeds():
    rows = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            link = getattr(e, "link", None)
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            if not link or not title:
                continue
            # Clean HTML from title and summary
            title = clean_html(title)
            summary = clean_html(summary)
            if not title:  # Skip if title is empty after cleaning
                continue
            doc_id = sha16(link)
            published = parse_pub_date(e).isoformat()
            source = feed.feed.get("title", url) if getattr(feed, 'feed', None) else url
            rows.append(dict(doc_id=doc_id, url=link, title=title, summary=summary, published=published, source=source))
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    return df


def main():
    ensure_dirs()
    print("Loading embedding model...")
    cache_dir = os.path.join(DATA_DIR, "fastembed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)
    # Determine embedding dimension by a small probe
    try:
        import numpy as np
        probe_vec = list(embedder.embed(["dimension-probe"], batch_size=1, parallel=None))[0]
        dim = int(np.array(probe_vec).shape[-1])
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding model: {e}")

    print("Loading existing metadata and index...")
    meta = load_meta()
    index = load_index(dim)

    existing_ids = set(meta["doc_id"]) if not meta.empty else set()

    print("Fetching feeds...")
    fresh = fetch_feeds()
    new = fresh[~fresh["doc_id"].isin(existing_ids)].reset_index(drop=True)

    if new.empty:
        print("No new articles.")
        return

    texts = (new["title"].fillna("") + " [SEP] " + new["summary"].fillna("")).tolist()
    print(f"Embedding {len(texts)} new articles...")
    # fastembed returns generator of np arrays
    vecs = list(embedder.embed(texts, batch_size=64, parallel=None))
    emb = np.vstack(vecs).astype("float32")
    # Normalize to use cosine via inner product
    faiss.normalize_L2(emb)

    # Build topic vectors (once per run) and assign categories to new articles
    print("Assigning categories...")
    topic_vecs = build_topic_vectors(embedder)
    cats = assign_categories_to_embeddings(emb, topic_vecs, top_k=2, min_sim=0.25)
    new["categories"] = [json.dumps(c) for c in cats]

    index.add(emb)

    save_index(index)
    # Ensure meta has categories column
    if "categories" not in meta.columns:
        meta["categories"] = ["[]"] * len(meta)
    updated = pd.concat([meta, new], ignore_index=True)
    save_meta(updated)
    print(f"Added {len(new)} new articles. Total: {len(updated)}")


if __name__ == "__main__":
    main()
