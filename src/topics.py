TOPIC_SEEDS = {
    "politics": ["politics", "government policy", "elections", "parliament"],
    "business": ["business", "economy", "markets", "finance"],
    "technology": ["technology", "software", "gadgets", "startups", "cloud"],
    "science": ["science", "research", "space", "physics", "biology"],
    "health": ["health", "medicine", "public health", "wellness"],
    "sports": ["sports", "football", "cricket", "tennis", "olympics"],
    "entertainment": ["entertainment", "movies", "music", "celebrity"],
    "world": ["world news", "international", "global affairs"],
    "india": ["India", "Indian news", "Indian politics", "Indian economy"],
    "ai": ["artificial intelligence", "machine learning", "AI research"],
    "climate": ["climate change", "environment", "sustainability"],
    "education": ["education", "university", "learning", "curriculum"],
}


def normalize_topic_name(name: str) -> str:
    return name.strip().lower()


def build_topic_vectors(embedder, topics=None):
    import numpy as np
    topics = topics or list(TOPIC_SEEDS.keys())
    topic_vecs = {}
    for t in topics:
        seeds = TOPIC_SEEDS.get(t, [t])
        vecs = list(embedder.embed(seeds, batch_size=32, parallel=None))
        arr = np.vstack(vecs).astype("float32")
        # average and L2 normalize
        arr = arr.mean(axis=0, keepdims=True)
        from faiss import normalize_L2
        normalize_L2(arr)
        topic_vecs[normalize_topic_name(t)] = arr[0]
    return topic_vecs


def assign_categories_to_embeddings(embeddings, topic_vecs, top_k=2, min_sim=0.25):
    """
    embeddings: np.ndarray [N, D], L2-normalized
    topic_vecs: dict[str, np.ndarray[D]] L2-normalized
    returns: list[list[str]] categories per row
    """
    import numpy as np
    topics = list(topic_vecs.keys())
    T = np.stack([topic_vecs[t] for t in topics], axis=0).astype("float32")
    # cosine via dot product
    sims = embeddings @ T.T  # [N, T]
    idx = np.argsort(-sims, axis=1)[:, :top_k]
    out = []
    for i in range(embeddings.shape[0]):
        cats = []
        for j in idx[i]:
            if sims[i, j] >= min_sim:
                cats.append(topics[j])
        out.append(cats)
    return out
