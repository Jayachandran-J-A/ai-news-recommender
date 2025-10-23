import os
import pandas as pd
import numpy as np
import faiss
from fastembed import TextEmbedding
import json
from datetime import datetime
from dateutil import parser as dtparser
from .topics import build_topic_vectors, normalize_topic_name

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
META_CSV = os.path.join(DATA_DIR, "meta.csv")
INDEX_FP = os.path.join(DATA_DIR, "index.faiss")
MODEL_FP = os.path.join(DATA_DIR, "models", "xgb_mind.json")
NRMS_MODEL_FP = os.path.join(DATA_DIR, "..", "models", "nrms_model.pt")
PROFILES_FP = os.path.join(DATA_DIR, "user_profiles.json")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global cache for models
_XGB_MODEL = None
_ENSEMBLE_MODEL = None


def load_resources():
    if not os.path.exists(META_CSV):
        raise RuntimeError("Metadata not found. Please run: python -m src.ingest_rss")
    if not os.path.exists(INDEX_FP):
        raise RuntimeError("Vector index not found. Please run: python -m src.ingest_rss")
    meta = pd.read_csv(META_CSV)
    index = faiss.read_index(INDEX_FP)
    cache_dir = os.path.join(DATA_DIR, "fastembed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)
    return meta, index, model


def _load_xgb_model():
    global _XGB_MODEL
    if _XGB_MODEL is not None:
        return _XGB_MODEL
    if not os.path.exists(MODEL_FP):
        return None
    try:
        import xgboost as xgb
        _XGB_MODEL = xgb.Booster()
        _XGB_MODEL.load_model(MODEL_FP)
        return _XGB_MODEL
    except Exception:
        return None


def _load_ensemble_model():
    """Load ensemble model combining NRMS + XGBoost"""
    global _ENSEMBLE_MODEL
    if _ENSEMBLE_MODEL is not None:
        return _ENSEMBLE_MODEL
    
    # Check if NRMS model exists
    if not os.path.exists(NRMS_MODEL_FP):
        print("⚠️ NRMS model not found, falling back to XGBoost only")
        return None
    
    # Check if XGBoost model exists
    if not os.path.exists(MODEL_FP):
        print("⚠️ XGBoost model not found, ensemble requires both models")
        return None
    
    try:
        from src.ensemble import EnsembleRecommender
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _ENSEMBLE_MODEL = EnsembleRecommender(
            nrms_model_path=NRMS_MODEL_FP,
            xgb_model_path=MODEL_FP,
            device=device,
            nrms_weight=0.6  # 60% NRMS, 40% XGBoost
        )
        print(f"✅ Ensemble model loaded on {device}")
        return _ENSEMBLE_MODEL
    except Exception as e:
        print(f"⚠️ Failed to load ensemble model: {e}")
        # Try XGBoost only as fallback
        return None


def _encode_texts(texts, model):
    import numpy as np
    vecs = list(model.embed(texts, batch_size=64, parallel=None))
    arr = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(arr)
    return arr


def _user_vector_from_urls(urls, meta, model):
    df = meta[meta["url"].isin(urls)]
    if df.empty:
        return None
    texts = (df["title"].fillna("") + " [SEP] " + df["summary"].fillna("")).tolist()
    vecs = _encode_texts(texts, model)
    return vecs.mean(axis=0, keepdims=True)


def _load_user_history(session_id="default", max_age_days=30):
    """Load user's click history with recency weighting"""
    if not os.path.exists(PROFILES_FP):
        return []
    try:
        with open(PROFILES_FP, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        if session_id not in profiles:
            return []
        clicks = profiles[session_id].get("clicks", [])
        # Filter by age and add recency weights
        now = datetime.utcnow()
        weighted = []
        for c in clicks:
            try:
                ts = dtparser.parse(c["timestamp"])
                age_days = (now - ts).total_seconds() / (24 * 3600)
                if age_days <= max_age_days:
                    weight = np.exp(-age_days / 7)  # exponential decay, half-life ~7 days
                    weighted.append({"url": c["url"], "weight": weight})
            except Exception:
                continue
        return weighted
    except Exception:
        return []


def recommend_for_user(urls=None, query=None, k=20, exclude_clicked=True, categories=None, session_id=None):
    meta, index, model = load_resources()

    cache_dir = os.path.join(DATA_DIR, "fastembed_cache")
    topic_vecs = build_topic_vectors(model)

    # Load user history if session_id provided
    if session_id:
        history = _load_user_history(session_id)
        # Use weighted average of historical clicks
        if history and not urls:
            hist_urls = [h["url"] for h in history]
            urls = hist_urls  # Fallback to history if no explicit URLs provided
        elif history and urls:
            # Merge: prioritize explicit urls but add historical context
            hist_urls = [h["url"] for h in history[:10]]  # Top 10 recent
            urls = list(set(urls + hist_urls))

    cat_vec = None
    if categories:
        cats = [normalize_topic_name(c) for c in categories]
        # average of topic vectors
        import numpy as np
        vecs = [topic_vecs[c] for c in cats if c in topic_vecs]
        if vecs:
            cat_vec = np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True).astype("float32")
            faiss.normalize_L2(cat_vec)

    if urls:
        uvec = _user_vector_from_urls(urls, meta, model)
    elif query:
        uvec = _encode_texts([query], model)
    else:
        # cold start: rely on categories if provided
        if cat_vec is None:
            # fallback to most recent
            out = meta.sort_values("published", ascending=False).head(k)
            return out.to_dict(orient="records")
        uvec = cat_vec

    if uvec is None:
        out = meta.sort_values("published", ascending=False).head(k)
        return out.to_dict(orient="records")

    extra = len(urls) if urls and exclude_clicked else 0
    scores, idxs = index.search(uvec, min(k + extra, index.ntotal))
    idxs = idxs[0]
    scores = scores[0]

    # Filter out indices that are out of bounds (index/meta desync)
    valid_mask = idxs < len(meta)
    idxs = idxs[valid_mask]
    scores = scores[valid_mask]
    
    if len(idxs) == 0:
        return []

    candidates = meta.iloc[idxs].copy()
    candidates["score"] = scores

    if urls and exclude_clicked:
        candidates = candidates[~candidates["url"].isin(urls)]

    try:
        candidates["published"] = pd.to_datetime(candidates["published"], utc=True)
        max_t = candidates["published"].max()
        decay = np.exp(-(max_t - candidates["published"]).dt.total_seconds() / (3600*24*3))
        candidates["final_score"] = 0.85 * candidates["score"] + 0.1 * decay
    except Exception:
        candidates["final_score"] = candidates["score"]

    # Category boost: if article categories intersect desired categories
    if categories:
        def cat_boost(row):
            try:
                ac = set(json.loads(row.get("categories", "[]")))
            except Exception:
                ac = set()
            want = set([normalize_topic_name(c) for c in categories])
            inter = len(ac.intersection(want))
            return 0.05 * inter  # +5% per matching category up to 10%
        candidates["final_score"] = candidates["final_score"] + candidates.apply(cat_boost, axis=1)

    # Get top-100 candidates for re-ranking
    top_candidates = candidates.sort_values("final_score", ascending=False).head(min(100, len(candidates)))
    
    # Try ensemble model first (NRMS + XGBoost), fallback to XGBoost only
    ensemble_model = _load_ensemble_model()
    xgb_model = _load_xgb_model()
    
    if ensemble_model is not None and len(top_candidates) > 1:
        try:
            import torch
            import numpy as np
            from dateutil import parser as dtparser
            
            # Prepare features for ensemble
            feats = []
            nrms_inputs_list = []
            
            # Get user history embeddings
            history_embs = []
            if urls:
                hist_df = meta[meta["url"].isin(urls[:50])]  # max 50 history items
                if not hist_df.empty:
                    hist_texts = (hist_df["title"].fillna("") + " [SEP] " + hist_df["summary"].fillna("")).tolist()
                    hist_vecs = _encode_texts(hist_texts, model)
                    history_embs = hist_vecs
            
            # Pad history to 50 items
            if len(history_embs) == 0:
                history_embs = np.zeros((50, 384), dtype=np.float32)
            else:
                if len(history_embs) < 50:
                    padding = np.zeros((50 - len(history_embs), 384), dtype=np.float32)
                    history_embs = np.vstack([history_embs, padding])
                else:
                    history_embs = history_embs[:50]
            
            history_mask = np.array([1.0] * min(len(history_embs), 50) + [0.0] * max(0, 50 - len(history_embs)), dtype=np.float32)
            
            # Get candidate embeddings
            cand_texts = (top_candidates["title"].fillna("") + " [SEP] " + top_candidates["summary"].fillna("")).tolist()
            cand_embs = _encode_texts(cand_texts, model)
            
            # Build XGBoost features
            for idx, row in top_candidates.iterrows():
                sim = row.get("score", 0.5)
                try:
                    pub = row.get("published")
                    if isinstance(pub, str):
                        pub = dtparser.parse(pub)
                    if pub and hasattr(pub, 'timestamp'):
                        age_hours = (pd.Timestamp.now(tz='UTC') - pd.Timestamp(pub, tz='UTC')).total_seconds() / 3600
                        rec = max(0.0, 1.0 - age_hours / (24*7))
                    else:
                        rec = 0.5
                except Exception:
                    rec = 0.5
                try:
                    ac = set(json.loads(row.get("categories", "[]")))
                    want = set([normalize_topic_name(c) for c in (categories or [])])
                    overlap = float(len(ac.intersection(want)) > 0) if want else 0.5
                except Exception:
                    overlap = 0.5
                feats.append([sim, rec, overlap])
            
            # Prepare NRMS inputs
            nrms_inputs = {
                'history_embs': torch.FloatTensor(history_embs),
                'history_mask': torch.FloatTensor(history_mask),
                'candidate_embs': torch.FloatTensor(cand_embs)
            }
            
            # Get ensemble predictions
            xgb_features = np.array(feats, dtype=np.float32)
            ensemble_scores = ensemble_model.predict(nrms_inputs, xgb_features)
            
            top_candidates["ml_score"] = ensemble_scores
            top_candidates = top_candidates.sort_values("ml_score", ascending=False).head(k)
            print(f"✅ Used ensemble model for ranking")
            
        except Exception as e:
            print(f"⚠️ Ensemble failed ({e}), falling back to XGBoost")
            # Fallback to XGBoost only
            if xgb_model is not None:
                try:
                    import xgboost as xgb
                    from dateutil import parser as dtparser
                    feats = []
                    for _, row in top_candidates.iterrows():
                        sim = row.get("score", 0.5)
                        try:
                            pub = row.get("published")
                            if isinstance(pub, str):
                                pub = dtparser.parse(pub)
                            if pub and hasattr(pub, 'timestamp'):
                                age_hours = (pd.Timestamp.now(tz='UTC') - pd.Timestamp(pub, tz='UTC')).total_seconds() / 3600
                                rec = max(0.0, 1.0 - age_hours / (24*7))
                            else:
                                rec = 0.5
                        except Exception:
                            rec = 0.5
                        try:
                            ac = set(json.loads(row.get("categories", "[]")))
                            want = set([normalize_topic_name(c) for c in (categories or [])])
                            overlap = float(len(ac.intersection(want)) > 0) if want else 0.5
                        except Exception:
                            overlap = 0.5
                        feats.append([sim, rec, overlap])
                    X = np.array(feats, dtype=np.float32)
                    dmat = xgb.DMatrix(X)
                    preds = xgb_model.predict(dmat)
                    top_candidates["ml_score"] = preds
                    top_candidates = top_candidates.sort_values("ml_score", ascending=False).head(k)
                except Exception:
                    top_candidates = top_candidates.head(k)
            else:
                top_candidates = top_candidates.head(k)
                
    elif xgb_model is not None and len(top_candidates) > 1:
        # XGBoost only (ensemble not available)
        try:
            import xgboost as xgb
            from dateutil import parser as dtparser
            feats = []
            for _, row in top_candidates.iterrows():
                sim = row.get("score", 0.5)
                try:
                    pub = row.get("published")
                    if isinstance(pub, str):
                        pub = dtparser.parse(pub)
                    if pub and hasattr(pub, 'timestamp'):
                        age_hours = (pd.Timestamp.now(tz='UTC') - pd.Timestamp(pub, tz='UTC')).total_seconds() / 3600
                        rec = max(0.0, 1.0 - age_hours / (24*7))
                    else:
                        rec = 0.5
                except Exception:
                    rec = 0.5
                try:
                    ac = set(json.loads(row.get("categories", "[]")))
                    want = set([normalize_topic_name(c) for c in (categories or [])])
                    overlap = float(len(ac.intersection(want)) > 0) if want else 0.5
                except Exception:
                    overlap = 0.5
                feats.append([sim, rec, overlap])
            X = np.array(feats, dtype=np.float32)
            dmat = xgb.DMatrix(X)
            preds = xgb_model.predict(dmat)
            top_candidates["ml_score"] = preds
            top_candidates = top_candidates.sort_values("ml_score", ascending=False).head(k)
            print(f"✅ Used XGBoost model for ranking")
        except Exception as e:
            top_candidates = top_candidates.head(k)
    else:
        top_candidates = top_candidates.head(k)
    
    return top_candidates[["title", "url", "source", "published", "final_score", "categories", "summary"]].to_dict(orient="records")


if __name__ == "__main__":
    recs = recommend_for_user(query="latest AI research breakthroughs", k=10)
    for r in recs:
        print(f"{r['final_score']:.3f} | {r['title']} | {r['source']} | {r['url']}")
