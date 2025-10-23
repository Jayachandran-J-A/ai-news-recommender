"""
Minimal MIND evaluator and trainer for a hybrid ranker:
- Downloads MINDsmall or full MIND (configurable)
- Builds content embeddings with fastembed
- Trains a gradient boosted ranker (XGBoost) using features: [cosine(u, v), recency (hours), category overlap]
- Evaluates on dev impressions with NDCG@10 and AUC

Note: This is a CPU-friendly baseline to add a DS component to the project.
"""
import os
import gzip
import shutil
import tarfile
import zipfile
import urllib.request
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from fastembed import TextEmbedding
import faiss

from dateutil import parser as dtparser
from datetime import timezone

def _roc_auc_score_simple(y_true, y_score):
    import numpy as _np
    y_true = _np.asarray(y_true)
    y_score = _np.asarray(y_score)
    # rank scores (average ties)
    order = _np.argsort(y_score)
    ranks = _np.empty_like(order, dtype=_np.float64)
    ranks[order] = _np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = _np.unique(y_score, return_inverse=True, return_counts=True)
    sums = _np.bincount(inv, ranks)
    avg_ranks = sums / counts
    ranks = avg_ranks[inv]
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return _np.nan
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mind"))
EMB_CACHE = os.path.join(DATA_ROOT, "embeddings.npy")
NEWS_CSV = os.path.join(DATA_ROOT, "news.csv")
BEHAVIOR_TRAIN = os.path.join(DATA_ROOT, "behaviors_train.tsv")
BEHAVIOR_DEV = os.path.join(DATA_ROOT, "behaviors_dev.tsv")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

MIND_URLS = {
    "small": {
        "news": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    },
    "large": {
        "news": "https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip",
    },
}


def _download(url: str, dest: str):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(
            "Failed to download. Public access may be blocked. "
            f"Please download manually and place the file at {dest}. Error: {e}"
        )
    return dest


def _unzip(zipfp: str, dest_dir: str):
    with zipfile.ZipFile(zipfp, 'r') as z:
        z.extractall(dest_dir)


def _find_file(root: str, name: str) -> str:
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def load_mind_split(size: str = "small") -> Tuple[pd.DataFrame, pd.DataFrame]:
    urls = MIND_URLS[size]
    train_zip = _download(urls["news"], os.path.join(DATA_ROOT, f"MIND_{size}_train.zip"))
    dev_zip = _download(urls["dev"], os.path.join(DATA_ROOT, f"MIND_{size}_dev.zip"))

    train_dir = os.path.join(DATA_ROOT, f"train_{size}")
    dev_dir = os.path.join(DATA_ROOT, f"dev_{size}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)

    _unzip(train_zip, train_dir)
    _unzip(dev_zip, dev_dir)

    # Expect news.tsv and behaviors.tsv (possibly in nested folder)
    train_news_fp = _find_file(train_dir, "news.tsv")
    train_beh_fp = _find_file(train_dir, "behaviors.tsv")
    dev_news_fp = _find_file(dev_dir, "news.tsv")
    dev_beh_fp = _find_file(dev_dir, "behaviors.tsv")

    train_news = pd.read_table(train_news_fp, header=None)
    train_news.columns = ["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"]

    dev_news = pd.read_table(dev_news_fp, header=None)
    dev_news.columns = train_news.columns

    news = pd.concat([train_news, dev_news]).drop_duplicates(subset=["news_id"]).reset_index(drop=True)

    train_beh = pd.read_table(train_beh_fp, header=None)
    train_beh.columns = ["impression_id","user_id","time","history","impressions"]

    dev_beh = pd.read_table(dev_beh_fp, header=None)
    dev_beh.columns = train_beh.columns

    # Save copies for inspection
    os.makedirs(DATA_ROOT, exist_ok=True)
    news.to_csv(NEWS_CSV, index=False)
    train_beh.to_csv(BEHAVIOR_TRAIN, sep="\t", index=False)
    dev_beh.to_csv(BEHAVIOR_DEV, sep="\t", index=False)

    return news, (train_beh, dev_beh)


def build_news_embeddings(news: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    cache_dir = os.path.join(DATA_ROOT, "fastembed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)

    texts = (news["title"].fillna("") + " [SEP] " + news["abstract"].fillna("")).tolist()
    vecs = list(embedder.embed(texts, batch_size=128, parallel=None))
    emb = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(emb)

    id_map = {nid: i for i, nid in enumerate(news["news_id"].tolist())}
    np.save(EMB_CACHE, emb)
    return emb, id_map


def _parse_history(history: str) -> List[str]:
    if pd.isna(history) or not history:
        return []
    return history.split(' ')


def _parse_impressions(impr: str) -> List[Tuple[str, int]]:
    items = []
    for token in impr.split(' '):
        try:
            nid, label = token.split('-')
            items.append((nid, int(label)))
        except Exception:
            continue
    return items


def _user_vector(history_ids: List[str], id_map: Dict[str, int], emb: np.ndarray) -> np.ndarray:
    idx = [id_map[h] for h in history_ids if h in id_map]
    if not idx:
        return None
    u = emb[idx].mean(axis=0, keepdims=True)
    faiss.normalize_L2(u)
    return u


def make_pairwise_features(beh: pd.DataFrame, id_map: Dict[str, int], emb: np.ndarray, news: pd.DataFrame, max_rows: int = 200000):
    records = []
    for _, row in tqdm(beh.iterrows(), total=len(beh)):
        history_ids = _parse_history(row["history"])
        items = _parse_impressions(row["impressions"])  # list of (news_id, label)
        u = _user_vector(history_ids, id_map, emb)
        if u is None:
            continue
        for nid, label in items:
            j = id_map.get(nid)
            if j is None:
                continue
            v = emb[j:j+1]
            sim = float((u @ v.T)[0][0])
            # recency proxy: if news has timestamp in dataset (not provided), skip or set 0
            rec = 0.0
            # category overlap
            try:
                user_hist = news[news["news_id"].isin(history_ids)]
                cats_u = set(user_hist["category"].dropna().unique().tolist())
                cat_v = news.loc[news["news_id"] == nid, "category"].dropna()
                overlap = 1.0 if (len(cats_u) and len(cat_v) and cat_v.iloc[0] in cats_u) else 0.0
            except Exception:
                overlap = 0.0
            records.append((sim, rec, overlap, label))
            if len(records) >= max_rows:
                break
        if len(records) >= max_rows:
            break
    X = np.array([[r[0], r[1], r[2]] for r in records], dtype=np.float32)
    y = np.array([r[3] for r in records], dtype=np.int32)
    return X, y


def train_xgb_ranker(X: np.ndarray, y: np.ndarray):
    import xgboost as xgb
    # Binary classification as a simple baseline (clicked vs not)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 6,
        "eta": 0.2,
        "objective": "binary:logistic",
        "eval_metric": ["auc"],
        "verbosity": 1,
        "tree_method": "hist",
    }
    bst = xgb.train(params, dtrain, num_boost_round=200)
    return bst


def evaluate_model(bst, beh: pd.DataFrame, id_map: Dict[str, int], emb: np.ndarray, news: pd.DataFrame):
    import xgboost as xgb
    ndcg_k = 10
    ndcg_scores = []
    aucs = []

    for _, row in tqdm(beh.iterrows(), total=len(beh)):
        history_ids = _parse_history(row["history"])
        items = _parse_impressions(row["impressions"])
        if not items:
            continue
        u = _user_vector(history_ids, id_map, emb)
        if u is None:
            continue
        sims = []
        labels = []
        feats = []
        for nid, label in items:
            j = id_map.get(nid)
            if j is None:
                continue
            v = emb[j:j+1]
            sim = float((u @ v.T)[0][0])
            labels.append(label)
            # features as in training
            user_hist = news[news["news_id"].isin(history_ids)]
            cats_u = set(user_hist["category"].dropna().unique().tolist())
            cat_v = news.loc[news["news_id"] == nid, "category"].dropna()
            overlap = 1.0 if (len(cats_u) and len(cat_v) and cat_v.iloc[0] in cats_u) else 0.0
            feats.append([sim, 0.0, overlap])
        if not feats:
            continue
        X = np.array(feats, dtype=np.float32)
        dpred = xgb.DMatrix(X)
        preds = bst.predict(dpred)
        # NDCG@10
        order = np.argsort(-preds)
        gains = (2 ** np.array(labels) - 1)[order]
        discounts = 1 / np.log2(np.arange(2, 2 + len(gains)))
        dcg = (gains[:ndcg_k] * discounts[:ndcg_k]).sum()
        ideal = (2 ** np.sort(labels)[::-1] - 1)
        idcg = (ideal[:ndcg_k] * discounts[:ndcg_k]).sum()
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
        # AUC on this impression set if both classes present
        if len(set(labels)) > 1:
            try:
                auc_val = _roc_auc_score_simple(labels, preds)
                if not np.isnan(auc_val):
                    aucs.append(auc_val)
            except Exception:
                pass
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0, float(np.mean(aucs)) if aucs else 0.0


if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)
    print("Loading MIND (small) ...")
    news, (train_beh, dev_beh) = load_mind_split(size="small")

    print("Building embeddings ...")
    emb, id_map = build_news_embeddings(news)

    print("Generating training features ...")
    X_train, y_train = make_pairwise_features(train_beh.sample(frac=0.2, random_state=42), id_map, emb, news, max_rows=200000)

    print("Training XGBoost baseline ...")
    bst = train_xgb_ranker(X_train, y_train)
    # Save model for serving-time re-ranking
    models_dir = os.path.join(DATA_ROOT, "..", "models")
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    model_fp = os.path.join(models_dir, "xgb_mind.json")
    import xgboost as xgb
    bst.save_model(model_fp)
    print(f"Saved model to {model_fp}")

    print("Evaluating on dev ...")
    ndcg10, auc = evaluate_model(bst, dev_beh.sample(frac=0.2, random_state=43), id_map, emb, news)

    print({"NDCG@10": ndcg10, "AUC": auc})
