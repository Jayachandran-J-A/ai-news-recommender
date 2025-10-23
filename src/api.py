from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from .recommend import recommend_for_user as recommend_basic
from .recommend_advanced import recommend_for_user as recommend_advanced, load_resources
from .trending import extract_trending_terms
import os
import pandas as pd
import faiss
import json
from datetime import datetime
from .recommend import DATA_DIR, META_CSV, INDEX_FP
import hashlib

app = FastAPI(title="News Recommender - High Performance")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

PROFILES_FP = os.path.join(DATA_DIR, "user_profiles.json")

# Global cache
_RECOMMENDATION_CACHE = {}
_CACHE_TTL = 120  # 2 minutes


def _get_cache_key(query, categories, session_id):
    """Generate cache key"""
    key_str = f"{query}|{','.join(sorted(categories or []))}|{session_id or 'anon'}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _check_cache(key):
    """Check cache validity"""
    if key in _RECOMMENDATION_CACHE:
        cached_time, result = _RECOMMENDATION_CACHE[key]
        age = (datetime.now() - cached_time).total_seconds()
        if age < _CACHE_TTL:
            return result
    return None


@app.on_event("startup")
async def startup_event():
    """Pre-load resources on startup"""
    print("🚀 Loading embedding model and FAISS index...")
    load_resources()
    
    # Try to load ensemble model
    from .recommend import _load_ensemble_model, _load_xgb_model
    ensemble = _load_ensemble_model()
    xgb = _load_xgb_model()
    
    if ensemble:
        print("✅ Ensemble model loaded (NRMS + XGBoost)")
    elif xgb:
        print("✅ XGBoost model loaded")
    else:
        print("⚠️ Using baseline vector search only")
    
    print("✅ API ready!")


@app.get("/recommend")
def recommend(urls: Optional[List[str]] = Query(default=None), query: Optional[str] = None, k: int = 10, categories: Optional[List[str]] = Query(default=None), session_id: Optional[str] = None):
    try:
        # Check cache first
        cache_key = _get_cache_key(query, categories, session_id)
        cached_result = _check_cache(cache_key)
        if cached_result is not None:
            return {"items": cached_result, "cached": True}
        
        # Get fresh recommendations using advanced engine
        items = recommend_advanced(urls=urls, query=query, k=k, categories=categories, session_id=session_id)
        
        # Format with match_percentage
        formatted = []
        for item in items:
            formatted.append({
                "title": item["title"],
                "url": item["url"],
                "source": item["source"],
                "published": item["published"],
                "match_percentage": round(item.get("final_score", 60), 1),
                "categories": item["categories"]
            })
        
        # Cache result
        _RECOMMENDATION_CACHE[cache_key] = (datetime.now(), formatted)
        
        return {"items": formatted, "cached": False}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/debug/info")
def debug_info():
    info = {"meta_exists": os.path.exists(META_CSV), "index_exists": os.path.exists(INDEX_FP)}
    if info["meta_exists"]:
        try:
            meta = pd.read_csv(META_CSV)
            info["meta_len"] = len(meta)
            info["sample_titles"] = meta["title"].head(3).tolist()
        except Exception as e:
            info["meta_error"] = str(e)
    if info["index_exists"]:
        try:
            index = faiss.read_index(INDEX_FP)
            info["index_ntotal"] = index.ntotal
        except Exception as e:
            info["index_error"] = str(e)
    return info


@app.post("/click")
def record_click(url: str, session_id: str = "default"):
    """Record user click for personalization"""
    try:
        # Load or initialize profiles
        if os.path.exists(PROFILES_FP):
            with open(PROFILES_FP, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        else:
            profiles = {}
        
        # Add click to user's history
        if session_id not in profiles:
            profiles[session_id] = {"clicks": []}
        
        profiles[session_id]["clicks"].append({
            "url": url,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep last 100 clicks per user
        profiles[session_id]["clicks"] = profiles[session_id]["clicks"][-100:]
        
        # Save profiles
        os.makedirs(os.path.dirname(PROFILES_FP), exist_ok=True)
        with open(PROFILES_FP, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        
        return {"status": "ok", "clicks": len(profiles[session_id]["clicks"])}
    except Exception as e:
        return {"error": str(e)}


@app.get("/trending")
def get_trending(hours: int = 72, top_n: int = 10):
    """Get trending keywords from recent articles (default: last 72 hours)"""
    try:
        trends = extract_trending_terms(hours=hours, top_n=top_n)
        return {"trends": [{"term": term, "count": count} for term, count in trends]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/refresh")
def refresh_news():
    """
    Manually trigger news refresh from RSS feeds
    Runs ingestion in background and rebuilds index
    """
    try:
        import subprocess
        import sys
        
        # Clear recommendation cache
        global _RECOMMENDATION_CACHE
        _RECOMMENDATION_CACHE.clear()
        
        # Run ingestion in background (non-blocking)
        subprocess.Popen(
            [sys.executable, "-m", "src.ingest_rss"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return {
            "status": "started",
            "message": "News refresh started in background. New articles will be available in 1-2 minutes."
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/refresh/status")
def refresh_status():
    """Check last update time and article count"""
    try:
        if os.path.exists(META_CSV):
            meta = pd.read_csv(META_CSV)
            # Get the most recent article's timestamp
            if 'published' in meta.columns and len(meta) > 0:
                last_update = meta['published'].max()
            else:
                last_update = "Unknown"
            
            return {
                "total_articles": len(meta),
                "last_update": last_update,
                "sources": len(meta['source'].unique()) if 'source' in meta.columns else 0
            }
        else:
            return {"error": "No articles indexed yet"}
    except Exception as e:
        return {"error": str(e)}

