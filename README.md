# 🚀 AI-Powered News Recommender System# AI-Powered News Recommender System 🚀



> **Data Science Capstone Project** - Intelligent news recommendation using Machine Learning, Deep Learning, and Vector SearchA complete **Data Science Capstone Project** featuring an intelligent news recommendation system with:

- **Machine Learning**: XGBoost re-ranker + BGE embeddings

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)- **Vector Search**: FAISS semantic similarity

![React](https://img.shields.io/badge/React-18+-61DAFB.svg)- **Modern UI**: React + TypeScript with shadcn/ui

![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)- **Real-time Data**: RSS crawler with background scheduler

![License](https://img.shields.io/badge/License-MIT-green.svg)- **Personalization**: Session-based click tracking



## 📋 Table of Contents## 🏗️ Architecture

- [Features](#-features)

- [Architecture](#-architecture)```

- [Project Structure](#-project-structure)┌─────────────────┐

- [Quick Start](#-quick-start)│  React Frontend │ (Vite, TypeScript, shadcn/ui)

- [Usage](#-usage)│  Port 8080      │

- [Technologies](#-technologies)└────────┬────────┘

- [Documentation](#-documentation)         │ HTTP API

         ↓

## ✨ Features┌─────────────────┐

│  FastAPI Server │ (Python, CORS enabled)

### 🤖 Machine Learning & AI│  Port 8003      │

- **Deep Learning**: NRMS (Neural News Recommendation with Multi-head Self-attention)└────────┬────────┘

- **Traditional ML**: XGBoost gradient boosting for re-ranking         │

- **Embeddings**: BGE-small (384-dim) semantic vectors with ONNX runtime    ┌────┴────┬────────────┬──────────┐

- **Vector Search**: FAISS IndexFlatIP for cosine similarity    ↓         ↓            ↓          ↓

- **Ensemble Model**: 60% NRMS + 40% XGBoost for optimal performance┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐

│RSS Feed│ │ FAISS  │ │XGBoost │ │Trending│

### 📰 News Processing│Crawler │ │ Index  │ │Re-rank │ │Extract │

- **20+ News Sources**: BBC, Reuters, CNN, TechCrunch, NDTV, The Hindu, etc.└────────┘ └────────┘ └────────┘ └────────┘

- **Auto Categorization**: 12 topics (Technology, Politics, Sports, Business, etc.)    │         │            │          │

- **Trending Analysis**: TF-IDF keyword extraction from last 72 hours    ↓         ↓            ↓          ↓

- **Real-time Updates**: Background RSS crawler every 30 minutes┌──────────────────────────────────────────┐

│           Data Layer                      │

### 🎯 User Experience│  - meta.csv (377 articles)               │

- **Smart Recommendations**: Personalized "For You" feed based on reading history│  - index.faiss (BGE embeddings)          │

- **Cold Start Solution**: Interest selection modal for new users│  - user_profiles.json (click tracking)   │

- **Semantic Search**: AI-powered search with category filters└──────────────────────────────────────────┘

- **Responsive Design**: Mobile-first UI with shadcn/ui components```

- **Live Updates**: Auto-refresh every 2 minutes

## 🎯 Features

## 🏗️ Architecture

### Machine Learning & NLP

```- **BGE-small Embeddings**: 384-dim semantic vectors (ONNX runtime)

┌─────────────────────────────────────────────────────────────┐- **FAISS Vector Search**: IndexFlatIP for cosine similarity

│                     FRONTEND (React + TypeScript)            │- **XGBoost Re-ranker**: Learned model trained on MIND dataset

│                         Port 8080                            │- **Category Tagging**: 12 topics auto-assigned using seed vectors

└────────────────────────┬────────────────────────────────────┘- **Trending Extraction**: TF-IDF keyword/bigram analysis

                         │ HTTP/REST API

                         ▼### User Experience

┌─────────────────────────────────────────────────────────────┐- ✨ **Onboarding Modal**: Interest selection for cold start

│              BACKEND (FastAPI + Python)                      │- 🔍 **Smart Search**: Semantic search + category filters

│                    Port 8003                                 │- 📊 **Trending Topics**: Hot keywords from last 72 hours

│         ┌───────────┴───────────┬──────────────┬──────────┐ │- 🎯 **Personalized Feed**: "For You" based on click history

│         ▼                       ▼              ▼          ▼ │- ♻️ **Auto-Refresh**: Live updates every 2 minutes

│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │- 📱 **Responsive Design**: Mobile-first, works on all devices

│  │ RSS Feed │  │   Ensemble   │  │  FAISS   │  │Trending │ │

│  │ Crawler  │  │Recommender   │  │  Index   │  │Extractor│ │### Backend

│  │(20 srcs) │  │NRMS+XGBoost  │  │ (Vector  │  │(TF-IDF) │ │- 🌐 **20+ News Sources**: BBC, NDTV, TechCrunch, Reuters, CNN, etc.

│  └──────────┘  └──────────────┘  │  Search) │  └─────────┘ │- ⏰ **Background Scheduler**: Crawls RSS every 30 minutes

└──────────────────────┬──────────────────────────────────────┘- 🔗 **Click Tracking**: Session-based personalization

                       │- 📈 **RESTful API**: FastAPI with Swagger docs

                       ▼

┌─────────────────────────────────────────────────────────────┐## 🚀 Quick Start

│                    DATA LAYER                                │

│  meta.csv | index.faiss | user_profiles.json | models/      │### Prerequisites

└─────────────────────────────────────────────────────────────┘- Python 3.13

```- Node.js 18+ and npm

- Git

## 📁 Project Structure

### 1. Clone & Setup Backend

```

news-recommender/```bash

├── 📄 README.md                    # Project documentationcd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

├── 📄 requirements.txt             # Python dependencies

├── 📄 .gitignore                   # Git ignore rules# Create virtual environment

├── 🔧 setup.bat                    # Automated setup scriptpython -m venv .venv

├── ▶️ start.bat                    # Start both servers.\.venv\Scripts\Activate.ps1

│

├── 📁 src/                         # Backend source code# Install dependencies

│   ├── api.py                      # FastAPI applicationpip install -r requirements.txt

│   ├── recommend.py                # Recommendation engine

│   ├── nrms.py                     # NRMS neural model# Initial data ingestion (fetches articles, builds index)

│   ├── ensemble.py                 # Ensemble modelpython -m src.ingest_rss

│   └── ingest_rss.py              # RSS feed crawler```

│

├── 📁 frontend/                    # React frontend### 2. Setup Frontend

│   ├── src/                        # Source code

│   └── package.json                # Dependencies```bash

│cd nexusnews-ai-main

├── 📁 scripts/                     # Utility scriptsnpm install

│   ├── train_nrms.py              # Train NRMS model```

│   └── rebuild_index.py           # Rebuild FAISS index

│### 3. Start Both Servers

├── 📁 tests/                       # Test files

│   ├── test_api.py                # API tests**Option A: Using Startup Script (Recommended)**

│   └── evaluate_all.py            # Model evaluation```powershell

│.\start.ps1

├── 📁 docs/                        # Documentation```

│   ├── QUICK_START.md             # Quick start guideThis opens both servers in new windows and launches the UI in your browser.

│   └── Report/                     # Capstone report

│**Option B: Manual (Two Terminals)**

├── 📁 data/                        # Data directory

│   ├── meta.csv                    # Article metadataTerminal 1 - Backend:

│   └── index.faiss                 # Vector index```bash

│cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

└── 📁 models/                      # Trained models.\.venv\Scripts\Activate.ps1

    ├── nrms_best.pt               # NRMS weightspython -m uvicorn src.server:app --port 8003

    └── xgb_model.pkl              # XGBoost model```

```

Terminal 2 - Frontend:

## 🚀 Quick Start```bash

cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender\nexusnews-ai-main"

### Prerequisitesnpm run dev

- **Python 3.8+** ([Download](https://www.python.org/downloads/))```

- **Node.js 18+** ([Download](https://nodejs.org/))

- **Git** ([Download](https://git-scm.com/downloads))### 4. Access the Application



### Installation- **🎨 Frontend UI**: http://localhost:8080

- **🔌 Backend API**: http://localhost:8003

#### Automated Setup (Recommended) ⚡- **📚 API Docs**: http://localhost:8003/docs (Swagger UI)



```bash## 📊 Model Details

# Clone the repository

git clone https://github.com/YOUR_USERNAME/news-recommender.git### Embeddings

cd news-recommender- **Model**: `BAAI/bge-small-en-v1.5` (ONNX optimized)

- **Dimensions**: 384

# Run automated setup (handles everything!)- **Normalization**: L2 (for cosine similarity via inner product)

setup.bat- **Cache**: Stored in `data/fastembed_cache/`

```

### Vector Index

The setup script will:- **Type**: FAISS IndexFlatIP (exact search)

- ✅ Check Python and Node.js- **Size**: 377 articles (dynamically grows)

- ✅ Create virtual environment- **Similarity**: Cosine similarity (normalized vectors)

- ✅ Install dependencies- **Storage**: `data/index.faiss`

- ✅ Fetch initial news articles

### Re-ranking (Optional)

### Starting the Application- **Model**: XGBoost gradient boosting

- **Features**: [cosine_sim, recency_hours, category_overlap]

```bash- **Training**: MIND dataset (Microsoft News Dataset)

start.bat- **Metrics**: NDCG@10, AUC

```- **Fallback**: Uses heuristic scores if model not trained



This opens:### Categories

- 🌐 Frontend: **http://localhost:8080**Politics, Business, Technology, Science, Health, Sports, Entertainment, World, India, AI, Climate, Education

- 🔗 Backend: **http://localhost:8003**

- 📚 API Docs: **http://localhost:8003/docs**## 📁 Project Structure



## 📖 Usage```

news-recommender/

### For Users├── src/                          # Python backend

│   ├── api.py                    # FastAPI endpoints

1. **Select Interests**: Choose topics in onboarding modal│   ├── server.py                 # CORS + background scheduler

2. **Browse Feed**: View personalized "For You" articles│   ├── recommend.py              # ML recommendation engine

3. **Search**: Use semantic search with filters│   ├── ingest_rss.py             # RSS crawler + embeddings

4. **Trending**: Check hot topics│   ├── topics.py                 # Category taxonomy + seed vectors

│   ├── trending.py               # Keyword extraction

### For Developers│   ├── eval_mind.py              # XGBoost training pipeline

│   ├── config_feeds.py           # RSS feed URLs

#### Run Tests│   └── utils.py                  # Helpers (hash, date parsing)

```bash├── nexusnews-ai-main/            # React frontend

.venv\Scripts\activate│   ├── src/

python tests/test_complete_system.py│   │   ├── components/           # UI components (shadcn/ui)

python tests/evaluate_all.py│   │   ├── lib/

```│   │   │   └── api.ts            # API client (TypeScript)

│   │   ├── pages/

#### Train Models│   │   │   └── Index.tsx         # Main page

```bash│   │   └── App.tsx

python scripts/train_nrms.py --epochs 10│   ├── .env                      # API URL config

```│   └── package.json

├── data/

#### API Endpoints│   ├── index.faiss               # Vector index

│   ├── meta.csv                  # Article metadata

- `GET /recommend` - Personalized recommendations│   ├── user_profiles.json        # Click tracking

- `GET /search` - Semantic search│   └── fastembed_cache/          # Model cache

- `GET /trending` - Trending topics├── requirements.txt              # Python dependencies

- `POST /click` - Track clicks├── start.ps1                     # Quick startup script

└── README.md                     # This file

Full docs: http://localhost:8003/docs```



## 🛠️ Technologies## 🔧 API Endpoints



### Backend### `GET /recommend`

- **FastAPI** - Web frameworkGet personalized article recommendations

- **PyTorch** - Deep learning (NRMS)- **Query Params**:

- **XGBoost** - Gradient boosting  - `query`: Search text (default: "news")

- **FAISS** - Vector search  - `categories[]`: Filter by categories (multiple)

- **FastEmbed** - BGE embeddings  - `k`: Number of results (default: 20)

  - `session_id`: User session for personalization

### Frontend- **Returns**: `{ items: Article[] }`

- **React 18** - UI library

- **TypeScript** - Type safety### `POST /click`

- **Vite** - Build toolTrack article click for personalization

- **shadcn/ui** - Components- **Body**: `url`, `session_id`

- **Tailwind CSS** - Styling- **Returns**: `{ status: "ok", clicks: number }`



## 📚 Documentation### `GET /trending`

Get trending keywords

See `docs/` folder:- **Query Params**:

- [Quick Start Guide](docs/QUICK_START.md)  - `hours`: Time window (default: 72)

- [Testing Guide](docs/TESTING_GUIDE.md)  - `top_n`: Number of keywords (default: 10)

- [Training Guide](docs/COLAB_TRAINING_GUIDE.md)- **Returns**: `{ trends: [{term, count}] }`

- [Troubleshooting](docs/TROUBLESHOOTING.md)

- [Project Report](docs/Report/)### `GET /debug/info`

System diagnostics

## 🐛 Troubleshooting- **Returns**: Metadata + index stats



**Port in use?**## 🧪 Testing

```bash

netstat -ano | findstr :8003### Test Backend

taskkill /F /PID <PID>```bash

```# Check system status

curl http://localhost:8003/debug/info

**No articles?**

```bash# Test recommendations

python -m src.ingest_rsscurl "http://localhost:8003/recommend?query=AI&k=5"

```

# Test trending

**Import errors?**curl "http://localhost:8003/trending?hours=72&top_n=8"

```bash```

# Make sure you're in project root

cd news-recommender### Test Frontend

.venv\Scripts\activate1. Open http://localhost:8080

```2. Select interests in onboarding

3. Verify articles load

## 📄 License4. Try search, filters, trending clicks

5. Check browser DevTools → Network tab for API calls

MIT License

## 📚 Training XGBoost Model (Optional)

## 🙏 Acknowledgments

The system works without the trained model (uses heuristic scoring), but for better recommendations:

- MIND Dataset - Microsoft

- NRMS Paper - Wu et al. (EMNLP 2019)1. Download MIND dataset manually (due to Azure access restrictions)

- FastEmbed - Qdrant2. Place in `data/mind/` directory

- shadcn/ui - UI Components3. Train model:

```bash

---python -m src.eval_mind

```

⭐ **Star this repo** if helpful!4. Model saved to `data/models/xgb_mind.json`

5. Restart server → model auto-loads

## 🐛 Troubleshooting

### Backend won't start
- **Error**: `ModuleNotFoundError: No module named 'faiss'`
- **Fix**: Activate virtual environment first: `.\.venv\Scripts\Activate.ps1`

### No articles showing
- **Cause**: Index/metadata out of sync
- **Fix**: Rebuild index: `python rebuild_index.py`

### CORS errors
- **Check**: CORS already enabled in `src/server.py`
- **Verify**: Frontend `.env` has correct `VITE_API_URL`

### Trending not loading
- **Needs**: Sufficient articles (50+) for keyword extraction
- **Check**: Backend logs for `/trending` errors

## 📖 Documentation

- `INTEGRATION_COMPLETE.md` - Full integration guide
- `nexusnews-ai-main/README.md` - Frontend-specific docs
- `http://localhost:8003/docs` - API documentation (Swagger)

## 🎓 For Capstone Presentation

### Demo Checklist
- [ ] Show onboarding with interest selection
- [ ] Display personalized feed loading
- [ ] Filter by categories (live API calls)
- [ ] Search with semantic similarity
- [ ] Click article → show tracking
- [ ] Display trending topics widget
- [ ] Toggle auto-refresh
- [ ] Open DevTools → show API calls
- [ ] Explain ML pipeline (embeddings → FAISS → XGBoost)

### Key Metrics
- **Articles Indexed**: 377+
- **Sources**: 20+ RSS feeds
- **Categories**: 12 topics
- **Embedding Dim**: 384
- **Response Time**: <500ms
- **Refresh Interval**: 30 minutes

## 📝 Notes

- **Respect robots.txt**: Uses public RSS feeds only
- **No article scraping**: Metadata only (title, summary, URL)
- **Privacy**: Session IDs stored locally (no PII)
- **Scalability**: Can handle 10K+ articles with current architecture

## 🔗 Technologies Used

**Backend**: Python, FastAPI, FAISS, fastembed, XGBoost, pandas, feedparser
**Frontend**: React, TypeScript, Vite, shadcn/ui, TanStack Query, Tailwind CSS
**ML/NLP**: BGE embeddings, FAISS vector search, XGBoost ranking
**Data**: RSS feeds, MIND dataset (optional)

---

**Status**: ✅ Fully Functional
**Last Updated**: October 16, 2025
**License**: MIT (Educational Project)

