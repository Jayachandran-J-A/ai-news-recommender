# 🚀 AI-Powered News Recommender System# 🚀 AI-Powered News Recommender System# AI-Powered News Recommender System 🚀



> **Data Science Capstone Project** - Intelligent news recommendation using Deep Learning, Machine Learning, and Vector Search



![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)> **Data Science Capstone Project** - Intelligent news recommendation using Machine Learning, Deep Learning, and Vector SearchA complete **Data Science Capstone Project** featuring an intelligent news recommendation system with:

![React](https://img.shields.io/badge/React-18+-61DAFB.svg)

![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)- **Machine Learning**: XGBoost re-ranker + BGE embeddings

![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)

![License](https://img.shields.io/badge/License-MIT-green.svg)![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)- **Vector Search**: FAISS semantic similarity



A complete AI-powered news recommendation system featuring **NRMS** (Neural News Recommendation with Multi-head Self-attention), **XGBoost ensemble**, and **FAISS vector search** with a modern React frontend.![React](https://img.shields.io/badge/React-18+-61DAFB.svg)- **Modern UI**: React + TypeScript with shadcn/ui



---![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)- **Real-time Data**: RSS crawler with background scheduler



## 📋 Table of Contents![License](https://img.shields.io/badge/License-MIT-green.svg)- **Personalization**: Session-based click tracking

- [Features](#-features)

- [Architecture](#-architecture)

- [Quick Start](#-quick-start)

- [Project Structure](#-project-structure)## 📋 Table of Contents## 🏗️ Architecture

- [API Documentation](#-api-documentation)

- [Technologies](#-technologies)- [Features](#-features)

- [Testing](#-testing)

- [Architecture](#-architecture)```

---

- [Project Structure](#-project-structure)┌─────────────────┐

## ✨ Features

- [Quick Start](#-quick-start)│  React Frontend │ (Vite, TypeScript, shadcn/ui)

### 🤖 AI & Machine Learning

- **NRMS Model**: Neural news recommendation with multi-head self-attention (PyTorch)- [Usage](#-usage)│  Port 8080      │

- **XGBoost Ensemble**: Gradient boosting for intelligent re-ranking

- **BGE Embeddings**: 384-dimensional semantic vectors (BAAI/bge-small-en-v1.5)- [Technologies](#-technologies)└────────┬────────┘

- **FAISS Search**: Fast approximate nearest neighbor search

- **Hybrid Approach**: 60% NRMS + 40% XGBoost for optimal performance- [Documentation](#-documentation)         │ HTTP API



### 📰 News Processing         ↓

- **20+ RSS Sources**: BBC, Reuters, CNN, TechCrunch, NDTV, The Hindu, and more

- **Auto-Categorization**: 12 topics (Technology, Politics, Sports, Business, etc.)## ✨ Features┌─────────────────┐

- **Trending Analysis**: TF-IDF keyword extraction from recent articles

- **Background Updates**: Automatic news refresh every 30 minutes│  FastAPI Server │ (Python, CORS enabled)



### 🎯 User Experience### 🤖 Machine Learning & AI│  Port 8003      │

- **Personalized Feed**: "For You" recommendations based on reading history

- **Cold Start Solution**: Interest selection onboarding for new users- **Deep Learning**: NRMS (Neural News Recommendation with Multi-head Self-attention)└────────┬────────┘

- **Semantic Search**: AI-powered search with multi-category filters

- **Modern UI**: React + TypeScript + Tailwind CSS + shadcn/ui- **Traditional ML**: XGBoost gradient boosting for re-ranking         │

- **Live Dashboard**: Real-time trending topics and updates

- **Embeddings**: BGE-small (384-dim) semantic vectors with ONNX runtime    ┌────┴────┬────────────┬──────────┐

---

- **Vector Search**: FAISS IndexFlatIP for cosine similarity    ↓         ↓            ↓          ↓

## 🏗️ Architecture

- **Ensemble Model**: 60% NRMS + 40% XGBoost for optimal performance┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐

```

┌─────────────────────────────────────────────────────────────┐│RSS Feed│ │ FAISS  │ │XGBoost │ │Trending│

│                   FRONTEND (React + TypeScript)              │

│                         Port 8080                            │### 📰 News Processing│Crawler │ │ Index  │ │Re-rank │ │Extract │

│  Components: Feed, Search, Trending, Categories, Modal      │

└────────────────────────┬────────────────────────────────────┘- **20+ News Sources**: BBC, Reuters, CNN, TechCrunch, NDTV, The Hindu, etc.└────────┘ └────────┘ └────────┘ └────────┘

                         │ HTTP REST API

                         ▼- **Auto Categorization**: 12 topics (Technology, Politics, Sports, Business, etc.)    │         │            │          │

┌─────────────────────────────────────────────────────────────┐

│                  BACKEND (FastAPI + Python)                  │- **Trending Analysis**: TF-IDF keyword extraction from last 72 hours    ↓         ↓            ↓          ↓

│                        Port 8003                             │

│                                                              │- **Real-time Updates**: Background RSS crawler every 30 minutes┌──────────────────────────────────────────┐

│   ┌────────────┐  ┌────────────┐  ┌──────────┐            │

│   │ RSS Crawler│  │  Ensemble  │  │  FAISS   │            ││           Data Layer                      │

│   │ (20 feeds) │→ │ Recommender│→ │  Index   │            │

│   └────────────┘  │NRMS+XGBoost│  │(Vectors) │            │### 🎯 User Experience│  - meta.csv (377 articles)               │

│                   └────────────┘  └──────────┘            │

│                          ↓                                   │- **Smart Recommendations**: Personalized "For You" feed based on reading history│  - index.faiss (BGE embeddings)          │

│                   ┌────────────┐                            │

│                   │  Trending  │                            │- **Cold Start Solution**: Interest selection modal for new users│  - user_profiles.json (click tracking)   │

│                   │  Extractor │                            │

│                   └────────────┘                            │- **Semantic Search**: AI-powered search with category filters└──────────────────────────────────────────┘

└──────────────────────┬──────────────────────────────────────┘

                       │- **Responsive Design**: Mobile-first UI with shadcn/ui components```

                       ▼

┌─────────────────────────────────────────────────────────────┐- **Live Updates**: Auto-refresh every 2 minutes

│                      DATA LAYER                              │

│  • meta.csv (Article metadata)                              │## 🎯 Features

│  • index.faiss (Vector embeddings)                          │

│  • user_profiles.json (Click tracking)                      │## 🏗️ Architecture

│  • models/ (NRMS weights, XGBoost model)                    │

└─────────────────────────────────────────────────────────────┘### Machine Learning & NLP

```

```- **BGE-small Embeddings**: 384-dim semantic vectors (ONNX runtime)

---

┌─────────────────────────────────────────────────────────────┐- **FAISS Vector Search**: IndexFlatIP for cosine similarity

## 🚀 Quick Start

│                     FRONTEND (React + TypeScript)            │- **XGBoost Re-ranker**: Learned model trained on MIND dataset

### Prerequisites

- **Python 3.8+** - [Download](https://www.python.org/downloads/)│                         Port 8080                            │- **Category Tagging**: 12 topics auto-assigned using seed vectors

- **Node.js 18+** - [Download](https://nodejs.org/)

- **Git** - [Download](https://git-scm.com/downloads/)└────────────────────────┬────────────────────────────────────┘- **Trending Extraction**: TF-IDF keyword/bigram analysis



### Installation                         │ HTTP/REST API



#### Option 1: Automated Setup (Recommended) ⚡                         ▼### User Experience



```bash┌─────────────────────────────────────────────────────────────┐- ✨ **Onboarding Modal**: Interest selection for cold start

# Clone the repository

git clone https://github.com/Jayachandran-J-A/ai-news-recommender.git│              BACKEND (FastAPI + Python)                      │- 🔍 **Smart Search**: Semantic search + category filters

cd ai-news-recommender

│                    Port 8003                                 │- 📊 **Trending Topics**: Hot keywords from last 72 hours

# Run automated setup (installs everything!)

setup.bat│         ┌───────────┴───────────┬──────────────┬──────────┐ │- 🎯 **Personalized Feed**: "For You" based on click history

```

│         ▼                       ▼              ▼          ▼ │- ♻️ **Auto-Refresh**: Live updates every 2 minutes

The setup script will:

- ✅ Verify Python and Node.js installation│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │- 📱 **Responsive Design**: Mobile-first, works on all devices

- ✅ Create virtual environment

- ✅ Install Python dependencies│  │ RSS Feed │  │   Ensemble   │  │  FAISS   │  │Trending │ │

- ✅ Install frontend dependencies

- ✅ Fetch initial news articles│  │ Crawler  │  │Recommender   │  │  Index   │  │Extractor│ │### Backend

- ✅ Build FAISS index

│  │(20 srcs) │  │NRMS+XGBoost  │  │ (Vector  │  │(TF-IDF) │ │- 🌐 **20+ News Sources**: BBC, NDTV, TechCrunch, Reuters, CNN, etc.

#### Option 2: Manual Setup

│  └──────────┘  └──────────────┘  │  Search) │  └─────────┘ │- ⏰ **Background Scheduler**: Crawls RSS every 30 minutes

```bash

# 1. Create virtual environment└──────────────────────┬──────────────────────────────────────┘- 🔗 **Click Tracking**: Session-based personalization

python -m venv .venv

.venv\Scripts\activate  # Windows                       │- 📈 **RESTful API**: FastAPI with Swagger docs

# source .venv/bin/activate  # Linux/Mac

                       ▼

# 2. Install Python dependencies

pip install -r requirements.txt┌─────────────────────────────────────────────────────────────┐## 🚀 Quick Start



# 3. Install frontend dependencies│                    DATA LAYER                                │

cd frontend

npm install│  meta.csv | index.faiss | user_profiles.json | models/      │### Prerequisites

cd ..

└─────────────────────────────────────────────────────────────┘- Python 3.13

# 4. Fetch initial data

python -m src.ingest_rss```- Node.js 18+ and npm

```

- Git

### Starting the Application

## 📁 Project Structure

```bash

# Simply run### 1. Clone & Setup Backend

start.bat

``````



This will:news-recommender/```bash

- Start **Backend API** on http://localhost:8003

- Start **Frontend UI** on http://localhost:8080├── 📄 README.md                    # Project documentationcd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

- Open your browser automatically

- Show **API docs** at http://localhost:8003/docs├── 📄 requirements.txt             # Python dependencies



---├── 📄 .gitignore                   # Git ignore rules# Create virtual environment



## 📁 Project Structure├── 🔧 setup.bat                    # Automated setup scriptpython -m venv .venv



```├── ▶️ start.bat                    # Start both servers.\.venv\Scripts\Activate.ps1

ai-news-recommender/

├── 📄 README.md                    # This file│

├── 📄 requirements.txt             # Python dependencies

├── 📄 .gitignore                   # Git ignore rules├── 📁 src/                         # Backend source code# Install dependencies

├── 🔧 setup.bat                    # Automated setup script

├── ▶️ start.bat                    # Startup script│   ├── api.py                      # FastAPI applicationpip install -r requirements.txt

│

├── 📁 src/                         # Backend source code│   ├── recommend.py                # Recommendation engine

│   ├── api.py                      # FastAPI application

│   ├── recommend.py                # Recommendation engine│   ├── nrms.py                     # NRMS neural model# Initial data ingestion (fetches articles, builds index)

│   ├── recommend_advanced.py       # Advanced recommender

│   ├── nrms.py                     # NRMS neural model│   ├── ensemble.py                 # Ensemble modelpython -m src.ingest_rss

│   ├── ensemble.py                 # Ensemble model

│   ├── ingest_rss.py              # RSS feed crawler│   └── ingest_rss.py              # RSS feed crawler```

│   ├── topics.py                   # Category classification

│   ├── trending.py                 # Trending extraction│

│   ├── config_feeds.py            # RSS feed URLs

│   ├── metrics.py                  # Evaluation metrics├── 📁 frontend/                    # React frontend### 2. Setup Frontend

│   ├── mind_dataset.py            # MIND dataset loader

│   └── utils.py                    # Utility functions│   ├── src/                        # Source code

│

├── 📁 frontend/                    # React frontend│   └── package.json                # Dependencies```bash

│   ├── src/

│   │   ├── components/             # UI components│cd nexusnews-ai-main

│   │   ├── pages/                  # Page components

│   │   ├── lib/                    # Utilities & API client├── 📁 scripts/                     # Utility scriptsnpm install

│   │   └── App.tsx                 # Main app component

│   ├── package.json                # Frontend dependencies│   ├── train_nrms.py              # Train NRMS model```

│   └── vite.config.ts              # Vite configuration

││   └── rebuild_index.py           # Rebuild FAISS index

├── 📁 scripts/                     # Utility scripts

│   ├── train_nrms.py              # Train NRMS model│### 3. Start Both Servers

│   ├── rebuild_index.py           # Rebuild FAISS index

│   ├── setup_xgboost.py           # XGBoost setup├── 📁 tests/                       # Test files

│   └── quick_test.py              # Quick system test

││   ├── test_api.py                # API tests**Option A: Using Startup Script (Recommended)**

├── 📁 tests/                       # Test files

│   ├── test_api.py                # API tests│   └── evaluate_all.py            # Model evaluation```powershell

│   ├── test_complete_system.py    # Integration tests

│   ├── evaluate_all.py            # Model evaluation│.\start.ps1

│   └── health_check.py            # Health check script

│├── 📁 docs/                        # Documentation```

├── 📁 docs/                        # Documentation

│   ├── QUICK_START.md             # Quick start guide│   ├── QUICK_START.md             # Quick start guideThis opens both servers in new windows and launches the UI in your browser.

│   ├── TESTING_GUIDE.md           # Testing instructions

│   ├── COLAB_TRAINING_GUIDE.md    # Model training guide│   └── Report/                     # Capstone report

│   └── TROUBLESHOOTING.md         # Common issues

││**Option B: Manual (Two Terminals)**

├── 📁 data/                        # Data directory

│   ├── meta.csv                    # Article metadata├── 📁 data/                        # Data directory

│   ├── index.faiss                 # FAISS vector index

│   └── fastembed_cache/           # Embedding model cache│   ├── meta.csv                    # Article metadataTerminal 1 - Backend:

│

├── 📁 models/                      # Trained models│   └── index.faiss                 # Vector index```bash

│   ├── nrms_best.pt               # NRMS model weights

│   └── xgb_mind.json              # XGBoost model│cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

│

└── 📁 templates/                   # HTML templates└── 📁 models/                      # Trained models.\.venv\Scripts\Activate.ps1

```

    ├── nrms_best.pt               # NRMS weightspython -m uvicorn src.server:app --port 8003

---

    └── xgb_model.pkl              # XGBoost model```

## 📡 API Documentation

```

### Core Endpoints

Terminal 2 - Frontend:

#### `GET /recommend`

Get personalized article recommendations.## 🚀 Quick Start```bash



**Query Parameters:**cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender\nexusnews-ai-main"

- `query` (string): Search text (default: "news")

- `categories[]` (array): Filter by categories### Prerequisitesnpm run dev

- `k` (int): Number of results (default: 20)

- `session_id` (string): User session for personalization- **Python 3.8+** ([Download](https://www.python.org/downloads/))```



**Response:**- **Node.js 18+** ([Download](https://nodejs.org/))

```json

{- **Git** ([Download](https://git-scm.com/downloads))### 4. Access the Application

  "items": [

    {

      "title": "Article Title",

      "summary": "Article summary...",### Installation- **🎨 Frontend UI**: http://localhost:8080

      "url": "https://...",

      "category": "Technology",- **🔌 Backend API**: http://localhost:8003

      "published": "2025-10-23T10:00:00",

      "source": "TechCrunch",#### Automated Setup (Recommended) ⚡- **📚 API Docs**: http://localhost:8003/docs (Swagger UI)

      "score": 0.95

    }

  ]

}```bash## 📊 Model Details

```

# Clone the repository

#### `GET /search`

Semantic search across articles.git clone https://github.com/YOUR_USERNAME/news-recommender.git### Embeddings



**Query Parameters:**cd news-recommender- **Model**: `BAAI/bge-small-en-v1.5` (ONNX optimized)

- `query` (string, required): Search text

- `category` (string, optional): Filter by category- **Dimensions**: 384

- `k` (int): Number of results (default: 20)

# Run automated setup (handles everything!)- **Normalization**: L2 (for cosine similarity via inner product)

#### `GET /trending`

Get trending topics and keywords.setup.bat- **Cache**: Stored in `data/fastembed_cache/`



**Query Parameters:**```

- `hours` (int): Time window in hours (default: 72)

- `top_n` (int): Number of trends (default: 10)### Vector Index



**Response:**The setup script will:- **Type**: FAISS IndexFlatIP (exact search)

```json

{- ✅ Check Python and Node.js- **Size**: 377 articles (dynamically grows)

  "keywords": ["AI", "Climate", "Technology"],

  "bigrams": ["artificial intelligence", "climate change"]- ✅ Create virtual environment- **Similarity**: Cosine similarity (normalized vectors)

}

```- ✅ Install dependencies- **Storage**: `data/index.faiss`



#### `POST /click`- ✅ Fetch initial news articles

Track article click for personalization.

### Re-ranking (Optional)

**Body:**

```json### Starting the Application- **Model**: XGBoost gradient boosting

{

  "url": "article_url",- **Features**: [cosine_sim, recency_hours, category_overlap]

  "session_id": "user_session_id"

}```bash- **Training**: MIND dataset (Microsoft News Dataset)

```

start.bat- **Metrics**: NDCG@10, AUC

#### `GET /categories`

Get list of available categories.```- **Fallback**: Uses heuristic scores if model not trained



**Response:**

```json

{This opens:### Categories

  "categories": ["Technology", "Politics", "Sports", ...]

}- 🌐 Frontend: **http://localhost:8080**Politics, Business, Technology, Science, Health, Sports, Entertainment, World, India, AI, Climate, Education

```

- 🔗 Backend: **http://localhost:8003**

**Full API Documentation:** http://localhost:8003/docs (Swagger UI)

- 📚 API Docs: **http://localhost:8003/docs**## 📁 Project Structure

---



## 🛠️ Technologies

## 📖 Usage```

### Backend

- **FastAPI** - Modern Python web frameworknews-recommender/

- **PyTorch** - Deep learning framework for NRMS

- **XGBoost** - Gradient boosting for ranking### For Users├── src/                          # Python backend

- **FAISS** - Facebook AI Similarity Search

- **FastEmbed** - Efficient embedding generation (BGE)│   ├── api.py                    # FastAPI endpoints

- **Pandas** - Data manipulation

- **Feedparser** - RSS feed parsing1. **Select Interests**: Choose topics in onboarding modal│   ├── server.py                 # CORS + background scheduler

- **scikit-learn** - ML utilities

2. **Browse Feed**: View personalized "For You" articles│   ├── recommend.py              # ML recommendation engine

### Frontend

- **React 18** - UI library3. **Search**: Use semantic search with filters│   ├── ingest_rss.py             # RSS crawler + embeddings

- **TypeScript** - Type-safe JavaScript

- **Vite** - Fast build tool4. **Trending**: Check hot topics│   ├── topics.py                 # Category taxonomy + seed vectors

- **Tailwind CSS** - Utility-first CSS

- **shadcn/ui** - Beautiful UI components│   ├── trending.py               # Keyword extraction

- **TanStack Query** - Data fetching

- **Lucide Icons** - Icon library### For Developers│   ├── eval_mind.py              # XGBoost training pipeline



### Machine Learning│   ├── config_feeds.py           # RSS feed URLs

- **NRMS** - Neural News Recommendation (EMNLP 2019)

- **BGE** - BAAI General Embeddings#### Run Tests│   └── utils.py                  # Helpers (hash, date parsing)

- **Word2Vec** - Word embeddings

- **TF-IDF** - Term frequency analysis```bash├── nexusnews-ai-main/            # React frontend

- **Cosine Similarity** - Vector similarity metric

.venv\Scripts\activate│   ├── src/

---

python tests/test_complete_system.py│   │   ├── components/           # UI components (shadcn/ui)

## 🧪 Testing

python tests/evaluate_all.py│   │   ├── lib/

### Run Tests

```│   │   │   └── api.ts            # API client (TypeScript)

```bash

# Activate virtual environment│   │   ├── pages/

.venv\Scripts\activate

#### Train Models│   │   │   └── Index.tsx         # Main page

# Run all tests

python tests/test_complete_system.py```bash│   │   └── App.tsx



# Evaluate modelspython scripts/train_nrms.py --epochs 10│   ├── .env                      # API URL config

python tests/evaluate_all.py

```│   └── package.json

# Health check

python tests/health_check.py├── data/



# API tests#### API Endpoints│   ├── index.faiss               # Vector index

python tests/test_api.py

```│   ├── meta.csv                  # Article metadata



### Train Models- `GET /recommend` - Personalized recommendations│   ├── user_profiles.json        # Click tracking



```bash- `GET /search` - Semantic search│   └── fastembed_cache/          # Model cache

# Train NRMS model (requires MIND dataset)

python scripts/train_nrms.py --epochs 10 --batch_size 64- `GET /trending` - Trending topics├── requirements.txt              # Python dependencies



# Setup XGBoost- `POST /click` - Track clicks├── start.ps1                     # Quick startup script

python scripts/setup_xgboost.py

└── README.md                     # This file

# Rebuild FAISS index

python scripts/rebuild_index.pyFull docs: http://localhost:8003/docs```

```



---

## 🛠️ Technologies## 🔧 API Endpoints

## 📚 Documentation



Comprehensive guides in `docs/` folder:

### Backend### `GET /recommend`

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started quickly

- **[Testing Guide](docs/TESTING_GUIDE.md)** - Run tests and evaluate- **FastAPI** - Web frameworkGet personalized article recommendations

- **[Training Guide](docs/COLAB_TRAINING_GUIDE.md)** - Train models on Google Colab

- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions- **PyTorch** - Deep learning (NRMS)- **Query Params**:



---- **XGBoost** - Gradient boosting  - `query`: Search text (default: "news")



## 🐛 Troubleshooting- **FAISS** - Vector search  - `categories[]`: Filter by categories (multiple)



### Common Issues- **FastEmbed** - BGE embeddings  - `k`: Number of results (default: 20)



**Port already in use:**  - `session_id`: User session for personalization

```bash

# Find process using port 8003### Frontend- **Returns**: `{ items: Article[] }`

netstat -ano | findstr :8003

# Kill the process- **React 18** - UI library

taskkill /F /PID <PID>

```- **TypeScript** - Type safety### `POST /click`



**No articles showing:**- **Vite** - Build toolTrack article click for personalization

```bash

# Re-run data ingestion- **shadcn/ui** - Components- **Body**: `url`, `session_id`

python -m src.ingest_rss

```- **Tailwind CSS** - Styling- **Returns**: `{ status: "ok", clicks: number }`



**Import errors:**

```bash

# Ensure virtual environment is activated## 📚 Documentation### `GET /trending`

.venv\Scripts\activate

# Reinstall dependenciesGet trending keywords

pip install -r requirements.txt

```See `docs/` folder:- **Query Params**:



**Frontend won't start:**- [Quick Start Guide](docs/QUICK_START.md)  - `hours`: Time window (default: 72)

```bash

cd frontend- [Testing Guide](docs/TESTING_GUIDE.md)  - `top_n`: Number of keywords (default: 10)

# Delete node_modules and reinstall

Remove-Item -Recurse -Force node_modules- [Training Guide](docs/COLAB_TRAINING_GUIDE.md)- **Returns**: `{ trends: [{term, count}] }`

npm install

```- [Troubleshooting](docs/TROUBLESHOOTING.md)



---- [Project Report](docs/Report/)### `GET /debug/info`



## 📊 Performance MetricsSystem diagnostics



- **Response Time**: < 500ms average## 🐛 Troubleshooting- **Returns**: Metadata + index stats

- **Embedding Generation**: 50-100ms per article

- **FAISS Search**: < 10ms for top-K retrieval

- **News Sources**: 20+ RSS feeds

- **Update Frequency**: Every 30 minutes**Port in use?**## 🧪 Testing

- **Supported Categories**: 12 topics

- **Model Accuracy**: NDCG@10: 0.75+ (on MIND dataset)```bash



---netstat -ano | findstr :8003### Test Backend



## 🤝 Contributingtaskkill /F /PID <PID>```bash



This is an academic capstone project. For suggestions or issues:```# Check system status



1. Check existing issues on GitHubcurl http://localhost:8003/debug/info

2. Create detailed bug reports

3. Submit pull requests with clear descriptions**No articles?**



---```bash# Test recommendations



## 📄 Licensepython -m src.ingest_rsscurl "http://localhost:8003/recommend?query=AI&k=5"



MIT License - See [LICENSE](LICENSE) file for details.```



---# Test trending



## 👥 Author**Import errors?**curl "http://localhost:8003/trending?hours=72&top_n=8"



**Jayachandran J A** - Data Science Capstone Project 2025```bash```



---# Make sure you're in project root



## 🙏 Acknowledgmentscd news-recommender### Test Frontend



- **MIND Dataset** - Microsoft News Dataset.venv\Scripts\activate1. Open http://localhost:8080

- **NRMS Paper** - Chuhan Wu et al. (EMNLP 2019)

- **FastEmbed** - Qdrant team for efficient embeddings```2. Select interests in onboarding

- **shadcn/ui** - Beautiful React component library

- **FAISS** - Facebook AI Research team3. Verify articles load



---## 📄 License4. Try search, filters, trending clicks



## 📞 Support5. Check browser DevTools → Network tab for API calls



For issues and questions:MIT License

- **GitHub Issues**: https://github.com/Jayachandran-J-A/ai-news-recommender/issues

- **Documentation**: See `docs/` folder## 📚 Training XGBoost Model (Optional)

- **API Docs**: http://localhost:8003/docs

## 🙏 Acknowledgments

---

The system works without the trained model (uses heuristic scoring), but for better recommendations:

⭐ **Star this repository** if you found it helpful!

- MIND Dataset - Microsoft

🔗 **Repository**: https://github.com/Jayachandran-J-A/ai-news-recommender

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

