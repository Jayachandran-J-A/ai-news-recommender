#  AI-Powered News Recommender System

> **Data Science Capstone Project** - Intelligent news recommendation using Deep Learning, Machine Learning, and Vector Search

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete AI-powered news recommendation system featuring **NRMS**, **XGBoost ensemble**, and **FAISS vector search**.

---

##  Features

###  AI & Machine Learning
- **NRMS**: Neural news recommendation with multi-head self-attention
- **XGBoost**: Gradient boosting for re-ranking
- **BGE Embeddings**: 384-dim semantic vectors
- **FAISS**: Fast vector similarity search
- **Ensemble**: 60% NRMS + 40% XGBoost

###  News Processing
- **20+ Sources**: BBC, Reuters, CNN, TechCrunch, etc.
- **12 Categories**: Auto-classification
- **Trending**: TF-IDF keyword extraction
- **Auto-Update**: Every 30 minutes

###  User Features
- Personalized recommendations
- Semantic search
- Interest onboarding
- Modern React UI

---

##  Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Git

### Installation

```powershell
# Clone repository
git clone https://github.com/Jayachandran-J-A/ai-news-recommender.git
cd ai-news-recommender

# Run automated setup
setup.bat

# Start application
start.bat
```

Access at:
- Frontend: http://localhost:8080
- Backend API: http://localhost:8003
- API Docs: http://localhost:8003/docs

---

##  Project Structure

```
ai-news-recommender/
 src/              # Backend Python code
 frontend/         # React TypeScript app
 scripts/          # Utility scripts
 tests/            # Test files
 docs/             # Documentation
 data/             # Data files
 models/           # Trained models
```

---

##  Technologies

**Backend**: FastAPI, PyTorch, XGBoost, FAISS, FastEmbed

**Frontend**: React, TypeScript, Tailwind CSS, shadcn/ui

**ML**: NRMS (EMNLP 2019), BGE embeddings, TF-IDF

---

##  Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [Testing Guide](docs/TESTING_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

##  Testing

```bash
# Activate environment
.venv\Scripts\activate

# Run tests
python tests/test_complete_system.py
python tests/evaluate_all.py
```

---

##  API Endpoints

- GET /recommend - Personalized recommendations
- GET /search - Semantic search
- GET /trending - Trending topics
- POST /click - Track user clicks

Full docs: http://localhost:8003/docs

---

##  License

MIT License

---

##  Author

**Jayachandran J A** - Data Science Capstone 2025

---

 **Star this repo!**

 https://github.com/Jayachandran-J-A/ai-news-recommender
