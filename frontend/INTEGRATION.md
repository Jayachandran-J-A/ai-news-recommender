# NexusNews AI - UI Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
cd nexusnews-ai-main
npm install
```

### 2. Start Backend (Terminal 1)
```bash
cd ..
python -m uvicorn src.server:app --port 8003
```

### 3. Start Frontend (Terminal 2)
```bash
cd nexusnews-ai-main
npm run dev
```

Visit **http://localhost:5173**

## API Integration Complete

All components now connect to your FastAPI backend at `localhost:8003`.

### Endpoints Used
- `GET /recommend` - Fetch personalized articles
- `POST /click` - Track article clicks
- `GET /trending` - Get trending keywords
- `GET /debug/info` - System stats

## Features Working
✅ Real-time article recommendations
✅ Session-based personalization  
✅ Click tracking
✅ Category filtering
✅ Search
✅ Trending topics
✅ Auto-refresh

See full README.md in this directory for details.
