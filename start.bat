@echo off
REM ============================================
REM  AI News Recommender - Smart Startup Script
REM  Checks dependencies and starts servers
REM ============================================

echo.
echo ========================================
echo   AI NEWS RECOMMENDER SYSTEM
echo   Enhanced with NRMS + XGBoost
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Check if frontend dependencies exist
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not found!
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Quick dependency check
echo [INFO] Verifying Python dependencies...
python -c "import fastapi, uvicorn, torch, xgboost, faiss, fastembed" 2>nul
if errorlevel 1 (
    echo [WARNING] Some Python dependencies are missing!
    echo Installing/updating dependencies...
    pip install -r requirements.txt -q
)

REM Check if data exists
if not exist "data\meta.csv" (
    echo.
    echo [WARNING] No data found! Running initial data ingestion...
    echo This will take 2-3 minutes...
    python -m src.ingest_rss
    echo.
)

echo [1/3] Starting Backend API Server...
echo        Loading XGBoost + NRMS ensemble models...
start "🚀 Backend API (Port 8003)" cmd /k "cd /d %~dp0 && .venv\Scripts\activate.bat && python -m uvicorn src.api:app --host 0.0.0.0 --port 8003 --reload"

echo [2/3] Waiting for backend to initialize...
echo        This may take 10-15 seconds to load AI models...
timeout /t 12 /nobreak >nul

echo [3/3] Starting React Frontend...
start "🎨 Frontend UI (Port 8080)" cmd /k "cd /d %~dp0frontend && npm run dev -- --host 0.0.0.0 --port 8080"

echo.
echo Waiting for frontend to build and start...
timeout /t 10 /nobreak >nul

echo.
echo ========================================
echo   🎉 AI NEWS RECOMMENDER READY!
echo ========================================
echo.
echo 🌐 Frontend:     http://localhost:8080
echo 🔗 Backend API:  http://localhost:8003
echo 📚 API Docs:     http://localhost:8003/docs
echo.
echo ✅ Models Loaded:
echo   • XGBoost Enhanced Ranking
echo   • NRMS Neural Attention
echo   • FAISS Vector Search
echo   • BGE Embeddings
echo.
echo � Features:
echo   • Smart personalized recommendations
echo   • Real-time news from 20+ sources
echo   • Semantic search with AI
echo   • Trending topics extraction
echo.
echo Press Ctrl+C in the terminal windows to stop the servers.
echo.

REM Open browser automatically
timeout /t 3 /nobreak >nul
start http://localhost:8080

echo Browser opened! Check the terminal windows for logs.
echo.
pause
