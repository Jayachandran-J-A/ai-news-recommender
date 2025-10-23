@echo off
REM ============================================
REM  AI News Recommender - Automated Setup
REM  Handles all dependencies and environment setup
REM ============================================

echo.
echo ========================================
echo   AI NEWS RECOMMENDER - SETUP
echo ========================================
echo.

REM Check Python installation
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo [OK] Python found!

REM Check Node.js installation
echo.
echo [2/7] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH!
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)
node --version
npm --version
echo [OK] Node.js and npm found!

REM Create virtual environment if it doesn't exist
echo.
echo [3/7] Setting up Python virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created!
) else (
    echo [OK] Virtual environment already exists!
)

REM Activate virtual environment and install Python dependencies
echo.
echo [4/7] Installing Python dependencies...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies!
    echo Please check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo [OK] Python dependencies installed!

REM Install frontend dependencies
echo.
echo [5/7] Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    echo Installing npm packages (this may take a few minutes)...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies!
        cd ..
        pause
        exit /b 1
    )
    echo [OK] Frontend dependencies installed!
) else (
    echo [OK] Frontend dependencies already installed!
)
cd ..

REM Create data directory structure
echo.
echo [6/7] Creating data directories...
if not exist "data" mkdir data
if not exist "data\mind" mkdir data\mind
if not exist "models" mkdir models
echo [OK] Data directories ready!

REM Initial data ingestion (fetch news articles)
echo.
echo [7/7] Fetching initial news articles...
echo This will take 2-3 minutes to download and process articles...
python -m src.ingest_rss
if errorlevel 1 (
    echo [WARNING] Initial data ingestion had some errors.
    echo You can run it again with: python -m src.ingest_rss
) else (
    echo [OK] Initial data loaded successfully!
)

echo.
echo ========================================
echo   ✅ SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo   1. Run "start.bat" to launch the application
echo   2. Open http://localhost:8080 in your browser
echo   3. Check API docs at http://localhost:8003/docs
echo.
echo For more information, see README.md
echo.
pause
