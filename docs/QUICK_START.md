# 🚀 QUICK START REFERENCE

## Three Ways to Start Your Project:

### Method 1: PowerShell Script (Recommended)
```powershell
.\start.ps1
```
- ✅ Checks prerequisites automatically
- ✅ Kills conflicting processes
- ✅ Opens browser automatically
- ✅ Shows helpful error messages

---

### Method 2: Batch File (Alternative)
```cmd
start.bat
```
- ✅ Works without PowerShell execution policy issues
- ✅ Simple Windows Command Prompt
- ✅ Opens servers in separate windows

---

### Method 3: Manual (If scripts fail)

**Terminal 1 - Backend:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.server:app --port 8003
```

**Terminal 2 - Frontend:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender\nexusnews-ai-main"
npm run dev
```

**Browser:**
```
http://localhost:8080
```

---

## Common Issues:

### "Scripts are disabled"
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Port already in use"
```powershell
# Kill process on port 8003
Get-NetTCPConnection -LocalPort 8003 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Kill process on port 8080
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### "No articles found"
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.ingest_rss
```

---

## URLs to Remember:

| Service | URL |
|---------|-----|
| **Frontend UI** | http://localhost:8080 |
| **Backend API** | http://localhost:8003 |
| **API Docs** | http://localhost:8003/docs |
| **Trending** | http://localhost:8003/trending |

---

## Stopping the Servers:

### If started with script:
- Close the two PowerShell/CMD windows that opened

### If started manually:
- Press `Ctrl+C` in each terminal

---

## Health Check:

Before starting, verify everything is ready:
```powershell
python health_check.py
```

---

## Getting Help:

See full troubleshooting guide:
```
TROUBLESHOOTING.md
```

See full documentation:
```
README.md
```

---

## First Time Setup:

If this is your first time running the project:

1. **Create virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Install frontend dependencies:**
```powershell
cd nexusnews-ai-main
npm install
cd ..
```

3. **Fetch articles:**
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.ingest_rss
```

4. **Start servers:**
```powershell
.\start.ps1
```

---

## Testing:

Run comprehensive tests:
```powershell
.\.venv\Scripts\Activate.ps1
python evaluate_recommendations.py
```

Interactive testing:
```powershell
python test_interactive.py
```

---

**Need more help? Check TROUBLESHOOTING.md**
