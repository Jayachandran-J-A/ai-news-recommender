# 🔧 TROUBLESHOOTING GUIDE - Start Script Issues

## Problem: start.ps1 doesn't work

### Quick Fix Options:

---

## Option 1: Use the Manual Start Script

Run this instead:
```powershell
.\start-manual.ps1
```

This will show you the commands to run in two separate terminals.

---

## Option 2: Run Commands Manually

### Step 1: Start Backend (Terminal 1)
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.server:app --port 8003
```

Wait until you see:
```
INFO:     Uvicorn running on http://127.0.0.1:8003 (Press CTRL+C to quit)
```

### Step 2: Start Frontend (Terminal 2 - New Window)
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender\nexusnews-ai-main"
npm run dev
```

Wait until you see:
```
  ➜  Local:   http://localhost:8080/
```

### Step 3: Open Browser
Open: http://localhost:8080

---

## Common Issues & Solutions

### Issue 1: "Virtual environment not found"

**Solution:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Issue 2: "Node modules not found"

**Solution:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender\nexusnews-ai-main"
npm install
```

---

### Issue 3: "Port 8003 already in use"

**Solution - Kill the process:**
```powershell
# Find process on port 8003
netstat -ano | findstr :8003

# Kill it (replace XXXX with PID from above)
taskkill /PID XXXX /F
```

Or use the script:
```powershell
Get-NetTCPConnection -LocalPort 8003 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

### Issue 4: "Port 8080 already in use"

**Solution - Kill the process:**
```powershell
# Find process on port 8080
netstat -ano | findstr :8080

# Kill it (replace XXXX with PID from above)
taskkill /PID XXXX /F
```

Or use the script:
```powershell
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

### Issue 5: "No articles found in data/meta.csv"

**Solution - Fetch articles:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"
.\.venv\Scripts\Activate.ps1
python -m src.ingest_rss
```

Wait 2-3 minutes for articles to be fetched.

---

### Issue 6: "Python not found"

**Solution:**
Check Python installation:
```powershell
python --version
```

If not found, download from: https://www.python.org/downloads/

---

### Issue 7: "npm not found"

**Solution:**
Check Node.js installation:
```powershell
node --version
npm --version
```

If not found, download from: https://nodejs.org/

---

### Issue 8: "Scripts are disabled on this system"

**Error:**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution - Enable scripts (Run PowerShell as Administrator):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again:
```powershell
.\start.ps1
```

---

### Issue 9: Backend starts but shows errors

**Check backend logs in the terminal window.**

Common errors:

**"ModuleNotFoundError":**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**"RuntimeError: Metadata not found":**
```powershell
python -m src.ingest_rss
```

---

### Issue 10: Frontend starts but shows blank page

**Check browser console (F12).**

Common issues:

**"Failed to fetch":**
- Backend not running on port 8003
- Check: http://localhost:8003/

**"CORS error":**
- Backend should have CORS enabled (already configured)
- Restart backend

---

## Testing if Everything Works

### Test Backend:
```powershell
# Should return: {"message":"News Recommender API"}
Invoke-WebRequest -Uri "http://localhost:8003/" -UseBasicParsing
```

### Test Frontend:
Open browser: http://localhost:8080
- Should see onboarding modal
- Select interests
- Should see article cards

---

## Still Not Working?

### Complete Fresh Start:

1. **Close all terminals and browser tabs**

2. **Clean restart:**
```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

# Kill any running processes
Get-Process | Where-Object {$_.Path -like "*uvicorn*"} | Stop-Process -Force
Get-Process | Where-Object {$_.Path -like "*node*"} | Stop-Process -Force

# Start fresh
.\start.ps1
```

3. **If start.ps1 still fails, use manual method:**
   - Open PowerShell Terminal 1 → Run backend commands
   - Open PowerShell Terminal 2 → Run frontend commands
   - Open browser → http://localhost:8080

---

## Quick Health Check

Run this to verify everything is set up:
```powershell
python health_check.py
```

This will tell you exactly what's wrong.

---

## Need More Help?

Check these files:
- `README.md` - Full setup instructions
- `TESTING_GUIDE.md` - Testing information
- `PRESENTATION_GUIDE.md` - Presentation help

Or check the terminal output for specific error messages.
