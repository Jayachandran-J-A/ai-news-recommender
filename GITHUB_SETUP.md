# 📦 GitHub Setup Guide

## Uploading Your Project to GitHub

Follow these steps to upload your news recommender system to GitHub.

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in repository details:
   - **Repository name**: `news-recommender` or `ai-news-recommender`
   - **Description**: "AI-Powered News Recommendation System - Data Science Capstone Project"
   - **Visibility**: Choose **Public** (recommended for portfolio) or **Private**
   - **DO NOT** check "Initialize with README" (we already have one)
4. Click **"Create repository"**

### Step 2: Initialize Git (If Not Already Done)

Open PowerShell in your project directory and run:

```powershell
cd "c:\Users\jayac\Projects\Data-Science-CapstoneProject\Idea 2\news-recommender"

# Initialize git repository
git init

# Check git status
git status
```

### Step 3: Add Files to Git

```powershell
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Commit with a message
git commit -m "Initial commit: AI News Recommender System with NRMS + XGBoost"
```

### Step 4: Connect to GitHub

Copy the commands from GitHub (they'll look like this, but with your username):

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/news-recommender.git

# Verify remote
git remote -v

# Set branch name to main
git branch -M main
```

### Step 5: Push to GitHub

```powershell
# Push code to GitHub
git push -u origin main
```

**If prompted for credentials:**
- Use your GitHub username
- For password, use a **Personal Access Token** (not your GitHub password)
  - Create one at: https://github.com/settings/tokens
  - Check "repo" permissions
  - Copy the token and use it as password

### Step 6: Verify Upload

1. Go to your GitHub repository URL
2. Refresh the page
3. You should see all your files!

## 📋 What Gets Uploaded (and What Doesn't)

### ✅ Included (Version Controlled)
- Source code (`src/`, `frontend/src/`)
- Documentation (`docs/`, `README.md`)
- Configuration files (`requirements.txt`, `package.json`)
- Scripts (`scripts/`, `tests/`)
- Startup files (`setup.bat`, `start.bat`)

### ❌ Excluded (in .gitignore)
- Virtual environment (`.venv/`)
- Node modules (`frontend/node_modules/`)
- Cache files (`__pycache__/`, `.cache/`)
- Data files (`data/`, `Dataset-archive/`)
- Trained models (`models/*.pt`)
- User data (`user_profiles.json`)

**Why?** These are large files that can be regenerated. Users will run `setup.bat` to create them.

## 🔄 Future Updates

After making changes:

```powershell
# Check what changed
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Added new feature: trending topics visualization"

# Push to GitHub
git push
```

## 🌟 Making Your Repo Look Professional

### 1. Add Topics/Tags

On your GitHub repo page:
- Click the **⚙️ (Settings icon)** next to "About"
- Add topics: `machine-learning`, `deep-learning`, `news-recommendation`, `fastapi`, `react`, `pytorch`, `xgboost`, `capstone-project`

### 2. Add Repository Description

- Edit description: "AI-powered news recommender with NRMS, XGBoost ensemble, and FAISS vector search"
- Add website: Your deployed URL (if you deploy)

### 3. Pin Repository

- Go to your GitHub profile
- Click **"Customize your pins"**
- Select this repository to showcase it

### 4. Create Releases

When your project is complete:

```powershell
# Tag the release
git tag -a v1.0.0 -m "Version 1.0.0 - Complete Capstone Project"
git push origin v1.0.0
```

Then create a GitHub Release with:
- Release notes
- Highlights/features
- Setup instructions

## 🎓 For Capstone Submission

### Include in Your Report

Add your GitHub URL:
```
GitHub Repository: https://github.com/YOUR_USERNAME/news-recommender
```

### Create a Good README

Your README should have:
- ✅ Project title and description
- ✅ Features and architecture
- ✅ Installation instructions
- ✅ Usage examples
- ✅ Screenshots (add to `docs/screenshots/`)
- ✅ Technologies used
- ✅ Acknowledgments

### Optional: Add Screenshots

```powershell
# Create screenshots folder
mkdir docs\screenshots

# Add screenshots of your app
# Then commit them
git add docs/screenshots/
git commit -m "Added application screenshots"
git push
```

Update README.md to include images:
```markdown
## 📸 Screenshots

![Home Page](docs/screenshots/home.png)
![Search Results](docs/screenshots/search.png)
![Recommendations](docs/screenshots/recommendations.png)
```

## 🚀 Deployment (Optional)

### Backend Deployment
- **Railway**: https://railway.app (Free tier)
- **Render**: https://render.com (Free tier)
- **Heroku**: https://heroku.com

### Frontend Deployment
- **Vercel**: https://vercel.com (Free)
- **Netlify**: https://netlify.com (Free)
- **GitHub Pages**: For static builds

### Full Stack Deployment
- **Fly.io**: https://fly.io
- **DigitalOcean App Platform**: https://www.digitalocean.com/products/app-platform

## 🤝 Collaboration (If Working in Team)

### Clone Your Repo

Team members can clone:
```powershell
git clone https://github.com/YOUR_USERNAME/news-recommender.git
cd news-recommender
setup.bat
```

### Branching Strategy

```powershell
# Create feature branch
git checkout -b feature/new-feature

# Make changes, commit
git add .
git commit -m "Implemented new feature"

# Push branch
git push origin feature/new-feature

# Create Pull Request on GitHub
```

## 🔒 Security Notes

### ⚠️ Never Commit

- API keys
- Passwords
- Database credentials
- Personal information

### ✅ Use Environment Variables

Create `.env` file (already in .gitignore):
```env
API_KEY=your_api_key_here
DATABASE_URL=your_db_url
```

Load in code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
```

## 📞 Support

If you encounter issues:

1. **Authentication Failed**: Use Personal Access Token, not password
2. **Large Files Error**: Check if large files are in .gitignore
3. **Remote Already Exists**: Use `git remote set-url origin https://...`
4. **Merge Conflicts**: Pull before push: `git pull origin main`

## ✅ Checklist Before Uploading

- [ ] Code is clean and commented
- [ ] README.md is comprehensive
- [ ] .gitignore is properly configured
- [ ] Large files excluded (models, data, node_modules)
- [ ] Tests are passing
- [ ] Documentation is complete
- [ ] setup.bat and start.bat work correctly
- [ ] Repository description added
- [ ] Topics/tags added

---

**Good luck with your GitHub upload! 🎉**

For help, contact: your.email@example.com
