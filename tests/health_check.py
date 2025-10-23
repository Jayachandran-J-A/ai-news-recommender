"""
Quick Health Check - Verify your recommendation system is working

Run this first to ensure everything is set up correctly.
"""
import os
import sys

def check_files():
    """Check if required files exist"""
    print("📁 Checking Files...")
    
    checks = {
        "data/meta.csv": "Article metadata",
        "data/index.faiss": "Vector index",
        "src/recommend_advanced.py": "Recommendation engine",
        "src/config_feeds.py": "RSS feed configuration"
    }
    
    all_good = True
    for filepath, description in checks.items():
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        print(f"  {status} {description}: {filepath}")
        if not exists:
            all_good = False
    
    return all_good

def check_data():
    """Check if we have articles"""
    print("\n📊 Checking Data...")
    
    try:
        import pandas as pd
        meta = pd.read_csv("data/meta.csv")
        article_count = len(meta)
        
        print(f"  ✅ Found {article_count} articles")
        
        if article_count < 100:
            print(f"  ⚠️  Warning: Only {article_count} articles. Consider running refresh.")
            return True
        elif article_count < 500:
            print(f"  ℹ️  Info: {article_count} articles is decent. More is better!")
            return True
        else:
            print(f"  ✅ Great! {article_count} articles is excellent.")
            return True
            
    except Exception as e:
        print(f"  ❌ Error reading metadata: {e}")
        return False

def check_model():
    """Check if recommendation engine works"""
    print("\n🤖 Checking Recommendation Engine...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.recommend_advanced import recommend_for_user
        
        print("  ✅ Engine loaded successfully")
        
        # Try a simple recommendation
        print("  🧪 Testing recommendation...")
        results = recommend_for_user(
            session_id="health_check",
            user_interests=["Technology"],
            search_query="",
            selected_categories=[],
            top_k=3
        )
        
        if results:
            print(f"  ✅ Generated {len(results)} recommendations")
            print(f"  ✅ Top score: {results[0].get('final_score', 0):.1f}%")
            return True
        else:
            print(f"  ⚠️  No recommendations returned. Check if articles exist.")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_sources():
    """Check RSS feed configuration"""
    print("\n📡 Checking RSS Feeds...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.config_feeds import RSS_FEEDS
        
        feed_count = len(RSS_FEEDS)
        print(f"  ✅ Configured {feed_count} RSS feeds")
        
        if feed_count < 20:
            print(f"  ⚠️  Only {feed_count} feeds. Consider adding more for diversity.")
        elif feed_count < 50:
            print(f"  ℹ️  {feed_count} feeds is good!")
        else:
            print(f"  ✅ Excellent! {feed_count} feeds for great coverage.")
        
        # Count categories
        categories = set()
        for cat, url in RSS_FEEDS:
            categories.add(cat)
        
        print(f"  ✅ Covering {len(categories)} categories")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_backend():
    """Check if backend API is running"""
    print("\n🌐 Checking Backend API...")
    
    try:
        import requests
        response = requests.get("http://localhost:8003/", timeout=2)
        
        if response.status_code == 200:
            print("  ✅ Backend is running on port 8003")
            return True
        else:
            print(f"  ⚠️  Backend responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ⚠️  Backend not running. Start with: python -m src.server")
        return False
    except Exception as e:
        print(f"  ℹ️  Could not check backend: {e}")
        return False

def main():
    """Run all health checks"""
    print("\n" + "🏥"*40)
    print("  RECOMMENDATION SYSTEM HEALTH CHECK")
    print("🏥"*40 + "\n")
    
    results = {
        "Files": check_files(),
        "Data": check_data(),
        "Model": check_model(),
        "Sources": check_sources(),
        "Backend": check_backend()
    }
    
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    print(f"\n  Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 Everything looks good! Your system is ready.")
        print("\n  Next steps:")
        print("     1. Run: python evaluate_recommendations.py")
        print("     2. Or: python test_interactive.py")
        print("     3. Or: Open UI at http://localhost:8080")
    elif passed >= total - 1:
        print("\n  ✅ System is mostly ready. Minor issues detected.")
        print("     Check warnings above and fix if needed.")
    else:
        print("\n  ⚠️  System has issues. Please fix errors above.")
        
        if not results["Files"]:
            print("\n  💡 Missing files? Run: python -m src.ingest_rss")
        
        if not results["Model"]:
            print("  💡 Model issues? Check Python dependencies: pip install -r requirements.txt")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
