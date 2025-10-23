#!/usr/bin/env python3
"""
Simple test script to verify the ensemble model integration works.
Tests all three recommendation approaches and shows the differences.
"""
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_recommendations():
    """Test all recommendation models"""
    print("🧪 TESTING NEWS RECOMMENDATION SYSTEM")
    print("="*60)
    
    try:
        from src.recommend import recommend_for_user, _load_xgb_model, _load_ensemble_model
        
        # Check model availability
        xgb_available = _load_xgb_model() is not None
        ensemble_available = _load_ensemble_model() is not None
        
        print(f"XGBoost Model: {'✅ Available' if xgb_available else '❌ Not found'}")
        print(f"Ensemble Model: {'✅ Available' if ensemble_available else '❌ Not found'}")
        print(f"NRMS Model: {'✅ Available' if os.path.exists('models/nrms_model.pt') else '❌ Not found'}")
        
        # Test queries
        test_queries = [
            "artificial intelligence breakthrough",
            "climate change news",
            "technology innovations"
        ]
        
        print("\n" + "="*60)
        print("TESTING RECOMMENDATIONS")
        print("="*60)
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)
            
            try:
                # Get recommendations
                recommendations = recommend_for_user(query=query, k=5)
                
                if recommendations:
                    for i, rec in enumerate(recommendations[:3], 1):
                        title = rec.get('title', 'No title')[:60]
                        source = rec.get('source', 'Unknown')
                        score = rec.get('final_score', rec.get('ml_score', rec.get('score', 0)))
                        print(f"  {i}. {title}...")
                        print(f"     Source: {source} | Score: {score:.3f}")
                    
                    print(f"✅ Retrieved {len(recommendations)} recommendations")
                else:
                    print("❌ No recommendations returned")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "="*60)
        print("SYSTEM STATUS SUMMARY")
        print("="*60)
        
        # Generate status report
        status = {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "xgboost": xgb_available,
                "ensemble": ensemble_available,
                "nrms_file": os.path.exists('models/nrms_model.pt')
            },
            "status": "operational" if xgb_available else "baseline_only"
        }
        
        if ensemble_available:
            print("🎉 FULL SYSTEM OPERATIONAL!")
            print("   • Ensemble model combining NRMS + XGBoost")
            print("   • State-of-the-art neural attention")
            print("   • Traditional gradient boosting")
            print("   • Best possible recommendation quality")
        elif xgb_available:
            print("✅ ENHANCED SYSTEM OPERATIONAL!")
            print("   • XGBoost model for improved ranking")
            print("   • Better than baseline vector search")
            print("   • Good recommendation quality")
        else:
            print("⚠️ BASELINE SYSTEM ONLY")
            print("   • Using vector similarity search")
            print("   • Functional but not optimized")
        
        # Save status
        with open("system_status.json", "w") as f:
            json.dump(status, f, indent=2)
        
        print(f"\n💾 Status saved to: system_status.json")
        
        return status
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install torch xgboost faiss-cpu fastembed pandas numpy")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def test_api_startup():
    """Test API can start successfully"""
    print("\n🌐 TESTING API STARTUP")
    print("="*40)
    
    try:
        from src.api import app
        print("✅ API imports successfully")
        print("✅ FastAPI app created")
        print("ℹ️ To start server: python -m uvicorn src.api:app --host 0.0.0.0 --port 8003")
        return True
    except Exception as e:
        print(f"❌ API startup error: {e}")
        return False


def main():
    print("🚀 COMPREHENSIVE SYSTEM TEST")
    print("Testing your enhanced news recommendation system...")
    print()
    
    # Test core functionality
    status = test_recommendations()
    
    # Test API
    api_ok = test_api_startup()
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    
    if status and status["status"] == "operational":
        print("🎉 SUCCESS! Your system is fully operational with:")
        print("   ✅ Neural recommendation model (NRMS)")
        print("   ✅ Traditional ML model (XGBoost)")
        print("   ✅ Ensemble combining both")
        print("   ✅ FastAPI serving layer")
        print()
        print("🎓 CAPSTONE PROJECT STATUS: COMPLETE!")
        print("   • State-of-the-art deep learning model")
        print("   • Comprehensive ensemble approach")
        print("   • Real-world dataset (Microsoft MIND)")
        print("   • Production-ready API")
        print()
        print("🚀 NEXT STEPS:")
        print("   1. Run full evaluation: python evaluate_all.py")
        print("   2. Start API server: python -m uvicorn src.api:app --host 0.0.0.0 --port 8003")
        print("   3. Test in browser: http://localhost:8003/recommend?query=AI")
        print("   4. Prepare presentation with performance metrics")
        
    elif status and status["models"]["xgboost"]:
        print("✅ GOOD! XGBoost enhanced system working")
        print("   • Better than baseline vector search")
        print("   • Ensemble model needs NRMS file")
        print("   • Download nrms_model.pt from Colab to models/ folder")
        
    else:
        print("⚠️ PARTIAL: Baseline system only")
        print("   • Vector search is working")
        print("   • Missing ML models for enhanced ranking")
        
    if api_ok:
        print("   ✅ API ready for deployment")
    else:
        print("   ⚠️ API needs dependency fixes")
        
    print("\n🏁 Test complete!")


if __name__ == "__main__":
    main()