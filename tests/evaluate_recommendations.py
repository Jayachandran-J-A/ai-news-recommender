"""
Comprehensive Recommendation Quality Evaluation Script

This script helps you test and measure how good your recommendations are.
It tests multiple scenarios and provides detailed metrics.
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommend_advanced import recommend_for_user
from src.config_feeds import RSS_FEEDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROFILES_FP = os.path.join(DATA_DIR, "user_profiles.json")
META_CSV = os.path.join(DATA_DIR, "meta.csv")


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_user_profiles() -> Dict:
    """Load existing user profiles"""
    if os.path.exists(PROFILES_FP):
        with open(PROFILES_FP, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_metadata() -> pd.DataFrame:
    """Load article metadata"""
    if os.path.exists(META_CSV):
        return pd.read_csv(META_CSV)
    return pd.DataFrame()


def test_category_relevance():
    """Test 1: Category Matching - Do recommendations match requested categories?"""
    print_section("TEST 1: Category Relevance")
    
    test_cases = [
        {
            "name": "Technology Enthusiast",
            "interests": ["Technology", "AI", "Science"],
            "session_id": "test_tech_user",
            "expected_categories": ["Technology", "AI", "Science"]
        },
        {
            "name": "Sports Fan",
            "interests": ["Sports", "Cricket"],
            "session_id": "test_sports_user",
            "expected_categories": ["Sports", "Cricket"]
        },
        {
            "name": "Business Professional",
            "interests": ["Business", "Finance", "Economy"],
            "session_id": "test_business_user",
            "expected_categories": ["Business", "Finance", "Economy"]
        },
        {
            "name": "News Junkie (All Topics)",
            "interests": ["International", "Politics", "World"],
            "session_id": "test_news_user",
            "expected_categories": ["International", "Politics", "World"]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📋 Testing: {case['name']}")
        print(f"   Interests: {', '.join(case['interests'])}")
        
        # Get recommendations
        recommendations = recommend_for_user(
            session_id=case['session_id'],
            user_interests=case['interests'],
            search_query="",
            selected_categories=[],
            top_k=10
        )
        
        if not recommendations:
            print(f"   ❌ NO RECOMMENDATIONS RETURNED")
            results.append({
                "test": case['name'],
                "status": "FAILED",
                "avg_score": 0,
                "category_match_rate": 0
            })
            continue
        
        # Analyze results
        scores = [r.get('final_score', 0) for r in recommendations]
        avg_score = sum(scores) / len(scores)
        
        # Check category matching
        category_matches = 0
        for rec in recommendations:
            rec_categories = rec.get('categories', '').lower()
            for interest in case['interests']:
                if interest.lower() in rec_categories:
                    category_matches += 1
                    break
        
        match_rate = (category_matches / len(recommendations)) * 100
        
        print(f"   ✓ Got {len(recommendations)} recommendations")
        print(f"   ✓ Average Match Score: {avg_score:.1f}%")
        print(f"   ✓ Category Match Rate: {match_rate:.1f}% ({category_matches}/{len(recommendations)})")
        
        # Show top 3 recommendations
        print(f"\n   Top 3 Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. [{rec.get('final_score', 0):.1f}%] {rec.get('title', 'N/A')[:80]}")
            print(f"         Categories: {rec.get('categories', 'N/A')}")
        
        results.append({
            "test": case['name'],
            "status": "PASSED" if match_rate >= 70 else "NEEDS IMPROVEMENT",
            "avg_score": avg_score,
            "category_match_rate": match_rate,
            "total_recs": len(recommendations)
        })
    
    return results


def test_personalization():
    """Test 2: Personalization - Do recommendations improve with user behavior?"""
    print_section("TEST 2: Personalization (User Behavior Learning)")
    
    # Load user profiles
    profiles = load_user_profiles()
    
    if not profiles:
        print("❌ No user profiles found. Personalization requires user clicks.")
        print("   Tip: Use the app, click on articles you like, then run this test again.")
        return []
    
    results = []
    
    for session_id, profile in list(profiles.items())[:5]:  # Test first 5 users
        clicks = profile.get('clicks', [])
        interests = profile.get('interests', [])
        
        if not clicks:
            continue
        
        print(f"\n👤 User: {session_id}")
        print(f"   Interests: {', '.join(interests) if interests else 'None set'}")
        print(f"   Click History: {len(clicks)} articles")
        
        # Show clicked categories
        clicked_categories = [c.get('categories', '') for c in clicks]
        category_counter = Counter([cat for cats in clicked_categories for cat in cats.split(',') if cat.strip()])
        
        if category_counter:
            print(f"   Top Clicked Categories: {', '.join([f'{k} ({v})' for k, v in category_counter.most_common(3)])}")
        
        # Get recommendations
        recommendations = recommend_for_user(
            session_id=session_id,
            user_interests=interests,
            search_query="",
            selected_categories=[],
            top_k=10
        )
        
        if recommendations:
            scores = [r.get('final_score', 0) for r in recommendations]
            avg_score = sum(scores) / len(scores)
            
            print(f"   ✓ Recommendations: {len(recommendations)} articles")
            print(f"   ✓ Average Score: {avg_score:.1f}%")
            
            # Check if recommendations align with click history
            rec_categories = [r.get('categories', '') for r in recommendations]
            alignment = 0
            for top_cat, _ in category_counter.most_common(3):
                for rec_cat in rec_categories:
                    if top_cat.lower() in rec_cat.lower():
                        alignment += 1
                        break
            
            alignment_rate = (alignment / min(3, len(category_counter))) * 100 if category_counter else 0
            print(f"   ✓ Alignment with History: {alignment_rate:.1f}%")
            
            results.append({
                "session_id": session_id,
                "clicks": len(clicks),
                "avg_score": avg_score,
                "alignment": alignment_rate
            })
    
    return results


def test_diversity():
    """Test 3: Diversity - Are recommendations diverse enough?"""
    print_section("TEST 3: Diversity Check")
    
    # Get recommendations for a broad query
    recommendations = recommend_for_user(
        session_id="test_diversity",
        user_interests=["Technology", "Business", "Science"],
        search_query="",
        selected_categories=[],
        top_k=20
    )
    
    if not recommendations:
        print("❌ No recommendations returned")
        return {}
    
    # Analyze diversity
    sources = [r.get('source', 'Unknown') for r in recommendations]
    categories = [r.get('categories', '') for r in recommendations]
    
    source_dist = Counter(sources)
    category_list = [cat.strip() for cats in categories for cat in cats.split(',') if cat.strip()]
    category_dist = Counter(category_list)
    
    print(f"📊 Analyzed {len(recommendations)} recommendations:\n")
    
    print(f"   Source Diversity:")
    print(f"   ✓ Unique Sources: {len(source_dist)} out of {len(recommendations)}")
    for source, count in source_dist.most_common(5):
        print(f"      - {source}: {count} articles ({count/len(recommendations)*100:.1f}%)")
    
    print(f"\n   Category Diversity:")
    print(f"   ✓ Unique Categories: {len(category_dist)}")
    for cat, count in category_dist.most_common(5):
        print(f"      - {cat}: {count} articles")
    
    # Calculate diversity score (higher is better)
    source_diversity = len(source_dist) / len(recommendations) * 100
    category_diversity = len(category_dist) / len(recommendations) * 100
    
    print(f"\n   Diversity Scores:")
    print(f"   ✓ Source Diversity: {source_diversity:.1f}% (target: >50%)")
    print(f"   ✓ Category Diversity: {category_diversity:.1f}% (target: >30%)")
    
    return {
        "source_diversity": source_diversity,
        "category_diversity": category_diversity,
        "unique_sources": len(source_dist),
        "unique_categories": len(category_dist)
    }


def test_recency():
    """Test 4: Recency - Are recent articles prioritized?"""
    print_section("TEST 4: Recency Bias")
    
    recommendations = recommend_for_user(
        session_id="test_recency",
        user_interests=["Technology"],
        search_query="recent news",
        selected_categories=[],
        top_k=20
    )
    
    if not recommendations:
        print("❌ No recommendations returned")
        return {}
    
    # Parse publication dates
    now = datetime.now()
    ages = []
    
    for rec in recommendations:
        pub_str = rec.get('published', '')
        if pub_str:
            try:
                from dateutil import parser as dtparser
                pub_date = dtparser.parse(pub_str)
                age_hours = (now - pub_date).total_seconds() / 3600
                ages.append(age_hours)
            except:
                pass
    
    if ages:
        avg_age = sum(ages) / len(ages)
        articles_24h = sum(1 for age in ages if age <= 24)
        articles_48h = sum(1 for age in ages if age <= 48)
        articles_week = sum(1 for age in ages if age <= 168)
        
        print(f"📅 Article Freshness Analysis:\n")
        print(f"   ✓ Average Age: {avg_age:.1f} hours ({avg_age/24:.1f} days)")
        print(f"   ✓ Last 24 hours: {articles_24h}/{len(ages)} ({articles_24h/len(ages)*100:.1f}%)")
        print(f"   ✓ Last 48 hours: {articles_48h}/{len(ages)} ({articles_48h/len(ages)*100:.1f}%)")
        print(f"   ✓ Last 7 days: {articles_week}/{len(ages)} ({articles_week/len(ages)*100:.1f}%)")
        
        return {
            "avg_age_hours": avg_age,
            "last_24h_percent": articles_24h/len(ages)*100,
            "last_48h_percent": articles_48h/len(ages)*100
        }
    else:
        print("⚠️  Could not parse publication dates")
        return {}


def test_search_quality():
    """Test 5: Search Quality - Do search results match queries?"""
    print_section("TEST 5: Search Quality")
    
    search_queries = [
        "artificial intelligence",
        "climate change",
        "cricket",
        "stock market",
        "covid vaccine"
    ]
    
    results = []
    
    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        
        recommendations = recommend_for_user(
            session_id="test_search",
            user_interests=[],
            search_query=query,
            selected_categories=[],
            top_k=10
        )
        
        if not recommendations:
            print(f"   ❌ No results found")
            results.append({"query": query, "found": 0, "relevance": 0})
            continue
        
        # Check relevance (keywords in title or content)
        query_words = query.lower().split()
        relevant = 0
        
        for rec in recommendations:
            title = rec.get('title', '').lower()
            desc = rec.get('description', '').lower()
            
            if any(word in title or word in desc for word in query_words):
                relevant += 1
        
        relevance = (relevant / len(recommendations)) * 100
        avg_score = sum(r.get('final_score', 0) for r in recommendations) / len(recommendations)
        
        print(f"   ✓ Found: {len(recommendations)} articles")
        print(f"   ✓ Relevance: {relevance:.1f}% ({relevant}/{len(recommendations)} contain query terms)")
        print(f"   ✓ Average Score: {avg_score:.1f}%")
        print(f"   ✓ Top Result: {recommendations[0].get('title', 'N/A')[:80]}")
        
        results.append({
            "query": query,
            "found": len(recommendations),
            "relevance": relevance,
            "avg_score": avg_score
        })
    
    return results


def test_data_coverage():
    """Test 6: Data Coverage - How much content is available?"""
    print_section("TEST 6: Data Coverage")
    
    meta = load_metadata()
    
    if meta.empty:
        print("❌ No metadata found")
        return {}
    
    print(f"📚 Content Statistics:\n")
    print(f"   ✓ Total Articles: {len(meta)}")
    
    # Category distribution
    if 'categories' in meta.columns:
        all_categories = []
        for cats in meta['categories'].dropna():
            all_categories.extend([c.strip() for c in str(cats).split(',') if c.strip()])
        
        cat_dist = Counter(all_categories)
        print(f"\n   Category Distribution:")
        for cat, count in cat_dist.most_common(10):
            print(f"      - {cat}: {count} articles")
    
    # Source distribution
    if 'source' in meta.columns:
        source_dist = meta['source'].value_counts()
        print(f"\n   Top Sources:")
        for source, count in source_dist.head(10).items():
            print(f"      - {source}: {count} articles")
    
    # Recency
    if 'published' in meta.columns:
        try:
            from dateutil import parser as dtparser
            now = datetime.now()
            
            ages = []
            for pub_str in meta['published'].dropna():
                try:
                    pub_date = dtparser.parse(str(pub_str))
                    age_hours = (now - pub_date).total_seconds() / 3600
                    ages.append(age_hours)
                except:
                    pass
            
            if ages:
                articles_24h = sum(1 for age in ages if age <= 24)
                articles_week = sum(1 for age in ages if age <= 168)
                
                print(f"\n   Freshness:")
                print(f"      - Last 24 hours: {articles_24h} articles")
                print(f"      - Last 7 days: {articles_week} articles")
                print(f"      - Oldest article: {max(ages)/24:.1f} days ago")
        except Exception as e:
            print(f"\n   ⚠️  Could not analyze recency: {e}")
    
    print(f"\n   Configured Sources: {len(RSS_FEEDS)} RSS feeds")
    
    return {
        "total_articles": len(meta),
        "sources": len(meta['source'].unique()) if 'source' in meta.columns else 0
    }


def generate_summary_report(all_results: Dict):
    """Generate a summary report with recommendations"""
    print_section("EVALUATION SUMMARY")
    
    print("🎯 Recommendation System Quality Report\n")
    
    # Overall assessment
    issues = []
    strengths = []
    
    # Check category relevance
    if 'category_tests' in all_results and all_results['category_tests']:
        avg_category_match = sum(r['category_match_rate'] for r in all_results['category_tests']) / len(all_results['category_tests'])
        avg_score = sum(r['avg_score'] for r in all_results['category_tests']) / len(all_results['category_tests'])
        
        print(f"✓ Category Matching: {avg_category_match:.1f}% accuracy")
        print(f"✓ Average Match Score: {avg_score:.1f}%")
        
        if avg_category_match >= 80:
            strengths.append("Excellent category matching")
        elif avg_category_match >= 60:
            strengths.append("Good category matching")
        else:
            issues.append("Category matching needs improvement")
        
        if avg_score >= 85:
            strengths.append("High match scores achieved")
        else:
            issues.append("Match scores below target (85%+)")
    
    # Check diversity
    if 'diversity' in all_results:
        div = all_results['diversity']
        print(f"\n✓ Source Diversity: {div.get('source_diversity', 0):.1f}%")
        print(f"✓ Category Diversity: {div.get('category_diversity', 0):.1f}%")
        
        if div.get('source_diversity', 0) >= 50:
            strengths.append("Good source diversity")
        else:
            issues.append("Limited source diversity - articles from too few sources")
    
    # Check search quality
    if 'search_tests' in all_results and all_results['search_tests']:
        avg_relevance = sum(r['relevance'] for r in all_results['search_tests']) / len(all_results['search_tests'])
        print(f"\n✓ Search Relevance: {avg_relevance:.1f}%")
        
        if avg_relevance >= 70:
            strengths.append("Search results are relevant")
        else:
            issues.append("Search results may not match queries well")
    
    # Check data coverage
    if 'coverage' in all_results:
        cov = all_results['coverage']
        print(f"\n✓ Total Articles: {cov.get('total_articles', 0)}")
        print(f"✓ Unique Sources: {cov.get('sources', 0)}")
        
        if cov.get('total_articles', 0) >= 500:
            strengths.append("Good content coverage")
        else:
            issues.append("Limited article database - consider adding more sources")
    
    # Recommendations
    print("\n" + "-"*80)
    
    if strengths:
        print("\n💪 Strengths:")
        for s in strengths:
            print(f"   ✓ {s}")
    
    if issues:
        print("\n⚠️  Areas for Improvement:")
        for i in issues:
            print(f"   • {i}")
        
        print("\n📋 Recommendations:")
        if any("match score" in i.lower() for i in issues):
            print("   1. Fine-tune feature weights in recommend_advanced.py")
            print("   2. Collect more user behavior data (clicks) for personalization")
        
        if any("diversity" in i.lower() for i in issues):
            print("   3. Add more RSS feed sources in config_feeds.py")
        
        if any("article database" in i.lower() for i in issues):
            print("   4. Run manual refresh to fetch more articles")
            print("   5. Schedule regular background crawling")
    
    if not issues:
        print("\n🎉 Your recommendation system is performing well!")
    
    print("\n" + "-"*80)


def main():
    """Run all evaluation tests"""
    print("\n" + "🎯"*40)
    print("  NEWS RECOMMENDATION SYSTEM EVALUATION")
    print("🎯"*40)
    
    all_results = {}
    
    try:
        # Test 1: Category Relevance
        all_results['category_tests'] = test_category_relevance()
        
        # Test 2: Personalization
        all_results['personalization'] = test_personalization()
        
        # Test 3: Diversity
        all_results['diversity'] = test_diversity()
        
        # Test 4: Recency
        all_results['recency'] = test_recency()
        
        # Test 5: Search Quality
        all_results['search_tests'] = test_search_quality()
        
        # Test 6: Data Coverage
        all_results['coverage'] = test_data_coverage()
        
        # Generate summary
        generate_summary_report(all_results)
        
        print("\n✅ Evaluation complete!")
        print("\nTip: Run this script regularly to track improvements over time.")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
