"""
Interactive Recommendation Testing Tool

Use this to manually test recommendations and see detailed results.
"""
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommend_advanced import recommend_for_user


def print_article(article, rank):
    """Pretty print an article recommendation"""
    print(f"\n{'='*80}")
    print(f"Rank #{rank}")
    print(f"{'='*80}")
    print(f"Title: {article.get('title', 'N/A')}")
    print(f"Source: {article.get('source', 'N/A')}")
    print(f"Categories: {article.get('categories', 'N/A')}")
    print(f"Match Score: {article.get('final_score', 0):.1f}%")
    print(f"Published: {article.get('published', 'N/A')}")
    print(f"\nDescription: {article.get('description', 'N/A')[:200]}...")
    print(f"URL: {article.get('url', 'N/A')}")
    
    # Show score breakdown if available
    if 'semantic_score' in article:
        print(f"\n📊 Score Breakdown:")
        print(f"   Semantic Match: {article.get('semantic_score', 0):.1f}%")
        print(f"   Category Match: {article.get('category_score', 0):.1f}%")
        print(f"   Recency: {article.get('recency_score', 0):.1f}%")
        print(f"   User Behavior: {article.get('behavior_score', 0):.1f}%")


def test_scenario_1():
    """Scenario 1: Tech Enthusiast"""
    print("\n" + "🚀"*40)
    print("  SCENARIO 1: Technology Enthusiast")
    print("🚀"*40)
    
    print("\nUser Profile:")
    print("  Interests: Technology, AI, Machine Learning, Programming")
    print("  Search: (none)")
    print("  Categories: (none)")
    
    recommendations = recommend_for_user(
        session_id="test_tech_enthusiast",
        user_interests=["Technology", "AI", "Machine Learning", "Programming"],
        search_query="",
        selected_categories=[],
        top_k=5
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations\n")
    
    for i, article in enumerate(recommendations, 1):
        print_article(article, i)
    
    return recommendations


def test_scenario_2():
    """Scenario 2: Sports Fan with Search"""
    print("\n" + "⚽"*40)
    print("  SCENARIO 2: Sports Fan Searching for Cricket")
    print("⚽"*40)
    
    print("\nUser Profile:")
    print("  Interests: Sports, Cricket")
    print("  Search: 'cricket match'")
    print("  Categories: (none)")
    
    recommendations = recommend_for_user(
        session_id="test_sports_fan",
        user_interests=["Sports", "Cricket"],
        search_query="cricket match",
        selected_categories=[],
        top_k=5
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations\n")
    
    for i, article in enumerate(recommendations, 1):
        print_article(article, i)
    
    return recommendations


def test_scenario_3():
    """Scenario 3: Category Filter Only"""
    print("\n" + "📰"*40)
    print("  SCENARIO 3: Browsing International News")
    print("📰"*40)
    
    print("\nUser Profile:")
    print("  Interests: (none)")
    print("  Search: (none)")
    print("  Categories: International, World")
    
    recommendations = recommend_for_user(
        session_id="test_category_browser",
        user_interests=[],
        search_query="",
        selected_categories=["International", "World"],
        top_k=5
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations\n")
    
    for i, article in enumerate(recommendations, 1):
        print_article(article, i)
    
    return recommendations


def test_scenario_4():
    """Scenario 4: Complex Query"""
    print("\n" + "🎯"*40)
    print("  SCENARIO 4: Complex Query - Business + Climate + Search")
    print("🎯"*40)
    
    print("\nUser Profile:")
    print("  Interests: Business, Economy, Climate")
    print("  Search: 'renewable energy investment'")
    print("  Categories: Business")
    
    recommendations = recommend_for_user(
        session_id="test_complex_query",
        user_interests=["Business", "Economy", "Climate"],
        search_query="renewable energy investment",
        selected_categories=["Business"],
        top_k=5
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations\n")
    
    for i, article in enumerate(recommendations, 1):
        print_article(article, i)
    
    return recommendations


def custom_test():
    """Let user create their own test"""
    print("\n" + "🎨"*40)
    print("  CUSTOM TEST")
    print("🎨"*40)
    
    print("\nCreate your own test scenario:")
    
    # Get user inputs
    interests_input = input("\nEnter interests (comma-separated, or press Enter to skip): ").strip()
    interests = [i.strip() for i in interests_input.split(',')] if interests_input else []
    
    search_query = input("Enter search query (or press Enter to skip): ").strip()
    
    categories_input = input("Enter category filters (comma-separated, or press Enter to skip): ").strip()
    categories = [c.strip() for c in categories_input.split(',')] if categories_input else []
    
    top_k = input("How many recommendations? (default 5): ").strip()
    top_k = int(top_k) if top_k.isdigit() else 5
    
    print("\n" + "-"*80)
    print("Testing with:")
    print(f"  Interests: {interests if interests else '(none)'}")
    print(f"  Search: '{search_query}' " if search_query else "  Search: (none)")
    print(f"  Categories: {categories if categories else '(none)'}")
    print(f"  Results: Top {top_k}")
    print("-"*80)
    
    recommendations = recommend_for_user(
        session_id="test_custom",
        user_interests=interests,
        search_query=search_query,
        selected_categories=categories,
        top_k=top_k
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations\n")
    
    for i, article in enumerate(recommendations, 1):
        print_article(article, i)
    
    return recommendations


def compare_scenarios():
    """Compare recommendations across different scenarios"""
    print("\n" + "📊"*40)
    print("  COMPARISON TEST")
    print("📊"*40)
    
    print("\nTesting how recommendations change with different inputs...")
    
    # Same interest, different searches
    scenarios = [
        {
            "name": "Tech + AI Search",
            "interests": ["Technology"],
            "search": "artificial intelligence",
            "categories": []
        },
        {
            "name": "Tech + Crypto Search",
            "interests": ["Technology"],
            "search": "cryptocurrency",
            "categories": []
        },
        {
            "name": "Tech + No Search",
            "interests": ["Technology"],
            "search": "",
            "categories": []
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        recs = recommend_for_user(
            session_id=f"test_compare_{scenario['name']}",
            user_interests=scenario['interests'],
            search_query=scenario['search'],
            selected_categories=scenario['categories'],
            top_k=3
        )
        
        results.append({
            'name': scenario['name'],
            'count': len(recs),
            'top_title': recs[0].get('title', 'N/A') if recs else 'N/A',
            'top_score': recs[0].get('final_score', 0) if recs else 0,
            'articles': recs
        })
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Results: {result['count']}")
        print(f"  Top Score: {result['top_score']:.1f}%")
        print(f"  Top Article: {result['top_title'][:60]}...")
    
    return results


def main():
    """Main interactive menu"""
    while True:
        print("\n" + "="*80)
        print("  🎯 INTERACTIVE RECOMMENDATION TESTING TOOL")
        print("="*80)
        print("\nChoose a test scenario:")
        print("  1. Tech Enthusiast (AI, Programming)")
        print("  2. Sports Fan Searching for Cricket")
        print("  3. Category Filter Only (International News)")
        print("  4. Complex Query (Business + Climate + Search)")
        print("  5. Custom Test (Create your own)")
        print("  6. Compare Scenarios")
        print("  7. Run All Scenarios")
        print("  0. Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            test_scenario_1()
        elif choice == '2':
            test_scenario_2()
        elif choice == '3':
            test_scenario_3()
        elif choice == '4':
            test_scenario_4()
        elif choice == '5':
            custom_test()
        elif choice == '6':
            compare_scenarios()
        elif choice == '7':
            test_scenario_1()
            test_scenario_2()
            test_scenario_3()
            test_scenario_4()
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        input("\n\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
