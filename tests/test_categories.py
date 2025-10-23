from src.recommend_advanced import recommend_for_user

print("\n" + "="*70)
print(" AI MODEL TEST - Category Matching + Recent Articles")
print("="*70 + "\n")

# Test 1: With categories (boosts category score)
print("TEST 1: User interests = [Technology, AI]\n")
results = recommend_for_user(
    categories=["technology", "ai"],
    k=10
)

for i, r in enumerate(results, 1):
    score = r.get('final_score', 0)
    title = r['title'][:55]
    cats = r.get('categories', '[]')
    print(f"{i}. [{score:.1f}%] {title}...")
    print(f"   Categories: {cats}\n")

print("\n" + "="*70)
avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
print(f" Average Match Score: {avg_score:.1f}%")
print(f" Articles >= 90%: {sum(1 for r in results if r.get('final_score', 0) >= 90)}")
print(f" Articles >= 80%: {sum(1 for r in results if r.get('final_score', 0) >= 80)}")
print(f" Articles >= 70%: {sum(1 for r in results if r.get('final_score', 0) >= 70)}")
print("="*70 + "\n")

# Test 2: Query + Categories
print("\n" + "="*70)
print(" TEST 2: Query='artificial intelligence' + Categories=[AI, Technology]")
print("="*70 + "\n")

results2 = recommend_for_user(
    query="artificial intelligence machine learning",
    categories=["ai", "technology"],
    k=10
)

for i, r in enumerate(results2, 1):
    score = r.get('final_score', 0)
    title = r['title'][:55]
    print(f"{i}. [{score:.1f}%] {title}...")

print("\n" + "="*70)
avg_score2 = sum(r.get('final_score', 0) for r in results2) / len(results2)
print(f" Average Match Score: {avg_score2:.1f}%")
print(f" Articles >= 90%: {sum(1 for r in results2 if r.get('final_score', 0) >= 90)}")
print(f" Articles >= 85%: {sum(1 for r in results2 if r.get('final_score', 0) >= 85)}")
print(f" Articles >= 80%: {sum(1 for r in results2 if r.get('final_score', 0) >= 80)}")
print("="*70)
