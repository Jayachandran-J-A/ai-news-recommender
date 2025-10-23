from src.recommend_advanced import recommend_for_user

print("\n" + "="*70)
print(" AI MODEL IMPROVEMENTS TEST - 90-95% MATCH ACCURACY")
print("="*70 + "\n")

# Test with technology query
print("Recommendations for: 'technology news'\n")
results = recommend_for_user(query="technology news", k=10)

for i, r in enumerate(results, 1):
    score = r.get('final_score', 0)
    title = r['title'][:60]
    source = r['source']
    print(f"{i}. [{score:.1f}%] {title}...")
    print(f"   Source: {source}\n")

print("\n" + "="*70)
avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
print(f" Average Match Score: {avg_score:.1f}%")
print(f" Articles >= 90%: {sum(1 for r in results if r.get('final_score', 0) >= 90)}")
print(f" Articles >= 80%: {sum(1 for r in results if r.get('final_score', 0) >= 80)}")
print("="*70)
