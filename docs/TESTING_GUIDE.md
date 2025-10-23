# Testing and Evaluating Your Recommendation System

This guide explains how to test and measure the quality of your news recommendation system.

## 🎯 Quick Start

### Method 1: Automated Comprehensive Evaluation (Recommended)

Run the full evaluation suite to test all aspects of your recommendation system:

```powershell
# Activate your virtual environment first
.\.venv\Scripts\Activate.ps1

# Run comprehensive evaluation
python evaluate_recommendations.py
```

**This will test:**
- ✅ Category Matching (Do recommendations match user interests?)
- ✅ Personalization (Does it learn from user clicks?)
- ✅ Diversity (Are recommendations diverse enough?)
- ✅ Recency (Are recent articles prioritized?)
- ✅ Search Quality (Do search results make sense?)
- ✅ Data Coverage (How much content is available?)

**Expected Output:**
```
🎯 TEST 1: Category Relevance
   ✓ Got 10 recommendations
   ✓ Average Match Score: 87.3%
   ✓ Category Match Rate: 90% (9/10)
   
🎯 TEST 2: Personalization
   ✓ Recommendations align with click history
   
... and more
```

---

### Method 2: Interactive Manual Testing

Test specific scenarios interactively:

```powershell
python test_interactive.py
```

This provides a menu where you can:
1. Test predefined scenarios (Tech enthusiast, Sports fan, etc.)
2. Create custom tests with your own interests/searches
3. Compare how recommendations change with different inputs
4. See detailed score breakdowns for each article

**Perfect for:**
- Testing specific user personas
- Debugging why certain articles appear
- Understanding score calculations
- Comparing different queries

---

## 📊 What Makes a "Good" Recommendation?

### 1. **Match Score (Target: 85-95%)**
   - This is the overall confidence score
   - Combines semantic similarity, category match, recency, etc.
   - **Good:** 85%+ consistently
   - **Needs Work:** Below 70%

### 2. **Category Relevance (Target: 80%+)**
   - % of recommendations matching user's interests
   - If user likes "Technology", how many tech articles do they get?
   - **Good:** 80%+ of results match interests
   - **Needs Work:** Below 60%

### 3. **Diversity (Target: Source >50%, Category >30%)**
   - Articles should come from varied sources
   - Should span multiple categories (not all the same)
   - **Good:** Articles from 10+ different sources
   - **Needs Work:** All from 2-3 sources only

### 4. **Recency (Target: 50%+ within 24h)**
   - Recent news should appear first
   - **Good:** 50%+ of recommendations from last 24 hours
   - **Needs Work:** Most articles are days old

### 5. **Search Relevance (Target: 70%+)**
   - Search results should contain query terms
   - **Good:** 70%+ results mention search keywords
   - **Needs Work:** Irrelevant results

---

## 🔍 How to Interpret Results

### Example: Good System ✅
```
✓ Category Matching: 85.0% accuracy
✓ Average Match Score: 87.5%
✓ Source Diversity: 65.0%
✓ Search Relevance: 80.0%
✓ Total Articles: 674

💪 Strengths:
   ✓ Excellent category matching
   ✓ High match scores achieved
   ✓ Good source diversity
```

### Example: Needs Improvement ⚠️
```
✓ Category Matching: 55.0% accuracy
✓ Average Match Score: 68.0%
✓ Source Diversity: 35.0%
✓ Total Articles: 150

⚠️  Areas for Improvement:
   • Category matching needs improvement
   • Match scores below target (85%+)
   • Limited article database

📋 Recommendations:
   1. Fine-tune feature weights in recommend_advanced.py
   2. Add more RSS feed sources
   3. Run manual refresh to fetch more articles
```

---

## 🧪 Testing Strategies

### A. Before Making Changes
1. Run `python evaluate_recommendations.py` to get baseline metrics
2. Save the results (take a screenshot or copy output)
3. Make your changes to the code
4. Run evaluation again and compare

### B. Testing User Personas
Use `test_interactive.py` to test specific user types:

**Tech Enthusiast:**
- Interests: Technology, AI, Programming
- Should get: Tech news, AI updates, coding articles

**Sports Fan:**
- Interests: Sports, Cricket
- Should get: Sports news, cricket matches, scores

**Business Professional:**
- Interests: Business, Finance, Economy
- Should get: Market news, economic updates, company news

### C. Testing Edge Cases
Test these scenarios to find problems:

1. **Empty Interests** - What happens if user has no interests set?
2. **Rare Categories** - Search for niche topics (e.g., "quantum computing")
3. **Multiple Categories** - Combine unrelated interests (Sports + Technology)
4. **Very Specific Search** - Use specific queries ("India vs Australia cricket score")

---

## 📈 Improving Your System

### If Match Scores Are Low (<80%)

**Problem:** Articles don't match user interests well

**Solutions:**
1. **Adjust weights in `recommend_advanced.py`:**
   ```python
   # Line ~280 in recommend_advanced.py
   # Increase category weight if category matching is important
   final = (
       0.40 * category +      # Increase this (was 0.38)
       0.25 * semantic +      # Decrease this (was 0.28)
       0.12 * recency +
       0.12 * behavior +
       0.06 * title +
       0.05 * source
   )
   ```

2. **Collect more user behavior data:**
   - Use the app and click on articles you like
   - The system learns from clicks and improves over time

### If Diversity Is Low

**Problem:** All recommendations from same source/category

**Solutions:**
1. **Add more RSS feeds in `config_feeds.py`:**
   ```python
   RSS_FEEDS = [
       ("Technology", "https://new-tech-site.com/rss"),
       # Add more diverse sources
   ]
   ```

2. **Run manual refresh:**
   - Click "Refresh News" button in the UI
   - Or run: `python -m src.ingest_rss`

### If Search Doesn't Work Well

**Problem:** Search results don't match query

**Solutions:**
1. **Increase semantic weight for searches** in `recommend_advanced.py`
2. **Ensure sufficient articles** in database (run refresh)
3. **Check embedding model** is loaded correctly

### If Articles Are Too Old

**Problem:** Recommendations show old articles

**Solutions:**
1. **Increase recency weight** in `recommend_advanced.py`
2. **Run regular refreshes** to fetch new articles
3. **Check RSS feeds** are active and publishing

---

## 🔧 Advanced: Measuring Specific Metrics

### Precision@K (How many top results are relevant?)

```python
# After getting recommendations
relevant = 0
for rec in recommendations[:10]:  # Top 10
    if user_interest.lower() in rec.get('categories', '').lower():
        relevant += 1

precision = relevant / 10
print(f"Precision@10: {precision*100}%")  # Should be 70%+
```

### User Satisfaction (A/B Testing)

1. Show 10 recommendations to user
2. Ask: "How many of these interest you?"
3. Calculate: interested_count / 10
4. Target: 60%+ satisfaction

### Click-Through Rate (CTR)

Track in production:
```python
CTR = (articles_clicked / articles_shown) * 100
# Good CTR for news: 5-15%
```

---

## 📝 Creating Your Evaluation Report

After running tests, document your findings:

### Template:

```markdown
# Recommendation System Evaluation - [Date]

## Overall Score: [X/100]

### Metrics
- Match Score: X%
- Category Accuracy: X%
- Source Diversity: X%
- Search Relevance: X%
- Total Articles: X

### Strengths
- [What works well]

### Weaknesses
- [What needs improvement]

### Action Items
1. [Specific change to make]
2. [Another improvement]

### Next Evaluation: [Date]
```

---

## 🎓 Best Practices

1. **Test Regularly** - Run evaluation after every major change
2. **Keep Baselines** - Save results to track improvement over time
3. **Test With Real Users** - Nothing beats actual user feedback
4. **Monitor Production** - Track click rates, user engagement
5. **Iterate** - Small improvements compound over time

---

## 🚀 Quick Commands Reference

```powershell
# Full evaluation suite
python evaluate_recommendations.py

# Interactive testing
python test_interactive.py

# Test specific categories (existing script)
python test_categories.py

# Fetch fresh articles
python -m src.ingest_rss

# Rebuild index after adding sources
python rebuild_index.py

# Start backend to test via UI
python -m src.server

# Check API directly
python test_api.py
```

---

## 💡 Tips

- **Before Demo:** Run evaluation to ensure system is performing well
- **After Changes:** Always test to catch regressions
- **With Users:** Ask for feedback on recommendations they see
- **Over Time:** Track metrics weekly to see trends

---

## ❓ Troubleshooting

**Q: Evaluation script shows errors?**
- Ensure backend dependencies installed: `pip install -r requirements.txt`
- Check data exists: `data/meta.csv` and `data/index.faiss` should exist

**Q: All scores are 0?**
- Run: `python -m src.ingest_rss` to fetch articles first
- Check: `data/meta.csv` has content

**Q: How to improve from 85% to 90%+?**
- Collect more user behavior data (clicks)
- Fine-tune weights based on your specific use case
- Add more high-quality sources
- Train custom model on labeled data (advanced)

---

## 📚 Further Reading

- `AI_MODEL_IMPROVEMENTS.md` - Details on the multi-signal ranking system
- `recommend_advanced.py` - Source code with detailed comments
- `SYSTEM_STATUS.md` - Current system capabilities

---

**Remember:** Good recommendations are subjective! The best test is real user feedback. 🎯
