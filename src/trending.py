"""Extract trending keywords from recent articles"""
import os
import re
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
from .recommend import META_CSV

# Comprehensive stop words to exclude
STOP_WORDS = {
    # Basic articles, pronouns, conjunctions
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'them', 'their', 'theirs', 
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'we', 'us', 'our', 'ours', 'they',
    
    # Question words and common verbs
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'then', 'too', 'very',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'before',
    'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'during', 'inside',
    'into', 'near', 'off', 'onto', 'outside', 'over', 'through', 'toward', 'under',
    'underneath', 'until', 'unto', 'upon', 'within', 'without',
    
    # Common verbs and adjectives
    's', 't', 'just', 'don', 'now', 'new', 'says', 'said', 'say', 'get', 'gets', 'getting',
    'got', 'gotten', 'make', 'makes', 'making', 'made', 'go', 'goes', 'going', 'went', 'gone',
    'see', 'sees', 'seeing', 'saw', 'seen', 'know', 'knows', 'knowing', 'knew', 'known',
    'take', 'takes', 'taking', 'took', 'taken', 'use', 'uses', 'using', 'used',
    'find', 'finds', 'finding', 'found', 'give', 'gives', 'giving', 'gave', 'given',
    'tell', 'tells', 'telling', 'told', 'work', 'works', 'working', 'worked',
    'call', 'calls', 'calling', 'called', 'try', 'tries', 'trying', 'tried',
    'ask', 'asks', 'asking', 'asked', 'need', 'needs', 'needing', 'needed',
    'feel', 'feels', 'feeling', 'felt', 'become', 'becomes', 'becoming', 'became',
    'leave', 'leaves', 'leaving', 'left', 'put', 'puts', 'putting', 'mean', 'means',
    'keep', 'keeps', 'keeping', 'kept', 'let', 'lets', 'letting', 'begin', 'begins',
    'seem', 'seems', 'seemed', 'turn', 'turns', 'turned', 'show', 'shows', 'showed',
    'help', 'helps', 'helped', 'talk', 'talks', 'talked', 'provide', 'provides',
    'allow', 'allows', 'allowed', 'include', 'includes', 'included', 'continue',
    'continues', 'continued', 'set', 'sets', 'setting', 'run', 'runs', 'running', 'ran',
    'move', 'moves', 'moving', 'moved', 'like', 'likes', 'liked', 'live', 'lives', 'lived',
    'believe', 'believes', 'believed', 'hold', 'holds', 'holding', 'held', 'bring',
    'brings', 'bringing', 'brought', 'happen', 'happens', 'happened', 'write', 'writes',
    'wrote', 'written', 'sit', 'sits', 'sitting', 'sat', 'stand', 'stands', 'stood',
    'lose', 'loses', 'lost', 'pay', 'pays', 'paid', 'meet', 'meets', 'met', 'run',
    'include', 'whether', 'likely', 'still', 'also', 'well', 'back', 'even', 'way',
    'because', 'any', 'there', 'think', 'first', 'two', 'three', 'four', 'five', 'much', 'good', 'really',
    'one', 'many', 'year', 'years', 'time', 'times', 'day', 'days', 'today', 'week', 'weeks',
    'month', 'months', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october',
    'november', 'december', 'people', 'person', 'user', 'users', 'while', 'during', 'since',
    'company', 'companies', 'world', 'news', 'report', 'reports', 'according', 'however',
    'things', 'thing', 'something', 'everything', 'nothing', 'anything', 'anyone', 'someone',
    'everyone', 'nobody', 'somebody', 'everybody', 'anywhere', 'somewhere', 'everywhere', 'nowhere',
    'model', 'models', 'agent', 'agents', 'system', 'systems', 'online', 'offline',
    'performance', 'data', 'information', 'content', 'service', 'services', 'based',
    'available', 'support', 'supports', 'feature', 'features', 'version', 'latest',
    'platform', 'platforms', 'announced', 'announces', 'launch', 'launched', 'releases',
    'human', 'humans', 'power', 'another', 'others', 'rather', 'whether', 'either',
    'offer', 'offers', 'offering', 'showing', 'shows', 'shown', 'looks', 'looking',
    # News source names (to filter out)
    'engadget', 'reuters', 'bloomberg', 'techcrunch', 'verge', 'wired', 'guardian',
    'times', 'forbes', 'cnet', 'zdnet', 'axios', 'politico', 'economist',
    
    # HTML/URL artifacts
    'href', 'http', 'https', 'www', 'com', 'org', 'net', 'html', 'amp', 'div', 'span',
    'style', 'class', 'img', 'src', 'alt', 'title', 'rel', 'target', 'nofollow', 'blank',
    'font', 'family', 'text', 'align', 'left', 'right', 'center', 'color', 'size',
    'border', 'width', 'height', 'margin', 'padding', 'display', 'position',
    
    # RSS feed artifacts
    'continue', 'reading', 'read', 'more', 'full', 'article', 'story', 'post', 'click',
    'here', 'via', 'subscribe', 'follow', 'share', 'comment', 'comments', 'posted',
    'published', 'updated', 'source', 'image', 'video', 'photo', 'picture',
    'originally', 'appeared', 'originally appeared', 'copyright', 'reserved', 'rights',
    
    # Noise words
    'nan', 'null', 'none', 'undefined', 'true', 'false', 'yes', 'etc',
}


def clean_html(text):
    """Remove HTML tags and decode entities"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_trending_terms(hours=24, top_n=10, min_length=3):
    """
    Extract trending keywords from recent articles.
    
    Args:
        hours: Only consider articles published in last N hours
        top_n: Return top N most frequent terms
        min_length: Minimum character length for a term
    
    Returns:
        List of (term, count) tuples
    """
    if not os.path.exists(META_CSV):
        return []
    
    try:
        meta = pd.read_csv(META_CSV)
        meta['published'] = pd.to_datetime(meta['published'], utc=True, errors='coerce')
        
        # Filter recent articles
        cutoff = pd.Timestamp.now(tz='UTC') - timedelta(hours=hours)
        recent = meta[meta['published'] >= cutoff]
        
        if recent.empty:
            return []
        
        # Combine title and summary text, clean HTML
        texts = []
        for _, row in recent.iterrows():
            title = clean_html(str(row.get('title', '')))
            summary = clean_html(str(row.get('summary', '')))
            texts.append(f"{title} {summary}")
        
        # Tokenize and count (focus on meaningful content words)
        word_counts = Counter()
        proper_noun_bonus = Counter()  # Track capitalized words (likely proper nouns)
        
        for text in texts:
            # Extract words (letters only, min length)
            words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
            # Filter stop words and very common generic words
            words = [w for w in words if w not in STOP_WORDS and len(w) >= 5]  # Increased to 5 chars
            word_counts.update(words)
        
        # Track proper nouns (capitalized words in original text) for bonus scoring
        for text in texts:
            proper_nouns = re.findall(r'\b[A-Z][a-z]{4,}\b', text)  # Capitalized words (not acronyms)
            proper_noun_bonus.update([w.lower() for w in proper_nouns])
        
        # Also extract bigrams (two-word phrases)
        bigram_counts = Counter()
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
            words = [w for w in words if w not in STOP_WORDS and len(w) >= 4]
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts.update(bigrams)
        
        # Quality filter: Only include terms that appear in multiple articles
        num_articles = len(texts)
        min_articles = max(2, int(num_articles * 0.05))  # At least 5% of articles or 2 articles
        
        # Filter bigrams - require appearing in at least 2 articles with decent frequency
        quality_bigrams = []
        for term, count in bigram_counts.items():
            if count >= 3:  # Increased minimum threshold
                # Check if bigram contains at least one substantive word (6+ chars)
                words_in_bigram = term.split()
                if any(len(w) >= 6 for w in words_in_bigram):
                    quality_bigrams.append((term, count * 2.0))  # Prioritize bigrams more
        
        # Filter single words - require higher frequency, bonus for proper nouns
        quality_words = []
        for term, count in word_counts.items():
            if count >= 5 and len(term) >= 5:  # Higher thresholds
                # Give bonus to proper nouns (names, places, organizations)
                bonus = 1.5 if proper_noun_bonus.get(term, 0) >= 3 else 1.0
                quality_words.append((term, count * bonus))
        
        # Combine and get top terms
        all_terms = quality_bigrams + quality_words
        all_terms.sort(key=lambda x: x[1], reverse=True)
        
        return [(term, int(count)) for term, count in all_terms[:top_n]]
    
    except Exception as e:
        print(f"Error extracting trending terms: {e}")
        return []


if __name__ == "__main__":
    trends = extract_trending_terms(hours=168, top_n=15)  # Last week
    print("Trending terms:")
    for term, count in trends:
        print(f"  {term}: {count}")
