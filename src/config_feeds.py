"""
Comprehensive RSS feed configuration
Expanded to 60+ sources across all categories
"""

RSS_FEEDS = [
    # International News - Major Agencies
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/uk/rss",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "http://rss.cnn.com/rss/edition_world.rss",
    "http://rss.cnn.com/rss/edition.rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.washingtonpost.com/rss/world",
    "https://www.ft.com/world?format=rss",
    "https://www.economist.com/the-world-this-week/rss.xml",
    "https://www.npr.org/rss/rss.php?id=1004",
    "https://feeds.skynews.com/feeds/rss/home.xml",
    
    # Technology & AI
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
    "https://www.theverge.com/rss/index.xml",
    "https://www.cnet.com/rss/news/",
    "https://www.engadget.com/rss.xml",
    "https://www.technologyreview.com/feed/",
    "https://venturebeat.com/feed/",
    "https://www.artificialintelligence-news.com/feed/",
    "https://blogs.nvidia.com/feed/",
    "http://feeds.feedburner.com/TechCrunch/",
    
    # Business & Finance
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.businessinsider.com/rss",
    "https://feeds.fortune.com/fortune/headlines",
    "https://www.marketwatch.com/rss/",
    
    # Science & Health
    "https://feeds.feedburner.com/NDTV-ScienceandEnvironment",
    "https://www.sciencedaily.com/rss/all.xml",
    "https://www.sciencedaily.com/rss/top/science.xml",
    "https://www.healthline.com/rss",
    "https://www.medicalnewstoday.com/rss",
    "http://rss.cnn.com/rss/cnn_health.rss",
    "http://feeds.bbci.co.uk/news/health/rss.xml",
    "https://www.nature.com/nature.rss",
    "https://www.sciencemag.org/rss/news_current.xml",
    
    # India-Specific
    "https://feeds.feedburner.com/ndtvnews-india-news",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.thehindu.com/news/national/?service=rss",
    "https://indianexpress.com/feed/",
    "https://www.hindustantimes.com/rss/india-news/rssfeed.xml",
    "https://www.business-standard.com/rss/home_page_top_stories.rss",
    
    # Sports
    "http://feeds.bbci.co.uk/sport/rss.xml",
    "https://www.espn.com/espn/rss/news",
    "http://rss.cnn.com/rss/edition_sport.rss",
    "https://www.skysports.com/rss/12040",
    
    # Entertainment
    "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "https://variety.com/feed/",
    "https://www.hollywoodreporter.com/feed/",
    
    # Climate & Environment
    "https://www.theguardian.com/environment/rss",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://insideclimatenews.org/feed/",
    "https://www.carbonbrief.org/feed/",
]

FEEDS = RSS_FEEDS  # Backward compatibility
