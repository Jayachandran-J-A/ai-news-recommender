import { useState, useEffect } from "react";
import { OnboardingModal } from "@/components/OnboardingModal";
import { Navigation } from "@/components/Navigation";
import { CategoryFilter } from "@/components/CategoryFilter";
import { TrendingWidget } from "@/components/TrendingWidget";
import { NewsCard } from "@/components/NewsCard";
import { LoadingGrid } from "@/components/LoadingSkeleton";
import { EmptyState } from "@/components/EmptyState";
import { ErrorBanner } from "@/components/ErrorBanner";
import { Button } from "@/components/ui/button";
import { Clock, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { 
  fetchRecommendations, 
  fetchTrending, 
  trackArticleClick,
  refreshNews,
  getSessionId,
  parseCategories,
  type Article as ApiArticle,
  type TrendingKeyword
} from "@/lib/api";

interface Article {
  title: string;
  source: string;
  url: string;
  published?: string;
  categories?: string[];
  score?: number;
  image?: string;
}

const CATEGORIES = [
  "all",
  "technology",
  "ai",
  "politics",
  "business",
  "health",
  "science",
  "sports",
  "entertainment",
  "world",
  "india",
  "climate",
  "education",
];

const Index = () => {
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [userInterests, setUserInterests] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [articles, setArticles] = useState<Article[]>([]);
  const [trendingKeywords, setTrendingKeywords] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [sessionId] = useState(() => getSessionId());

  useEffect(() => {
    const hasOnboarded = localStorage.getItem("newsai_onboarded");
    const savedInterests = localStorage.getItem("newsai_interests");

    if (!hasOnboarded) {
      setShowOnboarding(true);
    } else if (savedInterests) {
      try {
        setUserInterests(JSON.parse(savedInterests));
      } catch (e) {
        console.error("Failed to parse saved interests:", e);
      }
    }

    // Load initial data
    loadArticles();
    loadTrendingKeywords();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadArticles(true);
    }, 2 * 60 * 1000); // 2 minutes

    return () => clearInterval(interval);
  }, [autoRefresh, searchQuery, selectedCategories]);

  const loadTrendingKeywords = async () => {
    try {
      const trends = await fetchTrending(168, 8);  // 7 days to capture older data
      setTrendingKeywords(trends.map(t => t.term));
    } catch (err) {
      console.error("Failed to load trending keywords:", err);
      // Non-critical error, don't show to user
    }
  };

  const loadArticles = async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);

    try {
      const params: {
        query?: string;
        categories?: string[];
        k?: number;
      } = {
        k: 20,
      };

      // Add query if present
      if (searchQuery.trim()) {
        params.query = searchQuery.trim();
      } else {
        params.query = "news"; // Default query
      }

      // Add categories if selected
      if (selectedCategories.length > 0 && !selectedCategories.includes("all")) {
        params.categories = selectedCategories;
      }

      const apiArticles = await fetchRecommendations(params);
      
      // Convert API articles to UI format
      const convertedArticles: Article[] = apiArticles.map(article => ({
        title: article.title,
        source: article.source,
        url: article.url,
        published: article.published,
        categories: parseCategories(article.categories),
        score: article.final_score,
      }));

      setArticles(convertedArticles);
      
      if (silent && convertedArticles.length > articles.length) {
        toast.success(`${convertedArticles.length - articles.length} new articles loaded`);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load articles";
      setError(errorMessage);
      console.error("Failed to load articles:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleOnboardingComplete = (categories: string[]) => {
    setUserInterests(categories);
    localStorage.setItem("newsai_onboarded", "true");
    localStorage.setItem("newsai_interests", JSON.stringify(categories));
    setShowOnboarding(false);
    // DON'T set selectedCategories - personal interests are separate from filters
    toast.success("Preferences saved! Your feed is now personalized.");
    // Reload with new interests (uses userInterests in background, not filters)
    loadArticles();
  };

  const handleCategoryToggle = (category: string) => {
    if (category === "all") {
      setSelectedCategories([]);
    } else {
      setSelectedCategories((prev) =>
        prev.includes(category)
          ? prev.filter((c) => c !== category)
          : [...prev, category]
      );
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  const handleArticleRead = async (url: string) => {
    try {
      // Track the click for personalization
      await trackArticleClick(url);
      console.log("Article clicked:", { url, sessionId });
      
      // Open article in new tab
      window.open(url, '_blank', 'noopener,noreferrer');
      
      toast.success("Article opened in new tab!");
    } catch (err) {
      console.error("Failed to track click:", err);
      // Still open the article even if tracking fails
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const handleManualRefresh = async () => {
    setIsRefreshing(true);
    toast.info("Fetching latest news from 60+ sources...");
    
    try {
      const result = await refreshNews();
      if (result.status === 'started') {
        toast.success(result.message || "News refresh started!");
        
        // Reload articles after crawl completes (60 seconds)
        setTimeout(() => {
          loadArticles();
          toast.success("News feed updated with fresh articles!");
        }, 60000);
      } else {
        toast.error(result.error || 'Failed to start refresh');
      }
    } catch (err) {
      console.error('Manual refresh error:', err);
      toast.error('Refresh failed: ' + String(err));
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleTrendingClick = (keyword: string) => {
    setSearchQuery(keyword);
    toast.info(`Searching for: ${keyword}`);
  };

  const handleReset = () => {
    setSearchQuery("");
    setSelectedCategories([]);
  };

  const handleRecent = () => {
    setSearchQuery("recent news");
    setSelectedCategories([]);
    toast.info("Fetching recent articles");
  };

  const handleForYou = () => {
    if (userInterests.length > 0) {
      setSelectedCategories(userInterests);
      setSearchQuery("");
      toast.info("Loading personalized feed");
    } else {
      toast.info("Select your interests to get personalized recommendations");
      setShowOnboarding(true);
    }
  };

  useEffect(() => {
    loadArticles();
  }, [searchQuery, selectedCategories]);

  return (
    <div className="min-h-screen bg-background">
      <OnboardingModal
        isOpen={showOnboarding}
        onClose={handleOnboardingComplete}
      />

      <Navigation
        onSearch={handleSearch}
        onOpenInterests={() => setShowOnboarding(true)}
        autoRefresh={autoRefresh}
        onToggleAutoRefresh={() => setAutoRefresh(!autoRefresh)}
        onManualRefresh={handleManualRefresh}
        isRefreshing={isRefreshing}
      />

      <CategoryFilter
        categories={CATEGORIES}
        selectedCategories={selectedCategories}
        onToggleCategory={handleCategoryToggle}
      />

      {error && (
        <ErrorBanner
          message={error}
          onRetry={() => loadArticles()}
          onDismiss={() => setError(null)}
        />
      )}

      <main className="container mx-auto px-4 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main Content */}
          <div className="flex-1">
            {/* Action Buttons */}
            <div className="flex gap-3 mb-6">
              <Button
                onClick={handleRecent}
                variant="outline"
                className="gap-2"
              >
                <Clock className="h-4 w-4" />
                Recent
              </Button>
              <Button
                onClick={handleForYou}
                className="gap-2 bg-gradient-to-r from-primary to-primary-glow"
              >
                <Sparkles className="h-4 w-4" />
                For You
              </Button>
            </div>

            {/* Articles Grid */}
            {loading ? (
              <LoadingGrid />
            ) : articles.length === 0 ? (
              <EmptyState onReset={handleReset} />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 fade-in">
                {articles.map((article, index) => (
                  <NewsCard
                    key={`${article.url}-${index}`}
                    article={article}
                    onRead={handleArticleRead}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <aside className="lg:w-80 space-y-6">
            <TrendingWidget
              keywords={trendingKeywords}
              onKeywordClick={handleTrendingClick}
            />
          </aside>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 py-8 bg-card">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <p>Powered by BGE Embeddings + XGBoost + FAISS</p>
            <div className="flex gap-4">
              <a href="#" className="hover:text-foreground transition-colors">
                About
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Privacy
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                GitHub
              </a>
            </div>
            <p className="text-xs">
              {articles.length} articles loaded | Session: {sessionId.slice(0, 12)}...
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
