// API client for News Recommender backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8003';

export interface Article {
  title: string;
  url: string;
  source: string;
  published: string;
  final_score?: number;
  match_percentage?: number;  // New: 90-95% target match scores
  categories: string; // JSON string of array
  summary?: string; // Article summary/description
}

export interface TrendingKeyword {
  term: string;
  count: number;
}

export interface RecommendResponse {
  items: Article[];
  error?: string;
}

export interface TrendingResponse {
  trends: TrendingKeyword[];
  error?: string;
}

export interface DebugInfo {
  meta_exists: boolean;
  index_exists: boolean;
  meta_len?: number;
  index_ntotal?: number;
  sample_titles?: string[];
}

export interface ClickPayload {
  url: string;
  session_id: string;
}

// Get session ID from localStorage or create new one
export const getSessionId = (): string => {
  let sessionId = localStorage.getItem('newsai_session_id');
  if (!sessionId) {
    sessionId = `user_${Math.random().toString(36).substr(2, 9)}_${Date.now()}`;
    localStorage.setItem('newsai_session_id', sessionId);
  }
  return sessionId;
};

// Fetch recommended articles
export const fetchRecommendations = async (params: {
  query?: string;
  categories?: string[];
  k?: number;
}): Promise<Article[]> => {
  const sessionId = getSessionId();
  const queryParams = new URLSearchParams();
  
  queryParams.append('session_id', sessionId);
  queryParams.append('k', (params.k || 20).toString());
  
  if (params.query) {
    queryParams.append('query', params.query);
  }
  
  if (params.categories && params.categories.length > 0) {
    params.categories.forEach(cat => {
      queryParams.append('categories', cat);
    });
  }
  
  const response = await fetch(`${API_BASE_URL}/recommend?${queryParams.toString()}`);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const data: RecommendResponse = await response.json();
  
  if (data.error) {
    throw new Error(data.error);
  }
  
  return data.items || [];
};

// Track article click
export const trackArticleClick = async (url: string): Promise<void> => {
  const sessionId = getSessionId();
  const formData = new URLSearchParams();
  formData.append('url', url);
  formData.append('session_id', sessionId);
  
  await fetch(`${API_BASE_URL}/click`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData.toString(),
  });
};

// Fetch trending keywords
export const fetchTrending = async (hours: number = 72, topN: number = 10): Promise<TrendingKeyword[]> => {
  const response = await fetch(`${API_BASE_URL}/trending?hours=${hours}&top_n=${topN}`);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const data: TrendingResponse = await response.json();
  
  if (data.error) {
    throw new Error(data.error);
  }
  
  return data.trends || [];
};

// Get debug info about the system
export const fetchDebugInfo = async (): Promise<DebugInfo> => {
  const response = await fetch(`${API_BASE_URL}/debug/info`);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return response.json();
};

// Parse categories from JSON string
export const parseCategories = (categoriesStr: string): string[] => {
  try {
    return JSON.parse(categoriesStr);
  } catch {
    return [];
  }
};

// Format relative time
export const formatRelativeTime = (dateStr: string): string => {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  } catch {
    return dateStr;
  }
};

// Trigger manual news refresh
export const refreshNews = async (): Promise<{status: string; message: string; error?: string}> => {
  try {
    const response = await fetch(`${API_BASE_URL}/refresh`, {
      method: 'POST',
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Refresh news error:', error);
    return {
      status: 'error',
      message: 'Failed to trigger news refresh',
      error: String(error)
    };
  }
};

// Get refresh status
export const getRefreshStatus = async (): Promise<{total_articles: number; last_update: string; sources: number; error?: string}> => {
  try {
    const response = await fetch(`${API_BASE_URL}/refresh/status`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get refresh status error:', error);
    return {
      total_articles: 0,
      last_update: 'Unknown',
      sources: 0,
      error: String(error)
    };
  }
};

