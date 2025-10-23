import { useState } from "react";
import { Search, X, RefreshCw, Sparkles, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ModeToggle } from "@/components/mode-toggle";

interface NavigationProps {
  onSearch: (query: string) => void;
  onOpenInterests: () => void;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
  onManualRefresh: () => void;  // New prop for manual refresh
  isRefreshing?: boolean;  // New prop to show loading state
}

export function Navigation({
  onSearch,
  onOpenInterests,
  autoRefresh,
  onToggleAutoRefresh,
  onManualRefresh,
  isRefreshing = false,
}: NavigationProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(searchQuery);
  };

  const clearSearch = () => {
    setSearchQuery("");
    onSearch("");
  };

  return (
    <nav className="sticky top-0 z-40 w-full border-b bg-card/95 backdrop-blur-md shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col lg:flex-row items-center gap-4">
          {/* Logo & Branding */}
          <div className="flex items-center gap-3 lg:min-w-[240px]">
            <div className="p-2 rounded-xl bg-gradient-to-br from-primary to-primary-glow">
              <Sparkles className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
                NexusNews AI
              </h1>
              <p className="text-xs text-muted-foreground">Intelligent News Recommender</p>
            </div>
          </div>

          {/* Search Bar */}
          <form onSubmit={handleSearch} className="flex-1 w-full lg:max-w-2xl">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search news or enter keywords..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-10 h-11 bg-muted/50 border-border/50 focus:bg-card transition-colors"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={clearSearch}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              )}
            </div>
          </form>

          {/* Actions */}
          <div className="flex items-center gap-2">
            {/* Theme Toggle */}
            <ModeToggle />

            {/* Manual Refresh Button */}
            <Button
              onClick={onManualRefresh}
              variant="outline"
              className="gap-2"
              disabled={isRefreshing}
              title="Fetch latest news from sources"
            >
              <Download className={`h-4 w-4 ${isRefreshing ? "animate-bounce" : ""}`} />
              <span className="hidden sm:inline">
                {isRefreshing ? "Fetching..." : "Refresh News"}
              </span>
            </Button>

            <Button
              onClick={onOpenInterests}
              variant="outline"
              className="gap-2"
            >
              <Sparkles className="h-4 w-4" />
              <span className="hidden sm:inline">Interests</span>
            </Button>

            <button
              onClick={onToggleAutoRefresh}
              title={autoRefresh ? "Disable auto-refresh" : "Enable auto-refresh"}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200
                ${
                  autoRefresh
                    ? "bg-success/10 border-success text-success hover:bg-success/20"
                    : "border-border hover:bg-muted"
                }
              `}
            >
              <RefreshCw
                className={`h-4 w-4 ${autoRefresh ? "animate-spin" : ""}`}
              />
              <span className="text-sm font-medium hidden sm:inline">
                {autoRefresh ? "Live" : "Paused"}
              </span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
