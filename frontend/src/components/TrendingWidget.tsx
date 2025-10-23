import { Flame } from "lucide-react";

interface TrendingWidgetProps {
  keywords: string[];
  onKeywordClick: (keyword: string) => void;
}

export function TrendingWidget({ keywords, onKeywordClick }: TrendingWidgetProps) {
  if (keywords.length === 0) return null;

  return (
    <div className="sticky top-[152px] bg-card rounded-2xl border shadow-sm p-6 hover-lift">
      <div className="flex items-center gap-2 mb-4">
        <div className="p-2 rounded-lg bg-accent/10">
          <Flame className="h-5 w-5 text-accent" />
        </div>
        <h3 className="text-lg font-bold">Trending Now</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {keywords.map((keyword, index) => (
          <button
            key={index}
            onClick={() => onKeywordClick(keyword)}
            className="px-3 py-1.5 rounded-lg bg-accent/5 hover:bg-accent/10 border border-accent/20 text-sm font-medium text-accent-foreground transition-all duration-200 hover:scale-105"
          >
            #{keyword}
          </button>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t text-xs text-muted-foreground">
        Updated from last 72 hours
      </div>
    </div>
  );
}
