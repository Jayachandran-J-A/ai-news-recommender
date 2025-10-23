import { Search, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  onReset: () => void;
}

export function EmptyState({ onReset }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      <div className="p-4 rounded-full bg-muted/50 mb-6">
        <Search className="h-12 w-12 text-muted-foreground" />
      </div>
      <h3 className="text-2xl font-bold mb-2">No articles found</h3>
      <p className="text-muted-foreground mb-6 text-center max-w-md">
        Try different keywords or categories, or reset your filters to see more news.
      </p>
      <Button
        onClick={onReset}
        variant="outline"
        className="gap-2"
      >
        <RefreshCw className="h-4 w-4" />
        Reset Filters
      </Button>
    </div>
  );
}
