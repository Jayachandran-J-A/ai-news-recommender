import { Filter } from "lucide-react";

interface CategoryFilterProps {
  categories: string[];
  selectedCategories: string[];
  onToggleCategory: (category: string) => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  all: "All",
  technology: "Technology",
  "ai-ml": "AI/ML",
  politics: "Politics",
  business: "Business",
  health: "Health",
  science: "Science",
  sports: "Sports",
  entertainment: "Entertainment",
  world: "World News",
  climate: "Climate",
  finance: "Finance",
  education: "Education",
};

export function CategoryFilter({
  categories,
  selectedCategories,
  onToggleCategory,
}: CategoryFilterProps) {
  const isSelected = (category: string) => {
    if (category === "all") return selectedCategories.length === 0;
    return selectedCategories.includes(category);
  };

  return (
    <div className="w-full border-b bg-card/50 backdrop-blur-sm sticky top-[88px] z-30">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center gap-3 overflow-x-auto scrollbar-hide">
          <div className="flex items-center gap-2 text-muted-foreground flex-shrink-0">
            <Filter className="h-4 w-4" />
            <span className="text-sm font-medium">Filter:</span>
          </div>

          <div className="flex gap-2">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => onToggleCategory(category)}
                className={`
                  px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 whitespace-nowrap
                  ${
                    isSelected(category)
                      ? "bg-primary text-primary-foreground shadow-md"
                      : "bg-muted hover:bg-muted/80 text-foreground border border-border"
                  }
                `}
              >
                {CATEGORY_LABELS[category] || category}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
