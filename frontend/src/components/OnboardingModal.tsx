import { useState } from "react";
import { X, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: (selectedCategories: string[]) => void;
}

const CATEGORIES = [
  { id: "technology", label: "Technology", icon: "💻" },
  { id: "ai-ml", label: "AI/ML", icon: "🤖" },
  { id: "politics", label: "Politics", icon: "🏛️" },
  { id: "business", label: "Business", icon: "💼" },
  { id: "health", label: "Health", icon: "🏥" },
  { id: "science", label: "Science", icon: "🔬" },
  { id: "sports", label: "Sports", icon: "⚽" },
  { id: "entertainment", label: "Entertainment", icon: "🎬" },
  { id: "world", label: "World News", icon: "🌍" },
  { id: "climate", label: "Climate", icon: "🌱" },
  { id: "finance", label: "Finance", icon: "📈" },
  { id: "education", label: "Education", icon: "📚" },
];

export function OnboardingModal({ isOpen, onClose }: OnboardingModalProps) {
  const [selected, setSelected] = useState<string[]>([]);

  if (!isOpen) return null;

  const toggleCategory = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]
    );
  };

  const handleSubmit = () => {
    onClose(selected.length > 0 ? selected : ["technology", "world"]);
  };

  const handleSkip = () => {
    onClose(["technology", "world"]);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-card rounded-2xl shadow-xl border border-border animate-in zoom-in-95 duration-300">
        <button
          onClick={handleSkip}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-muted transition-colors text-foreground"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="p-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-primary to-primary-glow">
              <Sparkles className="h-6 w-6 text-primary-foreground" />
            </div>
            <h2 className="text-3xl font-bold text-foreground">Discover News That Matters to You</h2>
          </div>
          
          <p className="text-muted-foreground text-lg mb-8">
            Select topics you're interested in to get personalized recommendations
          </p>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-8">
            {CATEGORIES.map((category) => (
              <button
                key={category.id}
                onClick={() => toggleCategory(category.id)}
                className={`
                  p-4 rounded-xl border-2 transition-all duration-200 text-foreground
                  ${
                    selected.includes(category.id)
                      ? "border-primary bg-primary/10 shadow-md scale-105"
                      : "border-border hover:border-primary/50 hover:bg-muted/50"
                  }
                `}
              >
                <div className="text-2xl mb-2">{category.icon}</div>
                <div className="text-sm font-medium">{category.label}</div>
              </button>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              onClick={handleSubmit}
              className="flex-1 h-12 text-base font-semibold bg-gradient-to-r from-primary to-primary-glow hover:shadow-glow transition-all duration-300"
              disabled={selected.length === 0}
            >
              Get Personalized Feed
              {selected.length > 0 && (
                <span className="ml-2 px-2 py-0.5 rounded-full bg-primary-foreground/20 text-xs">
                  {selected.length}
                </span>
              )}
            </Button>
            <Button
              onClick={handleSkip}
              variant="outline"
              className="sm:w-auto h-12"
            >
              Skip for now
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
