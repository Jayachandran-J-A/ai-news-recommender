import { AlertCircle, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorBannerProps {
  message: string;
  onRetry: () => void;
  onDismiss: () => void;
}

export function ErrorBanner({ message, onRetry, onDismiss }: ErrorBannerProps) {
  return (
    <div className="fixed top-24 left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-4 animate-in slide-in-from-top duration-300">
      <div className="bg-destructive/10 border border-destructive/50 rounded-lg p-4 shadow-lg backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">
              {message}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={onRetry}
              size="sm"
              variant="outline"
              className="gap-2 border-destructive/50 hover:bg-destructive/20"
            >
              <RefreshCw className="h-3 w-3" />
              Retry
            </Button>
            <button
              onClick={onDismiss}
              className="p-1 rounded hover:bg-destructive/20 transition-colors"
            >
              <X className="h-4 w-4 text-destructive" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
