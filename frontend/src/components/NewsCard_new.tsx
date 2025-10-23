import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { parseCategories, formatRelativeTime } from "@/lib/api";
import { Clock, TrendingUp } from "lucide-react";

interface NewsCardProps {
  title: string;
  source: string;
  published: string;
  url: string;
  categories?: string;
  match_percentage?: number;
  onRead: (url: string) => void;
}

export const NewsCard = ({ title, source, published, url, categories, match_percentage, onRead }: NewsCardProps) => {
  const categoryList = parseCategories(categories);
  const timeAgo = formatRelativeTime(published);
  
  // Determine badge color based on match percentage
  const getMatchBadgeColor = (score: number) => {
    if (score >= 90) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 border-green-300";
    if (score >= 80) return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 border-blue-300";
    if (score >= 70) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 border-yellow-300";
    return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200 border-gray-300";
  };

  return (
    <Card className="hover:shadow-lg transition-all duration-200 cursor-pointer group hover:border-primary/50" onClick={() => onRead(url)}>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <CardTitle className="group-hover:text-primary transition-colors line-clamp-2 flex-1 text-base">
            {title}
          </CardTitle>
          {match_percentage !== undefined && match_percentage > 0 && (
            <Badge className={`${getMatchBadgeColor(match_percentage)} flex items-center gap-1 shrink-0 font-semibold border`}>
              <TrendingUp className="w-3 h-3" />
              {match_percentage.toFixed(0)}%
            </Badge>
          )}
        </div>
        <CardDescription className="flex items-center gap-2 text-sm flex-wrap mt-1.5">
          <span className="font-medium text-foreground/80">{source}</span>
          <span className="text-muted-foreground">•</span>
          <span className="flex items-center gap-1 text-muted-foreground">
            <Clock className="w-3 h-3" />
            {timeAgo}
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-1.5">
          {categoryList.map((cat) => (
            <Badge key={cat} variant="secondary" className="text-xs capitalize">
              {cat}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
