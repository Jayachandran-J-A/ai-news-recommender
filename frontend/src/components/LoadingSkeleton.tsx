export function LoadingSkeleton() {
  return (
    <div className="bg-card rounded-xl border shadow-sm overflow-hidden animate-pulse">
      <div className="aspect-video w-full bg-muted shimmer" />
      <div className="p-5 space-y-3">
        <div className="flex gap-2">
          <div className="h-6 w-20 bg-muted rounded-full shimmer" />
          <div className="h-6 w-24 bg-muted rounded-full shimmer" />
        </div>
        <div className="space-y-2">
          <div className="h-5 bg-muted rounded shimmer" />
          <div className="h-5 bg-muted rounded w-3/4 shimmer" />
        </div>
        <div className="h-4 w-32 bg-muted rounded shimmer" />
        <div className="h-10 bg-muted rounded-lg shimmer" />
      </div>
    </div>
  );
}

export function LoadingGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {[...Array(6)].map((_, i) => (
        <LoadingSkeleton key={i} />
      ))}
    </div>
  );
}
