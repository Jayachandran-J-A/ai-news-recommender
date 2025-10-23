"""
Comprehensive evaluation metrics for news recommendation systems.
Implements NDCG@k, MRR, Hit Rate, AUC, diversity, and other metrics.
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at rank k.
    
    Args:
        relevance_scores: Binary relevance labels (1 = clicked, 0 = not clicked)
        k: Cutoff rank
    
    Returns:
        DCG@k score
    """
    relevance_scores = np.array(relevance_scores)[:k]
    if len(relevance_scores) == 0:
        return 0.0
    
    # DCG formula: sum(rel_i / log2(i + 2)) for i in range(k)
    gains = relevance_scores / np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains)


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at rank k.
    
    Args:
        relevance_scores: Binary relevance labels (1 = clicked, 0 = not clicked)
        k: Cutoff rank
    
    Returns:
        NDCG@k score (normalized to [0, 1])
    """
    dcg = dcg_at_k(relevance_scores, k)
    
    # Ideal DCG: sort relevance scores in descending order
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr_score(relevance_scores: List[float]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first relevant item.
    
    Args:
        relevance_scores: Binary relevance labels (1 = clicked, 0 = not clicked)
    
    Returns:
        MRR score
    """
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Hit Rate@k: 1 if any relevant item in top-k, else 0.
    
    Args:
        relevance_scores: Binary relevance labels
        k: Cutoff rank
    
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = relevance_scores[:k]
    return 1.0 if any(rel > 0 for rel in top_k) else 0.0


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Area Under ROC Curve.
    
    Args:
        y_true: Binary labels (1 = clicked, 0 = not clicked)
        y_pred: Predicted scores
    
    Returns:
        AUC score
    """
    from sklearn.metrics import roc_auc_score
    
    if len(np.unique(y_true)) < 2:
        # Only one class present
        return 0.5
    
    return roc_auc_score(y_true, y_pred)


def diversity_score(recommended_items: List[str], item_categories: Dict[str, str]) -> float:
    """
    Intra-list diversity: ratio of unique categories in recommendations.
    
    Args:
        recommended_items: List of recommended item IDs
        item_categories: Mapping from item_id to category
    
    Returns:
        Diversity score (0 to 1)
    """
    if not recommended_items:
        return 0.0
    
    categories = [item_categories.get(item, 'unknown') for item in recommended_items]
    unique_categories = len(set(categories))
    
    return unique_categories / len(categories)


def novelty_score(recommended_items: List[str], item_popularity: Dict[str, int], total_users: int) -> float:
    """
    Novelty: -log2(popularity) averaged over recommendations.
    Recommending less popular items = higher novelty.
    
    Args:
        recommended_items: List of recommended item IDs
        item_popularity: Mapping from item_id to click count
        total_users: Total number of users in system
    
    Returns:
        Novelty score (higher = more novel)
    """
    if not recommended_items:
        return 0.0
    
    novelties = []
    for item in recommended_items:
        popularity = item_popularity.get(item, 0)
        # Add smoothing to avoid log(0)
        prob = (popularity + 1) / (total_users + 1)
        novelties.append(-np.log2(prob))
    
    return np.mean(novelties)


class RecommenderEvaluator:
    """
    Comprehensive evaluator for recommendation models.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Args:
            k_values: List of k values for top-k metrics (e.g., [5, 10, 20])
        """
        self.k_values = k_values
        self.results = defaultdict(list)
    
    def evaluate_single_user(
        self,
        predictions: List[Tuple[str, float]],  # (item_id, score)
        ground_truth: List[str],  # clicked item IDs
        item_categories: Dict[str, str] = None,
        item_popularity: Dict[str, int] = None,
        total_users: int = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            predictions: List of (item_id, score) tuples sorted by score descending
            ground_truth: List of clicked item IDs
            item_categories: Optional category mapping for diversity
            item_popularity: Optional popularity mapping for novelty
            total_users: Optional total user count for novelty
        
        Returns:
            Dictionary of metric scores
        """
        if not predictions:
            return {f'ndcg@{k}': 0.0 for k in self.k_values}
        
        # Extract predicted item IDs
        predicted_items = [item_id for item_id, _ in predictions]
        
        # Create binary relevance vector
        relevance = [1.0 if item_id in ground_truth else 0.0 for item_id in predicted_items]
        
        metrics = {}
        
        # NDCG@k
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = ndcg_at_k(relevance, k)
        
        # MRR
        metrics['mrr'] = mrr_score(relevance)
        
        # Hit Rate@k
        for k in self.k_values:
            metrics[f'hit@{k}'] = hit_rate_at_k(relevance, k)
        
        # Diversity (if category info provided)
        if item_categories:
            for k in self.k_values:
                top_k_items = predicted_items[:k]
                metrics[f'diversity@{k}'] = diversity_score(top_k_items, item_categories)
        
        # Novelty (if popularity info provided)
        if item_popularity and total_users:
            for k in self.k_values:
                top_k_items = predicted_items[:k]
                metrics[f'novelty@{k}'] = novelty_score(top_k_items, item_popularity, total_users)
        
        return metrics
    
    def add_user_result(self, metrics: Dict[str, float]):
        """Add metrics for a single user to aggregate results."""
        for metric_name, value in metrics.items():
            self.results[metric_name].append(value)
    
    def get_aggregate_results(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all users.
        
        Returns:
            Dictionary with mean and std for each metric
        """
        aggregate = {}
        
        for metric_name, values in self.results.items():
            if values:
                aggregate[f'{metric_name}_mean'] = np.mean(values)
                aggregate[f'{metric_name}_std'] = np.std(values)
        
        return aggregate
    
    def print_results(self):
        """Print formatted evaluation results."""
        results = self.get_aggregate_results()
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Group by metric type
        metric_groups = {
            'NDCG': [k for k in results.keys() if 'ndcg' in k and 'mean' in k],
            'Hit Rate': [k for k in results.keys() if 'hit' in k and 'mean' in k],
            'MRR': [k for k in results.keys() if 'mrr' in k and 'mean' in k],
            'Diversity': [k for k in results.keys() if 'diversity' in k and 'mean' in k],
            'Novelty': [k for k in results.keys() if 'novelty' in k and 'mean' in k],
        }
        
        for group_name, metrics in metric_groups.items():
            if metrics:
                print(f"\n{group_name}:")
                for metric in sorted(metrics):
                    mean_val = results[metric]
                    std_key = metric.replace('_mean', '_std')
                    std_val = results.get(std_key, 0)
                    print(f"  {metric.replace('_mean', ''):20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n" + "="*60)


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    output_file: str = None
) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Args:
        results_dict: Dictionary mapping model_name -> metric_results
        output_file: Optional CSV file to save comparison
    
    Returns:
        DataFrame with comparison table
    """
    # Extract mean metrics for each model
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {'model': model_name}
        for metric_name, value in results.items():
            if '_mean' in metric_name:
                clean_name = metric_name.replace('_mean', '')
                row[clean_name] = value
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric (NDCG@10)
    if 'ndcg@10' in df.columns:
        df = df.sort_values('ndcg@10', ascending=False)
    
    # Print formatted table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Comparison saved to: {output_file}")
    
    return df


if __name__ == '__main__':
    # Example usage
    print("Testing evaluation metrics...")
    
    # Simulate predictions and ground truth
    predictions = [
        ('item1', 0.95),
        ('item2', 0.87),
        ('item3', 0.76),
        ('item4', 0.65),
        ('item5', 0.54),
        ('item6', 0.43),
        ('item7', 0.32),
        ('item8', 0.21),
        ('item9', 0.15),
        ('item10', 0.08),
    ]
    
    ground_truth = ['item2', 'item5', 'item11']  # item11 not in predictions
    
    # Test individual metrics
    predicted_items = [item_id for item_id, _ in predictions]
    relevance = [1.0 if item_id in ground_truth else 0.0 for item_id in predicted_items]
    
    print(f"\nPredicted items: {predicted_items[:5]}...")
    print(f"Ground truth: {ground_truth}")
    print(f"Relevance vector: {relevance}")
    
    print(f"\nNDCG@5:  {ndcg_at_k(relevance, 5):.4f}")
    print(f"NDCG@10: {ndcg_at_k(relevance, 10):.4f}")
    print(f"MRR:     {mrr_score(relevance):.4f}")
    print(f"Hit@5:   {hit_rate_at_k(relevance, 5):.4f}")
    print(f"Hit@10:  {hit_rate_at_k(relevance, 10):.4f}")
    
    # Test evaluator
    evaluator = RecommenderEvaluator(k_values=[5, 10])
    
    for i in range(5):  # Simulate 5 users
        metrics = evaluator.evaluate_single_user(predictions, ground_truth)
        evaluator.add_user_result(metrics)
    
    evaluator.print_results()
