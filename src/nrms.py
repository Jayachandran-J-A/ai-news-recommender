"""
NRMS: Neural News Recommendation with Multi-Head Self-Attention.

Based on: "Neural News Recommendation with Multi-Head Self-Attention" (EMNLP 2019)
Paper: https://www.aclweb.org/anthology/D19-1671/

Architecture:
1. News Encoder: Multi-head self-attention over news embeddings
2. User Encoder: Multi-head self-attention over clicked news history
3. Click Predictor: Dot product + sigmoid for CTR prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            output: Attended features of shape (batch, seq_len, d_model)
            attention_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            # Expand mask for broadcasting: (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # attended: (batch, num_heads, seq_len, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        attended = attended.view(batch_size, seq_len, self.d_model)
        # (batch, seq_len, d_model)
        
        # Output projection
        output = self.W_o(attended)
        output = self.dropout(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(x + output)
        
        return output, attention_weights


class NewsEncoder(nn.Module):
    """
    Encode a single news article using multi-head self-attention.
    """
    
    def __init__(self, emb_dim: int = 384, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            emb_dim: Dimension of input embeddings (from FastEmbed)
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        # Project embedding to hidden dimension
        self.projection = nn.Linear(emb_dim, hidden_dim)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # Query vector for attention pooling
        self.query = nn.Parameter(torch.randn(hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, news_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            news_emb: News embedding of shape (batch, emb_dim)
        
        Returns:
            news_repr: News representation of shape (batch, hidden_dim)
            attention_weights: Attention weights for interpretability
        """
        batch_size = news_emb.shape[0]
        
        # Project to hidden dimension
        x = self.projection(news_emb)  # (batch, hidden_dim)
        
        # Add sequence dimension for self-attention
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply self-attention
        x, attn_weights = self.attention(x)  # (batch, 1, hidden_dim)
        
        # Remove sequence dimension
        news_repr = x.squeeze(1)  # (batch, hidden_dim)
        
        return news_repr, attn_weights


class UserEncoder(nn.Module):
    """
    Encode user from browsing history using multi-head self-attention.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multi-head self-attention over history
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # Query vector for attention pooling
        self.query = nn.Parameter(torch.randn(hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, history_reprs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history_reprs: History news representations of shape (batch, hist_len, hidden_dim)
            mask: Optional mask of shape (batch, hist_len)
        
        Returns:
            user_repr: User representation of shape (batch, hidden_dim)
            attention_weights: Attention weights for interpretability
        """
        # Apply self-attention over history
        attended, attn_weights = self.attention(history_reprs, mask)
        # attended: (batch, hist_len, hidden_dim)
        
        # Additive attention pooling
        # Compute attention scores using learnable query
        query = self.query.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        scores = torch.matmul(attended, query.transpose(-2, -1)).squeeze(-1)
        # scores: (batch, hist_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get pooling weights
        pooling_weights = F.softmax(scores, dim=-1)  # (batch, hist_len)
        
        # Weighted sum
        user_repr = torch.matmul(pooling_weights.unsqueeze(1), attended).squeeze(1)
        # user_repr: (batch, hidden_dim)
        
        return user_repr, pooling_weights


class NRMS(nn.Module):
    """
    Neural News Recommendation with Multi-Head Self-Attention.
    """
    
    def __init__(
        self,
        emb_dim: int = 384,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            emb_dim: Dimension of input embeddings (from FastEmbed)
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        # News encoder
        self.news_encoder = NewsEncoder(emb_dim, hidden_dim, num_heads, dropout)
        
        # User encoder
        self.user_encoder = UserEncoder(hidden_dim, num_heads, dropout)
    
    def encode_news(self, news_emb: torch.Tensor) -> torch.Tensor:
        """Encode a single news article."""
        news_repr, _ = self.news_encoder(news_emb)
        return news_repr
    
    def encode_user(self, history_embs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode user from history of news embeddings."""
        batch_size, hist_len, emb_dim = history_embs.shape
        
        # Encode each news in history
        history_embs_flat = history_embs.view(batch_size * hist_len, emb_dim)
        history_reprs_flat, _ = self.news_encoder(history_embs_flat)
        history_reprs = history_reprs_flat.view(batch_size, hist_len, self.hidden_dim)
        
        # Encode user from history representations
        user_repr, _ = self.user_encoder(history_reprs, mask)
        
        return user_repr
    
    def forward(
        self,
        history_embs: torch.Tensor,
        candidate_emb: torch.Tensor,
        history_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            history_embs: User history embeddings of shape (batch, hist_len, emb_dim)
            candidate_emb: Candidate news embedding of shape (batch, emb_dim)
            history_mask: Optional mask of shape (batch, hist_len)
        
        Returns:
            scores: Click probability of shape (batch, 1)
        """
        # Encode user
        user_repr = self.encode_user(history_embs, history_mask)
        
        # Encode candidate news
        candidate_repr = self.encode_news(candidate_emb)
        
        # Compute click score: dot product + sigmoid
        scores = torch.sum(user_repr * candidate_repr, dim=-1, keepdim=True)
        scores = torch.sigmoid(scores)
        
        return scores
    
    def predict_batch(
        self,
        history_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
        history_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict scores for multiple candidates per user.
        
        Args:
            history_embs: User history embeddings of shape (batch, hist_len, emb_dim)
            candidate_embs: Candidate news embeddings of shape (batch, num_candidates, emb_dim)
            history_mask: Optional mask of shape (batch, hist_len)
        
        Returns:
            scores: Click probabilities of shape (batch, num_candidates)
        """
        batch_size, num_candidates, emb_dim = candidate_embs.shape
        
        # Encode user (once per batch)
        user_repr = self.encode_user(history_embs, history_mask)
        # user_repr: (batch, hidden_dim)
        
        # Encode all candidate news
        candidate_embs_flat = candidate_embs.view(batch_size * num_candidates, emb_dim)
        candidate_reprs_flat = self.encode_news(candidate_embs_flat)
        candidate_reprs = candidate_reprs_flat.view(batch_size, num_candidates, self.hidden_dim)
        
        # Compute scores: batch matrix multiplication
        # user_repr: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        # candidate_reprs: (batch, num_candidates, hidden_dim) -> (batch, hidden_dim, num_candidates)
        scores = torch.bmm(
            user_repr.unsqueeze(1),
            candidate_reprs.transpose(1, 2)
        ).squeeze(1)
        # scores: (batch, num_candidates)
        
        scores = torch.sigmoid(scores)
        
        return scores


if __name__ == '__main__':
    # Test NRMS model
    print("Testing NRMS model...")
    
    # Model parameters
    emb_dim = 384  # BGE-small embedding size
    hidden_dim = 256
    num_heads = 8
    batch_size = 16
    hist_len = 20
    num_candidates = 5
    
    # Create model
    model = NRMS(emb_dim=emb_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    history_embs = torch.randn(batch_size, hist_len, emb_dim)
    candidate_emb = torch.randn(batch_size, emb_dim)
    history_mask = torch.ones(batch_size, hist_len)
    history_mask[:, hist_len//2:] = 0  # Mask out second half
    
    # Test forward pass (single candidate)
    print("\n1. Testing single candidate prediction...")
    scores = model(history_embs, candidate_emb, history_mask)
    print(f"   Input shapes: history={history_embs.shape}, candidate={candidate_emb.shape}")
    print(f"   Output shape: {scores.shape}")
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Test batch prediction (multiple candidates)
    print("\n2. Testing batch prediction (multiple candidates)...")
    candidate_embs = torch.randn(batch_size, num_candidates, emb_dim)
    scores_batch = model.predict_batch(history_embs, candidate_embs, history_mask)
    print(f"   Input shapes: history={history_embs.shape}, candidates={candidate_embs.shape}")
    print(f"   Output shape: {scores_batch.shape}")
    print(f"   Score range: [{scores_batch.min():.4f}, {scores_batch.max():.4f}]")
    
    # Test backward pass
    print("\n3. Testing backward pass...")
    loss = F.binary_cross_entropy(scores, torch.rand(batch_size, 1))
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed successfully")
    
    # Test individual encoders
    print("\n4. Testing individual encoders...")
    news_repr = model.encode_news(candidate_emb)
    print(f"   News encoding shape: {news_repr.shape}")
    
    user_repr = model.encode_user(history_embs, history_mask)
    print(f"   User encoding shape: {user_repr.shape}")
    
    print("\n✓ All NRMS model tests passed!")
