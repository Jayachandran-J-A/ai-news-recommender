"""
MIND Dataset parser for PyTorch training.
Parses behaviors.tsv and news.tsv into training samples.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from fastembed import TextEmbedding


class MINDDataset(Dataset):
    """
    PyTorch Dataset for MIND (Microsoft News Dataset).
    
    Each sample is:
    - user_history: List of news IDs the user clicked
    - candidate_news: News ID to score
    - label: 1 if clicked, 0 if not clicked
    """
    
    def __init__(
        self,
        behaviors_file: str,
        news_file: str,
        embedding_model: TextEmbedding = None,
        max_history_len: int = 50,
        neg_sampling_ratio: int = 4,
        mode: str = 'train'
    ):
        """
        Args:
            behaviors_file: Path to behaviors.tsv
            news_file: Path to news.tsv
            embedding_model: FastEmbed model for generating embeddings
            max_history_len: Maximum number of history items per user
            neg_sampling_ratio: Negative samples per positive sample
            mode: 'train' or 'test'
        """
        self.max_history_len = max_history_len
        self.neg_sampling_ratio = neg_sampling_ratio
        self.mode = mode
        
        print(f"Loading MIND dataset in {mode} mode...")
        
        # Load news data
        print(f"Loading news from: {news_file}")
        self.news_data = self._load_news(news_file)
        print(f"Loaded {len(self.news_data)} news articles")
        
        # Generate or load embeddings
        if embedding_model is not None:
            print("Generating embeddings for news articles...")
            self.news_embeddings = self._generate_embeddings(embedding_model)
            print(f"Generated embeddings: {len(self.news_embeddings)} articles")
        else:
            self.news_embeddings = {}
        
        # Load behavior data
        print(f"Loading behaviors from: {behaviors_file}")
        self.samples = self._load_behaviors(behaviors_file)
        print(f"Created {len(self.samples)} training samples")
    
    def _load_news(self, news_file: str) -> Dict[str, Dict]:
        """
        Load news.tsv into memory.
        
        Format: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
        """
        news_data = {}
        
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else ''
                subcategory = parts[2] if len(parts) > 2 else ''
                title = parts[3] if len(parts) > 3 else ''
                abstract = parts[4] if len(parts) > 4 else ''
                
                news_data[news_id] = {
                    'title': title,
                    'abstract': abstract,
                    'category': category,
                    'subcategory': subcategory,
                    'text': f"{title}. {abstract}"  # Combined text for embedding
                }
        
        return news_data
    
    def _generate_embeddings(self, embedding_model: TextEmbedding) -> Dict[str, np.ndarray]:
        """Generate embeddings for all news articles."""
        news_embeddings = {}
        
        # Collect all texts
        news_ids = []
        texts = []
        
        for news_id, news_info in self.news_data.items():
            news_ids.append(news_id)
            texts.append(news_info['text'])
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = news_ids[i:i+batch_size]
            
            embeddings = list(embedding_model.embed(batch_texts))
            
            for news_id, emb in zip(batch_ids, embeddings):
                news_embeddings[news_id] = emb
            
            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {i + batch_size}/{len(texts)} articles")
        
        return news_embeddings
    
    def _load_behaviors(self, behaviors_file: str) -> List[Tuple]:
        """
        Load behaviors.tsv and create training samples.
        
        Format: impression_id, user_id, time, history, impressions
        - history: space-separated news IDs that user clicked before
        - impressions: space-separated news_id-label pairs (e.g., "N12345-1 N67890-0")
        """
        samples = []
        
        with open(behaviors_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                user_id = parts[1]
                history_str = parts[3]
                impressions_str = parts[4]
                
                # Parse history
                if history_str:
                    history = history_str.split()
                else:
                    history = []
                
                # Filter out history items not in news_data
                history = [nid for nid in history if nid in self.news_data]
                
                # Truncate history to max length (keep most recent)
                if len(history) > self.max_history_len:
                    history = history[-self.max_history_len:]
                
                # Skip users with no history
                if not history:
                    continue
                
                # Parse impressions
                impressions = impressions_str.split()
                pos_candidates = []
                neg_candidates = []
                
                for imp in impressions:
                    if '-' not in imp:
                        continue
                    news_id, label = imp.split('-')
                    
                    # Skip if news not in news_data
                    if news_id not in self.news_data:
                        continue
                    
                    if label == '1':
                        pos_candidates.append(news_id)
                    else:
                        neg_candidates.append(news_id)
                
                # Create samples: (history, candidate, label)
                # Positive samples
                for pos_news in pos_candidates:
                    samples.append((user_id, history.copy(), pos_news, 1))
                
                # Negative samples (subsample for training)
                if self.mode == 'train' and neg_candidates:
                    # Sample neg_sampling_ratio negatives per positive
                    n_neg = min(len(neg_candidates), len(pos_candidates) * self.neg_sampling_ratio)
                    sampled_neg = np.random.choice(neg_candidates, size=n_neg, replace=False)
                    
                    for neg_news in sampled_neg:
                        samples.append((user_id, history.copy(), neg_news, 0))
                else:
                    # In test mode, use all negatives
                    for neg_news in neg_candidates:
                        samples.append((user_id, history.copy(), neg_news, 0))
                
                # Progress
                if (line_num + 1) % 10000 == 0:
                    print(f"  Processed {line_num + 1} behavior logs, {len(samples)} samples created")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a single sample.
        
        Returns:
            Dictionary with:
            - user_id: User ID string
            - history_ids: List of news IDs in user history
            - history_embs: Tensor of shape (history_len, emb_dim)
            - candidate_id: Candidate news ID
            - candidate_emb: Tensor of shape (emb_dim,)
            - label: 1 or 0
        """
        user_id, history, candidate, label = self.samples[idx]
        
        # Get embeddings
        history_embs = []
        for news_id in history:
            if news_id in self.news_embeddings:
                history_embs.append(self.news_embeddings[news_id])
        
        if not history_embs:
            # Fallback: zero embedding
            history_embs = [np.zeros(384)]  # BGE-small embedding size
        
        candidate_emb = self.news_embeddings.get(
            candidate,
            np.zeros(384)
        )
        
        return {
            'user_id': user_id,
            'history_ids': history,
            'history_embs': torch.FloatTensor(np.array(history_embs)),
            'candidate_id': candidate,
            'candidate_emb': torch.FloatTensor(candidate_emb),
            'label': torch.FloatTensor([label])
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    Handles variable-length history sequences.
    """
    # Find max history length in batch
    max_hist_len = max(len(sample['history_ids']) for sample in batch)
    emb_dim = batch[0]['history_embs'].shape[1]
    
    # Pad history embeddings
    history_embs_padded = []
    history_masks = []
    
    for sample in batch:
        hist_len = sample['history_embs'].shape[0]
        
        # Pad to max_hist_len
        if hist_len < max_hist_len:
            padding = torch.zeros(max_hist_len - hist_len, emb_dim)
            padded_hist = torch.cat([sample['history_embs'], padding], dim=0)
            mask = torch.cat([
                torch.ones(hist_len),
                torch.zeros(max_hist_len - hist_len)
            ])
        else:
            padded_hist = sample['history_embs']
            mask = torch.ones(hist_len)
        
        history_embs_padded.append(padded_hist)
        history_masks.append(mask)
    
    # Stack into batch tensors
    return {
        'user_ids': [sample['user_id'] for sample in batch],
        'history_embs': torch.stack(history_embs_padded),  # (batch, max_hist_len, emb_dim)
        'history_masks': torch.stack(history_masks),  # (batch, max_hist_len)
        'candidate_embs': torch.stack([sample['candidate_emb'] for sample in batch]),  # (batch, emb_dim)
        'labels': torch.stack([sample['label'] for sample in batch])  # (batch, 1)
    }


def create_dataloaders(
    train_behaviors: str,
    val_behaviors: str,
    news_file: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_history_len: int = 50
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_behaviors: Path to train behaviors.tsv
        val_behaviors: Path to validation behaviors.tsv
        news_file: Path to news.tsv
        batch_size: Batch size
        num_workers: Number of worker processes
        max_history_len: Max history length
    
    Returns:
        (train_loader, val_loader)
    """
    # Initialize embedding model (shared)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Create datasets
    train_dataset = MINDDataset(
        behaviors_file=train_behaviors,
        news_file=news_file,
        embedding_model=embedding_model,
        max_history_len=max_history_len,
        neg_sampling_ratio=4,
        mode='train'
    )
    
    val_dataset = MINDDataset(
        behaviors_file=val_behaviors,
        news_file=news_file,
        embedding_model=embedding_model,
        max_history_len=max_history_len,
        neg_sampling_ratio=1,
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    import sys
    
    data_dir = "Dataset-archive/MINDsmall_train"
    
    if not os.path.exists(data_dir):
        print(f"Dataset not found at: {data_dir}")
        print("Please update the path to your MIND dataset")
        sys.exit(1)
    
    behaviors_file = os.path.join(data_dir, "behaviors.tsv")
    news_file = os.path.join(data_dir, "news.tsv")
    
    print("Testing MIND dataset loading...")
    
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Create small test dataset
    dataset = MINDDataset(
        behaviors_file=behaviors_file,
        news_file=news_file,
        embedding_model=embedding_model,
        max_history_len=20,
        neg_sampling_ratio=2,
        mode='train'
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Test sample retrieval
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  User ID: {sample['user_id']}")
    print(f"  History length: {len(sample['history_ids'])}")
    print(f"  History embedding shape: {sample['history_embs'].shape}")
    print(f"  Candidate embedding shape: {sample['candidate_emb'].shape}")
    print(f"  Label: {sample['label'].item()}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"  history_embs: {batch['history_embs'].shape}")
    print(f"  history_masks: {batch['history_masks'].shape}")
    print(f"  candidate_embs: {batch['candidate_embs'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    print("\n✓ Dataset loading test passed!")
