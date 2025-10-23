"""
Train NRMS model on MIND dataset.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nrms import NRMS
from src.mind_dataset import MINDDataset, collate_fn
from src.metrics import RecommenderEvaluator, auc_score
from fastembed import TextEmbedding


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        history_embs = batch['history_embs'].to(device)
        history_masks = batch['history_masks'].to(device)
        candidate_embs = batch['candidate_embs'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        scores = model(history_embs, candidate_embs, history_masks)
        
        # Compute loss
        loss = criterion(scores, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_loss = loss.item() * len(labels)
        total_loss += batch_loss
        total_samples += len(labels)
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Free memory
        del history_embs, history_masks, candidate_embs, labels, scores, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / total_samples


def evaluate_epoch(model, dataloader, criterion, device, epoch):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_scores = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            history_embs = batch['history_embs'].to(device)
            history_masks = batch['history_masks'].to(device)
            candidate_embs = batch['candidate_embs'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            scores = model(history_embs, candidate_embs, history_masks)
            
            # Compute loss
            loss = criterion(scores, labels)
            
            # Update metrics
            batch_loss = loss.item() * len(labels)
            total_loss += batch_loss
            total_samples += len(labels)
            
            # Store predictions for AUC
            all_scores.extend(scores.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Update progress bar
            avg_loss = total_loss / total_samples
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Free memory
            del history_embs, history_masks, candidate_embs, labels, scores, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples
    
    # Compute AUC
    auc = auc_score(np.array(all_labels), np.array(all_scores))
    
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser(description='Train NRMS model on MIND dataset')
    parser.add_argument('--data_dir', type=str, default='Dataset-archive/MINDsmall_train',
                        help='Path to MIND dataset directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--max_history_len', type=int, default=50,
                        help='Maximum history length')
    parser.add_argument('--neg_sampling_ratio', type=int, default=4,
                        help='Negative sampling ratio')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NRMS TRAINING ON MIND DATASET")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num heads: {args.num_heads}")
    print(f"Dropout: {args.dropout}")
    print(f"Max history length: {args.max_history_len}")
    print(f"Negative sampling ratio: {args.neg_sampling_ratio}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # File paths
    behaviors_file = os.path.join(args.data_dir, 'behaviors.tsv')
    news_file = os.path.join(args.data_dir, 'news.tsv')
    
    if not os.path.exists(behaviors_file) or not os.path.exists(news_file):
        print(f"\nError: Dataset files not found at {args.data_dir}")
        print("Please ensure behaviors.tsv and news.tsv exist in the data directory")
        return
    
    # Initialize embedding model (shared for all datasets)
    print("\nInitializing embedding model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Create datasets
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    train_dataset = MINDDataset(
        behaviors_file=behaviors_file,
        news_file=news_file,
        embedding_model=embedding_model,
        max_history_len=args.max_history_len,
        neg_sampling_ratio=args.neg_sampling_ratio,
        mode='train'
    )
    
    # Split into train/val
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nSplit dataset: {train_size} train, {val_size} validation")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    model = NRMS(
        emb_dim=384,  # BGE-small embedding size
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_auc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_auc = evaluate_epoch(model, val_loader, criterion, device, epoch)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            model_path = os.path.join(args.output_dir, 'nrms_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
                'args': vars(args)
            }, model_path)
            print(f"  ✓ Saved best model (AUC: {val_auc:.4f}) to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'nrms_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'val_loss': val_loss,
        'args': vars(args),
        'history': history
    }, final_model_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'nrms_best.pt')}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
