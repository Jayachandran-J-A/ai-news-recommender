"""Quick test for MIND dataset loading."""
import os
import sys
from fastembed import TextEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mind_dataset import MINDDataset
from torch.utils.data import DataLoader
from src.mind_dataset import collate_fn

print("Testing MIND dataset loading with small sample...")

data_dir = "Dataset-archive/MINDsmall_train"
behaviors_file = os.path.join(data_dir, "behaviors_test.tsv")
news_file = os.path.join(data_dir, "news.tsv")

if not os.path.exists(behaviors_file):
    print(f"Test file not found: {behaviors_file}")
    print("Creating test file from first 100 lines...")
    os.system(f'Get-Content "Dataset-archive\\MINDsmall_train\\behaviors.tsv" -Head 100 | Out-File -FilePath "{behaviors_file}" -Encoding UTF8')

print("\n1. Initializing embedding model...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("\n2. Creating dataset...")
dataset = MINDDataset(
    behaviors_file=behaviors_file,
    news_file=news_file,
    embedding_model=embedding_model,
    max_history_len=20,
    neg_sampling_ratio=2,
    mode='train'
)

print(f"\n3. Dataset created with {len(dataset)} samples")

if len(dataset) > 0:
    print("\n4. Testing sample retrieval...")
    sample = dataset[0]
    print(f"   User ID: {sample['user_id']}")
    print(f"   History length: {len(sample['history_ids'])}")
    print(f"   History embedding shape: {sample['history_embs'].shape}")
    print(f"   Candidate embedding shape: {sample['candidate_emb'].shape}")
    print(f"   Label: {sample['label'].item()}")
    
    print("\n5. Testing dataloader...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"   Batch shapes:")
    print(f"     history_embs: {batch['history_embs'].shape}")
    print(f"     history_masks: {batch['history_masks'].shape}")
    print(f"     candidate_embs: {batch['candidate_embs'].shape}")
    print(f"     labels: {batch['labels'].shape}")
    
    print("\n✓ Dataset loading test passed!")
else:
    print("\n✗ No samples created. Check dataset files.")
