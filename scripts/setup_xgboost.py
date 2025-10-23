#!/usr/bin/env python3
"""
Quick XGBoost model training for ensemble completion.
Creates a simple XGBoost model for the ensemble system.
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def create_synthetic_training_data():
    """Create synthetic training data for XGBoost model"""
    print("Creating synthetic training data for XGBoost...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Features: [cosine_similarity, recency_score, category_overlap]
    # Simulate realistic news recommendation scenarios
    
    # High quality articles: high similarity, recent, relevant categories
    high_quality = np.random.normal(0.8, 0.1, (n_samples // 2, 3))
    high_quality[:, 1] = np.clip(np.random.normal(0.8, 0.15, n_samples // 2), 0, 1)  # recency
    high_quality[:, 2] = np.random.choice([0, 1], n_samples // 2, p=[0.3, 0.7])  # category match
    high_labels = np.ones(n_samples // 2)
    
    # Low quality articles: lower similarity, older, irrelevant categories
    low_quality = np.random.normal(0.4, 0.15, (n_samples // 2, 3))
    low_quality[:, 1] = np.clip(np.random.normal(0.3, 0.2, n_samples // 2), 0, 1)  # recency
    low_quality[:, 2] = np.random.choice([0, 1], n_samples // 2, p=[0.8, 0.2])  # category match
    low_labels = np.zeros(n_samples // 2)
    
    # Combine data
    X = np.vstack([high_quality, low_quality])
    y = np.hstack([high_labels, low_labels])
    
    # Clip values to realistic ranges
    X[:, 0] = np.clip(X[:, 0], 0, 1)  # cosine similarity [0, 1]
    X[:, 1] = np.clip(X[:, 1], 0, 1)  # recency score [0, 1]
    X[:, 2] = np.clip(X[:, 2], 0, 1)  # category overlap [0, 1]
    
    return X, y

def train_xgboost_model():
    """Train XGBoost model for news recommendation ranking"""
    print("🚀 Training XGBoost model...")
    
    # Generate training data
    X, y = create_synthetic_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc'
    )
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"✅ Model trained! Test AUC: {auc:.4f}")
    
    # Save model
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgb_mind.json")
    
    # Convert to Booster and save (for compatibility with existing code)
    booster = model.get_booster()
    booster.save_model(model_path)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Test loading
    test_booster = xgb.Booster()
    test_booster.load_model(model_path)
    print("✅ Model loading test successful")
    
    return model_path

def main():
    print("🎯 Quick XGBoost Model Setup")
    print("This creates a basic XGBoost model for the ensemble system.")
    print()
    
    try:
        model_path = train_xgboost_model()
        
        print("\n" + "="*50)
        print("✅ SUCCESS!")
        print(f"XGBoost model ready at: {model_path}")
        print()
        print("🚀 Next steps:")
        print("1. Run: python test_complete_system.py")
        print("2. Start API: python -m uvicorn src.api:app --host 0.0.0.0 --port 8003")
        print("3. Test ensemble: your system now has XGBoost + NRMS!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()