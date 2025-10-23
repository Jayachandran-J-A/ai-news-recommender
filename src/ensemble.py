import numpy as np
import torch
import xgboost as xgb

class EnsembleRecommender:
    """
    Combines NRMS (neural) and XGBoost (tree-based) models for news recommendation.
    Uses weighted average of both model scores for best accuracy.
    """
    def __init__(self, nrms_model_path, xgb_model_path, device='cpu', nrms_weight=0.6):
        # Load NRMS model
        from src.nrms import NRMS
        checkpoint = torch.load(nrms_model_path, map_location=device, weights_only=False)
        self.nrms = NRMS()
        self.nrms.load_state_dict(checkpoint['model_state_dict'])
        self.nrms.eval()
        self.nrms.to(device)
        self.device = device
        # Load XGBoost model
        self.xgb = xgb.Booster()
        self.xgb.load_model(xgb_model_path)
        # Ensemble weight (tune for best results)
        self.nrms_weight = nrms_weight
        self.xgb_weight = 1 - nrms_weight

    def predict(self, nrms_inputs, xgb_features):
        """
        nrms_inputs: dict with keys 'history_embs', 'history_mask', 'candidate_embs' (torch tensors)
        xgb_features: numpy array of shape [num_candidates, num_features]
        Returns: ensemble scores (numpy array)
        """
        # NRMS prediction
        with torch.no_grad():
            for k in nrms_inputs:
                nrms_inputs[k] = nrms_inputs[k].to(self.device)
            nrms_scores = self.nrms(
                nrms_inputs['history_embs'].unsqueeze(0),
                nrms_inputs['history_mask'].unsqueeze(0),
                nrms_inputs['candidate_embs'].unsqueeze(0)
            ).cpu().numpy().flatten()
        # XGBoost prediction
        dmatrix = xgb.DMatrix(xgb_features)
        xgb_scores = self.xgb.predict(dmatrix)
        # Weighted average
        ensemble_scores = self.nrms_weight * nrms_scores + self.xgb_weight * xgb_scores
        return ensemble_scores

    def set_weights(self, nrms_weight):
        self.nrms_weight = nrms_weight
        self.xgb_weight = 1 - nrms_weight
