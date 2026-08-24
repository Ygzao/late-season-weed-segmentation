from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label

# 3x3 structuring element for 8-neighborhood connectivity (N_8)
STRUCT_8 = np.ones((3, 3), dtype=int)


class FocalTverskyLoss(nn.Module):
    """
    Standard Focal Tversky Loss computed globally at the pixel level.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Flatten to compute pixel-level intersections/unions
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        
        tp = (probs_f * targets_f).sum(dim=1)
        fp = (probs_f * (1.0 - targets_f)).sum(dim=1)
        fn = ((1.0 - probs_f) * targets_f).sum(dim=1)
        
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        per_sample = torch.pow((1.0 - tversky), self.gamma)
        
        if weights is not None:
            return (per_sample * weights).mean()
        return per_sample.mean()


class ComponentBalancedLoss(nn.Module):
    """
    Component-Balanced Loss (L_CB).
    Isolates 8-connected components in the ground-truth mask and averages Tversky loss
    individually across components to equalize size-based gradient imbalances.
    Supports optional minimum component footprint filtering (min_size) and
    bounded area-scaling regularization (cap_threshold).
    """
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        eps: float = 1e-6,
        min_size: int = 0,
        cap_threshold: float | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.min_size = min_size
        self.cap_threshold = cap_threshold

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        batch_size = logits.size(0)
        loss_sum = 0.0
        
        targets_np = targets.detach().cpu().numpy()
        
        for i in range(batch_size):
            target_tile = targets_np[i, 0]
            prob_tile = probs[i, 0]
            target_tile_tensor = targets[i, 0]
            
            # Label disjoint components using 8-neighborhood connectivity (N_8)
            cc, num_ref = label(target_tile, structure=STRUCT_8)
            
            valid_components = []
            for r_idx in range(1, num_ref + 1):
                comp_size = int((cc == r_idx).sum())
                if comp_size >= self.min_size:
                    valid_components.append((r_idx, comp_size))
                    
            if len(valid_components) == 0:
                # Background-only tile (K = 0): defaults to tile-level Focal-Tversky Loss
                tp = (prob_tile * target_tile_tensor).sum()
                fp = (prob_tile * (1.0 - target_tile_tensor)).sum()
                fn = ((1.0 - prob_tile) * target_tile_tensor).sum()
                tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
                loss_sum += torch.pow((1.0 - tversky), self.gamma)
            else:
                # Component balancing across K ground-truth components
                comp_loss_sum = 0.0
                weight_sum = 0.0
                for r_idx, comp_size in valid_components:
                    comp_mask = torch.from_numpy(cc == r_idx).to(logits.device).float()
                    tp = (prob_tile * comp_mask).sum()
                    fn = ((1.0 - prob_tile) * comp_mask).sum()
                    
                    # Penalize false negatives inside this component
                    tversky_comp = (tp + self.eps) / (tp + self.beta * fn + self.eps)
                    l_k = torch.pow((1.0 - tversky_comp), self.gamma)
                    
                    if self.cap_threshold is not None:
                        w_k = min(1.0, float(comp_size) / float(self.cap_threshold))
                    else:
                        w_k = 1.0
                        
                    comp_loss_sum += w_k * l_k
                    weight_sum += w_k
                    
                # False positive global background penalty
                bg_mask = (1.0 - target_tile_tensor)
                fp = (prob_tile * bg_mask).sum()
                fp_loss = self.alpha * fp / (bg_mask.sum() + self.eps)
                
                loss_sum += (comp_loss_sum / (weight_sum + self.eps if self.cap_threshold is not None else len(valid_components))) + fp_loss
                
        return loss_sum / batch_size

