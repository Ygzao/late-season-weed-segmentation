from __future__ import annotations
import numpy as np
from scipy.ndimage import label

# 3x3 structuring element for 8-neighborhood connectivity (N_8)
STRUCT_8 = np.ones((3, 3), dtype=int)


def filter_predictions(pred_tile: np.ndarray, prob_tile: np.ndarray, threshold: float = 0.70) -> np.ndarray:
    """
    Confidence-Guided Component Filtering (CF) post-processing algorithm.
    
    Identifies 8-connected components in the binary prediction map, computes the 
    average prediction probability (confidence) for each component, and removes 
    any component whose average confidence is below the threshold.
    
    Parameters:
        pred_tile (np.ndarray): Binary prediction mask (0 or 1) of shape (H, W).
        prob_tile (np.ndarray): Continuous prediction probability map [0, 1] of shape (H, W).
        threshold (float): Confidence threshold parameter (default = 0.70).
        
    Returns:
        np.ndarray: Filtered binary prediction mask of shape (H, W).
    """
    if pred_tile.sum() == 0:
        return pred_tile.copy()
        
    cc_pred, num_pred = label(pred_tile, structure=STRUCT_8)
    filtered_mask = np.zeros_like(pred_tile)
    
    for idx in range(1, num_pred + 1):
        comp_mask = (cc_pred == idx)
        mean_prob = prob_tile[comp_mask].mean()
        
        if mean_prob >= threshold:
            filtered_mask[comp_mask] = 1.0
            
    return filtered_mask

