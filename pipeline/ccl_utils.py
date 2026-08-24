from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
from skimage.morphology import convex_hull_image
from scipy.ndimage import label as scipy_label

# 3x3 structuring element for 8-neighborhood connectivity (N_8)
STRUCT_8 = np.ones((3, 3), dtype=int)


def compute_spatial_descriptors(mask: np.ndarray) -> dict:
    """
    Computes spatial configuration descriptors for a binary mask (0 or 1, or 0 or 255).
    
    Parameters:
        mask (np.ndarray): Binary mask array of shape (H, W).
        
    Returns:
        dict: Containing:
            - 'weed_ratio' (WR): Ratio of target pixels to total pixels.
            - 'n_patches' (N): Count of disjoint connected components using 8-connectivity.
            - 'largest_patch_fraction' (LPF): Dominance metric (area of largest component / total target area).
            - 'hull_ratio' (HR): Dispersion metric (area of convex hull / total target area).
    """
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
        
    # Ensure binary format (0 or 255 for cv2)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    H, W = binary_mask.shape
    total_pixels = H * W
    
    target_sum = (binary_mask > 0).sum()
    wr = float(target_sum / total_pixels)
    
    if target_sum == 0:
        return {
            'weed_ratio': 0.0,
            'n_patches': 0,
            'largest_patch_fraction': 0.0,
            'hull_ratio': 1.0
        }
        
    # Compute Patch Count (N) using OpenCV 8-connectivity Connected Component Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    n_patches = int(max(0, num_labels - 1))
    
    # Compute Largest Patch Fraction (LPF)
    if n_patches > 0:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_patch_size = sizes.max()
        lpf = float(largest_patch_size / target_sum)
    else:
        lpf = 0.0
        
    # Compute Convex Hull Ratio (HR)
    try:
        hull = convex_hull_image(binary_mask > 0)
        hull_area = hull.sum()
        hr = float(hull_area / target_sum)
    except Exception:
        hr = 1.0
        
    return {
        'weed_ratio': wr,
        'n_patches': n_patches,
        'largest_patch_fraction': lpf,
        'hull_ratio': hr
    }


def compute_non_geometric_descriptors(rgb_img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Computes the three non-geometric radiometric scene descriptors described in Section 2.7:
    - 'delta_ExG': Weed-background Excess Green (ExG = 2G - R - B) contrast.
    - 'mean_L': Mean tile luminance (L*) in CIELAB color space.
    - 'var_bg_ExG': Variance of ExG across background crop and soil pixels.
    """
    img_f = rgb_img.astype(np.float64)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    mask_bool = (mask > 0.5) if mask.max() <= 1.0 else (mask > 127)
    bg_bool = ~mask_bool

    R, G, B = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    ExG = 2.0 * G - R - B

    L_star = rgb2lab(img_f / 255.0)[:, :, 0]
    mean_L = float(L_star.mean())

    if mask_bool.sum() > 0 and bg_bool.sum() > 0:
        delta_ExG = float(abs(ExG[mask_bool].mean() - ExG[bg_bool].mean()))
        var_bg_ExG = float(ExG[bg_bool].var())
    else:
        delta_ExG = 0.0
        var_bg_ExG = float(ExG.var())

    return {
        'delta_ExG': delta_ExG,
        'mean_L': mean_L,
        'var_bg_ExG': var_bg_ExG
    }


def scipy_label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    8-connected component labeling using SciPy.
    Used primarily during training loss computation and evaluation to isolate components.
    """
    return scipy_label(mask, structure=STRUCT_8)

