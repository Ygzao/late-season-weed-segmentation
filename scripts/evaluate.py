import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import label

# Setup path for local packages import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.unet import UNetEfficientNet, SplitSegDataset
from pipeline.confidence_filter import filter_predictions

# 3x3 structuring element for 8-neighborhood connectivity (N_8)
STRUCT_8 = np.ones((3, 3), dtype=int)


def compute_component_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Computes tile-level overlap-based component detection statistics (Section 2.9):
    - comp_precision (CP): proportion of predicted components overlapping >= 1 reference component
    - comp_recall (CR): proportion of reference components overlapping >= 1 predicted component
    - comp_f1 (F1): 2 * CP * CR / (CP + CR)
    - fp_components: false-positive clutter components in tile
    - missed_components: missed reference components in tile
    """
    cc_gt, num_gt = label(gt_mask, structure=STRUCT_8)
    cc_pred, num_pred = label(pred_mask, structure=STRUCT_8)
    
    recalled = 0
    missed = 0
    for idx in range(1, num_gt + 1):
        comp = (cc_gt == idx)
        if (pred_mask[comp] == 1.0).sum() > 0:
            recalled += 1
        else:
            missed += 1
            
    precise_preds = 0
    fps = 0
    for idx in range(1, num_pred + 1):
        comp = (cc_pred == idx)
        if (gt_mask[comp] == 1.0).sum() > 0:
            precise_preds += 1
        else:
            fps += 1
            
    cp = precise_preds / num_pred if num_pred > 0 else 1.0
    cr = recalled / num_gt if num_gt > 0 else 1.0
    f1 = (2.0 * cp * cr) / (cp + cr + 1e-7) if (cp + cr) > 0 else 0.0

    return {
        "comp_precision": cp,
        "comp_recall": cr,
        "comp_f1": f1,
        "fp_components": fps,
        "missed_components": missed,
    }


def format_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Weed Segmentation Models on Test Split")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to RGB images folder or dataset root")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to ground truth masks folder or dataset root")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Path to checkpoints folder (.pt files)")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to splits folder")
    parser.add_argument("--include-supplementary", action="store_true", help="Also report tau_CF = 0.60 and Table S2 seed matrix")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using evaluation device: {device}")

    test_csv = Path(args.splits_dir) / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found in {args.splits_dir}")

    test_dataset = SplitSegDataset(test_csv, Path(args.images_dir), Path(args.masks_dir), augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Loaded {len(test_dataset)} test split tiles.")

    seeds = [42, 101, 202, 303, 404]

    all_treatments = {
        "Baseline": {"loss_prefix": "unet_baseline_seed_", "use_cf": False, "threshold": None, "main": True},
        "Baseline-CF (0.60)": {"loss_prefix": "unet_baseline_seed_", "use_cf": True, "threshold": 0.60, "main": False},
        "Baseline-CF (0.70)": {"loss_prefix": "unet_baseline_seed_", "use_cf": True, "threshold": 0.70, "main": True},
        "CB": {"loss_prefix": "unet_comp_balanced_loss_seed_", "use_cf": False, "threshold": None, "main": True},
        "CB-CF (0.60)": {"loss_prefix": "unet_comp_balanced_loss_seed_", "use_cf": True, "threshold": 0.60, "main": False},
        "CB-CF (0.70)": {"loss_prefix": "unet_comp_balanced_loss_seed_", "use_cf": True, "threshold": 0.70, "main": True},
    }

    results_table2 = []
    results_s2 = []

    for name, config in all_treatments.items():
        if not config["main"] and not args.include_supplementary:
            continue

        print(f"Evaluating configuration: {name}...")
        seed_scores = []

        for seed in seeds:
            ckpt_name = f"{config['loss_prefix']}{seed}.pt"
            ckpt_path = Path(args.checkpoints_dir) / ckpt_name

            if not ckpt_path.exists():
                ckpt_name = f"unet_comp_balanced_seed_{seed}.pt" if "comp_balanced" in config["loss_prefix"] else f"unet_baseline_seed_{seed}.pt"
                ckpt_path = Path(args.checkpoints_dir) / ckpt_name

            if not ckpt_path.exists():
                print(f"Skipping Seed {seed}: Checkpoint {ckpt_path} not found.")
                continue

            model = UNetEfficientNet(pretrained=False).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
            model.eval()

            weed_ious = []
            two_class_mious = []
            comp_precs = []
            comp_recs = []
            comp_f1s = []
            fp_counts = []
            miss_counts = []

            with torch.no_grad():
                for batch in test_loader:
                    imgs = batch["image"].to(device)
                    masks = batch["mask"].cpu().numpy()

                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()

                    for i in range(imgs.size(0)):
                        prob_tile = probs[i, 0]
                        gt_tile = (masks[i, 0] > 0.5).astype(np.float32)
                        pred_tile = (prob_tile >= 0.50).astype(np.float32)

                        if config["use_cf"]:
                            pred_tile = filter_predictions(pred_tile, prob_tile, threshold=config["threshold"])

                        # Pixel-level IoUs (Section 2.9)
                        tp = ((pred_tile == 1.0) & (gt_tile == 1.0)).sum()
                        fp = ((pred_tile == 1.0) & (gt_tile == 0.0)).sum()
                        fn = ((pred_tile == 0.0) & (gt_tile == 1.0)).sum()
                        tn = ((pred_tile == 0.0) & (gt_tile == 0.0)).sum()

                        weed_iou = tp / (tp + fp + fn + 1e-7)
                        bg_iou = tn / (tn + fp + fn + 1e-7)
                        two_class_miou = (weed_iou + bg_iou) / 2.0

                        weed_ious.append(float(weed_iou))
                        two_class_mious.append(float(two_class_miou))

                        # Component-level metrics (computed per tile-seed evaluation, then averaged)
                        cm = compute_component_metrics(pred_tile, gt_tile)
                        comp_precs.append(cm["comp_precision"])
                        comp_recs.append(cm["comp_recall"])
                        comp_f1s.append(cm["comp_f1"])
                        fp_counts.append(cm["fp_components"])
                        miss_counts.append(cm["missed_components"])

            seed_scores.append({
                "seed": seed,
                "weed_iou": float(np.mean(weed_ious)),
                "miou_2class": float(np.mean(two_class_mious)),
                "comp_precision": float(np.mean(comp_precs)),
                "comp_recall": float(np.mean(comp_recs)),
                "comp_f1": float(np.mean(comp_f1s)),
                "fp_per_tile": float(np.mean(fp_counts)),
                "miss_per_tile": float(np.mean(miss_counts)),
            })

        if len(seed_scores) > 0:
            df_seed = pd.DataFrame(seed_scores)
            results_table2.append({
                "Configuration": name,
                "Foreground Weed IoU": f"{df_seed['weed_iou'].mean():.4f} ± {df_seed['weed_iou'].std():.3f}",
                "Two-Class mIoU": f"{df_seed['miou_2class'].mean():.4f} ± {df_seed['miou_2class'].std():.3f}",
                "Comp. Precision": f"{df_seed['comp_precision'].mean():.4f}",
                "Comp. Recall": f"{df_seed['comp_recall'].mean():.4f}",
                "Comp. F1": f"{df_seed['comp_f1'].mean():.4f}",
                "FP / Tile": f"{df_seed['fp_per_tile'].mean():.3f}",
            })

            s2_row = {"Configuration": name}
            for _, r in df_seed.iterrows():
                s2_row[f"Seed {int(r['seed'])}"] = f"{r['miou_2class']:.4f}"
            s2_row["Mean ± SD"] = f"{df_seed['miou_2class'].mean():.4f} ± {df_seed['miou_2class'].std():.4f}"
            results_s2.append(s2_row)

    df_summary = pd.DataFrame(results_table2)
    print("\n" + "=" * 100)
    print("TABLE 2: MULTI-SEED BENCHMARK SUMMARY ON 162 TEST TILES")
    print("=" * 100)
    print(format_table(df_summary))
    print("=" * 100)

    if args.include_supplementary and len(results_s2) > 0:
        df_s2 = pd.DataFrame(results_s2)
        print("\n" + "=" * 100)
        print("TABLE S2: TWO-CLASS MACRO mIoU BY RANDOM SEED AND MODEL CONFIGURATION")
        print("=" * 100)
        print(format_table(df_s2))
        print("=" * 100)


if __name__ == "__main__":
    main()
