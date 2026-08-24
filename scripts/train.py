import os
import sys
import math
import copy
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Setup path for local packages import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.unet import UNetEfficientNet, SplitSegDataset
from models.losses import FocalTverskyLoss, ComponentBalancedLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def lr_at_step(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return base_lr * cosine


@torch.no_grad()
def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.50).float()

    preds_f = preds.view(-1)
    targets_f = targets.view(-1)

    tp = (preds_f * targets_f).sum().item()
    fp = (preds_f * (1.0 - targets_f)).sum().item()
    fn = ((1.0 - preds_f) * targets_f).sum().item()

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return {"miou": float(iou), "dice": float(dice)}


def main():
    parser = argparse.ArgumentParser(description="Train U-Net + EfficientNet-B0 for Late-Season Weed Segmentation")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to RGB images folder or dataset root")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to ground truth masks folder or dataset root")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to split CSV indexes folder")
    parser.add_argument("--loss", type=str, choices=["tversky", "cb"], default="tversky", help="Loss function ('tversky' or 'cb')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (e.g. 42, 101, 202, 303, 404)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs (default: 80)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial AdamW learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear learning rate warmup epochs (default: 3)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on validation mIoU (0 to disable)")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum component pixel footprint for CB loss")
    parser.add_argument("--cap-threshold", type=float, default=None, help="Optional component area cap threshold for CB loss")
    parser.add_argument("--save-path", type=str, default=None, help="Output path for the trained model checkpoint (.pt)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {args.seed} | Loss: {args.loss}")

    # Build datasets
    splits_path = Path(args.splits_dir)
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    
    train_dataset = SplitSegDataset(train_csv, Path(args.images_dir), Path(args.masks_dir), augment=True, seed=args.seed)
    val_dataset = SplitSegDataset(val_csv, Path(args.images_dir), Path(args.masks_dir), augment=False, seed=args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model & optimizer
    model = UNetEfficientNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Selection of loss function
    if args.loss == "cb":
        criterion_train = ComponentBalancedLoss(
            alpha=0.7, beta=0.3, gamma=0.75, min_size=args.min_size, cap_threshold=args.cap_threshold
        )
    else:
        criterion_train = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

    # Validation criterion is always standard Focal Tversky Loss for consistency
    criterion_val = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    global_step = 0

    best_val_loss = float("inf")
    best_val_miou = -1.0
    patience_left = args.patience
    best_state = None

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        train_dice = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            msks = batch["mask"].to(device)

            current_lr = lr_at_step(args.lr, global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion_train(logits, msks)
            loss.backward()
            optimizer.step()
            global_step += 1

            train_loss += loss.item()
            m = compute_batch_metrics(logits.detach(), msks.detach())
            train_miou += m["miou"]
            train_dice += m["dice"]

        train_loss /= max(1, len(train_loader))
        train_miou /= max(1, len(train_loader))
        train_dice /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                msks = batch["mask"].to(device)
                logits = model(imgs)
                loss = criterion_val(logits, msks)
                val_loss += loss.item()
                m = compute_batch_metrics(logits, msks)
                val_miou += m["miou"]
                val_dice += m["dice"]

        val_loss /= max(1, len(val_loader))
        val_miou /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train [Loss: {train_loss:.4f} mIoU: {train_miou:.4f} Dice: {train_dice:.4f}] | "
            f"Val [Loss: {val_loss:.4f} mIoU: {val_miou:.4f} Dice: {val_dice:.4f}]"
        )

        # Retain optimal model checkpoint based on minimum validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        # Early stopping tracking on validation mIoU
        if val_miou > best_val_miou + 1e-6:
            best_val_miou = val_miou
            patience_left = args.patience
        elif args.patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping triggered at epoch {epoch} (best Val mIoU: {best_val_miou:.4f}).")
                break

    if args.save_path and best_state is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, args.save_path)
        print(f"Saved optimal model checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
