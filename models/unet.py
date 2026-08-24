from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T


class SegPairDataset(Dataset):
    """
    Dataset class for loading image and mask pairs with optional data augmentation.
    """
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.image_paths = sorted([p for p in self.images_dir.glob("*.png") if p.is_file()])
        if len(self.image_paths) == 0:
            self.image_paths = sorted([p for p in self.images_dir.rglob("*.png") if p.is_file() and "masks" not in p.parts])
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No PNGs found in {images_dir}")

        self.mask_paths = {}
        missing = []
        for ip in self.image_paths:
            mp = self._resolve_mask_path(ip)
            if mp is None:
                missing.append(ip.name)
            else:
                self.mask_paths[ip] = mp
        if missing:
            raise FileNotFoundError(f"Masks missing for {len(missing)} images. Example: {missing[:5]}")

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_img = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_msk = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST)

    def _resolve_mask_path(self, img_path: Path) -> Path | None:
        candidates = [
            self.masks_dir / img_path.name,
            self.masks_dir / "masks" / img_path.name,
        ]
        for sub in ["train", "val", "test"]:
            candidates.append(self.masks_dir / sub / "masks" / img_path.name)
            candidates.append(self.masks_dir / "split" / sub / "masks" / img_path.name)
        for c in candidates:
            if c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _augment_pair(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rng.random() < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal
            msk = torch.flip(msk, dims=[2])
        if self.rng.random() < 0.5:
            img = torch.flip(img, dims=[1])  # vertical
            msk = torch.flip(msk, dims=[1])

        if self.rng.random() < 0.5:
            angle = float(self.rng.uniform(-15, 15))
            img = torchvision.transforms.functional.rotate(img, angle=angle, interpolation=T.InterpolationMode.BILINEAR)
            msk = torchvision.transforms.functional.rotate(msk, angle=angle, interpolation=T.InterpolationMode.NEAREST)

        if self.rng.random() < 0.7:
            cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
            img = cj(img)

        if self.rng.random() < 0.2:
            blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
            img = blur(img)

        return img, msk

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        ip = self.image_paths[idx]
        mp = self.mask_paths.get(ip, self.masks_dir / ip.name)

        img = self._load_image(ip)
        msk = self._load_mask(mp)

        img = self.resize_img(img)
        msk = self.resize_msk(msk)

        img_t = self.to_tensor(img)
        img_t = self.normalize(img_t)

        msk_t = self.to_tensor(msk)
        msk_t = (msk_t > 0.0).float()

        if self.augment:
            img_t, msk_t = self._augment_pair(img_t, msk_t)

        return {"image": img_t, "mask": msk_t, "name": ip.name}


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetEfficientNet(nn.Module):
    """
    U-Net model with an EfficientNet-B0 encoder backbone.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load torchvision's EfficientNet-B0
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.efficientnet_b0(weights=weights)
        self.features = backbone.features

        # Map features block to U-Net encoder stages
        self.enc1 = nn.Sequential(self.features[0], self.features[1]) # Out: 16
        self.enc2 = self.features[2]                                  # Out: 24
        self.enc3 = self.features[3]                                  # Out: 40
        self.enc4 = nn.Sequential(self.features[4], self.features[5]) # Out: 112
        self.enc5 = nn.Sequential(self.features[6], self.features[7]) # Out: 320 (Bottleneck input)
        
        self.center = ConvBNReLU(320, 320)
        
        # Decoder
        self.up4 = UpBlock(320, 112, 128)
        self.up3 = UpBlock(128, 40, 64)
        self.up2 = UpBlock(64, 24, 32)
        self.up1 = UpBlock(32, 16, 16)
        
        # Upsample head back to full image size
        self.head_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # 1/2
        x2 = self.enc2(x1) # 1/4
        x3 = self.enc3(x2) # 1/8
        x4 = self.enc4(x3) # 1/16
        x5 = self.enc5(x4) # 1/32
        
        # Center
        c = self.center(x5)
        
        # Decoder
        d4 = self.up4(c, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        
        # Head
        out = self.head_up(d1)
        logits = self.head(out)
        
        return logits


class SplitSegDataset(SegPairDataset):
    """
    Subclass of SegPairDataset that filters image/mask loading based on a split CSV index.
    Supports both flat (images/, masks/) and split-nested (split/{train,val,test}/{images,masks}/) layouts.
    """
    def __init__(
        self,
        split_csv: Path | str,
        images_dir: Path | str,
        masks_dir: Path | str,
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        
        # Read the split index
        df = pd.read_csv(split_csv)
        col = 'tile_id' if 'tile_id' in df.columns else 'filename'
        subset = str(df['subset'].iloc[0]) if 'subset' in df.columns and len(df) > 0 else Path(split_csv).stem

        self.image_paths = []
        self.mask_paths = {}
        missing_imgs = []
        missing_msks = []

        for fn in df[col].values:
            ip = self._resolve_file(self.images_dir, fn, subset, "images")
            if ip is None:
                missing_imgs.append(fn)
                continue
            self.image_paths.append(ip)
            mp = self._resolve_file(self.masks_dir, fn, subset, "masks")
            if mp is None:
                missing_msks.append(fn)
            else:
                self.mask_paths[ip] = mp

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"None of the {len(df)} image files listed in {split_csv} were found in {images_dir}"
            )
        if missing_imgs:
            raise FileNotFoundError(
                f"{len(missing_imgs)} images listed in {split_csv} were missing in {images_dir}. Example: {missing_imgs[:5]}"
            )
        if missing_msks:
            raise FileNotFoundError(
                f"Masks missing for {len(missing_msks)} images in {masks_dir}. Example: {missing_msks[:5]}"
            )
            
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_img = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_msk = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST)

    @staticmethod
    def _resolve_file(base_dir: Path, filename: str, subset: str, kind: str) -> Path | None:
        candidates = [
            base_dir / filename,
            base_dir / kind / filename,
            base_dir / subset / kind / filename,
            base_dir / "split" / subset / kind / filename,
        ]
        for s in ["train", "val", "test"]:
            candidates.append(base_dir / s / kind / filename)
            candidates.append(base_dir / "split" / s / kind / filename)
        for c in candidates:
            if c.exists():
                return c
        return None


