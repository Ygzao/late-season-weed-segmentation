# Beyond Weed Coverage: Component-Balanced Learning Improves Fragmented Late-Season UAV Weed Segmentation

Official PyTorch implementation and benchmark repository for the manuscript:

> **Beyond Weed Coverage: Component-Balanced Learning Improves Fragmented Late-Season UAV Weed Segmentation**  
> David Ygzao<sup>1,2</sup>, Lior Blank<sup>2</sup>, Hanan Eizenberg<sup>2</sup>, and Mohsen B. Mesgaran<sup>3</sup>  
> <sup>1</sup>The Hebrew University of Jerusalem, Israel · <sup>2</sup>ARO Volcani Institute, Israel · <sup>3</sup>University of California, Davis, USA

---

## 1. Overview

Late-season UAV weed mapping is challenging because canopy closure obscures weeds and produces optical fragmentation—overlying crop branches split a single underlying weed plant into multiple disconnected 2D components in the image plane. Furthermore, standard pixel-level objectives (e.g., Cross-Entropy or Focal-Tversky loss) bias optimization toward large, contiguous weed patches, resulting in high omission rates for small, fragmented targets.

This repository provides:
1. **Spatial Configuration Diagnostics (`pipeline/ccl_utils.py`)**: Extraction of 8-connected target patch count ($N_{\text{patches}}$), largest patch fraction ($LPF$), convex hull ratio ($HR$), and non-geometric radiometric descriptors ($\Delta ExG$, $\overline{L}$, $\sigma^2_{\text{bg}}$).
2. **Component-Balanced Loss (`models/losses.py`)**: $\mathcal{L}_{\text{CB}}$ equalizes target-class gradient contributions across the $K$ distinct 8-connected ground-truth components ($1/K$) rather than aggregating all foreground pixels globally.
3. **Component Confidence Filtering (`pipeline/confidence_filter.py`)**: A two-stage post-processing filter (CF) that evaluates candidate 8-connected components predicted at $\tau_p = 0.50$ by their mean interior probability $S(C^{\text{pred}}_i)$ and prunes low-confidence clutter below $\tau_{\text{CF}} = 0.70$.

---

## 2. Repository Structure

```text
├── CITATION.cff                                   # Citation metadata
├── LICENSE                                        # MIT License
├── README.md                                      # Repository documentation
├── environment.yml                                # Conda environment specification
├── requirements.txt                               # Pip dependencies
├── models/
│   ├── unet.py                                    # U-Net + EfficientNet-B0 & dataset loaders
│   └── losses.py                                  # Focal-Tversky Loss & Component-Balanced Loss
├── pipeline/
│   ├── ccl_utils.py                               # Spatial & radiometric descriptor extraction
│   └── confidence_filter.py                       # Component Confidence Filtering (CF)
├── scripts/
│   ├── train.py                                   # Model training script (AdamW, warmup + cosine LR)
│   └── evaluate.py                                # Multi-seed benchmark evaluation (Table 2 & Table S2)
├── data/
│   ├── split/                                     # RGB UAV tiles & binary masks (train, val, test)
│   ├── splits/                                    # Stratified split indices (train.csv, val.csv, test.csv)
│   ├── table_final_ablation_seed_level_v11.csv    # Disaggregated tile-seed benchmark metrics (N = 4,860)
│   ├── table_final_ablation_summary_v11.csv       # Seed-level summary across configurations
│   ├── table_component_recall_quantile_bins_v11.csv # Size-stratified component recall (Q1-Q4)
│   └── table_morphology_env_regression_data.csv   # Spatial & radiometric descriptors for OLS/partial corr.
└── figures/
    ├── generate_all_figures.py                    # Reproduces manuscript Figures 2-7
    ├── fig1_system_architecture.png               # Figure 1: Training & inference pipeline
    ├── fig2_feature_descriptives.png              # Figure 2: Contrasting spatial configurations (A-F)
    ├── fig3_foreground_weed_iou.png               # Figure 3: Baseline foreground weed IoU across strata
    ├── fig4_statistical_proof.png                 # Figure 4: Hierarchical OLS R^2 & partial correlations
    ├── fig5_precision_recall_centroids.png        # Figure 5: Component precision-recall & error centroids
    ├── fig6_size_stratified_recall.png            # Figure 6: Component recall across area quartiles Q1-Q4
    ├── fig7_qualitative_predictions.png           # Figure 7: Qualitative predictions on test tiles A-D
    └── figA1_training_dynamics.png                # Figure S1: Training and validation trajectories
```

---

## 3. Environment Setup

### Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate weedseg-release
```

### Pip
```bash
pip install -r requirements.txt
```

---

## 4. Dataset & Pretrained Checkpoints

The dataset comprises $881$ RGB UAV tiles ($256 \times 256$ pixels, $\approx 0.977\text{ cm/pixel}$) collected across four commercial cotton fields east of Los Banos, California:
- **Training split (`data/splits/train.csv`)**: $n = 614$ tiles ($70\%$, from Fields B & C / internal codes `H1`, `H2`)
- **Validation split (`data/splits/val.csv`)**: $n = 105$ tiles ($12\%$, from Fields B & C / internal codes `H1`, `H2`)
- **Test split (`data/splits/test.csv`)**: $n = 162$ tiles ($18\%$), combining within-field spatial holdouts from Fields B (`H1`, $n=56$) and C (`H2`, $n=59$) with out-of-field holdouts from unseen Fields A (`E2`, $n=21$) and D (`H3`, $n=26$).

### Directory Layout
`SplitSegDataset` automatically supports both the archive's split-nested structure (`split/{train,val,test}/{images,masks}/`) and a flat directory structure (`images/` and `masks/`):

```text
# Option A: Extracted directly from late_season_weed_dataset.zip
data/
└── split/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

# Option B: Flat folders
/path/to/dataset/
├── images/
└── masks/
```

If `data/late_season_weed_dataset.zip` and `checkpoints/model_checkpoints.zip` are present locally or downloaded from the accompanying Zenodo release / GitHub Release assets, extract them via:
```bash
unzip data/late_season_weed_dataset.zip -d data/
unzip checkpoints/model_checkpoints.zip -d .
```

---

## 5. Training

To train the U-Net + EfficientNet-B0 segmentation network following the manuscript protocol (AdamW, initial learning rate $3 \times 10^{-4}$, weight decay $10^{-4}$, batch size $32$, 3-epoch linear warm-up with cosine decay, up to $80$ epochs with early stopping patience of $10$ on validation $\text{mIoU}$, and optimal checkpoint selection by minimum validation loss), run:

```bash
python scripts/train.py \
  --images-dir "data/split" \
  --masks-dir "data/split" \
  --splits-dir "data/splits" \
  --loss cb \
  --seed 42 \
  --batch-size 32 \
  --epochs 80 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --save-path "checkpoints/unet_comp_balanced_loss_seed_42.pt"
```
*(Use `--loss tversky` for the Baseline Focal-Tversky configuration, and `--seed` $\in \{42, 101, 202, 303, 404\}$ for multi-seed replication).*

---

## 6. Evaluation & Benchmark Reproduction

To evaluate the five pretrained random seeds ($\{42, 101, 202, 303, 404\}$) across the $162$ test tiles and reproduce **Table 2** (as well as **Table S2** and the $\tau_{\text{CF}} = 0.60$ supplementary configurations), run:

```bash
python scripts/evaluate.py \
  --images-dir "data/split" \
  --masks-dir "data/split" \
  --checkpoints-dir "checkpoints" \
  --include-supplementary
```

### Multi-Seed Benchmark Summary (Table 2)

| Configuration | Foreground Weed IoU | Two-Class mIoU | Comp. Precision | Comp. Recall | Comp. $F_1$ | FP / Tile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $0.4950 \pm 0.011$ | $0.6320 \pm 0.008$ | $0.7249$ | $0.9062$ | $0.7491$ | $1.196$ |
| **Baseline-CF ($0.70$)** | $0.4899 \pm 0.021$ | **$0.6402 \pm 0.018$** | **$0.8540$** | $0.8480$ | $0.7920$ | **$0.362$** |
| **CB** | $0.5032 \pm 0.014$ | $0.6364 \pm 0.012$ | $0.6296$ | **$0.9466$** | $0.7108$ | $1.527$ |
| **CB-CF ($0.70$)** | **$0.5059 \pm 0.014$** | $0.6392 \pm 0.013$ | $0.7691$ | $0.9269$ | **$0.8046$** | $0.706$ |

---

## 7. Reproducing Publication Figures

To regenerate the publication figures (`fig2` through `fig7` and `figA1`) from the precomputed statistical tables in `data/` and (optionally) the raw test imagery and checkpoints:

```bash
python figures/generate_all_figures.py \
  --images-dir "data/split" \
  --masks-dir "data/split" \
  --checkpoints-dir "checkpoints" \
  --data-dir "data" \
  --output-dir "figures"
```
*(Note: If `--images-dir` and `--masks-dir` are omitted, the script automatically skips image-dependent Figures 2 and 7 and compiles Figures 3, 4, 5, 6, and S1 directly from the statistical tables in `data/`).*

---

## 8. Citation

If you find this code or dataset useful in your research, please cite:

```bibtex
@article{ygzao2026beyond,
  title={Beyond Weed Coverage: Component-Balanced Learning Improves Fragmented Late-Season UAV Weed Segmentation},
  author={Ygzao, David and Blank, Lior and Eizenberg, Hanan and Mesgaran, Mohsen B.},
  year={2026}
}
```

## 9. License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
