# Brain Tumor MRI Classification

Comparative study of six deep learning architectures — **ResNet50**, **VGG16**, **EfficientNetB0**, **ViT-Base/16**, **GCN (Superpixel-based)**, and an **Improved GCN** — for multi-class brain tumor classification from MRI scans.

## Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** (Masoud Nickparvar, Kaggle)

- **Classes (4):** `glioma`, `meningioma`, `notumor`, `pituitary`
- **Original size:** 7,023 MRI images (5,712 train / 1,311 test)
- **Preprocessing:** All splits are cleaned with a **16×16 (256-bit) perceptual hash** — the single, permanent deduplication method used across every notebook in this repo — to remove exact and near-duplicate images and prevent train↔val↔test leakage before any model is trained.

## Models & Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|---|---|---|
| **ResNet50** (Transfer Learning) | 93.39% | 91.10% | 94.35% | 92.22% | 94.09% | 93.39% | 93.40% |
| **VGG16** (Transfer Learning) | 93.86% | 92.70% | 94.91% | 93.36% | 94.50% | 93.86% | 93.76% |
| **EfficientNetB0** (Transfer Learning) | 88.01% | 87.02% | 89.13% | 87.44% | 88.57% | 88.01% | 87.74% |
| **ViT-Base/16** (Fine-Tuned) | **94.97%** | 94.20% | 95.68% | 94.65% | 95.44% | 94.97% | 94.93% |
| **GCN** (Superpixel + RAG) | 75.13% | 73.00%* | 77.00%* | 73.41% | 76.00%* | 75.00%* | 74.00%* |
| **Improved GCN** |90.09%  |  |  |  |  |  |  |

`*` GCN's macro/weighted precision, recall, and weighted-F1 are reported to 2 decimal places, matching the notebook's `classification_report` output precision. Accuracy and macro-F1 are exact values from the notebook's stored results summary.

**Best-performing model:** ViT-Base/16, driven by global self-attention and full end-to-end fine-tuning of all 85.8M parameters.

## Repository Structure

```
brain-tumor-mri-classification-research/
├── notebooks/
│   ├── resnet50-cse475.ipynb        # ResNet50 transfer learning baseline
│   ├── vgg16-cse475.ipynb           # VGG16 transfer learning baseline
│   ├── efficientnetb0-cse475.ipynb  # EfficientNetB0 transfer learning baseline
│   ├── vit-cse475.ipynb             # Vision Transformer (ViT-Base/16), fine-tuned
│   └── gcn-cse475.ipynb             # GCN (superpixel graph) baseline + Improved GCN
├── reports/
│   └── ...                          # Word/PDF reports, EDA, comparison tables
└── README.md
```

## Methodology

- **Deduplication:** 16×16 average-hash (256-bit) fingerprinting applied identically to train, validation, and test splits — chosen over MD5 (misses re-compressed near-duplicates) and an 8×8 hash (too coarse, collides across different classes). Applied in two passes: within-split dedup, then cross-split leakage removal.
- **Class imbalance:** Handled via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`, applied in the loss function for every model.
- **CNN baselines (ResNet50, VGG16, EfficientNetB0):** Two-phase transfer learning — frozen-backbone head training, followed by fine-tuning the last convolutional block(s) at a reduced learning rate.
- **ViT-Base/16:** Full end-to-end fine-tuning (`timm` implementation) with AdamW, linear warmup + cosine LR decay, and mixed-precision training.
- **GCN:** Superpixel segmentation (SLIC) → region adjacency graph → hand-crafted node features → graph convolutional network, evaluated with 5-fold stratified cross-validation and McNemar's significance test against the ViT baseline.
- **Interpretability:** Grad-CAM (CNNs), attention-map visualization (ViT), and GNNExplainer / superpixel Grad-CAM / superpixel LIME (GCN).
- **Evaluation:** Held-out test set — accuracy, precision/recall/F1 (macro & weighted), confusion matrices, MCC, balanced accuracy, Cohen's Kappa, and ROC-AUC where applicable.

## Requirements

```
tensorflow>=2.11
torch
torchvision
timm
scikit-learn
pandas
numpy
matplotlib
seaborn
Pillow
```

## Course Context

Developed as part of **CSE475 (Machine Learning)** coursework at **East West University**.

## License

For academic and research purposes.
