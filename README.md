# 🫁 Chest X-Ray Multi-Disease Classification

A deep learning system for automated multi-label classification of pulmonary diseases from chest X-ray images. Built using PyTorch with four CNN architectures benchmarked on a subset of the NIH ChestX-ray14 dataset, with Grad-CAM explainability and a full-stack React + Django web interface.

> **Team:** Kirti Kour, Ghulam Qadir, Qurban Ali  
> **Platform:** Kaggle (NVIDIA Tesla T4 GPU)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)

---

## Overview

This project tackles **multi-label chest X-ray disease classification** — a clinically significant problem where a single X-ray can show multiple co-occurring pathologies. Unlike single-label classification, each disease is treated as an independent binary decision, making this a harder and more realistic medical imaging task.

The pipeline covers:
- Data loading and preprocessing from the NIH ChestX-ray14 dataset
- Multi-hot label encoding for 8 disease classes
- Training and fine-tuning of 4 pretrained CNN architectures
- Evaluation using Macro AUC and Macro F1
- Grad-CAM heatmap generation for model explainability
- Deployment-ready model export

---

## Dataset

**NIH ChestX-ray14** — one of the largest publicly available chest X-ray datasets.

| Property | Value |
|---|---|
| Total records in CSV | 112,120 |
| Subset used | 6,000 images |
| Filter | Images with ≥1 of 8 target disease labels |
| Image size | 224×224 (299×299 for InceptionV3) |
| Input format | RGB (3-channel) |

### Disease Labels & Distribution (6k subset)

| Disease | Count |
|---|---|
| Infiltration | 2,554 |
| Effusion | 1,743 |
| Atelectasis | 1,493 |
| Nodule | 825 |
| Mass | 735 |
| Pneumothorax | 659 |
| Cardiomegaly | 354 |
| Pneumonia | 175 |

> Note: Class imbalance exists — Infiltration dominates while Pneumonia is rare. This affects per-class F1 scores.

### Preprocessing & Augmentation

- **Training:** Random resized crop, horizontal flip, rotation (±10°), color jitter, normalization (ImageNet mean/std)
- **Validation/Test:** Center crop, normalization only
- Images converted to RGB since all CNN backbones expect 3-channel input

---

## Models

All four models were initialized with **ImageNet pretrained weights** and fine-tuned end-to-end. A custom 2-layer classifier head replaced the original output layer for each.

### Regularization Strategy

| Technique | Detail |
|---|---|
| Dropout | p=0.5 in classifier head |
| Weight Decay | L2 = 1e-4 in Adam optimizer |
| Early Stopping | Patience = 5 epochs |

### Training Config

```
Batch size:   32
Epochs:       20 (with early stopping)
Optimizer:    Adam (lr=1e-4, weight_decay=1e-4)
Loss:         BCEWithLogitsLoss (multi-label)
Device:       NVIDIA Tesla T4
Seed:         42
```

---

## Results

### Final Benchmark (Test Set)

| Rank | Model | Macro AUC | Macro F1 |
|---|---|---|---|
| 🥇 1 | **EfficientNet-B0** | **0.7400** | 0.2891 |
| 🥈 2 | VGG16 | 0.7352 | 0.3137 |
| 🥉 3 | ResNet50 | 0.7308 | 0.3138 |
| 4 | InceptionV3 | 0.7188 | 0.2637 |

**🏆 Best Model: EfficientNet-B0 — Macro AUC = 0.7400**

### Key Observations

- **EfficientNet-B0** achieved the best AUC-to-parameter ratio, consistent with its compound scaling design. It was saved as the deployment model (`deployment_model.pth`).
- **VGG16 and ResNet50** had nearly identical Macro F1 (~0.31), slightly outperforming EfficientNet on F1 while losing on AUC.
- **InceptionV3** ranked last on both metrics, possibly due to its larger input size (299×299) creating a distribution mismatch with the 224-trained preprocessing pipeline.
- **Effusion and Cardiomegaly** tend to yield the highest per-class AUC as they produce distinct, large-area visual patterns.
- **Mass and Nodule** are the hardest classes — small lesion size and visual similarity to surrounding tissue make them difficult to localize and classify.

---

## Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to visualize which regions of the X-ray each model uses for its predictions.

Key findings:
- Models correctly focused on **lung parenchyma** for respiratory diseases (Effusion, Atelectasis, Pneumonia)
- Models correctly highlighted the **cardiac silhouette** for Cardiomegaly
- This validates that models are learning clinically meaningful features rather than spurious correlations (e.g., scan borders, labels, or artifacts)

---

## Tech Stack

```
Deep Learning:   PyTorch, TorchVision, TorchMetrics
Models:          EfficientNet-B0, VGG16, ResNet50, InceptionV3
Explainability:  pytorch-grad-cam
Data:            Pandas, NumPy, scikit-learn
Visualization:   Matplotlib, Seaborn
Web Interface:   React.js (frontend), Django (backend)
Platform:        Kaggle (NVIDIA Tesla T4)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/chest-xray-classification.git
cd chest-xray-classification

# Install dependencies
pip install torch torchvision torchmetrics grad-cam scikit-learn pandas matplotlib seaborn

# Or install from requirements
pip install -r requirements.txt
```

### Dataset Setup (Kaggle)

1. Go to: https://www.kaggle.com/datasets/nih-chest-xrays/data
2. Add it to your Kaggle notebook via **+ Add Data**
3. Update `DATA_DIR` in the config to `/kaggle/input/data/`

---

## Usage

### Run the Notebook

Open `chestxrayanalysis.ipynb` on Kaggle with GPU enabled (Tesla T4 recommended) and run all cells sequentially.

### Load the Saved Model

```python
import torch

checkpoint = torch.load('deployment_model.pth')
print(checkpoint['model_name'])   # efficientnet_b0
print(checkpoint['macro_auc'])    # 0.7400
print(checkpoint['labels'])       # ['Atelectasis', 'Cardiomegaly', ...]
```

---

## Limitations

- **Class imbalance:** Infiltration accounts for ~42% of samples while Pneumonia has only 175 examples. Techniques like weighted sampling or focal loss were not applied.
- **Subset training:** Only 6,000 of 112,120 images were used for feasibility. Full dataset training would likely push AUC significantly higher (CheXNet reports ~0.84 on DenseNet121).
- **No radiologist validation:** Grad-CAM regions were not verified against expert annotations.
- **Grayscale-to-RGB conversion:** X-rays are inherently single-channel; the RGB conversion is an architectural constraint, not a clinical one.
- **No demographic analysis:** Patient age/gender splits were not analyzed for model bias.

---

## References

- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
- Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays", 2017
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
