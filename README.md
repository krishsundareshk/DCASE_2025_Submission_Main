# DCASE_2025_Submission_Main

A unified, end-to-end pipeline for self-supervised contrastive learning on RGB spectrogram patches across all machine types.

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Spectrogram Conversion (`convert_rgb.py`)](#spectrogram-conversion-convert_rgbpy)  
- [Attribute Handling](#attribute-handling)  
- [Training (`train_joint.py`)](#training-train_jointpy)  
- [Evaluation (`eval_joint.py`)](#evaluation-eval_jointpy)  
- [Repository Layout](#repository-layout)  
- [Usage Examples](#usage-examples)  
- [Scripts](#scripts)  
- [License](#license)  

---

## Prerequisites

- **Python** ≥ 3.8  
- **CUDA** (optional, for GPU acceleration)  
- **Dependencies** (install via `pip install -r requirements.txt`):
  ```
  torch>=1.10
  torchvision
  pandas
  numpy
  scikit-learn
  librosa
  matplotlib
  pillow
  tqdm
  ```

---

## Installation

1. **Clone repository**  
   ```bash
   git clone https://github.com/yourusername/DCASE_2025_Submission_Main.git
   cd DCASE_2025_Submission_Main
   ```

2. **Create & activate virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Organize `.wav` files**  
   ```
   training_data/{machine_type}/train
   training_data/{machine_type}/supplemental
   ```
   - `train`: main training set  
   - `supplemental`: any extra “normal” recordings

2. **Verify structure**  
   Example:
   ```
   training_data/
   ├── BandSealer/
   │   ├── train/
   │   └── supplemental/
   └── ToyRCCar/
       ├── train/
       └── supplemental/
   ```

---

## Spectrogram Conversion (`convert_rgb.py`)

Convert `.wav` files to 224×224 RGB spectrogram PNGs.

### Run:
```bash
python convert_rgb.py
```

### Resulting folders:
- Inputs:  
  ```
  training_data/{machine_type}/train
  training_data/{machine_type}/supplemental
  ```
- Outputs:  
  ```
  training_data/{machine_type}/trainRGB
  training_data/{machine_type}/supplementalRGB
  ```

Each PNG will match the corresponding WAV filename.

---

## Attribute Handling

If a machine type has metadata (attributes):

1. Include file:
   ```
   training_data/{machine_type}/attributes_00.csv
   ```

2. File format:
   - Required column: `filename` or `file_name` (matching PNG basenames)
   - Other columns: numerical or categorical attributes

3. Details:
   - Categorical attributes are encoded as integer class indices
   - Vectors are zero-padded to a global maximum attribute dimension

---

## Training (`train_joint.py`)

### Overview

- **Model**: ResNet-34 → 512-D → 128-D projection  
- **Patch size**: 32×32  
- **Attention**: Attribute-conditioned attention pooling  
- **Loss**: NT-Xent (temperature=0.1)  
- **Optimizer**: Adam (lr=2e-4)  
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)  
- **Early Stopping**: 25 epochs patience  
- **Checkpoints**:  
  ```
  checkpoints_submission25_BS256_Resnet34_stride16/epoch{n}.pth
  ```

### Hyperparameters

- `batch_size`: 256  
- `epochs`: 500  
- `max_patches`: 64  
- `stride`: 16  

### Run:
```bash
python train_joint.py
```

Automatically detects existing checkpoints and resumes if available.

---

## Evaluation (`eval_joint.py`)

Evaluate model at a fixed checkpoint (e.g. epoch 371) using Mahalanobis scoring.

### Steps:
- Extract embeddings from test split
- Fit Gaussian (EmpiricalCovariance) using “normal” samples from source & target domains
- Compute:
  - AUC
  - pAUC @ FPR ≤ 0.1

### Run:
```bash
python eval_joint.py
```

### Output:
Per machine and domain:
```
BandSealer
  source AUC = 0.8812   pAUC (FPR≤0.1) = 0.7453
  target AUC = 0.9024   pAUC (FPR≤0.1) = 0.7821
...
```

---

## Repository Layout

```
.
├── attention_pooling.py              # Attribute-conditioned pooling module
├── astra_attn_patch_dataset.py      # Patch dataset + attribute handling
├── convert_rgb.py                   # WAV → RGB spectrogram conversion
├── eval_joint.py                    # AUC + pAUC evaluation via Mahalanobis
├── train_joint.py                   # Joint training across machines
├── patch_attn_model.py              # ResNet + patch encoder model
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Usage Examples

```python
from patch_attn_model import PatchAttentionCLModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model with correct attribute dim
model = PatchAttentionCLModel(embed_dim=128, attr_dim=global_attr_dim)
model.load_state_dict(torch.load("checkpoints_submission25_BS256_Resnet34_stride16/epoch371.pth")["model_state"])
model.to(device)
model.eval()

# Forward pass
# patches: (B, N, 3, 224, 224)
# attrs:   (B, global_attr_dim)
embs = model(patches, B, N, attrs=attrs)  # → (B, D)
```

---

## Scripts

| Script            | Description                                        |
|-------------------|----------------------------------------------------|
| `convert_rgb.py`  | Converts WAV → 224×224 RGB spectrogram PNGs        |
| `train_joint.py`  | Trains ResNet-based patch encoder with attributes  |
| `eval_joint.py`   | Evaluates embeddings via Mahalanobis scoring       |

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.
