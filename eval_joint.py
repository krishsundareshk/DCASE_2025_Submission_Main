# eval_joint.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import pandas as pd

from astra_attn_patch_dataset import ASTRA_AttnPatchRGBDataset, ALL_MACHINE_TYPES
from patch_attn_model import PatchAttentionCLModel

# ─── CONFIG (must match train_joint.py) ────────────────────────────────
ROOT_DIR        = "training_data"
CHECKPOINT_DIR  = "checkpoints_submission25_BS256_Resnet34,stride16"
EVAL_EPOCH      = 371           # ← fixed epoch to evaluate
BATCH_SIZE      = 256
PATCH_SIZE      = 32
STRIDE          = 16
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# exactly the same attr set & casing as in train_joint.py
USE_ATTR = {"BandSealer", "CoffeeGrinder", "HomeCamera", "ToyRCCar"}
# ───────────────────────────────────────────────────────────────────────

def get_attr_dim(machine):
    """Number of attribute columns for fan/ToyCar/valve/gearbox, else 0."""
    if machine not in USE_ATTR:
        return 0
    path = os.path.join(ROOT_DIR, machine, "attributes_00.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        return len([c for c in df.columns
                    if c not in {"filename","file_name"}])
    return 0

# recompute global attribute dimension exactly as in training
attr_dims      = [get_attr_dim(m) for m in USE_ATTR]
GLOBAL_ATTR_DIM = max(attr_dims) if attr_dims else 0


class ASTRA_EvalRGBDataset(ASTRA_AttnPatchRGBDataset):
    """
    Deterministic evaluation dataset:
      - one centered view → patches
      - returns 'patches', 'attrs', 'label', 'filename'
    Pads each machine’s attrs to GLOBAL_ATTR_DIM.
    """
    def __init__(self, machine_type, split="test"):
        super().__init__(
            root_dir=ROOT_DIR,
            machine_type=machine_type,
            split=split,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            max_patches=None,
            global_attr_dim=GLOBAL_ATTR_DIM
        )
        self.base_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        patches = s["patches_1"]  # single view
        label   = 0 if "normal" in s["filename"] else 1
        return {
            "patches":  patches,          # (N,3,224,224)
            "attrs":    s["attrs"],       # (GLOBAL_ATTR_DIM,)
            "label":    label,
            "filename": s["filename"]
        }


def filter_by_domain_and_label(ds, domain, label_val):
    idxs = [
        i for i, p in enumerate(ds.samples)
        if domain in os.path.basename(p)
           and (0 if "normal" in os.path.basename(p) else 1) == label_val
    ]
    return Subset(ds, idxs)


def filter_by_domain(ds, domain):
    return Subset(
        ds,
        [i for i, p in enumerate(ds.samples) if domain in os.path.basename(p)]
    )


def extract_embeddings(loader, model):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", ncols=80):
            patches = batch["patches"].to(DEVICE)  # (B,N,3,224,224)
            attrs   = batch["attrs"].to(DEVICE)
            labs    = batch["label"]

            B, N, C, H, W = patches.shape
            z = model(patches, B, N, attrs=attrs)
            z = F.normalize(z, dim=1)

            embs.append(z.cpu().numpy())
            labels.extend(labs)
    return np.vstack(embs), np.array(labels)


def evaluate():
    # 1) instantiate model with the correct attr_dim
    model = PatchAttentionCLModel(embed_dim=128, attr_dim=GLOBAL_ATTR_DIM).to(DEVICE)

    # 2) load checkpoint
    ckpt_file = os.path.join(CHECKPOINT_DIR, f"epoch{EVAL_EPOCH}.pth")
    if not os.path.isfile(ckpt_file):
        print(f"❌ Checkpoint for epoch {EVAL_EPOCH} not found at {ckpt_file}")
        return

    ckpt = torch.load(ckpt_file, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"▶️  Loaded checkpoint: epoch{EVAL_EPOCH}.pth  (avg_loss={ckpt['avg_loss']:.4f})")

    # 3) per-machine evaluation
    for machine in ALL_MACHINE_TYPES:
        print(f"\n=== {machine} ===")
        train_ds = ASTRA_EvalRGBDataset(machine, split="train")
        test_ds  = ASTRA_EvalRGBDataset(machine, split="test")

        # gather normals in each domain
        src_norm = filter_by_domain_and_label(train_ds, "source", 0)
        tgt_norm = filter_by_domain_and_label(train_ds, "target", 0)
        if len(src_norm)==0 or len(tgt_norm)==0:
            print("  Skipping (missing normals).")
            continue

        src_loader = DataLoader(src_norm, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        tgt_loader = DataLoader(tgt_norm, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        src_embs, _ = extract_embeddings(src_loader, model)
        tgt_embs, _ = extract_embeddings(tgt_loader, model)

        mu_src = src_embs.mean(axis=0)
        mu_tgt = tgt_embs.mean(axis=0)
        cov    = EmpiricalCovariance().fit(np.vstack([src_embs, tgt_embs]))
        inv_cov = cov.precision_

        # score test samples
        for domain, mu in [("source", mu_src), ("target", mu_tgt)]:
            subset = filter_by_domain(test_ds, domain)
            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            te_embs, te_labels = extract_embeddings(loader, model)

            # Mahalanobis distance
            def mahal(x, μ):
                d = x - μ
                return np.array([v.dot(inv_cov).dot(v.T) for v in d])

            scores = mahal(te_embs, mu)
            # regular AUC
            auc = roc_auc_score(te_labels, scores) if len(set(te_labels))>1 else float("nan")
            # partial AUC at FPR<=0.1
            try:
                p_auc = roc_auc_score(te_labels, scores, max_fpr=0.1)
            except ValueError:
                p_auc = float("nan")

            print(f"  {domain} AUC = {auc:.4f}   pAUC (FPR≤0.1) = {p_auc:.4f}")


if __name__ == "__main__":
    evaluate()
