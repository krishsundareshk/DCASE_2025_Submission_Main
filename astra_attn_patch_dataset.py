# astra_attn_patch_dataset.py

import os
from glob import glob
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ALL_MACHINE_TYPES = ["AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera", "Polisher", "ScrewFeeder", "ToyPet","ToyRCCar"]
USE_ATTR = {"BandSealer", "CoffeeGrinder", "HomeCamera", "ToyRCCar"}

class ASTRA_AttnPatchRGBDataset(Dataset):
    """
    Joint dataset over ALL_MACHINE_TYPES, with optional per-image attrs for fan & ToyCar.
    Pads each attrs vector to global_attr_dim.
    """
    def __init__(self,
                 root_dir: str,
                 machine_type: str,
                 split: str = "train",
                 patch_size: int = 32,
                 stride: int = 16,
                 max_patches: int = None,
                 global_attr_dim: int = 0):
        assert machine_type in ALL_MACHINE_TYPES
        self.root_dir = root_dir
        self.machine_type = machine_type
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.global_attr_dim = global_attr_dim

        # 1) image list
        folder = os.path.join(root_dir, machine_type, f"{split}RGB")
        imgs = glob(os.path.join(folder, "*.png"))
        if split == "train":
            self.samples = sorted(imgs)
        else:
            self.samples = sorted([p for p in imgs
                                   if "source" in os.path.basename(p)
                                   or "target" in os.path.basename(p)])

        # 2) load attrs if needed
        self.machine_attr_dim = 0
        self.attr_map = {}
        if machine_type in USE_ATTR:
            csv_p = os.path.join(root_dir, machine_type, "attributes_00.csv")
            if os.path.isfile(csv_p):
                df = pd.read_csv(csv_p)
                # detect filename col
                if "filename" in df.columns:
                    fcol = "filename"
                elif "file_name" in df.columns:
                    fcol = "file_name"
                else:
                    raise ValueError(f"No filename col in {csv_p!r}")
                df["basename"] = df[fcol].astype(str).apply(os.path.basename)
                cols = [c for c in df.columns if c not in {fcol, "basename"}]
                # encode non-numeric
                for c in cols:
                    if not pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].astype("category").cat.codes
                self.machine_attr_dim = len(cols)
                for _, row in df.iterrows():
                    b = row["basename"]
                    vec = torch.tensor(row[cols].to_numpy(dtype=float), dtype=torch.float32)
                    self.attr_map[b] = vec

        # 3) transforms
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def extract_patches(self, img: torch.Tensor):
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.stride)
        patches = unfold(img.unsqueeze(0))   # (1, C*ps*ps, N)
        patches = patches.squeeze(0).T        # (N, C*ps*ps)
        patches = patches.view(-1, 3, self.patch_size, self.patch_size)
        return patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        v1 = self.base_transform(img)
        v2 = self.base_transform(img)
        p1 = self.extract_patches(v1)
        p2 = self.extract_patches(v2)

        if self.max_patches and p1.size(0) > self.max_patches:
            perm = torch.randperm(p1.size(0))
            sel = perm[:self.max_patches]
            p1 = p1[sel]; p2 = p2[sel]

        # build padded attr vector
        if self.machine_attr_dim > 0 and fname in self.attr_map:
            attrs = self.attr_map[fname]
        else:
            attrs = torch.zeros(self.machine_attr_dim, dtype=torch.float32)

        # pad to global_attr_dim
        if self.global_attr_dim > self.machine_attr_dim:
            pad = torch.zeros(self.global_attr_dim - self.machine_attr_dim)
            attrs = torch.cat([attrs, pad], dim=0)

        return {
            "patches_1": p1,
            "patches_2": p2,
            "attrs":     attrs,      # (global_attr_dim,)
            "machine":   self.machine_type,
            "filename":  fname
        }
