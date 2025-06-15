# dataset_inspect.py
import os
import pandas as pd
from astra_attn_patch_dataset import ASTRA_AttnPatchRGBDataset, ALL_MACHINE_TYPES, USE_ATTR

# ─── CONFIG ─────────────────────────────────────────────────
ROOT_DIR     = "training_data"   # <-- point to your base folder
STRIDE       = 16
MAX_PATCHES  = 64
BATCH_SIZE   = 256
# ────────────────────────────────────────────────────────────

def get_global_attr_dim():
    dims = []
    for m in USE_ATTR:
        path = os.path.join(ROOT_DIR, m, "attributes_00.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            cols = [c for c in df.columns if c not in {"filename","file_name"}]
            dims.append(len(cols))
    return max(dims) if dims else 0

if __name__ == "__main__":
    global_attr_dim = get_global_attr_dim()
    total = 0
    print("Per-machine training sample counts:")
    for m in ALL_MACHINE_TYPES:
        ds = ASTRA_AttnPatchRGBDataset(
            ROOT_DIR, m, split="train",
            patch_size=32, stride=STRIDE,
            max_patches=MAX_PATCHES,
            global_attr_dim=global_attr_dim
        )
        n = len(ds)
        print(f"  {m:10s}: {n:4d} samples")
        total += n

    iters = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nTOTAL samples = {total}")
    print(f"With batch_size={BATCH_SIZE} → {iters} iterations per epoch")
