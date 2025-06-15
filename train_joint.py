# train_joint.py

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from astra_attn_patch_dataset import ASTRA_AttnPatchRGBDataset, ALL_MACHINE_TYPES
from patch_attn_model import PatchAttentionCLModel, NTXentLoss

# ─── CONFIG ───────────────────────────────────────────────────────
ROOT_DIR        = "training_data"
CHECKPOINT_DIR  = "checkpoints_submission25_BS256_Resnet34,stride16"
BATCH_SIZE      = 256
EPOCHS          = 500
LEARNING_RATE   = 2e-4
TEMPERATURE     = 0.1
MAX_PATCHES     = 64
STRIDE          = 16
EARLYSTOP_PAT   = 25     # early stopping patience
LR_PATIENCE     = 10     # LR scheduler patience
LR_FACTOR       = 0.5    # LR reduction factor
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ATTR = {"BandSealer", "CoffeeGrinder", "HomeCamera", "ToyRCCar"}
# ─────────────────────────────────────────────────────────────────


def get_attr_dim(machine):
    if machine not in USE_ATTR:
        return 0
    import pandas as pd
    path = os.path.join(ROOT_DIR, machine, "attributes_00.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c not in {"filename", "file_name"}]
        return len(cols)
    return 0


def find_last_checkpoint():
    """Scan CHECKPOINT_DIR for epoch{n}.pth, return (n, path) or (0, None)."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0, None
    best_n, best_path = 0, None
    for fn in os.listdir(CHECKPOINT_DIR):
        if fn.startswith("epoch") and fn.endswith(".pth"):
            try:
                n = int(fn[len("epoch"):-4])
                if n > best_n:
                    best_n, best_path = n, os.path.join(CHECKPOINT_DIR, fn)
            except ValueError:
                continue
    return best_n, best_path


def main():
    # compute global attribute dim
    attr_dims       = [get_attr_dim(m) for m in USE_ATTR]
    global_attr_dim = max(attr_dims) if attr_dims else 0

    # prepare dataset
    datasets = []
    for m in ALL_MACHINE_TYPES:
        ds = ASTRA_AttnPatchRGBDataset(
            ROOT_DIR, m, split="train",
            patch_size=32, stride=STRIDE,
            max_patches=MAX_PATCHES,
            global_attr_dim=global_attr_dim
        )
        datasets.append(ds)
    joint_ds = ConcatDataset(datasets)
    loader   = DataLoader(
        joint_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # model + optimizer + loss + scheduler
    model     = PatchAttentionCLModel(embed_dim=128, attr_dim=global_attr_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True
    )
    criterion = NTXentLoss(temperature=TEMPERATURE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # resume if any
    last_epoch, ckpt_path = find_last_checkpoint()
    start_epoch = last_epoch + 1
    best_loss   = float("inf")
    no_improve  = 0

    if ckpt_path is not None:
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        best_loss = ckpt.get("avg_loss", best_loss)
    else:
        print("🚀 Starting training from scratch")

    # training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{EPOCHS}]", ncols=80)
        for batch in pbar:
            p1, p2 = batch["patches_1"].to(DEVICE), batch["patches_2"].to(DEVICE)
            attrs   = batch["attrs"].to(DEVICE)

            optimizer.zero_grad()
            z1   = model(p1, p1.size(0), p1.size(1), attrs=attrs)
            z2   = model(p2, p2.size(0), p2.size(1), attrs=attrs)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # step LR scheduler
        scheduler.step(avg_loss)

        # save checkpoint
        ckpt_out = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "avg_loss":    avg_loss
        }
        torch.save(ckpt_out, os.path.join(CHECKPOINT_DIR, f"epoch{epoch}.pth"))

        # early stopping
        if avg_loss < best_loss:
            best_loss, no_improve = avg_loss, 0
        else:
            no_improve += 1
            if no_improve >= EARLYSTOP_PAT:
                print(f"🛑 Early stopping at epoch {epoch} (no improvement in {EARLYSTOP_PAT}).")
                break

    print("✅ Joint training complete. Checkpoints in:", CHECKPOINT_DIR)


if __name__ == "__main__":
    # for Windows multiprocessing safety
    import multiprocessing
    multiprocessing.freeze_support()
    main()
