#!/usr/bin/env python3
"""
scripts/train_edsr.py — Train the GeoSR EDSR super-resolution model.

Usage:
    python scripts/train_edsr.py --epochs 50 --data_dir data/sentinel2/

Wald's Protocol self-supervised training:
  No expensive PlanetScope data needed.
  Uses only free Sentinel-2 imagery — downsample 4x to create LR/HR pairs.

Data directory structure expected:
    data/sentinel2/
        patch_0001.npy   # (4, H, W) float32 Sentinel-2 patches
        patch_0002.npy
        ...

If no data directory is provided, trains on procedurally generated synthetic
spectral patches — useful for architecture validation and CI/CD.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from shambaai.models.edsr import EDSR, SpectralFidelityLoss, walds_protocol_pair


# ── Synthetic dataset for validation ─────────────────────────────────────────

class SyntheticSentinel2Dataset(Dataset):
    """
    Procedurally generated Sentinel-2-like spectral patches.
    Encodes realistic spectral relationships:
      - NDVI range 0.1–0.8
      - Red-Edge responds proportionally to NIR
      - SWIR adds soil/water variation
    Used when no real Sentinel-2 data is available.
    """
    def __init__(self, n_samples: int = 2000, patch_size: int = 64):
        self.n = n_samples
        self.size = patch_size
        rng = np.random.default_rng(42)

        patches = []
        for _ in range(n_samples):
            # Base NDVI map with spatial variation
            ndvi_map = rng.uniform(0.1, 0.8, (patch_size, patch_size)).astype(np.float32)
            ndvi_map = self._smooth(ndvi_map)

            # Simulate 4 bands: B4 (Red), B8 (NIR), B5 (RedEdge), B11 (SWIR)
            B4  = (1 - ndvi_map) / (2 - ndvi_map + 1e-8) * rng.uniform(0.8, 1.2)
            B8  = ndvi_map * (B4 + 1e-4) + B4
            B5  = B8 * rng.uniform(0.5, 0.7) + rng.uniform(0, 0.05, (patch_size, patch_size)).astype(np.float32)
            B11 = rng.uniform(0.05, 0.35, (patch_size, patch_size)).astype(np.float32)

            patch = np.stack([B4, B8, B5, B11], axis=0).astype(np.float32)
            patch = np.clip(patch, 0, 1)
            patches.append(patch)

        self.patches = patches

    @staticmethod
    def _smooth(arr: np.ndarray, sigma: float = 4.0) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        hr = torch.from_numpy(self.patches[idx])
        lr, hr_target = walds_protocol_pair(hr, scale=4, noise_std=0.005)
        return lr, hr_target


class Sentinel2PatchDataset(Dataset):
    """Load real Sentinel-2 patches saved as .npy files."""
    def __init__(self, data_dir: str, patch_size: int = 64):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".npy")
        ])
        self.patch_size = patch_size
        assert len(self.files) > 0, f"No .npy files found in {data_dir}"
        print(f"[Dataset] Found {len(self.files)} Sentinel-2 patches in {data_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx]).astype(np.float32)
        # Crop or pad to patch_size
        _, H, W = patch.shape
        if H >= self.patch_size and W >= self.patch_size:
            y = np.random.randint(0, H - self.patch_size + 1)
            x = np.random.randint(0, W - self.patch_size + 1)
            patch = patch[:, y:y+self.patch_size, x:x+self.patch_size]
        hr = torch.from_numpy(np.clip(patch[:4], 0, 1))
        return walds_protocol_pair(hr, scale=4)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EDSR] Device: {device}")

    # Dataset
    if args.data_dir and os.path.isdir(args.data_dir):
        dataset = Sentinel2PatchDataset(args.data_dir, patch_size=args.patch_size)
    else:
        print(f"[EDSR] No data_dir found — using synthetic Sentinel-2 patches")
        dataset = SyntheticSentinel2Dataset(n_samples=args.n_synthetic, patch_size=args.patch_size)

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=(device=="cuda"))

    print(f"[EDSR] Train: {n_train} | Val: {n_val} | Batch: {args.batch_size}")

    # Model
    model = EDSR(
        n_bands=4, scale=4,
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[EDSR] Parameters: {total_params:,}")

    criterion = SpectralFidelityLoss(alpha=0.7, beta=0.2, gamma=0.1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for lr_batch, hr_batch in train_dl:
            lr_batch  = lr_batch.to(device)
            hr_batch  = hr_batch.to(device)
            optimizer.zero_grad()
            sr_batch  = model(lr_batch)
            loss      = criterion(sr_batch, hr_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * lr_batch.size(0)
        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for lr_batch, hr_batch in val_dl:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = model(lr_batch)
                val_loss += criterion(sr_batch, hr_batch).item() * lr_batch.size(0)
        val_loss /= n_val

        # PSNR on val set (higher = better, 30+ dB is good)
        psnr = -10 * np.log10(max(val_loss, 1e-10))

        flag = " ← best" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"PSNR={psnr:.1f}dB  lr={scheduler.get_last_lr()[0]:.2e}{flag}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "edsr_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }, ckpt_path)

    print(f"\n[EDSR] Training complete. Best checkpoint → {args.output_dir}/edsr_best.pt")
    print(f"[EDSR] Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShambaAI GeoSR EDSR model")
    parser.add_argument("--data_dir",    default=None,            help="Sentinel-2 .npy patches dir")
    parser.add_argument("--output_dir",  default="models/",       help="Checkpoint output dir")
    parser.add_argument("--epochs",      type=int,   default=50,  help="Training epochs")
    parser.add_argument("--batch_size",  type=int,   default=16,  help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-4,help="Learning rate")
    parser.add_argument("--n_feats",     type=int,   default=64,  help="Feature map depth")
    parser.add_argument("--n_resblocks", type=int,   default=16,  help="Residual blocks")
    parser.add_argument("--patch_size",  type=int,   default=64,  help="HR patch size")
    parser.add_argument("--n_synthetic", type=int,   default=2000,help="Synthetic samples if no data_dir")
    args = parser.parse_args()
    train(args)
