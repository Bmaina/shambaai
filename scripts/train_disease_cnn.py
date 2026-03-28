#!/usr/bin/env python3
"""
scripts/train_disease_cnn.py — Fine-tune CropDiseaseNet on crop disease images.

Usage:
    # With PlantVillage dataset (recommended):
    python scripts/train_disease_cnn.py --data_dir data/plantvillage/ --epochs 30

    # With custom East African field photos:
    python scripts/train_disease_cnn.py --data_dir data/field_photos/ --epochs 50

    # Quick validation (random synthetic images):
    python scripts/train_disease_cnn.py --synthetic --epochs 3

Data directory structure (ImageFolder format):
    data/plantvillage/
        maize_fall_armyworm/
            img001.jpg
            img002.jpg
        potato_late_blight/
            img001.jpg
        ...

Free datasets:
  - PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
  - CGIAR AI for Agriculture: https://github.com/CGIAR-SPIA
  - iPlant Africa: https://iplant.africa/datasets

Export to TFLite:
    After training, run:
    python scripts/export_tflite.py --checkpoint models/disease_cnn_best.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets

from shambaai.models.disease_cnn import (
    CropDiseaseNet, FocalLoss, get_transforms,
    DISEASE_CLASSES, NUM_CLASSES
)


# ── Synthetic dataset for CI/CD validation ────────────────────────────────────

class SyntheticCropDataset(Dataset):
    """
    Random RGB tensors with random class labels.
    Used only to validate the training loop runs — not for real accuracy.
    """
    def __init__(self, n_samples: int = 500, num_classes: int = NUM_CLASSES):
        self.n = n_samples
        self.num_classes = num_classes

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img   = torch.rand(3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


# ── Metrics ───────────────────────────────────────────────────────────────────

def accuracy(outputs: torch.Tensor, labels: torch.Tensor, top_k: int = 1) -> float:
    _, pred = outputs.topk(top_k, dim=1)
    correct = pred.eq(labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CropDiseaseNet] Device: {device}")

    # Dataset
    if args.synthetic:
        print("[CropDiseaseNet] Using synthetic data (architecture validation only)")
        dataset    = SyntheticCropDataset(n_samples=500)
        num_classes = NUM_CLASSES
        class_names = DISEASE_CLASSES
    elif args.data_dir and os.path.isdir(args.data_dir):
        full_ds = datasets.ImageFolder(
            args.data_dir, transform=get_transforms("train")
        )
        num_classes = len(full_ds.classes)
        class_names = full_ds.classes
        print(f"[CropDiseaseNet] {len(full_ds)} images | {num_classes} classes")
        print(f"  Classes: {class_names[:5]}{'...' if num_classes > 5 else ''}")
        dataset = full_ds
    else:
        raise ValueError("Provide --data_dir or --synthetic")

    # Train/val split
    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    if not args.synthetic:
        val_ds.dataset.transform = get_transforms("val")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=(device=="cuda"))

    print(f"[CropDiseaseNet] Train: {n_train} | Val: {n_val}")

    # Model
    model = CropDiseaseNet(num_classes=num_classes, pretrained=True).to(device)
    total  = sum(p.numel() for p in model.parameters())
    print(f"[CropDiseaseNet] Parameters: {total:,}")

    # Focal loss (handles class imbalance in disease datasets)
    criterion = FocalLoss(gamma=2.0).to(device)

    # Two-stage training:
    # Stage 1 — freeze backbone, train head only (warmup)
    # Stage 2 — unfreeze all, fine-tune with low LR
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_dl),
        epochs=args.epochs,
        pct_start=0.1,
    )

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at epoch 5 (stage 2)
        if epoch == 5:
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.1
            print("  [Stage 2] Backbone unfrozen — full fine-tune at lr×0.1")

        # Train
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * imgs.size(0)
            train_acc  += accuracy(logits, labels) * imgs.size(0)
        train_loss /= n_train
        train_acc  /= n_train

        # Validate
        model.eval()
        val_loss, val_acc, val_top5 = 0.0, 0.0, 0.0
        with torch.inference_mode():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits   = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                val_acc  += accuracy(logits, labels, top_k=1) * imgs.size(0)
                val_top5 += accuracy(logits, labels, top_k=min(5, num_classes)) * imgs.size(0)
        val_loss /= n_val
        val_acc  /= n_val
        val_top5 /= n_val

        flag = " ← best" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:>3}/{args.epochs}  "
            f"loss={train_loss:.3f}/{val_loss:.3f}  "
            f"acc={train_acc:.3f}/{val_acc:.3f}  "
            f"top5={val_top5:.3f}{flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.output_dir, "disease_cnn_best.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "num_classes": num_classes,
                "class_names": class_names,
                "config":      vars(args),
            }, ckpt)

    print(f"\n[CropDiseaseNet] Done. Best val acc: {best_val_acc:.3f}")
    print(f"[CropDiseaseNet] Checkpoint → {args.output_dir}/disease_cnn_best.pt")
    print(f"\nTo export TFLite:")
    print(f"  python scripts/export_tflite.py --checkpoint {args.output_dir}/disease_cnn_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ShambaAI CropDiseaseNet")
    parser.add_argument("--data_dir",   default=None,           help="ImageFolder root directory")
    parser.add_argument("--output_dir", default="models/",      help="Checkpoint save dir")
    parser.add_argument("--epochs",     type=int,   default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=32, help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--synthetic",  action="store_true",    help="Use synthetic data (CI/CD)")
    args = parser.parse_args()
    train(args)
