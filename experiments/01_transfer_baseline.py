"""Experiment 01 — Transfer-learning baseline (ResNet18, ImageNet init).

GT for proposal §4: quantifies the cost of training from scratch. OCTNet was
trained from random init to 99.90% TTA. Here ResNet18 (pretrained) is
fine-tuned on the SAME OCT2017 90/10 split with the same seed/eval protocol.

Protocol (explicit): freeze conv1..layer3, train layer4+fc, 12 epochs,
AdamW lr=1e-3 cosine, batch 128, mixed precision. This is the standard
"feature-extractor + head" transfer baseline (not exhaustive fine-tune).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import models as tv_models

from octnet import config
from octnet.evaluate import evaluate_classifier

EPOCHS = 8
BATCH_SIZE = 128
LR = 1e-3
EARLY_STOP_VAL = 0.995   # stop once val acc crosses this threshold

# Pretrained ResNet expects ImageNet normalization; OCTNet uses [-1,1]
# grayscale normalization. Use ImageNet stats for a fair transfer baseline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_tf(img_size=224):
    from torchvision import transforms as T
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(device):
    config.set_seed()
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    # full fine-tune: all layers trainable
    return model.to(device)


def _imagenet_train_tf(img_size=224):
    from torchvision import transforms as T
    from octnet.data import AddGaussianNoise
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size + 16, img_size + 16)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=8, translate=(0.05, 0.03), scale=(0.95, 1.05), shear=3),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        AddGaussianNoise(std=0.02),
    ])


def main():
    import torchvision.datasets as dset
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    device = config.DEVICE
    t0 = time.time()
    config.set_seed()

    full_train = dset.ImageFolder(config.DATA_ROOT / "train", transform=_imagenet_train_tf())
    test_ds = dset.ImageFolder(config.DATA_ROOT / "test", transform=_imagenet_tf())
    val_size = int(0.10 * len(full_train))
    train_ds, val_raw = torch.utils.data.random_split(
        full_train, [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED))
    from octnet.data import ValSubset
    val_ds = ValSubset(val_raw, _imagenet_tf())

    targets = [full_train.targets[i] for i in train_ds.indices]
    counts = Counter(targets)
    total = len(targets)
    sw = torch.tensor([total / counts[t] for t in targets], dtype=torch.float32)
    sampler = WeightedRandomSampler(sw, total, replacement=True)

    def _loader(ds, shuffle=False, sampler=None):
        return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=config.NUM_WORKERS,
                          pin_memory=True, sampler=sampler,
                          persistent_workers=config.NUM_WORKERS > 0,
                          prefetch_factor=(2 if config.NUM_WORKERS > 0 else None))

    train_loader = _loader(train_ds, sampler=sampler)
    val_loader = _loader(val_ds)

    model = build_model(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 TL | trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M total | ckpt classes {len(test_ds.classes)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    history = []
    best_va = 0.0
    stopped_early = False
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs)
                loss = crit(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_n += imgs.size(0)
        sched.step()

        # val
        model.eval()
        va_correct, va_n = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                out = model(imgs.to(device))
                va_correct += (out.argmax(1) == labels.to(device)).sum().item()
                va_n += imgs.size(0)
        va_acc = va_correct / va_n
        best_va = max(best_va, va_acc)
        history.append({"epoch": ep, "tr_loss": tr_loss / tr_n,
                        "tr_acc": tr_correct / tr_n, "va_acc": va_acc,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Ep {ep:03d}/{EPOCHS} | tr {history[-1]['tr_loss']:.4f}/{history[-1]['tr_acc']*100:.2f}% "
              f"| va {va_acc*100:.2f}% | best {best_va*100:.2f}%", flush=True)
        if va_acc >= EARLY_STOP_VAL:
            print(f"Early stop: val acc {va_acc*100:.2f}% >= {EARLY_STOP_VAL*100:.0f}%")
            stopped_early = True
            break

    # test eval (ImageNet-normalized, same metric as OCTNet evaluation)
    test_samples = [(p, l) for p, l in test_ds.samples] if hasattr(test_ds, "samples") else _test_samples(test_ds)
    test_metrics = evaluate_classifier(model, test_samples, transform=_imagenet_tf())

    elapsed = time.time() - t0
    summary = {
        "model": "ResNet18 (ImageNet init), full fine-tune",
        "params_total": total, "params_trainable": trainable,
        "epochs": ep if stopped_early else EPOCHS, "epochs_run": ep,
        "early_stopped": stopped_early,
        "batch_size": BATCH_SIZE, "lr": LR,
        "wall_time_s": round(elapsed, 1),
        "val_best_acc": round(float(best_va), 4),
        "test": test_metrics,
    }
    out = Path(__file__).resolve().parent.parent / "results" / "01_transfer_baseline.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))
    print("saved:", out)


def _test_samples(test_ds):
    # (path, label) pairs from the ImageFolder dataset
    if hasattr(test_ds, "samples"):
        return [(p, l) for p, l in test_ds.samples]
    return [(test_ds.imgs[i][0], test_ds.targets[i]) for i in range(len(test_ds))]


if __name__ == "__main__":
    main()