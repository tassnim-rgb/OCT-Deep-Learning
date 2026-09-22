"""Transforms, dataset helpers, and cross-dataset loading utilities."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config


class AddGaussianNoise:
    """Mimics OCT coherence speckle noise."""
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, t):
        return t + torch.randn_like(t) * self.std


def _norm():
    return transforms.Normalize([0.5] * 3, [0.5] * 3)


def build_train_tf(img_size=None):
    img_size = img_size or config.IMG_SIZE
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.03), scale=(0.95, 1.05), shear=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        _norm(),
        AddGaussianNoise(std=0.02),
    ])


def build_val_tf(img_size=None):
    img_size = img_size or config.IMG_SIZE
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _norm(),
    ])


def build_infer_tf(img_size=None):
    # identical to val transform; kept as a named alias for readability
    return build_val_tf(img_size)


def build_tta_tfs(img_size=None):
    img_size = img_size or config.IMG_SIZE
    return [
        transforms.Compose([transforms.Grayscale(3), transforms.Resize((img_size + 16, img_size + 16)),
                            transforms.CenterCrop(img_size), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Grayscale(3), transforms.Resize((img_size + 16, img_size + 16)),
                            transforms.RandomCrop(img_size), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Grayscale(3), transforms.Resize((img_size + 16, img_size + 16)),
                            transforms.RandomCrop(img_size), transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), _norm()]),
    ]


TRAIN_TF = build_train_tf()
VAL_TF = build_val_tf()
INFER_TF = build_infer_tf()
TTA_TFS = build_tta_tfs()


class ValSubset(torch.utils.data.Dataset):
    """Wrap a random_split subset and apply a (non-augmenting) transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_loaders(root=None, batch_size=None, num_workers=None, seed=None, val_frac=0.1):
    """Train/val/test loaders from an ImageFolder layout (train/, test/)."""
    root = root or config.DATA_ROOT
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    seed = seed or config.SEED

    full_train = datasets.ImageFolder(root / "train", transform=TRAIN_TF)
    test_ds = datasets.ImageFolder(root / "test", transform=VAL_TF)

    val_size = int(val_frac * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_subset = torch.utils.data.random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    val_ds = ValSubset(val_subset, VAL_TF)

    train_targets = [full_train.targets[i] for i in train_ds.indices]
    counts = Counter(train_targets)
    total = sum(counts.values())
    sw = torch.tensor([total / counts[t] for t in train_targets], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(sw, len(sw), replacement=True)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
              persistent_workers=num_workers > 0,
              prefetch_factor=(2 if num_workers > 0 else None))
    train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, full_train


# ── Cross-dataset helpers (octc8 / octid) ────────────────────────────────────

def load_shared_dataset(folder: str, model_classes=None, transform=None):
    """ImageFolder keeping only classes the model knows; remaps label indices
    to MODEL_CLASSES order. Returns the Subset with metadata."""
    model_classes = model_classes or config.CLASS_NAMES
    transform = transform or VAL_TF
    full_ds = datasets.ImageFolder(folder, transform=transform)
    shared = [c for c in model_classes if c in full_ds.classes]
    missing = [c for c in model_classes if c not in full_ds.classes]
    shared_set = set(shared)
    indices = [i for i, (_, label_idx) in enumerate(full_ds.samples)
               if full_ds.classes[label_idx] in shared_set]
    subset = Subset(full_ds, indices)
    subset.shared_classes = shared
    subset.full_ds = full_ds
    subset.model_classes = model_classes
    subset.label_remap = {full_ds.class_to_idx[c]: model_classes.index(c) for c in shared}
    return subset


def get_sample_list(subset) -> list[tuple[str, int]]:
    """list of (path, remapped_label_in_model_classes)."""
    return [(subset.full_ds.samples[i][0], subset.label_remap[subset.full_ds.samples[i][1]])
            for i in subset.indices]


def load_octc8_samples():
    subset = load_shared_dataset(str(config.OCTC8_ROOT / "test"), config.CLASS_NAMES, VAL_TF)
    return get_sample_list(subset), subset


# Interpretive map for OCT-C8's 8 classes → the model's 4 (OCT-C8 has no
# literal AMD/CSR/DR/MH counterparts in Kermany). Ambiguous by nature; used
# only as a deliberately hard cross-domain stress test.
OCTC8_SEMANTIC_MAP = {
    "AMD": "CNV", "CNV": "CNV", "CSR": "CNV", "DME": "DME",
    "DR": "DME", "DRUSEN": "DRUSEN", "MH": "DRUSEN", "NORMAL": "NORMAL",
}


def load_octc8_all_semantic():
    """All 8 OCT-C8 test classes (2800 images) mapped onto the model's 4
    classes via the INTERPRETIVE OCTC8_SEMANTIC_MAP above."""
    return _load_mapped(config.OCTC8_ROOT / "test", OCTC8_SEMANTIC_MAP)


def load_octid_samples():
    """OCTID (CSR/DR/MH/NORMAL) remapped onto the model's 4 classes via an
    INTERPRETIVE pathology mapping (the OCTID labels have no literal Kermany
    counterpart):
      DR -> DME    (diabetic retinopathy — diabetic counterpart of DME)
      CSR -> CNV   (subretinal/serous fluid — CNV's dominant OCT finding)
      MH -> DRUSEN (structural defect — no exact analog, nearest structural)
      NORMAL -> NORMAL
    Treated as a stress test (new domains, approximate labels), not as ground
    truth pathology identity."""
    octid_map = {"CSR": "CNV", "DR": "DME", "MH": "DRUSEN", "NORMAL": "NORMAL"}
    return _load_mapped(config.OCTID_ROOT, octid_map)


def _load_mapped(root, name_map):
    """Load an ImageFolder rooted at `root` where subdir names are mapped to
    model class names via name_map; returns (samples, subset)."""
    model_classes = config.CLASS_NAMES
    full_ds = datasets.ImageFolder(str(root), transform=VAL_TF)
    shared = sorted({m for c in full_ds.classes if (m := name_map.get(c, c)) in model_classes})
    subset = Subset(full_ds, [])
    subset.shared_classes = shared
    subset.full_ds = full_ds
    subset.model_classes = model_classes
    keep = []
    for i, (_, label_idx) in enumerate(full_ds.samples):
        mapped = name_map.get(full_ds.classes[label_idx], full_ds.classes[label_idx])
        if mapped in model_classes:
            keep.append(i)
    subset.indices = keep
    subset.label_remap = {
        label_idx: model_classes.index(name_map.get(full_ds.classes[label_idx],
                                                    full_ds.classes[label_idx]))
        for label_idx in set(full_ds.samples[i][1] for i in keep)
    }
    return get_sample_list(subset), subset