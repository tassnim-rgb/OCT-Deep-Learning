"""Evaluation helpers.

Classifier metrics (with corrected multi-class macro-F1, zero_division=0),
cross-dataset evaluation on remapped labels, and zoom-variant comparisons.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config, zoom as zoom_mod
from .data import VAL_TF, get_sample_list, load_octc8_samples, load_octid_samples
from .models import GradCAMpp, load_octnet


# ── Fast classifier-only inference ───────────────────────────────────────────

def classify_only(model, img_t, device=None) -> np.ndarray:
    """One image tensor (3,H,W) → softmax probs array (num_classes,). No CAM."""
    model.eval()
    device = device or config.DEVICE
    x = img_t.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
    return torch.softmax(logits, 1)[0].cpu().numpy()


def pil_to_infer(pil_image, transform=None):
    transform = transform or VAL_TF
    return transform(pil_image.convert("RGB"))


class SampleDataset(Dataset):
    """(path, label) sample list → tensors with a transform. Label indices are
    already remapped into the model's class order."""
    def __init__(self, sample_list, transform=None):
        self.samples = sample_list
        self.transform = transform or VAL_TF

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), int(label), str(path)


# ── Metrics ──────────────────────────────────────────────────────────────────

def summarize_metrics(true, pred, class_names=None):
    """accuracy + macro-F1 (corrected: zero_division=0) + per-class report."""
    class_names = class_names or config.CLASS_NAMES
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    per_class = {}
    cm = confusion_matrix(true, pred, labels=list(range(len(class_names))))
    for i, c in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "f1": round(float(f1), 4)}
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4),
            "per_class": per_class, "n": len(true)}


def evaluate_classifier(model, sample_list, class_names=None, batch_size=64,
                        progress=True, device=None, transform=None) -> dict:
    """Predict every (path, label) sample and return summary + arrays."""
    class_names = class_names or config.CLASS_NAMES
    device = device or config.DEVICE
    ds = SampleDataset(sample_list, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=config.NUM_WORKERS,
                        pin_memory=False)
    true, pred, paths = [], [], []
    it = tqdm(loader, desc="eval", leave=False) if progress else loader
    with torch.inference_mode():
        for batch in it:
            imgs, labels, ps = batch
            logits = model(imgs.to(device))
            preds = logits.argmax(1).cpu().numpy()
            true.extend(labels.numpy().tolist())
            pred.extend(preds.tolist())
            paths.extend(ps)
    metrics = summarize_metrics(true, pred, class_names)
    metrics.update({"true": true, "pred": pred, "paths": paths})
    return metrics


def cross_dataset_eval(model, dataset_key: str, class_names=None) -> dict:
    """Evaluate on the shared (4-class) subset of octc8 or octid with remapped
    labels. dataset_key in {'octc8','octid'}."""
    class_names = class_names or config.CLASS_NAMES
    if dataset_key == "octc8":
        _, subset = load_octc8_samples()
    elif dataset_key == "octid":
        _, subset = load_octid_samples()
    else:
        raise ValueError(dataset_key)
    samples = get_sample_list(subset)
    metrics = evaluate_classifier(model, samples, class_names)
    # attach remapped label ↔ original class names for reporting
    metrics["shared_classes"] = subset.shared_classes
    return {"dataset": dataset_key, "n": len(samples), **metrics}


def eval_zoom_variants(model, cam_fn, sample_list, variants=None,
                       class_names=None, progress=True, max_n=None) -> dict:
    """Run zoom-and-reanalyze over samples for each variant; report accuracy of
    final_pred, skip rate, disagreement rate, per-class detail."""
    class_names = class_names or config.CLASS_NAMES
    variants = variants or list(zoom_mod.VARIANTS)
    if max_n is not None:
        sample_list = sample_list[:max_n]

    results = {}
    for v in variants:
        kwargs = zoom_mod.VARIANTS[v]
        fn = zoom_mod.make_zoom_fn(model, cam_fn, class_names, **kwargs)
        true, finals, skips, disagree = [], [], [], []
        it = tqdm(sample_list, desc=f"zoom[{v}]", leave=False) if progress else sample_list
        for n_img, (path, label) in enumerate(it):
            pil = Image.open(path)
            r = fn(pil)
            true.append(label)
            finals.append(class_names.index(r["final_pred"]))
            skips.append(1 if r["zoom_skipped"] else 0)
            disagree.append(0 if r["zoom_skipped"] or r.get("agreement", True) else 1)
            if config.DEVICE.type == "cuda" and n_img % 200 == 199:
                torch.cuda.empty_cache()
        acc = accuracy_score(true, finals)
        macro_f1 = f1_score(true, finals, average="macro", zero_division=0)
        results[v] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "skip_rate": round(float(np.mean(skips)), 4),
            "disagreement_rate": round(float(np.mean(disagree)), 4),
            "n_zoom_called": int(len(skips) - sum(skips)),
        }
    return results


def load_pipeline_runtime(device=None, retriever_ckpt=None, corpus_path=None,
                          zoom_variant="guarded"):
    """Load OCTNet + GradCAMpp + zoom fn + retriever in one call (for app/scripts)."""
    from .retrieval import load_retriever

    device = device or config.DEVICE
    model = load_octnet(device=device)
    cam_fn = GradCAMpp(model, model.cam_layer, device=device)
    zoom_fn = zoom_mod.make_zoom_fn(model, cam_fn, config.CLASS_NAMES,
                                    **zoom_mod.VARIANTS[zoom_variant])
    retriever = load_retriever(ckpt_path=retriever_ckpt, kb_path=corpus_path, device=device)
    return model, cam_fn, zoom_fn, retriever