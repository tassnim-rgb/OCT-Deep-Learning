"""Experiment 03 — Zoom-and-reanalyze trade-off, measured across datasets.

Compares the three gate policies on the SAME model:
  baseline : always zoom (no guards)           — research found 42.30% on Kermany,
                                               but OCT-C8 54→84%, OCTID 92→100%
  guarded  : skip if conf>=0.75 or crop<15%    — research: keeps Kermany 99.70%,
                                               but nullifies cross-dataset gains
  margin   : skip only if conf>=0.75 AND       — proposed adaptive compromise:
             top1-top2 margin >= 0.35           zoom still runs on uncertain cases

Reference accuracies (classifier-only, from research notebooks):
  Kermany test 99.90% | OCT-C8 4-class subset 54% | OCTID 4-class 92%
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from octnet import config
from octnet.data import load_octc8_samples, load_octid_samples
from octnet.evaluate import evaluate_classifier, eval_zoom_variants, load_pipeline_runtime

RESULTS = Path(__file__).resolve().parent.parent / "results"
VARIANTS = ["baseline", "guarded", "margin"]


def main():
    model, cam_fn, _, _ = load_pipeline_runtime()
    out = {}

    # ── Kermany test (full 1000) ──────────────────────────────────────────
    import torchvision.datasets as dset
    from octnet.data import VAL_TF
    k_test = dset.ImageFolder(config.DATA_ROOT / "test", transform=VAL_TF)
    k_samples = [(p, l) for p, l in k_test.samples]
    print(f"Kermany test: {len(k_samples)} samples")
    k_cls = evaluate_classifier(model, k_samples, batch_size=128)
    out["kermany_classifier_only"] = {"accuracy": k_cls["accuracy"], "macro_f1": k_cls["macro_f1"]}
    print("kermany classifier-only acc:", k_cls["accuracy"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    kv = eval_zoom_variants(model, cam_fn, k_samples, variants=VARIANTS)
    out["kermany_zoom"] = kv
    for v in VARIANTS:
        print(f"  kermany [{v}] acc {kv[v]['accuracy']} skip {kv[v]['skip_rate']}")

    # ── OCT-C8 shared 4-class subset (all 1400, balanced) ────────────────
    c8_samples, _ = load_octc8_samples()
    print(f"OCT-C8 subset: {len(c8_samples)} samples")
    c8_cls = evaluate_classifier(model, c8_samples, batch_size=128)
    out["octc8_classifier_only"] = {"accuracy": c8_cls["accuracy"], "macro_f1": c8_cls["macro_f1"]}
    print("octc8 classifier-only acc:", c8_cls["accuracy"])
    c8v = eval_zoom_variants(model, cam_fn, c8_samples, variants=VARIANTS)
    out["octc8_zoom"] = c8v
    for v in VARIANTS:
        print(f"  octc8 [{v}] acc {c8v[v]['accuracy']} skip {c8v[v]['skip_rate']}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── OCTID 4-class subset ──────────────────────────────────────────────
    id_samples, _ = load_octid_samples()
    print(f"OCTID subset: {len(id_samples)} samples")
    id_cls = evaluate_classifier(model, id_samples, batch_size=128)
    out["octid_classifier_only"] = {"accuracy": id_cls["accuracy"], "macro_f1": id_cls["macro_f1"]}
    print("octid classifier-only acc:", id_cls["accuracy"])
    idv = eval_zoom_variants(model, cam_fn, id_samples, variants=VARIANTS)
    out["octid_zoom"] = idv
    for v in VARIANTS:
        print(f"  octid [{v}] acc {idv[v]['accuracy']} skip {idv[v]['skip_rate']}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (RESULTS / "03_zoom_tradeoff.json").write_text(json.dumps(out, indent=2))
    print("saved:", RESULTS / "03_zoom_tradeoff.json")


if __name__ == "__main__":
    main()