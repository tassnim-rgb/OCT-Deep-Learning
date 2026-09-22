"""Experiment 04 — Cross-dataset ablations with corrected metrics.

Addresses the proposal's ablation requirement where tools can actually matter:

  A   OCT-C8 literal shared subset (CNV/DME/DRUSEN/NORMAL, 1000 of 1400).
  A2  OCT-C8 ALL classes with an interpretive pathology map (2800 images,
      AMD→CNV CSR→CNV DR→DME MH→DRUSEN) — a deliberately hard stress test
      (this is the regime where the research noted 54% classifier accuracy).
  B   Kermany "uncertain slice" — the 100 test images with the lowest
      classifier confidence (ties broken by top1-top2 margin).
  C   LLM agent ablation — ONLY when CEREBRAS_API_KEY is set.

Conditions (vision-side): classifier-only, +guarded zoom, +margin zoom,
+baseline zoom. Metrics: accuracy + macro-F1 (corrected, zero_division=0).
The research notebooks' OCTID F1 was broken by a label-class count mismatch —
fixed here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from octnet import config, zoom as zoom_mod
from octnet.agent import make_tools, run_agent, ablation_run
from octnet.data import (VAL_TF, load_octc8_all_semantic, load_octc8_samples)
from octnet.evaluate import (GradCAMpp, classify_only, load_pipeline_runtime,
                             pil_to_infer, summarize_metrics)
from octnet.retrieval import load_retriever

RESULTS = Path(__file__).resolve().parent.parent / "results"
VARIANTS = ["guarded", "margin", "baseline"]


# ── shared evaluators (with periodic CUDA cache eviction) ──────────────────

def run_classifier(model, samples, desc):
    model.eval()
    true, pred = [], []
    with torch.inference_mode():
        for i, (path, label) in enumerate(tqdm(samples, desc=desc, leave=False)):
            x = pil_to_infer(Image.open(path)).unsqueeze(0).to(config.DEVICE)
            p = int(F.softmax(model(x), 1).argmax(1).item())
            true.append(label); pred.append(p)
    return summarize_metrics(true, pred)


def run_zoom(model, cam_fn, samples, variant, desc):
    fn = zoom_mod.make_zoom_fn(model, cam_fn, config.CLASS_NAMES,
                               **zoom_mod.VARIANTS[variant])
    true, pred = [], []
    for i, (path, label) in enumerate(tqdm(samples, desc=desc, leave=False)):
        r = fn(Image.open(path))
        true.append(label)
        pred.append(config.CLASS_NAMES.index(r["final_pred"]))
        if config.DEVICE.type == "cuda" and i % 300 == 299:
            torch.cuda.empty_cache()
    return summarize_metrics(true, pred)


def zoom_table(ab, label):
    print(label)
    for k, v in ab.items():
        print(f"  {k:18s} acc {v['accuracy']:.4f}  macro-F1 {v['macro_f1']:.4f}")


def uncertain_slice(model, sample_list, bottom_k=100):
    """The model's own least-trusted K test images: ranked by ascending
    confidence (ties broken by ascending top1-top2 margin). Deterministic,
    fixed-size — unlike a confidence threshold, this never collapses to a
    degenerate sample count on a near-perfect test set."""
    model.eval()
    scored = []
    for path, label in sample_list:
        probs = classify_only(model, pil_to_infer(Image.open(path)))
        top = np.sort(probs)[::-1]
        conf = float(top[0])
        margin = float(top[0] - top[1]) if len(top) > 1 else 1.0
        scored.append(((conf, margin), (path, label)))
    scored.sort(key=lambda x: x[0])            # ascending conf, then margin
    chosen = [sl for _, sl in scored[:bottom_k]]
    details = [sc for sc, _ in scored[:bottom_k]]
    return chosen, details


def main():
    model, cam_fn, _, _ = load_pipeline_runtime()
    out = {}

    # ── A. OCT-C8 shared subset ablations (all 1400, balanced) ──────────
    c8_samples, _ = load_octc8_samples()
    out["octc8_n"] = len(c8_samples)
    ab = {"classifier_only": run_classifier(model, c8_samples, "c8 classifier-only")}
    for v in VARIANTS:
        ab[f"zoom_{v}"] = run_zoom(model, cam_fn, c8_samples, v, f"c8 zoom[{v}]")
    out["octc8_ablations"] = ab
    zoom_table(ab, "OCT-C8 shared-subset ablations:")
    if config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── A2. OCT-C8 all-classes interpretive stress test ──────────────────
    c8all, _ = load_octc8_all_semantic()
    out["octc8_all_n"] = len(c8all)
    ab2 = {"classifier_only": run_classifier(model, c8all, "c8all classifier-only")}
    for v in VARIANTS:
        ab2[f"zoom_{v}"] = run_zoom(model, cam_fn, c8all, v, f"c8all zoom[{v}]")
    out["octc8_all_semantic"] = ab2
    zoom_table(ab2, "OCT-C8 all-classes (interpretive map) ablations:")
    if config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── B. Kermany uncertain slice (bottom-100 by confidence) ────────────
    import torchvision.datasets as dset
    k_test = dset.ImageFolder(config.DATA_ROOT / "test", transform=VAL_TF)
    k_samples = [(p, l) for p, l in k_test.samples]
    unc, details = uncertain_slice(model, k_samples, bottom_k=100)
    print(f"Kermany uncertain slice: {len(unc)} samples "
          f"(mean conf {np.mean([d[0] for d in details]):.3f}, "
          f"mean margin {np.mean([d[1] for d in details]):.3f})")
    u_ab = {"n": len(unc),
            "mean_conf": round(float(np.mean([d[0] for d in details])), 4),
            "mean_margin": round(float(np.mean([d[1] for d in details])), 4)}
    u_ab["classifier_only"] = run_classifier(model, unc, "uncertain classifier-only")
    for v in VARIANTS:
        u_ab[f"zoom_{v}"] = run_zoom(model, cam_fn, unc, v, f"uncertain zoom[{v}]")
    out["kermany_uncertain"] = u_ab
    zoom_table({k: v for k, v in u_ab.items() if isinstance(v, dict)},
               "Kermany uncertain-slice ablations:")
    if config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── C. LLM-dependent agent ablation (only with a key) ─────────────────
    has_key = bool(os.getenv("CEREBRAS_API_KEY"))
    if has_key:
        tools = make_tools(model, cam_fn, load_retriever(),
                           zoom_kwargs=zoom_mod.VARIANTS["guarded"])
        n = min(120, len(c8_samples))
        rng = np.random.RandomState(7)
        idx = rng.choice(len(c8_samples), n, replace=False)
        sample = [c8_samples[i] for i in idx]
        conds = {
            "Full pipeline": [],
            "No zoom (-Tool3)": ["zoom_reanalyze"],
            "No retrieval (-Tool4)": ["retrieve_knowledge"],
            "Classifier only (-2,3,4)": ["localize", "zoom_reanalyze", "retrieve_knowledge"],
        }
        res = {}
        for cond, disabled in conds.items():
            preds = []
            for path, label in tqdm(sample, desc=f"agent[{cond}]", leave=False):
                preds.append(ablation_run(path, tools, disabled))
            res[cond] = summarize_metrics([l for _, l in sample], preds)
            print(f"[agent] {cond:24s} acc {res[cond]['accuracy']:.4f} "
                  f"macro-F1 {res[cond]['macro_f1']:.4f}")
        out["agent_ablation_octc8"] = res
    else:
        out["agent_ablation_octc8"] = {
            "status": "SKIPPED — set CEREBRAS_API_KEY to run LLM ablation"}

    (RESULTS / "04_crossdataset_ablation.json").write_text(json.dumps(out, indent=2))
    print("saved:", RESULTS / "04_crossdataset_ablation.json")


if __name__ == "__main__":
    main()