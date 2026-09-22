"""Experiment 02 — Retrieval: measure current P@k on the XL corpus and
retrain the from-scratch contrastive encoder to beat it.

Baseline (from research notebooks): old 32-entry-corpus model reached
P@1/P@3 = 87.5%; the saved checkpoint already matches the 133-entry XL
vocab. We (a) measure the saved checkpoint on the XL corpus, (b) train a
fresh encoder on the full XL corpus, (c) report P@1/P@3 + per-class.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from octnet import config
from octnet.retrieval import (EVAL_QUERIES, load_corpus, load_retriever,
                              precision_at_k, train_embedder)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def evaluate(retriever, tag):
    p1 = precision_at_k(retriever, k=1)
    p3 = precision_at_k(retriever, k=3)
    print(f"[{tag}] P@1 {p1['precision']*100:.2f}%  P@3 {p3['precision']*100:.2f}%")
    print(f"[{tag}] per-class P@1: " + ", ".join(
        f"{k}:{v:.1f}%" for k, v in p1["per_class_pct"].items()))
    return {"p1": round(p1["precision"], 4), "p3": round(p3["precision"], 4),
            "per_class_p1": p1["per_class_pct"], "n_queries": p1["n"]}


def main():
    corpus = load_corpus(config.KB_PATH)
    print(f"corpus: {len(corpus)} entries | labels: {sorted(set(l for l, _ in corpus))}")

    # (a) saved checkpoint on XL corpus
    old = load_retriever(ckpt_path=config.CKPT_EMB, kb_path=config.KB_PATH)
    old_metrics = evaluate(old, "saved retrieval_encoder.pth")

    # (b) fresh train on full XL corpus
    config.set_seed()
    result = train_embedder(corpus, epochs=config.EMB_EPOCHS, save_path=RESULTS / "retrieval_encoder_xl.pth")
    new_encoder, tok = result["encoder"], result["tokenizer"]

    # rebuild retriever with the fresh encoder without re-tokenizing differently
    from octnet.retrieval import Retriever
    dev = config.DEVICE
    ids = torch.stack([tok.encode(t) for _, t in corpus]).to(dev)
    with torch.no_grad():
        embs = new_encoder(ids).cpu()
    new_ret = Retriever(encoder=new_encoder, tokenizer=tok, corpus=corpus,
                        corpus_embs=embs, device=dev)
    new_metrics = evaluate(new_ret, "fresh XL-trained encoder")

    # (c) plot training curve + embedding space
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(result["history"], lw=2)
    axes[0].set_title(f"NT-Xent loss (best {result['best_loss']:.4f})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    # PCA projection of corpus embeddings colored by label
    e = embs.numpy()
    e = (e - e.mean(0)) / (e.std(0) + 1e-8)
    u, s, vt = np.linalg.svd(e, full_matrices=False)
    proj = e @ vt[:2].T
    colors = {"CNV": "#d62728", "DME": "#1f77b4", "DRUSEN": "#2ca02c", "NORMAL": "#9467bd",
              "DIFFERENTIAL": "#8c564b"}
    labels = [l for l, _ in corpus]
    for lab in sorted(set(labels)):
        mask = [i for i, l in enumerate(labels) if l == lab]
        axes[1].scatter(proj[mask, 0], proj[mask, 1], s=20, label=lab, c=colors.get(lab))
    axes[1].set_title("Corpus embeddings (PCA)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "02_retrieval_embedding_xl.png", dpi=150)
    print("saved plot:", RESULTS / "02_retrieval_embedding_xl.png")

    out = {
        "corpus_entries": len(corpus),
        "vocab_size": tok.vocab_size,
        "saved_checkpoint_on_xl": old_metrics,
        "fresh_xl_trained": new_metrics,
        "improvement_p1_pp": round(new_metrics["p1"] - old_metrics["p1"], 4),
        "best_loss": round(result["best_loss"], 4),
    }
    (RESULTS / "02_retrieval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()