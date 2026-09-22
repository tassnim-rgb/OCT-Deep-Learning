"""Experiment 02b — Retrieval with hard-negative triplet training.

Targets the two observed DME failures:
  'intraretinal cysts diabetic'   -> DIFFERENTIAL docs win (token-similar)
  'hard exudates retina'          -> NORMAL docs win (embedding degeneracy)

Batches re-shuffle document order so doc-with-same-label pairs are useful;
explicit hard negatives force the encoder to separate query from
different-label docs that share tokens.
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
from octnet.retrieval import (Retriever, load_corpus, load_retriever,
                              precision_at_k, train_embedder_triplet)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_retriever(encoder, tok, corpus, device):
    ids = torch.stack([tok.encode(t) for _, t in corpus]).to(device)
    with torch.no_grad():
        embs = encoder(ids).cpu()
    return Retriever(encoder=encoder, tokenizer=tok, corpus=corpus,
                     corpus_embs=embs, device=device)


def main():
    corpus = load_corpus(config.KB_PATH)
    old = load_retriever(ckpt_path=config.CKPT_EMB, kb_path=config.KB_PATH)
    p1_old = precision_at_k(old, k=1)
    p3_old = precision_at_k(old, k=3)
    print(f"[saved] P@1 {p1_old['precision']*100:.2f}%  P@3 {p3_old['precision']*100:.2f}%")

    config.set_seed()
    res = train_embedder_triplet(corpus, epochs=60, save_path=RESULTS / "retrieval_encoder_xl_v2.pth")
    new = make_retriever(res["encoder"], res["tokenizer"], corpus, config.DEVICE)
    p1_new = precision_at_k(new, k=1)
    p3_new = precision_at_k(new, k=3)
    print(f"[triplet(s)] P@1 {p1_new['precision']*100:.2f}%  P@3 {p3_new['precision']*100:.2f}%")
    print("per-class P@1 triplet:", {k: f"{v:.1f}%" for k, v in p1_new["per_class_pct"].items()})

    # query-level diff on the two previously failing queries
    for q, correct in [("intraretinal cysts diabetic", "DME"), ("hard exudates retina", "DME")]:
        old_labs = [h["label"] for h in old.retrieve(q, top_k=3)]
        new_labs = [h["label"] for h in new.retrieve(q, top_k=3)]
        print(f"  Q {q!r}: saved->{old_labs}  triplet->{new_labs}  "
              f"{'OK' if correct in new_labs else 'MISS'}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(res["history"], lw=2)
    ax.set_title(f"Triplet + hard negatives loss (best {res['best_loss']:.4f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "02b_triplet_curve.png", dpi=150)

    out = {
        "saved": {"p1": p1_old["precision"], "p3": p3_old["precision"],
                  "per_class_p1": p1_old["per_class_pct"]},
        "triplet": {"p1": p1_new["precision"], "p3": p3_new["precision"],
                    "per_class_p1": p1_new["per_class_pct"],
                    "best_loss": res["best_loss"]},
        "delta_p3_pp": round(p3_new["precision"] - p3_old["precision"], 4),
        "artifact": str(RESULTS / "retrieval_encoder_xl_v2.pth"),
    }
    (RESULTS / "02b_retrieval_triplet.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()