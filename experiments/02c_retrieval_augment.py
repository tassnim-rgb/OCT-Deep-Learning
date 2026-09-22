"""Experiment 02c — Content-derived query augmentation for retrieval.

Failure mode observed in 02b: eval queries are clinical paraphrases
('hard exudates retina', 'intraretinal cysts diabetic') while training
queries are label templates ('DME finding'). The encoder never sees
clinical content words tied to labels.

Fix (no eval leakage): for each labeled corpus doc, mine its own content
tokens (stopword-filtered, len>3, filtered to words that actually appear
in corpus text) and synthesize extra queries as token combinations + the
doc's label. Any query string that equals an EVAL_QUERY is excluded.

Elsewhere identical protocol to the saved checkpoint (NT-Xent, batch 64).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from octnet import config
from octnet.models import WordTokenizer
from octnet.retrieval import (ContrastivePairDataset, EVAL_QUERIES, Retriever,
                              load_corpus, load_retriever, nt_xent_loss,
                              precision_at_k)

RESULTS = Path(__file__).resolve().parent.parent / "results"

STOP = set(("the", "and", "for", "with", "are", "can", "may", "from", "into",
            "that", "this", "these", "their", "than", "over", "under", "within",
            "without", "between", "also", "often", "usually", "show", "shows",
            "shown", "seen", "appear", "appears", "appearing", "associated",
            "indicates", "indicating", "characterized", "characteristic",
            "secondary", "primary", "result", "results", "resulting", "related",
            "however", "according", "computed", "measurement", "measurements",
            "used", "use", "using", "due", "such", "presence", "absence",
            "presence"))


def content_queries_for(doc_text, label, rng, n_per_doc=10, eval_set=None):
    words = [w.lower().strip(".,();:\"'-") for w in doc_text.split()]
    content = [w for w in words if len(w) > 3 and w not in STOP and w.isalpha()]
    if len(content) < 3:
        return []
    # keep the most frequent content words
    from collections import Counter
    freq = [w for w, _ in Counter(content).most_common(12)]
    queries = set()
    for _ in range(n_per_doc * 3):
        k = rng.randint(2, 3)
        combo = rng.sample(freq, min(k, len(freq)))
        q = " ".join(combo + [label.lower()])
        # also token-unordered variant
        q = q if rng.random() < 0.7 else " ".join(rng.sample(combo + [label.lower()], len(combo) + 1))
        if q in eval_set:
            continue
        queries.add(q)
        if len(queries) >= n_per_doc:
            break
    return list(queries)


def main():
    corpus = load_corpus(config.KB_PATH)
    old = load_retriever(ckpt_path=config.CKPT_EMB, kb_path=config.KB_PATH)
    p1_o, p3_o = precision_at_k(old, k=1), precision_at_k(old, k=3)
    print(f"[saved] P@1 {p1_o['precision']*100:.2f}%  P@3 {p3_o['precision']*100:.2f}%")

    all_texts = [t for _, t in corpus] + [l for l, _ in corpus]
    tokenizer = WordTokenizer(all_texts, max_len=config.EMB_MAX_LEN)

    eval_set = {q for q, _ in EVAL_QUERIES}
    rng = random.Random(123)
    extra = []
    for label, text in corpus:
        for q in content_queries_for(text, label, rng, n_per_doc=10, eval_set=eval_set):
            extra.append((q, text))
    print(f"content-augmented query pairs added: {len(extra)}")

    base = ContrastivePairDataset(corpus, tokenizer)
    pairs = []
    pairs = list(dict.fromkeys(pairs + extra))  # add augmented
    print(f"total training pairs: {len(pairs)}")

    class PairDS(Dataset):
        def __init__(self, pairs, tok):
            self.pairs = pairs
            self.tok = tok
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            q, d = self.pairs[i]
            return self.tok.encode(q), self.tok.encode(d)

    torch.manual_seed(config.SEED)
    from octnet.models import SmallEncoder
    from octnet import config as cfg
    encoder = SmallEncoder(vocab_size=tokenizer.vocab_size, embed_dim=cfg.EMB_DIM,
                           max_len=cfg.EMB_MAX_LEN).to(config.DEVICE)
    loader = DataLoader(PairDS(pairs, tokenizer), batch_size=64, shuffle=True, drop_last=True)
    if len(loader) == 0:
        loader = DataLoader(PairDS(pairs, tokenizer), batch_size=len(pairs), shuffle=True)

    optimizer = AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    best, best_state = float("inf"), None
    for ep in range(1, 81):
        encoder.train()
        tot, nb = 0.0, 0
        for q_ids, d_ids in loader:
            q_ids, d_ids = q_ids.to(config.DEVICE), d_ids.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent_loss(encoder(q_ids), encoder(d_ids), 0.07)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); nb += 1
        sched.step()
        avg = tot / max(nb, 1)
        if avg < best:
            best = avg
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:03d}/80 | loss {avg:.4f}" + (" ★" if avg == best else ""), flush=True)

    encoder.load_state_dict(best_state)
    encoder.eval()
    ids = torch.stack([tokenizer.encode(t) for _, t in corpus]).to(config.DEVICE)
    with torch.no_grad():
        embs = encoder(ids).cpu()
    new = Retriever(encoder=encoder, tokenizer=tokenizer, corpus=corpus,
                    corpus_embs=embs, device=config.DEVICE)
    p1_n, p3_n = precision_at_k(new, k=1), precision_at_k(new, k=3)
    print(f"[augmented] P@1 {p1_n['precision']*100:.2f}%  P@3 {p3_n['precision']*100:.2f}%")
    print("per-class P@1 augmented:", {k: f"{v:.1f}%" for k, v in p1_n["per_class_pct"].items()})

    for q, correct in [("intraretinal cysts diabetic", "DME"), ("hard exudates retina", "DME")]:
        labs = [h["label"] for h in new.retrieve(q, top_k=3)]
        print(f"  Q {q!r}: augmented->{labs} {'OK' if correct in labs else 'MISS'}")

    # label-filtered retrieval (the agent's actual usage when classifier is
    # decisive): restrict corpus to the query's label
    f1 = {q: (1.0 if new.retrieve(q, top_k=1, label_filter=c)[0]["label"] == c else 0.0)
          for q, c in EVAL_QUERIES}
    filtered_p1 = sum(f1.values()) / len(f1)
    print(f"[augmented + label-filtered] P@1 {filtered_p1*100:.2f}%")

    torch.save(encoder.state_dict(), RESULTS / "retrieval_encoder_xl_v2.pth")
    out = {
        "saved": {"p1": p1_o["precision"], "p3": p3_o["precision"],
                  "per_class_p1": p1_o["per_class_pct"]},
        "augmented": {"p1": p1_n["precision"], "p3": p3_n["precision"],
                      "per_class_p1": p1_n["per_class_pct"], "best_loss": best,
                      "n_pairs": len(pairs), "n_augmented": len(extra)},
        "filtered_p1_augmented": round(filtered_p1, 4),
        "artifact": str(RESULTS / "retrieval_encoder_xl_v2.pth"),
    }
    (RESULTS / "02c_retrieval_augment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()