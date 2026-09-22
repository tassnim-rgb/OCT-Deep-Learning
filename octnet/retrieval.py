"""Tool 4 — from-scratch contrastive retrieval encoder.

Loads the knowledge base (oct_corpus_xl.json by default), builds the
tokenizer/encoder, loads a checkpoint, and exposes retrieve().
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .models import SmallEncoder, WordTokenizer


def load_corpus(kb_path=None) -> list[tuple[str, str]]:
    kb_path = Path(kb_path or config.KB_PATH)
    with open(kb_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{"label": k, "text": v} for k, v in data.items()]
    return [(item["label"], item["text"]) for item in data]


class ContrastivePairDataset(Dataset):
    """Query/doc positive pairs; negatives handled in-batch by NT-Xent."""
    QUERY_TEMPLATES = [
        "{label} finding",
        "{label} OCT appearance",
        "{label} diagnosis",
        "{label} retinal features",
        "{label} clinical signs",
        "what does {label} look like on OCT",
        "{label} imaging characteristics",
    ]

    def __init__(self, corpus: list[tuple[str, str]], tokenizer: WordTokenizer):
        self.tokenizer = tokenizer
        self.pairs = []
        for label, text in corpus:
            for tmpl in self.QUERY_TEMPLATES:
                self.pairs.append((tmpl.format(label=label), text))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, d = self.pairs[idx]
        return self.tokenizer.encode(q), self.tokenizer.encode(d)


def nt_xent_loss(q_emb: torch.Tensor, d_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetrized NT-Xent with in-batch negatives (positive pairs on diagonal)."""
    sim = torch.matmul(q_emb, d_emb.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_q2d = torch.nn.functional.cross_entropy(sim, labels)
    loss_d2q = torch.nn.functional.cross_entropy(sim.T, labels)
    return (loss_q2d + loss_d2q) / 2


def build_tokenizer_encoder(corpus: list[tuple[str, str]], device=None) -> tuple[WordTokenizer, SmallEncoder]:
    device = device or config.DEVICE
    all_texts = [t for _, t in corpus] + [l for l, _ in corpus]
    tokenizer = WordTokenizer(all_texts, max_len=config.EMB_MAX_LEN)
    encoder = SmallEncoder(vocab_size=tokenizer.vocab_size,
                           embed_dim=config.EMB_DIM,
                           max_len=config.EMB_MAX_LEN).to(device)
    encoder.eval()
    return tokenizer, encoder


@dataclass
class Retriever:
    encoder: SmallEncoder
    tokenizer: WordTokenizer
    corpus: list[tuple[str, str]]
    corpus_embs: torch.Tensor
    device: torch.device

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        ids = torch.stack([self.tokenizer.encode(t) for t in texts]).to(self.device)
        return self.encoder(ids).cpu()

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 3, label_filter: str | None = None) -> list[dict]:
        q_emb = self.embed([query])
        scores = (self.corpus_embs @ q_emb.T).squeeze(1)
        if label_filter:
            mask = torch.tensor([l == label_filter for l, _ in self.corpus])
            scores = scores.masked_fill(~mask, -1.0)
        top_idx = scores.argsort(descending=True)[:top_k]
        return [
            {"rank": rank + 1, "label": self.corpus[i][0], "text": self.corpus[i][1],
             "score": float(scores[i])}
            for rank, i in enumerate(top_idx.tolist())
        ]


def load_retriever(ckpt_path=None, kb_path=None, device=None, strict=True) -> Retriever:
    """Load the from-scratch encoder + KB, verifying vocab compatibility."""
    ckpt_path = Path(ckpt_path or config.CKPT_EMB)
    device = device or config.DEVICE
    corpus = load_corpus(kb_path)
    tokenizer, encoder = build_tokenizer_encoder(corpus, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]

    if strict:
        encoder.load_state_dict(state)
    else:
        try:
            encoder.load_state_dict(state)
        except RuntimeError as e:
            print(f"[retrieval] checkpoint vocab mismatch ({e}); encoding corpus anyway.")
            return None  # caller must retrain

    corpus_embs = encoder(torch.stack([tokenizer.encode(t) for t, _ in corpus]).to(device)).cpu()
    return Retriever(encoder=encoder, tokenizer=tokenizer, corpus=corpus,
                     corpus_embs=corpus_embs, device=device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_embedder(corpus: list[tuple[str, str]], epochs=None, lr=None, temperature=None,
                   batch_size: int = 32, device=None, save_path=None, seed=None) -> dict:
    """Train the from-scratch contrastive encoder. Returns {'history', 'best_loss', 'encoder', 'tokenizer'}."""
    from . import config as cfg

    device = device or cfg.DEVICE
    epochs = epochs or cfg.EMB_EPOCHS
    lr = lr or cfg.EMB_LR
    temperature = temperature or cfg.EMB_TEMPERATURE
    save_path = Path(save_path) if save_path else None

    tokenizer, encoder = build_tokenizer_encoder(corpus, device)
    dataset = ContrastivePairDataset(corpus, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:  # corpus too small for full batch
        loader = DataLoader(dataset, batch_size=max(batch_size, len(dataset)), shuffle=True)

    optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    history = []
    for ep in range(1, epochs + 1):
        encoder.train()
        ep_loss = 0.0
        for q_ids, d_ids in loader:
            q_ids, d_ids = q_ids.to(device), d_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            q_emb = encoder(q_ids)
            d_emb = encoder(d_ids)
            loss = nt_xent_loss(q_emb, d_emb, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        avg = ep_loss / len(loader)
        history.append(avg)
        if avg < best_loss:
            best_loss = avg
            if save_path is not None:
                torch.save(encoder.state_dict(), save_path)
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:03d}/{epochs} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                  + (" ★" if avg == best_loss else ""))
    print(f"Best loss: {best_loss:.4f} — saved to {save_path}")
    encoder.eval()
    return {"history": history, "best_loss": best_loss, "encoder": encoder, "tokenizer": tokenizer}


# ── Triplet training with hard-negative mining ────────────────────────────────
# The NT-Xent variant above plateaus on the small corpus because random
# in-batch negatives rarely include the token-similar, different-label docs
# that actually fool retrieval. This variant mines hard negatives explicitly
# (token-overlap with the query among other labels) and uses a triplet margin
# loss — directly targeting the observed failure mode.

def token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    sa = set(w.lower() for w in a.split())
    sb = set(w.lower() for w in b.split())
    return len(sa & sb) / (len(sa | sb) + 1e-8)


def hard_negatives_for(query: str, corpus: list, label: str, k: int = 6):
    """Indices of docs with a different label, ranked by token overlap with the query."""
    cands = [(i, token_overlap(query, t)) for i, (l, t) in enumerate(corpus) if l != label]
    cands.sort(key=lambda x: -x[1])
    return [i for i, _ in cands[:k]], (cands[0][1] if cands else 0.0)


def train_embedder_triplet(corpus: list[tuple[str, str]], epochs: int = 40,
                           lr: float = 2e-4, margin: float = 0.2, hard_k: int = 6,
                           queries_per_step: int = 8, hard_per_query: int = 4,
                           device=None, save_path=None, seed: int = 42) -> dict:
    """Triplet training with explicit hard negatives. See module docstring."""
    import numpy as np

    device = device or config.DEVICE
    tokenizer, encoder = build_tokenizer_encoder(corpus, device)
    ds = ContrastivePairDataset(corpus, tokenizer)
    text2idx = {t: i for i, (_, t) in enumerate(corpus)}
    qpairs = [(q, text2idx[d]) for q, d in ds.pairs]

    cache = {}
    def hard_for(q, di):
        key = (q, di)
        if key not in cache:
            label = corpus[di][0]
            cache[key] = hard_negatives_for(q, corpus, label, k=hard_k)[0]
        return cache[key]

    rng = np.random.RandomState(seed)
    optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best, history = float("inf"), []
    best_save = None
    for ep in range(1, epochs + 1):
        encoder.train()
        total, nb = 0.0, 0
        rng.shuffle(qpairs)
        for start in range(0, len(qpairs), queries_per_step):
            chunk = qpairs[start:start + queries_per_step]
            texts, plan = [], []
            for q, di in chunk:
                hn = hard_for(q, di)[:hard_per_query]
                q_idx = len(texts); texts.append(q)
                p_idx = len(texts); texts.append(corpus[di][1])
                neg_start = len(texts)
                texts.extend(corpus[i][1] for i in hn)
                plan.append((q_idx, p_idx, list(range(neg_start, neg_start + len(hn)))))
            ids = torch.stack([tokenizer.encode(t) for t in texts]).to(device)
            embs = encoder(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for q_idx, p_idx, negs in plan:
                qe, pe = embs[q_idx], embs[p_idx]
                sim_pos = (qe * pe).sum()
                push = torch.relu(margin - sim_pos + (qe * embs[negs]).sum(1)).mean()
                loss = loss + (1 - sim_pos) + push
            loss = loss / len(plan)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); nb += 1
        scheduler.step()
        avg = total / max(nb, 1)
        history.append(avg)
        if avg < best:
            best = avg
            if save_path is not None:
                torch.save(encoder.state_dict(), save_path)
                best_save = str(save_path)
        if ep % 5 == 0 or ep == 1:
            print(f"Ep {ep:03d}/{epochs} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                  + (" ★" if avg == best else ""), flush=True)
    encoder.eval()
    return {"history": history, "best_loss": best, "encoder": encoder,
            "tokenizer": tokenizer, "save_path": best_save}


# ── Evaluation ────────────────────────────────────────────────────────────────

EVAL_QUERIES = [
    ("CNV finding", "CNV"),
    ("choroidal neovascularization OCT", "CNV"),
    ("subretinal fluid appearance", "CNV"),
    ("pigment epithelial detachment", "CNV"),
    ("DME finding", "DME"),
    ("diabetic macular edema OCT", "DME"),
    ("intraretinal cysts diabetic", "DME"),
    ("hard exudates retina", "DME"),
    ("DRUSEN finding", "DRUSEN"),
    ("drusen AMD macular degeneration", "DRUSEN"),
    ("RPE elevation dome shaped", "DRUSEN"),
    ("soft drusen OCT appearance", "DRUSEN"),
    ("NORMAL finding", "NORMAL"),
    ("normal healthy macula OCT", "NORMAL"),
    ("intact ellipsoid zone no fluid", "NORMAL"),
    ("foveal pit normal retina", "NORMAL"),
]


def precision_at_k(retriever: Retriever, queries=None, k: int = 3) -> dict:
    queries = queries or EVAL_QUERIES
    hits, by_class = 0, {}
    for query, correct in queries:
        labels = [r["label"] for r in retriever.retrieve(query, top_k=k)]
        hit = correct in labels
        hits += int(hit)
        by_class.setdefault(correct, []).append(int(hit))
    per_class = {c: (sum(v) / len(v) * 100) for c, v in sorted(by_class.items())}
    return {"precision": hits / len(queries), "n": len(queries),
            "per_class_pct": per_class}