"""Compile results/*.json into a compact markdown summary for the write-up."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pct(x, digits=2):
    return f"{x * 100:.{digits}f}"


def main():
    lines = []

    # ── 01 transfer baseline ───────────────────────────────────────────────
    r1 = load("01_transfer_baseline.json")
    if r1:
        t = r1.get("test", {})
        lines.append("## Experiment 01 — Transfer baseline (ResNet18 ImageNet)")
        lines.append("")
        lines.append(f"- model: {r1['model']} | params {r1['params_total']/1e6:.2f}M "
                     f"(trainable {r1['params_trainable']/1e6:.2f}M)")
        lines.append(f"- epochs run: {r1['epochs_run']} | early-stopped: {r1['early_stopped']} "
                     f"| wall time: {r1['wall_time_s']/60:.1f} min")
        lines.append(f"- val best: {pct(r1['val_best_acc'])}% | "
                     f"test acc: {pct(t.get('accuracy'))}% | test macro-F1: {pct(t.get('macro_f1'))}%")
        per = t.get("per_class") or {}
        lines.append("- test per-class F1: " + ", ".join(f"{k} {pct(v['f1'])}%" for k, v in per.items()))

    # ── 02 retrieval ───────────────────────────────────────────────────────
    r2 = load("02_retrieval.json")
    if r2:
        lines.append("")
        lines.append("## Experiment 02 — Retrieval on the 133-entry XL corpus")
        lines.append(f"- vocab: {r2['vocab_size']} | best loss: {r2['best_loss']:.4f}")
        for tag in ("saved_checkpoint_on_xl", "fresh_xl_trained"):
            m = r2[tag]
            lines.append(f"- {tag}: P@1 {pct(m['p1'])}% | P@3 {pct(m['p3'])}%  "
                         f"(per-class P@1: " + ", ".join(
                             f"{k} {v:.0f}%" for k, v in m["per_class_p1"].items()) + ")")

    r2b = load("02b_retrieval_triplet.json")
    if r2b:
        lines.append("")
        lines.append("### 02b hard-negative triplet (explored, regressed)")
        lines.append(f"- triplet: P@1 {pct(r2b['triplet']['p1'])}% | P@3 {pct(r2b['triplet']['p3'])}%"
                     f" (delta P@3 {r2b['delta_p3_pp'] * 100:+.1f} pp)")

    r2c = load("02c_retrieval_augment.json")
    if r2c:
        lines.append("")
        lines.append("### 02c content-query augmentation")
        lines.append(f"- augmented: P@1 {pct(r2c['augmented']['p1'])}% | "
                     f"P@3 {pct(r2c['augmented']['p3'])}% | "
                     f"label-filtered P@1 {pct(r2c['filtered_p1_augmented'])}%")

    # ── 03 zoom trade-off ──────────────────────────────────────────────────
    r3 = load("03_zoom_tradeoff.json")
    if r3:
        lines.append("")
        lines.append("## Experiment 03 — Zoom-and-reanalyze trade-off")
        lines.append("")
        lines.append("| dataset | policy | acc % | macro-F1 % | skip % |")
        lines.append("|---|---|---|---|---|")
        for ds in ("kermany", "octc8", "octid"):
            cls = r3.get(f"{ds}_classifier_only", {})
            lines.append(f"| {ds} | classifier-only | {pct(cls.get('accuracy'))} | "
                         f"{pct(cls.get('macro_f1'))} | — |")
            for v, m in (r3.get(f"{ds}_zoom") or {}).items():
                lines.append(f"| {ds} | {v} | {pct(m['accuracy'])} | "
                             f"{pct(m['macro_f1'])} | {pct(m['skip_rate'])} |")

    # ── 04 ablations ───────────────────────────────────────────────────────
    r4 = load("04_crossdataset_ablation.json")
    if r4:
        lines.append("")
        lines.append("## Experiment 04 — Cross-dataset ablations (corrected macro-F1)")
        ab = r4.get("octc8_ablations") or {}
        if ab:
            lines.append("")
            lines.append("### OCT-C8 shared 4-class subset")
            lines.append("| condition | acc % | macro-F1 % |")
            lines.append("|---|---|---|")
            for k, v in ab.items():
                lines.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['macro_f1'])} |")
        ab2 = r4.get("octc8_all_semantic") or {}
        if ab2:
            lines.append("")
            lines.append(f"### OCT-C8 all-classes interpretive stress test "
                         f"(n={r4.get('octc8_all_n')})")
            lines.append("| condition | acc % | macro-F1 % |")
            lines.append("|---|---|---|")
            for k, v in ab2.items():
                lines.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['macro_f1'])} |")
        u = r4.get("kermany_uncertain") or {}
        if u:
            lines.append("")
            lines.append(f"### Kermany uncertain slice (n={u.get('n')}, mean conf "
                         f"{u.get('mean_conf'):.3f}, mean margin {u.get('mean_margin'):.3f})")
            lines.append("| condition | acc % | macro-F1 % |")
            lines.append("|---|---|---|")
            for k, v in u.items():
                if isinstance(v, dict):
                    lines.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['macro_f1'])} |")
        ag = r4.get("agent_ablation_octc8")
        if isinstance(ag, dict) and "status" not in ag:
            lines.append("")
            lines.append("### LLM agent ablation (OCT-C8 subset)")
            lines.append("| condition | acc % | macro-F1 % |")
            lines.append("|---|---|---|")
            for k, v in ag.items():
                lines.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['macro_f1'])} |")
        elif isinstance(ag, dict):
            lines.append("")
            lines.append(f"- agent ablation: {ag['status']}")

    out = "\n".join(lines)
    (RESULTS / "RESULTS_SUMMARY.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()