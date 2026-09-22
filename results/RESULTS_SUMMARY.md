## Experiment 01 — Transfer baseline (ResNet18 ImageNet)

- model: ResNet18 (ImageNet init), full fine-tune | params 11.18M (trainable 11.18M)
- epochs run: 8 | early-stopped: False | wall time: 20.8 min
- val best: 96.89% | test acc: 99.60% | test macro-F1: 99.60%
- test per-class F1: CNV 99.21%, DME 99.80%, DRUSEN 99.60%, NORMAL 99.80%

## Experiment 02 — Retrieval on the 133-entry XL corpus
- vocab: 784 | best loss: 1.9295
- saved_checkpoint_on_xl: P@1 81.25% | P@3 81.25%  (per-class P@1: CNV 100%, DME 50%, DRUSEN 75%, NORMAL 100%)
- fresh_xl_trained: P@1 81.25% | P@3 81.25%  (per-class P@1: CNV 100%, DME 50%, DRUSEN 75%, NORMAL 100%)

### 02b hard-negative triplet (explored, regressed)
- triplet: P@1 68.75% | P@3 68.75% (delta P@3 -12.5 pp)

### 02c content-query augmentation
- augmented: P@1 62.50% | P@3 93.75% | label-filtered P@1 100.00%

## Experiment 03 — Zoom-and-reanalyze trade-off

| dataset | policy | acc % | macro-F1 % | skip % |
|---|---|---|---|---|
| kermany | classifier-only | 99.70 | 99.70 | — |
| kermany | baseline | 42.30 | 36.88 | 0.00 |
| kermany | guarded | 99.70 | 99.70 | 99.90 |
| kermany | margin | 99.70 | 99.70 | 99.90 |
| octc8 | classifier-only | 98.43 | 98.43 | — |
| octc8 | baseline | 40.43 | 35.64 | 0.00 |
| octc8 | guarded | 98.36 | 98.36 | 99.71 |
| octc8 | margin | 98.36 | 98.36 | 99.71 |
| octid | classifier-only | 74.27 | 60.07 | — |
| octid | baseline | 34.04 | 26.96 | 0.00 |
| octid | guarded | 74.08 | 59.81 | 97.68 |
| octid | margin | 74.08 | 59.81 | 97.68 |

## Experiment 04 — Cross-dataset ablations (corrected macro-F1)

### OCT-C8 shared 4-class subset
| condition | acc % | macro-F1 % |
|---|---|---|
| classifier_only | 98.43 | 98.43 |
| zoom_guarded | 98.36 | 98.36 |
| zoom_margin | 98.36 | 98.36 |
| zoom_baseline | 40.43 | 35.64 |

### OCT-C8 all-classes interpretive stress test (n=2800)
| condition | acc % | macro-F1 % |
|---|---|---|
| classifier_only | 76.93 | 76.26 |
| zoom_guarded | 77.11 | 76.49 |
| zoom_margin | 77.11 | 76.49 |
| zoom_baseline | 47.96 | 34.86 |

### Kermany uncertain slice (n=100, mean conf 0.937, mean margin 0.903)
| condition | acc % | macro-F1 % |
|---|---|---|
| classifier_only | 97.00 | 73.99 |
| zoom_guarded | 97.00 | 73.99 |
| zoom_margin | 97.00 | 73.99 |
| zoom_baseline | 33.00 | 21.34 |

- agent ablation: SKIPPED — set CEREBRAS_API_KEY to run LLM ablation