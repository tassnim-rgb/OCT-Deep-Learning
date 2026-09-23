# OCT Agentic Diagnostic Pipeline - Technical Report

**Repository:** `OCT-Deep-Learning` · **Author:** BELAGHIT Tassnim Alla
**Status:** Research software, not a medical device. See Disclaimer.

---

## 1. Abstract

This project builds an agentic diagnostic system for retinal OCT B-scan
classification. Instead of a single black-box label, the system produces an
interpretable result: a coarse pathology classification, a Grad-CAM++ heatmap
localizing the pathological region, a confidence-gated zoom-and-reanalyze step,
retrieved reference notes from a curated corpus, and a structured diagnostic
impression synthesized from all tool outputs. Every vision and retrieval
component is trained from scratch on public research datasets (no pretrained
weights). The full pipeline is additionally exported to ONNX and executes
in-browser on GitHub Pages via onnxruntime-web, producing byte-compatible
results with the Python pipeline.

## 2. Problem Statement

OCT B-scans must be graded for pathologies (CNV, DME, DRUSEN, NORMAL in the
Kermany formulation). Manual grading is slow and operator-dependent. A plain
classifier provides a single label without evidence. The project addresses the
question: can a multi-tool agentic pipeline give a *clinically useful*,
interpretable, and verifiable diagnostic reasoning trace rather than a bare
class label?

## 3. Motivation

Interpretable AI for medical imaging is a requirement for trust and eventual
adoption. A "zoom and reanalyze" strategy mirrors how clinicians inspect
regions of interest. A retrieval component lets the model justify decisions
with reference notes. The whole pipeline is deployed without a backend to make
the research results accessible to anyone with a browser, exercising real
client-side ML engineering (ONNX + WebAssembly).

## 4. Scientific / Engineering Context

The work is anchored in: Grad-CAM++ gradient-based visual explanations;
contrastive (NT-Xent) and triplet embeddings for retrieval; gated decision
policies for zoom; and cross-dataset evaluation as a proxy for
out-of-distribution robustness. A transfer baseline (ResNet18 ImageNet
fine-tune) contextualizes the from-scratch model's performance.

## 5. Mathematical Formulation

### 5.1 Classification

OCTNet maps a resized 224x224 grayscale-resampled image through a from-scratch
CNN (1.24M parameters) to 4 logits
`z = f_theta(x)`, then `p = softmax(z)`. Training uses cross-entropy with
label smoothing absent (plain CE); confidence is `max_k p_k`, margin is
`p_top1 - p_top2`.

### 5.2 Grad-CAM++ localization

For the predicted class `c`, the heatmap is computed with a weighted sum of
feature maps using the analytic second-derivative weighted combination
implemented as a static backward pass in the ONNX export
(`webui/export_onnx.py`, `cam.onnx`). A tight bounding box is derived by
thresholding the heatmap at `t = 0.4` (`octnet/zoom.py::get_cam_bbox`);
fallback is a center crop when nothing survives.

### 5.3 Zoom-and-reanalyze gates

Two gate policies over the first-pass prediction:

- **Guarded:** skip zoom when `confidence >= 0.75` (and skip tiny crops with
  `min_crop_area_frac = 0.15`).
- **Margin gate:** additionally skip when `(p_top1 - p_top2) >= 0.15`.

Baseline (unguarded) always re-classifies the cropped ROI.

### 5.4 Retrieval

A `SmallEncoder` maps texts to a 64-dim L2-normalized embedding space.
Training objectives:

- **NT-Xent:** `sim(q,d) = (q . d) / T` with temperature `T = 0.07`, optimized
  as cross-entropy over the corpus (query-doc contrastive pairs).
- **Triplet variant (explored):** `max(0, margin - s_pos + s_neg)` with
  margin `0.2` and hard negatives sampled by token overlap.

Retrieval metric is precision at k (P@1, P@3), optionally label-filtered to the
classifier's confident prediction class.

## 6. Dataset / Data Generation

Public research datasets only (no private or clinical data shipped):

| Dataset | Use | Notes |
|---|---|---|
| OCT2017 / Kermany | Train Tools 1-3 | 4 classes; train/ + test/ ImageFolder; 1000 balanced test images. Known near-duplicate overlap between train and test sets. |
| OCT-C8 | Cross-domain stress test | 8 classes; shared 4-class subset 1400 images; all-classes interpretive subset 2800 images. |
| OCTID | Transfer stress test | CSR/DR/MH/NORMAL remapped interpretively onto the model's 4 classes (no literal Kermany labels exist). |
| 133-entry curated corpus (`oct_corpus_xl.json`) | Tool 4 | CNV 35, DME 31, DRUSEN 27, NORMAL 22, DIFFERENTIAL 18; vocabulary 784. |

## 7. Preprocessing

Resize to (244,244) then random-crop to 224 for training (resize to 224 for
inference), grayscale resampled to 3 channels, normalize `(x-0.5)/0.5`,
plus train augmentation: horizontal flip, affine (rotate 8 deg, translate,
scale, shear), color jitter, Gaussian noise (std 0.02) mimicking OCT speckle.
Test-time augmentation (8 geometric transforms) is available for the
classifier.

## 8. Methodology

Pipeline: classify (Tool 1) - localize (Tool 2) - gated zoom (Tool 3) -
retrieve (Tool 4) - synthesize (Tool 5). The two app layers (Gradio assistant
UI and the OCT Workbench SPA + stdlib API) run the same deterministic pipeline
via `octnet/predict.py`. The hosted demo runs the identical pipeline in
JavaScript with onnxruntime-web, auto-detecting whether a Python backend is
reachable.

## 9. Model Architecture

- **OCTNet classifier:** from-scratch CNN, 1.24M parameters, 224x224 input,
  4-class output. No pretrained weights.
- **SmallEncoder:** text encoder with 784-token word vocabulary, 64-dim output
  embedding (trained on the XL corpus).
- **WordTokenizer:** vocab 784, matched to the corpus JSON.
- **ResNet18 transfer baseline:** ImageNet-initialized 11.18M parameters,
  full fine-tune, for comparison only.

## 10. Training Procedure

Seeds pinned (`SEED=42`). From-scratch OCTNet trained ~60 epochs to 99.70%
std / 99.90% TTA on the OCT2017 test set (research notebooks). Retrieval
encoder trained with NT-Xent on the XL corpus (best CE loss 1.9295; P@1 81.25%).
Triplet and augmentation variants trained as experiments (02b, 02c) and
reported honestly. Transfer baseline: 8 epochs, ~21 min on GPU, val best
96.89%.

## 11. Experimental Setup

Runs are scripted (`experiments/01..04`) and write JSON to `results/`. Datasets
require gitignored local folders; config via `octnet/config.py` and
`.env.example`. Zoom policies evaluated on balanced Kermany (1000), OCT-C8
shared subset (1400 / 2800), OCTID (517).

## 12. Evaluation Metrics

Accuracy, macro-F1 (corrected for label-class count mismatches in
`octnet/evaluate.py`), retrieval P@1 / P@3, zoom skip rate, and agreement
between passes. All numbers below are from `results/*.json` /
`results/RESULTS_SUMMARY.md`.

## 13. Results

**Classifier (test, OCT2017):** from-scratch OCTNet 99.70% std / 99.90% TTA
(research notebooks); independent transfer baseline ResNet18 99.60% acc /
99.60% macro-F1 (per-class F1: CNV 99.21, DME 99.80, DRUSEN 99.60, NORMAL
99.80).

**Zoom-and-reanalyze (balanced test sets):**

| Dataset | policy | acc % | macro-F1 % | zoom called |
|---|---|---|---|---|
| Kermany | classifier-only | 99.70 | 99.70 | - |
| Kermany | baseline (pure crop) | 42.30 | 36.88 | 1000/1000 |
| Kermany | guarded / margin | 99.70 | 99.70 | 1/1000 each |
| OCT-C8 | classifier-only | 98.43 | 98.43 | - |
| OCT-C8 | baseline | 40.43 | 35.64 | 1400/1400 |
| OCT-C8 | guarded / margin | 98.36 | 98.36 | 4/1400 each |
| OCTID | classifier-only | 74.27 | 60.07 | - |
| OCTID | baseline | 34.04 | 26.96 | 517/517 |
| OCTID | guarded / margin | 74.08 | 59.81 | 12/517 each |

**Retrieval (XL corpus):** P@1 81.25% / P@3 81.25% (per-class P@1: CNV 100,
NORMAL 100, DRUSEN 75, DME 50). Content-augmented variant: P@1 62.50%,
P@3 93.75%. Label-filtered (agent's confident-class path): P@1 100%.

## 14. Comparison With Baselines

- From-scratch OCTNet (99.70%/99.90% TTA) is competitive with ImageNet transfer
  (99.60%) at ~9x fewer parameters, with no pretrained weights; transfer's only
  advantage is training time.
- Baseline (unguarded) zoom is uniformly harmful (Kermany -57 pp, OCT-C8 -58 pp,
  OCTID -40 pp); the guarded gates nullify the harm. On the hard OCT-C8
  all-classes stress test (2800 images, classifier-only 76.93%) gated zoom adds
  a small real gain (+0.18 pp acc, +0.23 pp macro-F1).
- The margin gate adds nothing over the guarded gate on these data (honest
  negative result).

## 15. Visual Results

Grad-CAM++ heatmap overlays, ROI boxes, and zoom crops are produced by the web
UIs and shown in the live demo; result figures and confusion matrix
artifacts are committed under `results/` and the repo root. The demo's result
page toggles raw scan / highlighted / heatmap views.

## 16. Error Analysis

- OCT2017 train/test near-duplicate overlap means accuracies are upper bounds;
  OCT-C8/OCTID runs are the real stress tests.
- OCTID label mapping is interpretive; numbers there are transfer stress, not
  pathology identity.
- Two DME retrieval queries fail on legitimate synonymy ("intraretinal cysts
  diabetic" retrieves DIFFERENTIAL docs; "hard exudates retina" retrieves
  NORMAL docs), highlighting vocabulary mismatch in a 784-token corpus.
- Baseline zoom's apparent benefit on OCT-C8 in the research notebooks was an
  artifact of an AMD-only `samples[:100]` slice with index-aligned labels; a
  balanced evaluation refutes it (corrected in this repository).

## 17. Limitations

- Single-modality (B-scan only), 4-class ontology (Kermany formulation) and
  interpretive cross-dataset mappings. Not a segmentation or 3D-volume
  analysis.
- No external clinical validation; results are research-phase.
- Retrieval corpus is small (133 entries, 784-token vocab), so coverage is
  limited.
- The browser engine cannot decode TIFF uploads (draws honest UI message).

## 18. Reproducibility

`pip install -r requirements.txt`; run `experiments/01..04` scripts (results
land in `results/*.json`); run `python app_webui.py` for the local UI/API;
`webui/export_onnx.py` regenerates and validates the ONNX assets against
PyTorch. Seeds pinned. Weights and datasets are gitignored (see `.env.example`).

## 19. Future Work

Independent de-duplicated validation set; clinician study of ROI/retrieval
utility; margin-gate evaluation on larger out-of-distribution data; client-side
zoom TTA and DICOM loading in the browser engine; longer retrieval corpus with a
larger vocabulary.

## 20. Conclusion

The project demonstrates an interpretable, multi-tool OCT diagnosis pipeline
with from-scratch models, honest cross-dataset evaluation, gated zoom that
removes a harmful baseline behavior, and the first known in-browser
byte-compatible deployment of the full pipeline on GitHub Pages. It is
research-grade evidence of full-stack medical ML engineering.

## 21. References

1. D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases
   by Image-Based Deep Learning," Cell, 172(5):1122-1131, 2018 (OCT2017).
2. A. Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual
   Explanations for Deep Convolutional Networks," WACV, 2018.
3. T. Chen et al., "A Simple Framework for Contrastive Learning of Visual
   Representations" (NT-Xent), ICML, 2020.
4. Public benchmarks OCT-C8 and OCTID (layouts in `octnet/data.py`; OCTID
   labels used interpretively only).

---

## Disclaimer

Research software, not a medical device. All model outputs are produced by
models trained on public research datasets, have not been clinically
validated, and must never be used to diagnose or treat any condition.