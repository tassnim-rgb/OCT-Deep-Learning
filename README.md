# OCT Agentic Diagnostic Pipeline

An agentic diagnostic system for OCT retinal imaging, implemented per the
proposal in `main-4.pdf` (written for chest X-ray) but trained **from scratch**
on OCT data. Four specialist tools (all vision/retrieval components trained
from scratch, no pretrained weights) are orchestrated by an LLM that produces
a structured diagnostic impression.

> **Research software, not a medical device.** All model outputs (predictions,
> heatmaps, reference notes) are research results produced by models trained on
> public research datasets. They have not been clinically validated and must
> never be used to diagnose or treat any condition. Consult a qualified eye
> care professional for medical advice.

```
                     ┌─────────────────────────────────────────────┐
   OCT image ──────► │  Tool 1  coarse classify   (OCTNet,   1.24M)│
                     │  Tool 2  localize          (GradCAM++)      │
                     │  Tool 3  zoom-and-reanalyze (gated crop)    │
                     │  Tool 4  retrieve knowledge (SmallEncoder)  │
                     │  Tool 5  LLM orchestrator  (Cerebras/offline)│
                     │           ↓                                 │
                     │  {finding, confidence, localization,        │
                     │   supporting_evidence, justification,       │
                     │   uncertainty_flags}                        │
                     └─────────────────────────────────────────────┘
```

## Repository layout

```
projects/
├── octnet/                          # consolidated package (the product)
│   ├── config.py                    # paths, constants, device/seed
│   ├── models.py                    # OCTNet, GradCAM++, SmallEncoder, WordTokenizer
│   ├── data.py                      # transforms, loaders, cross-dataset helpers
│   ├── zoom.py                      # Tool 3: baseline / guarded / margin policies
│   ├── retrieval.py                 # Tool 4: corpus, NT-Xent + triplet training, eval
│   ├── agent.py                     # Tool 5: tools, registry, prompt, LLM+offline loop
│   └── evaluate.py                  # metrics (macro-F1 corrected), zoom/ablation evals
├── experiments/                     # research scripts closing the proposal's gaps
│   ├── 01_transfer_baseline.py      # ResNet18 (ImageNet) vs from-scratch OCTNet
│   ├── 02_retrain_retrieval.py      # P@k on the 133-entry XL corpus + retrain
│   ├── 02b_retrieval_triplet.py     # hard-negative triplet variant (explored)
│   ├── 02c_retrieval_augment.py     # content-query augmentation variant
│   ├── 03_zoom_tradeoff.py          # zoom policies across Kermany/OCT-C8/OCTID
│   └── 04_crossdataset_ablation.py  # ablations on OCT-C8 + uncertain slice
├── app_gradio.py                    # web UI (demo.serve → gradio) — http://127.0.0.1:7860
├── app_webui.py                     # OCT Workbench web UI + API — http://127.0.0.1:7861
├── octnet/predict.py                # clean inference API used by the web UI
├── webui/static/                    # OCT Workbench frontend (no build step)
│   ├── index.html                   # page shell
│   ├── style.css                    # design system (light/dark themes)
│   └── app.js                       # SPA router + pages
├── results/                         # JSON metrics + figures for the write-up
├── requirements.txt
└── *.ipynb                          # research log (canonical executed notebooks)
```

## Quickstart

```bash
# app (gradio, AI-assistant style)
python app_gradio.py                 # → http://127.0.0.1:7860

# app (OCT Workbench, modern research UI)
python app_webui.py                  # → http://127.0.0.1:7861

# experiments (results land in results/*.json)
python experiments/02_retrain_retrieval.py
python experiments/03_zoom_tradeoff.py
python experiments/04_crossdataset_ablation.py

# transfer baseline (GPU, ~1h)
python experiments/01_transfer_baseline.py
```

### OCT Workbench (web UI)

A dependency-free frontend plus a small stdlib HTTP API. It runs the same
deterministic pipeline as the Gradio app (classify → Grad-CAM++ → guarded zoom →
retrieval) through a clean inference module, `octnet/predict.py`; no ML code was
changed and no API keys are required.

* Landing page with a hero explaining the tool and recent analyses.
* Analyze page: drag-and-drop upload with client-side file preview (PNG/JPEG/
  BMP/TIFF/WebP, up to 30 MB) and real loading state.
* Result page: the scan front and centre, with a **Scan (raw) / Highlighted /
  Heatmap** toggle that overlays the Grad-CAM++ heatmap and the region-of-interest
  box directly on the image, plus the model's per-class probabilities, confidence,
  honest explanation, uncertainty flags, zoom verdict and retrieved reference
  notes. Every label distinguishes raw scan from model output, and each result
  carries the "research tool, not a diagnosis" note.
* Light/dark theme toggle, responsive layout, and local analysis history stored
  in `webui/history.json` (no database). History survives restarts.

API (used by the frontend, usable from curl):

```bash
curl -F "file=@scan.png" http://127.0.0.1:7861/api/analyze   # → analysis JSON
curl    http://127.0.0.1:7861/api/history                    # → recent analyses
curl    "http://127.0.0.1:7861/api/result?id=<id>"           # → one stored result
curl -X DELETE http://127.0.0.1:7861/api/history             # → clear history
```

`PORT` overrides the default port 7861. The pipeline (model + Grad-CAM + zoom +
retriever) is loaded once at startup and inference is serialised with a lock.

## Deployment (GitHub Pages)

The frontend (`webui/static/`) is a pure static site: vanilla HTML/CSS/JS, hash
routing, relative asset paths, and a `.nojekyll` marker, so it deploys to any
GitHub Pages project URL with no build step. The workflow
`.github/workflows/pages.yml` deploys `webui/static/` on every push to `main`.

**Live demo: `https://tassnim-rgb.github.io/OCT-Deep-Learning/`**

**The hosted demo is frontend-only and intentional.** OCT inference needs the
PyTorch pipeline and the trained weights, which live on the local machine and
are deliberately gitignored (never committed). The deployed frontend therefore
shows the full UI and, where a result would appear, explains that the inference
backend is not connected. It never fabricates predictions.

To get real inference, run the backend yourself:

```bash
python app_webui.py        # http://127.0.0.1:7861  (OCT Workbench UI + API)
python app_gradio.py       # http://127.0.0.1:7860  (AI-assistant UI)
```

To host the backend on a free/low-cost Python service later (Render, Railway,
Fly.io, Hugging Face Spaces, or any small VPS), deploy `app_webui.py` as-is:
the API is stdlib-only (no FastAPI/Flask dependency) and calls
`octnet/predict.py`, so it ports cleanly. You must place the gitignored weights
(`oct_best.pth`, `retrieval_encoder.pth`) and dataset on that server — they are
not in the repository.

The agent drives an LLM only when an endpoint is configured; otherwise it runs a
deterministic offline pipeline (classify → localize → gated zoom → retrieval →
synthesis). Any OpenAI-compatible endpoint works — resolve order: explicit
provider in the app's *Agent LLM settings* → env key → local Ollama:

| Provider | Key env var | Default model | Free? |
|---|---|---|---|
| Cerebras | `CEREBRAS_API_KEY` | `gpt-oss-120b` | free tier |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free tier |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` | `:free` models |
| Ollama (local) | none | `llama3.2` | yes — no key |
| Custom | `LLM_BASE_URL` | — | — |

```bash
export GROQ_API_KEY=gsk_…          # or CEREBRAS_API_KEY / OPENROUTER_API_KEY
python app_gradio.py               # agent auto-detects the free key you set

FAST_MODE=false python app_gradio.py   # full LLM reasoning (slower, richer)
```

Keys can also be pasted into the app's *Agent LLM settings* panel at runtime —
they are kept in memory for the session only and never written to disk. Keys
are never hardcoded; the app supports the **offline pipeline with no key at
all**.

## Artifacts

| Artifact | Role | Status |
|---|---|---|
| `oct_best.pth` | OCTNet classifier (from scratch, 1.24M params) | ✓ 99.70% std / 99.90% TTA on OCT2017 test |
| `retrieval_encoder.pth` | retrieval encoder (from scratch, trained on 133-entry XL corpus) | ✓ vocab 784 matches `oct_corpus_xl.json` |
| `oct_corpus_xl.json` | 133 reference entries (CNV 35 · DME 31 · DRUSEN 27 · NORMAL 22 · DIFFERENTIAL 18) | ✓ used by agent Tool 4 |

## Key results (measured by `experiments/`, see `results/*.json`)

**Zoom-and-reanalyze trade-off** (Tool 3) — measured on the full balanced
test sets (Kermany 1000, OCT-C8 shared-subset 1000, OCTID 517):

| Dataset | policy | acc % | macro-F1 % | zoom called |
|---|---|---|---|---|
| Kermany | classifier-only | 99.70 | 99.70 | — |
| Kermany | baseline (pure crop) | **42.30** | 36.88 | 1000/1000 |
| Kermany | guarded | 99.70 | 99.70 | 1/1000 |
| Kermany | margin gate | 99.70 | 99.70 | 1/1000 |
| OCT-C8 | classifier-only | 98.43 | 98.43 | — |
| OCT-C8 | baseline | 40.43 | 35.64 | 1400/1400 |
| OCT-C8 | guarded | 98.36 | 98.36 | 4/1400 |
| OCT-C8 | margin gate | 98.36 | 98.36 | 4/1400 |
| OCTID | classifier-only | 74.27 | 60.07 | — |
| OCTID | baseline | 34.04 | 26.96 | 517/517 |
| OCTID | guarded | 74.08 | 59.81 | 12/517 |
| OCTID | margin gate | 74.08 | 59.81 | 12/517 |

Reading: the research notebooks' 42.30% baseline (Kermany) reproduces exactly,
and the guarded policy's "keeps accuracy but skips 99.8%" is confirmed. The
claim that baseline zoom *improves* OCT-C8 (54→84%) does **not** survive a
balanced evaluation — it was an artifact of the notebook's `samples[:100]`
AMD-only slice with index-aligned labels. Measured properly, crop-then-
reclassify is uniformly harmful (Kermany −57 pp, OCT-C8 −58 pp, OCTID −40 pp).
The gated policies nullify the harm, and on the genuinely hard OCT-C8
all-classes stress test (2800 images, interpretive map; classifier-only
76.93%) the gated zoom gives a small *real* gain (+0.18 pp acc, +0.23 pp
macro-F1). The margin gate adds nothing over guarded on these data (an honest
negative result) — Tool 3's value is the ROI **visualization** for the LLM's
justification plus a small accuracy edge on OOD low-confidence images.

**Cross-dataset ablations** (Tool 4/5 vision-side, `results/04_crossdataset_ablation.json`):
classifier-only vs guarded/margin/baseline zoom on OCT-C8 (literal shared
subset 1400; all-8-classes interpretive stress test 2800) and the Kermany
uncertain slice (bottom-100 by confidence, mean conf 0.937). Corrected
macro-F1 throughout (the research notebooks' F1s were broken by label-class
count mismatches — fixed in `octnet/evaluate.py`).

**Transfer baseline** (proposal §4): ResNet18–ImageNet **full fine-tune**
(11.18M params, 8 epochs, ~21 min) reaches **99.60%** test acc / 99.60%
macro-F1 (val best 96.89%). From-scratch OCTNet (1.24M params, 60 epochs)
reaches 99.70% std / 99.90% TTA. So on OCT, a well-tuned from-scratch model is
competitive with transfer at ~9× fewer parameters and requires no pretrained
weights; transfer's advantage is only training time (less than half the
epochs). Full numbers in `results/01_transfer_baseline.json`.

**Retrieval quality** (Tool 4):
- Saved checkpoint on the XL corpus: **P@1 81.25% / P@3 81.25%** (CNV 100, NORMAL 100, DRUSEN 75, DME 50).
- Two DME queries fail on legitimate synonymy ("intraretinal cysts diabetic" →
  DIFFERENTIAL docs; "hard exudates retina" → NORMAL docs).
- Content-augmented variant trades P@1 (62.5%) for **P@3 93.75%**.
- Label-filtered retrieval (agent's confident-class path): **P@1 100%**.

## Notes & caveats

- **Dataset provenance.** OCT2017/Kermany train set is known to contain
  near-duplicate images of the test set — accuracies should be read as
  upper bounds; the cross-dataset runs (OCT-C8, OCTID) are the stress tests.
- **OCTID label mapping is interpretive** (DR→DME, CSR→CNV, MH→DRUSEN): OCTID
  has no literal Kermany class names; treat those numbers as transfer stress,
  not pathology identity (`octnet/data.py::_load_mapped`).
- **Secrets.** The research notebooks previously contained plaintext API keys
  (`GOOGLE_API_KEY` in `tool4_retrieval_embedder.ipynb`, `CEREBRAS_API_KEY` in
  `tool5_agent_orchestrator_full.ipynb`). These have been **scrubbed** from the
  repository (values replaced with `REPLACE_WITH_YOUR_*` placeholders) and
  notebook output cells were cleared before publishing. **Because those keys
  were at some point in plaintext files, revoke and reissue them on the
  provider consoles.** The package code (`octnet/agent.py`) reads keys from the
  environment only; the apps accept a key in memory for the session.
- Run-to-run variance: seeds are pinned (`SEED=42`) per experiment; multi-GPU
  nondeterminism can still shift ±0.05 pp.