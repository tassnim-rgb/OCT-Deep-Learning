# OCT Agentic Diagnostic Pipeline - Diagrams

Mermaid diagrams render on GitHub. They describe the implemented system only.

## 1. Inference pipeline

```mermaid
flowchart LR
    A["OCT B-scan input<br/>(PNG/JPEG)"] --> B["Preprocess<br/>grayscale x3, resize 224, (x-0.5)/0.5"]
    B --> C["Tool 1 OCTNet classify<br/>from-scratch CNN 1.24M"]
    C --> D["Tool 2 Grad-CAM++<br/>localize ROI"]
    D --> E["bbox extraction<br/>heatmap threshold 0.4"]
    E --> F{"Gate:<br/>conf >= 0.75?"}
    F -- "yes: skip zoom" --> G["Tool 4 retrieval<br/>SmallEncoder, corpus 133"]
    F -- "no: re-analyze crop" --> H["Tool 3 zoom<br/>crop ROI + re-classify"]
    H --> G
    G --> I["Tool 5 synthesize<br/>LLM or offline summarizer"]
    I --> J["Structured result JSON<br/>{finding, confidence, heatmap, ROI,<br/>evidence, justification, flags}"]
```

## 2. Deployment topologies

```mermaid
flowchart TB
    subgraph local["Local (python app_webui.py, port 7861)"]
        UI1["Web UI (webui/static)"] --> API["stdlib HTTP API"]
        API --> PRED["octnet/predict.py"]
        PRED --> PYT["PyTorch models<br/>oct_best.pth + retrieval_encoder.pth"]
    end

    subgraph hosted["GitHub Pages (no backend)"]
        UI2["Web UI (same static bundle)"] --> DETECT{"probe api/status"}
        DETECT -- "404: offline" --> ONNX["onnxruntime-web 1.20.0"]
        ONNX --> M["ONNX models<br/>classify.onnx, cam.onnx, encoder.onnx"]
    end

    local -- "identical result JSON schema" --> hosted
```

## 3. Zoom-and-reanalyze decision flow

```mermaid
flowchart TD
    P1["First pass on full image"] --> CAM["Grad-CAM++ heatmap"]
    CAM --> BB["tight bbox (t=0.4, pad 10%)"]
    BB --> CONF{"confidence >= 0.75<br/>and crop area >= 15%?"}
    CONF -- "yes" --> SKIP["zoom_skipped = true<br/>keep first-pass prediction"]
    CONF -- "no" --> ZOOM["re-classify cropped ROI"]
    ZOOM --> MERGE["blend weights, agreement check"]
    SKIP --> OUT["final prediction + evidence"]
    MERGE --> OUT
```

## 4. Experiment / evaluation flow

```mermaid
flowchart LR
    S["experiments/01..04 scripts<br/>SEED=42"] --> R["results/*.json"]
    R --> SUM["RESULTS_SUMMARY.md"]
    S --> EXP["OCT-C8, OCTID stress tests"]
    EXP --> METRICS["accuracy, corrected macro-F1,<br/>P@k, skip rate"]
```

All diagrams reflect code paths in `octnet/` (classify, zoom, retrieval,
evaluate) and `webui/` (in-browser ONNX engine).