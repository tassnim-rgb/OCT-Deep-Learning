"""OCT agentic diagnostic pipeline — consolidated package.

Modules
    config      shared paths, constants, runtime helpers
    models      OCTNet (from scratch) + GradCAM++ + retrieval SmallEncoder
    data        transforms, ImageFolder loaders, cross-dataset helpers
    zoom        Tool 3 zoom-and-reanalyze (baseline / guarded / margin gate)
    retrieval   Tool 4 contrastive retrieval (corpus, training, eval)
    agent       Tool 5 LLM orchestrator + offline fallback + ablation
    evaluate    classifier metrics, cross-dataset and zoom-variant evaluation
    predict     clean inference API for the web UI (analyze_image)
"""
from . import (  # noqa: F401
    agent,
    config,
    data,
    evaluate,
    models,
    predict,
    retrieval,
    zoom,
)

__all__ = ["config", "models", "data", "zoom", "retrieval", "agent", "evaluate", "predict"]
__version__ = "1.0.0"