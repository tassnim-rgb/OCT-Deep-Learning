"""Clean inference API for the web UI. Reuses the existing pipeline untouched.

``analyze_image()`` runs the deterministic pipeline (Tools 1-4: OCTNet
classifier, GradCAM++ localization, guarded zoom-and-reanalyze, contrastive
retrieval) and returns a structured result with the *real* model outputs plus
visualization images (PIL). Nothing here invents medical conclusions: every
field is derived from what the models actually produce.

The visualization helpers are intentionally re-implemented here (not imported
from app_gradio.py) so the ML-facing module stays free of UI code.
"""
from __future__ import annotations

import datetime as _dt
import io
import uuid
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw

from . import config, zoom as zoom_mod
from .data import build_infer_tf
from .evaluate import load_pipeline_runtime
from .retrieval import load_corpus

ImageSource = Union[str, Path, bytes, Image.Image]

_RUNTIME = None


def get_runtime():
    """Lazily load the pipeline runtime once (model + GradCAM + zoom + retriever)."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = load_pipeline_runtime(
            retriever_ckpt=config.CKPT_EMB, corpus_path=config.KB_PATH,
            zoom_variant="guarded")
    return _RUNTIME


def _to_pil(source: ImageSource) -> Image.Image:
    if isinstance(source, (str, Path)):
        with Image.open(source) as im:
            im.load()
        return im.convert("RGB")
    if isinstance(source, bytes):
        with Image.open(io.BytesIO(source)) as im:
            im.load()
        return im.convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    raise TypeError(f"unsupported image source: {type(source)!r}")


# ── Visualization (matching the look of the research app) ────────────────────

def _jet_colormap() -> np.ndarray:
    from matplotlib import cm
    return (cm.jet(np.linspace(0, 1, 256)) * 255).astype(np.uint8)[:, :3]


_JET = None


def heatmap_image(cam: np.ndarray, size: int = 224) -> Image.Image:
    """Raw GradCAM++ heatmap on a dark background."""
    global _JET
    if _JET is None:
        _JET = _jet_colormap()
    cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    idx = (cam_n * 255).astype(np.uint8)
    rgb = _JET[idx]
    img = Image.fromarray(rgb, "RGB").resize((size, size), Image.BILINEAR)
    return img


def overlay_image(pil: Image.Image, cam: np.ndarray, bbox=None,
                  alpha: float = 0.55, size: int = 224) -> Image.Image:
    """Scan + heatmap blend, plus the ROI rectangle (yellow) when given."""
    base = pil.resize((size, size), Image.BILINEAR)
    heat = heatmap_image(cam, size)
    out = Image.blend(base.convert("RGB"), heat, alpha)
    if bbox is not None:
        x0, y0, x1, y1 = [int(v * size / 224) for v in bbox]
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
    return out


def crop_image(pil: Image.Image, bbox, size: int = 224) -> Image.Image | None:
    """Crop the ROI (bbox in 224-space) from the original scan."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w, h = pil.size
    left, top = int(x0 * w / 224), int(y0 * h / 224)
    right, bottom = int(x1 * w / 224), int(y1 * h / 224)
    if right - left < 2 or bottom - top < 2:
        return None
    return pil.crop((left, top, right, bottom)).resize((size, size))


def downscale(pil: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = pil.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return pil


# ── The analysis itself ──────────────────────────────────────────────────────

def _region_of(cam) -> tuple[str, str, float]:
    """Peak-activation anatomical-ish region, spread description, spread frac."""
    h, w = cam.shape
    peak_y, peak_x = divmod(int(cam.argmax()), w)
    frac_x, frac_y = peak_x / w, peak_y / h
    horiz = "nasal" if frac_x < 0.4 else ("temporal" if frac_x > 0.6 else "central")
    vert = "superior" if frac_y < 0.4 else ("inferior" if frac_y > 0.6 else "mid")
    region = f"{vert}-{horiz}" if (horiz != "central" or vert != "mid") else "central foveal"
    spread = float((cam > 0.4).sum()) / cam.size
    spread_desc = "focal" if spread < 0.15 else ("moderate" if spread < 0.35 else "diffuse")
    return region, spread_desc, round(spread, 3)


def analyze_image(source: ImageSource, img_size: int = 224, top_k: int = 3) -> dict:
    """Run the full deterministic pipeline on one OCT scan.

    Returns a JSON-friendly dict (plus PIL images under ``images``). Every
    field is produced by the existing models; nothing is fabricated.
    """
    model, cam_fn, _, retriever = get_runtime()
    pil = _to_pil(source)
    infer_tf = build_infer_tf(img_size)

    # Tool 1+2+3 in one call: classify, then GradCAM++, then guarded zoom-recheck
    zr = zoom_mod.zoom_and_reanalyze(
        pil, model, cam_fn, config.CLASS_NAMES,
        **zoom_mod.VARIANTS["guarded"])
    cam = zr["cam"]
    bbox = zr["bbox"]                      # (x0, y0, x1, y1) in 224-space

    prediction = zr["final_pred"]
    confidence = float(zr["final_conf"])
    probs = {c: round(float(p), 4) for c, p in zr["final_probs"].items()}
    confidence = round(confidence, 4)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    margin = round(ranked[0][1] - ranked[1][1], 4)

    region, spread_desc, spread = _region_of(cam)

    # Tool 4: retrieve reference notes for the predicted class
    hits = retriever.retrieve(prediction, top_k=top_k)
    hits = [{"rank": h["rank"], "label": h["label"], "score": round(h["score"], 4),
             "text": h["text"]} for h in hits]

    # Honest flags (same logic as the offline agent pipeline)
    flags = []
    if confidence < 0.85:
        flags.append("low confidence")
    if margin < 0.15:
        flags.append("close runner-up")
    if not zr["zoom_skipped"] and not zr["agreement"]:
        flags.append("full image and zoomed region disagreed")

    # Concise explanation derived only from real outputs
    expl = (f"The model classified the full scan as {prediction} "
            f"with {confidence:.0%} confidence.")
    if zr["zoom_skipped"]:
        expl += " The prediction was clear enough that no zoomed re-check was needed."
    else:
        expl += (f" It zoomed into the highlighted region (which independently read as "
                 f"{zr['crop_pred']}) and combined both views into its final verdict.")
    expl += (f" The strongest signal was located {region} ({spread_desc} activation). "
             f"{len(hits)} reference {'note was' if len(hits) == 1 else 'notes were'} "
             f"retrieved for context.")
    if hits:
        expl += f" The closest match scored {hits[0]['score']:.3f}."

    evidence = [
        f"GradCAM++ activation is {spread_desc} ({spread:.0%} of the image), "
        f"peaking {region}.",
        ("Zoom re-analysis was skipped "
         f"({zr['skip_reason']})." if zr["zoom_skipped"] else
         f"Zoomed region read as {zr['crop_pred']} "
         f"({zr['crop_conf']:.0%}); combined verdict {zr['final_pred']}."),
        (f"Retrieved {len(hits)} reference snippet(s); top score "
         f"{hits[0]['score']:.3f}." if hits else "No reference notes retrieved."),
    ]

    steps = [
        {"name": "Classify full scan", "note": f"{prediction} ({confidence:.0%})"},
        {"name": "Localize with GradCAM++",
         "note": f"peak signal at {region}, {spread_desc} spread"},
        {"name": "Zoom-and-reanalyze",
         "note": (f"skipped ({zr['skip_reason']})" if zr["zoom_skipped"]
                  else f"crop read as {zr['crop_pred']}, combined verdict {prediction}")},
        {"name": "Retrieve reference notes", "note": f"{len(hits)} snippet(s)"},
    ]

    images = {
        "original": downscale(pil, 512),
        "overlay": overlay_image(pil, cam, bbox=bbox),
        "heatmap": heatmap_image(cam, 224),
        "crop": crop_image(pil, bbox) if not zr["zoom_skipped"] else None,
    }

    return {
        "schema": "octnet.analyze.v1",
        "id": str(uuid.uuid4()),
        "analyzed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": "OCTNet (trained from scratch, 1.24M params)",
        "device": config.DEVICE.type,
        "input_size": img_size,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probs,
        "margin": margin,
        "localization": {
            "region": region,
            "spread": spread,
            "spread_description": spread_desc,
            "bbox": [round(v / 224, 4) for v in bbox],   # normalized 0..1
        },
        "zoom": {
            "performed": not zr["zoom_skipped"],
            "skip_reason": zr["skip_reason"],
            "crop_pred": zr["crop_pred"],
            "crop_conf": round(zr["crop_conf"], 4) if zr["crop_conf"] is not None else None,
            "final_pred": zr["final_pred"],
            "final_conf": round(zr["final_conf"], 4),
            "agreement": bool(zr["agreement"]),
        },
        "retrieval": hits,
        "uncertainty_flags": flags,
        "explanation": expl,
        "evidence": evidence,
        "steps": steps,
        "images": images,
    }


def corpus_summary() -> list[dict]:
    """Knowledge-base overview for the landing page (real counts)."""
    from collections import Counter
    counts = Counter(l for l, _ in load_corpus(config.KB_PATH))
    return [{"label": k, "count": v} for k, v in sorted(counts.items())]