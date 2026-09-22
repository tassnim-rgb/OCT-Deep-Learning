"""Tool 3 — zoom-and-reanalyze. Both the baseline (unguarded) and guarded
versions from the research notebooks, plus an adaptive margin-based gate."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import config
from .data import build_infer_tf


def get_cam_bbox(cam: np.ndarray, threshold: float = 0.4) -> tuple[int, int, int, int]:
    """
    Threshold the GradCAM++ heatmap and return the tight bounding box
    (x0, y0, x1, y1) of the activated region (in CAM coordinates).
    Falls back to a centre crop if the threshold keeps nothing.
    """
    binary = (cam >= threshold).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        h, w = cam.shape
        return w // 4, h // 4, 3 * w // 4, 3 * h // 4
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


def zoom_and_reanalyze(
    pil_image: Image.Image,
    model,
    cam_fn,
    class_names: list[str] | None = None,
    img_size: int = 224,
    cam_threshold: float = 0.4,
    padding_frac: float = 0.10,
    confidence_gate: float | None = 0.75,
    margin_gate: float | None = None,
    min_crop_area_frac: float | None = 0.15,
    blend_alpha: float = 0.6,
) -> dict:
    """
    Full zoom-and-reanalyze pipeline.

    mode control via parameters:
      * baseline (no guards):      confidence_gate=None, min_crop_area_frac=None
      * guarded (from research):   confidence_gate=0.75, min_crop_area_frac=0.15,
                                   blend_alpha=0.6
      * adaptive margin gate:      margin_gate=0.15 (skip when top1−top2 margin
                                   ≥ gate and confidence ≥ 0.75)

    Returns a dict with keys original_pred/conf/probs, cam, bbox, crop_*,
    final_*, zoom_skipped, skip_reason, agreement, blend_weights.
    """
    class_names = class_names or config.CLASS_NAMES
    infer_tf = build_infer_tf(img_size)

    # Pass 1: full image
    img_t = infer_tf(pil_image)
    cam, orig_cls, orig_probs_arr = cam_fn(img_t)
    orig_conf = float(orig_probs_arr[orig_cls])
    orig_probs = {c: float(p) for c, p in zip(class_names, orig_probs_arr)}

    # Locate ROI (CAM coords → image pixels)
    cam_h, cam_w = cam.shape
    cx0, cy0, cx1, cy1 = get_cam_bbox(cam, cam_threshold)
    x0 = max(0, int(cx0 * img_size / cam_w) - int(img_size * padding_frac))
    y0 = max(0, int(cy0 * img_size / cam_h) - int(img_size * padding_frac))
    x1 = min(img_size, int(cx1 * img_size / cam_w) + int(img_size * padding_frac))
    y1 = min(img_size, int(cy1 * img_size / cam_h) + int(img_size * padding_frac))

    base = {
        "original_pred": class_names[orig_cls],
        "original_conf": orig_conf,
        "original_probs": orig_probs,
        "cam": cam,
        "bbox": (x0, y0, x1, y1),
        "crop_pred": None,
        "crop_conf": None,
        "crop_probs": None,
        "final_pred": class_names[orig_cls],
        "final_conf": orig_conf,
        "final_probs": orig_probs,
    }

    def skip(reason: str):
        return {**base, "zoom_skipped": True, "skip_reason": reason,
                "blend_weights": None, "agreement": True}

    # Guard 1: confidence / margin gate
    if confidence_gate is not None and orig_conf >= confidence_gate:
        return skip(f"confidence {orig_conf:.2f} >= gate {confidence_gate}")
    if margin_gate is not None:
        ranked = np.sort(orig_probs_arr)[::-1]
        margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else 1.0
        if (orig_conf >= (confidence_gate or 0.0)) and margin >= margin_gate:
            return skip(f"margin {margin:.2f} >= gate {margin_gate} (conf {orig_conf:.2f})")

    # Guard 2: minimum crop area
    crop_area = (x1 - x0) * (y1 - y0)
    image_area = img_size * img_size
    if min_crop_area_frac is not None and crop_area / image_area < min_crop_area_frac:
        return skip(f"crop area {crop_area / image_area * 100:.1f}% < min {min_crop_area_frac * 100:.0f}%")

    # Pass 2: crop + upsample
    crop_t = img_t[:, y0:y1, x0:x1].unsqueeze(0)
    crop_t = F.interpolate(crop_t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    crop_t = crop_t.squeeze(0)
    _, crop_cls, crop_probs_arr = cam_fn(crop_t)
    crop_conf = float(crop_probs_arr[crop_cls])

    # Guard 3: blend full + crop
    blended = blend_alpha * orig_probs_arr + (1 - blend_alpha) * crop_probs_arr
    final_cls = int(blended.argmax())
    final_conf = float(blended[final_cls])

    return {
        **base,
        "zoom_skipped": False,
        "skip_reason": None,
        "crop_pred": class_names[crop_cls],
        "crop_conf": crop_conf,
        "crop_probs": {c: float(p) for c, p in zip(class_names, crop_probs_arr)},
        "final_pred": class_names[final_cls],
        "final_conf": final_conf,
        "final_probs": {c: float(p) for c, p in zip(class_names, blended)},
        "blend_weights": f"{blend_alpha:.0%} full + {1 - blend_alpha:.0%} crop",
        "agreement": orig_cls == final_cls,
    }


def make_zoom_fn(model, cam_fn, class_names=None, **kwargs) -> Callable[[Image.Image], dict]:
    """Partial-apply a zoom config to a PIL image → result dict."""
    class_names = class_names or config.CLASS_NAMES

    def _zoom(pil_image: Image.Image) -> dict:
        return zoom_and_reanalyze(pil_image, model, cam_fn, class_names, **kwargs)

    return _zoom


VARIANTS = {
    # research baseline (tool3_baseline): always zoom, final = pure crop verdict
    "baseline": dict(confidence_gate=None, min_crop_area_frac=None, blend_alpha=0.0),
    # research guarded (tool3_guarded): skip when conf >= 0.75 or crop tiny;
    # 60% full + 40% crop blend otherwise
    "guarded":  dict(confidence_gate=0.75, min_crop_area_frac=0.15, blend_alpha=0.6),
    # adaptive middle ground: guarded, but only skip when the top-1/top-2
    # margin is also large (zoom still runs on decisive-looking-but-uncertain)
    "margin":   dict(confidence_gate=0.75, min_crop_area_frac=0.15, blend_alpha=0.6, margin_gate=0.35),
}