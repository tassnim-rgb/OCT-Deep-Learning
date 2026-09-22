"""Gradio web UI for the OCT agentic diagnostic pipeline.

Run:  python app_gradio.py           (then open http://127.0.0.1:7860)
Tabs:
  1. Automated diagnosis: upload or pick a sample OCT image; the agent
     (Tools 1-4, LLM-synthesized when an endpoint is configured) returns a
     structured finding + GradCAM overlay + zoom crop.
  2. Knowledge retrieval: free-text query -> top-k reference snippets.
  3. Zoom explorer: compare baseline / guarded / margin zoom policies on one image.

LLM agent backends (any OpenAI-compatible endpoint):
  Cerebras (CEREBRAS_API_KEY) · Groq (GROQ_API_KEY) · OpenRouter
  (OPENROUTER_API_KEY, incl. `:free` models) · local Ollama (no key) ·
  custom base URL. Keys may also be pasted in the UI at runtime. They live
  only in the session, never written to disk. Without any endpoint the agent
  uses the deterministic offline pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch
from PIL import Image as PILImage, ImageDraw

from octnet import config, zoom as zoom_mod
from octnet.agent import PROVIDERS, make_tools, resolve_llm_config, run_agent
from octnet.evaluate import GradCAMpp, load_pipeline_runtime
from octnet.models import load_octnet
from octnet.retrieval import load_corpus, load_retriever

# ═════════════════════════  Startup: load all weights once  ═════════════════════

model, cam_fn, zoom_fn, retriever = load_pipeline_runtime(
    retriever_ckpt=config.CKPT_EMB, corpus_path=config.KB_PATH)
tools = make_tools(model, cam_fn, retriever, zoom_kwargs=zoom_mod.VARIANTS["guarded"])

START_CFG = resolve_llm_config()
print(f"[app] models loaded | device={config.DEVICE} | "
      f"llm={START_CFG['provider'] if START_CFG else 'offline (no endpoint)'}")

CORPUS = load_corpus(config.KB_PATH)

# ═════════════════════════  Small HTML helpers  ═════════════════════════

CLASS_COLORS = {"CNV": "#E63946", "DME": "#F4A261", "DRUSEN": "#E9C46A", "NORMAL": "#2A9D8F"}

FRIENDLY_NAMES = {
    "CNV": "Wet AMD: abnormal new blood vessels (CNV)",
    "DME": "Diabetic eye swelling (DME)",
    "DRUSEN": "Drusen: early sign of age-related changes",
    "NORMAL": "No disease detected in this scan",
    "DIFFERENTIAL": "Differential notes (several possibilities)",
}


def _friendly(cls) -> str:
    return FRIENDLY_NAMES.get(str(cls).upper(), str(cls))


def _esc(v) -> str:
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cls_badge(cls) -> str:
    color = CLASS_COLORS.get(str(cls).upper(), "#64748B")
    return (f'<span class="oct-badge" style="background:{color}" '
            f'title="{_esc(_friendly(cls))}">{_esc(cls)}</span>')


def _chip(label: str, value: str, strong: bool = True) -> str:
    inner = f"<strong>{_esc(value)}</strong>" if strong else _esc(value)
    return f'<span class="oct-chip">{_esc(label)} · {inner}</span>'


def _hero_html() -> str:
    device = "GPU" if torch.cuda.is_available() else "CPU"
    agent = f"ready · {START_CFG['provider']}" if START_CFG else "turned off (no key needed to try)"
    chips = (
        _chip("AI model", "trained from scratch on OCT scans")
        + _chip("Running on", device)
        + _chip("AI assistant", agent)
        + _chip("Knowledge base", f"{len(CORPUS):,} eye-care notes")
        + _chip("Sample scans", f"{len(GALLERY):,} available")
    )
    return (
        '<div class="oct-hero">'
        '<div class="oct-hero-title">🩺 Your OCT scan, explained</div>'
        '<div class="oct-hero-sub">Upload a scan of the back of your eye and this research system '
        'will tell you, in plain words, what it sees: what it suspects, how confident it is, '
        "and exactly which part of the scan it focused on.</div>"
        f'<div class="oct-chips">{chips}</div></div>'
    )


def _corpus_chips_html() -> str:
    from collections import Counter
    counts = Counter(l for l, _ in CORPUS)
    chips = "".join(_chip(_friendly(k), str(v)) for k, v in sorted(counts.items()))
    return f'<div class="oct-chips oct-chips-dark">{chips}</div>'


def _finding_card(result: dict, cfg: dict | None) -> str:
    """Render the structured finding as a patient-friendly HTML card."""
    if result.get("error"):
        return (f'<div class="oct-alert">⚠️ <b>Something went wrong:</b> '
                f'{_esc(result["error"])}. Please try another scan.</div>')

    finding = result.get("finding", "n/a")
    conf = float(result.get("confidence") or 0.0)
    loc = _esc(result.get("localization", "n/a"))
    evidence = result.get("supporting_evidence") or []
    flags = result.get("uncertainty_flags") or []

    if result.get("_offline"):
        mode = "<span class='oct-chip-mode'>no AI assistant · offline mode</span>"
    elif result.get("_fast_fallback"):
        mode = "<span class='oct-chip-mode'>AI assistant skipped · system is highly confident</span>"
    else:
        mode = (f"<span class='oct-chip-mode'>AI assistant used · {_esc(cfg['provider'])}</span>"
                if cfg else "<span class='oct-chip-mode'>AI assistant used</span>")

    ev_items = "".join(f"<li>{_esc(e)}</li>" for e in evidence) or "<li>No other notes.</li>"
    if flags:
        flag_html = "".join(f'<span class="oct-flag">⚠ {_esc(f)}</span>' for f in flags)
    else:
        flag_html = '<span class="oct-flag oct-flag-ok">✓ nothing unusual to double-check</span>'
    pct = max(0.0, min(conf, 1.0))

    return (
        '<div class="oct-card oct-finding">'
        '<div class="oct-card-head">'
        '<span class="oct-eyebrow">WHAT THE SYSTEM FOUND</span>'
        f"{mode}</div>"
        '<div class="oct-finding-row">'
        f"{_cls_badge(finding)}"
        "<div>"
        f'<div class="oct-friendly">{_esc(_friendly(finding))}</div>'
        f'<div class="oct-loc">Highlighted area · {loc}</div></div></div>'
        '<div class="oct-conf">'
        f'<div class="oct-conf-label">How confident the system is · <b>{pct:.0%}</b></div>'
        '<div class="oct-conf-track">'
        f'<div class="oct-conf-fill" style="width:{pct * 100:.0f}%"></div></div></div>'
        '<div class="oct-ev">'
        '<div class="oct-ev-title">What the scan showed</div>'
        f"<ul>{ev_items}</ul></div>"
        '<div class="oct-flags">'
        f"{flag_html}</div>"
        '<div class="oct-note">⚠️ This is a <b>research tool</b>, not a medical diagnosis. '
        'Please discuss this result with your eye care professional.</div></div>'
    )


def _trace_html(result: dict) -> str:
    calls = result.get("_tool_calls") or []
    if not calls:
        return ('<div class="oct-card"><div class="oct-eyebrow">WHAT THE SYSTEM DID</div>'
                '<p class="oct-muted" style="margin:8px 0 0">Nothing to show here.</p></div>'
                if result.get("_offline") else
                '<div class="oct-card"><div class="oct-eyebrow">WHAT THE SYSTEM DID</div>'
                "<p class='oct-muted' style='margin:8px 0 0'>No steps were recorded.</p></div>")
    steps = []
    for i, c in enumerate(calls, 1):
        summary = json.dumps(c.get("result", {}))[:200]
        args = json.dumps(c.get("args", {}))[:80]
        steps.append(
            '<div class="oct-step">'
            f'<div class="oct-step-idx">{i}</div>'
            "<div><b>"
            f'{_esc(c["tool"])}</b>'
            f"<code>{_esc(args)} -> {_esc(summary)}</code>"
            "</div></div>"
        )
    return ('<div class="oct-card">'
            '<div class="oct-card-head"><span class="oct-eyebrow">WHAT THE SYSTEM DID, STEP BY STEP</span>'
            f'<span class="oct-chip oct-chip-count">{len(calls)} steps</span></div>'
            f'{"".join(steps)}</div>')


def _retrieval_html(hits: list[dict]) -> str:
    if not hits:
        return '<p class="oct-muted">No results.</p>'
    cards = []
    for h in hits:
        score = min(1.0, float(h.get("score") or 0.0))
        cards.append(
            '<div class="oct-card oct-rt">'
            '<div class="oct-rt-head">'
            f'<span class="oct-rank">#{h["rank"]}</span>'
            f'{_cls_badge(h["label"])}'
            f'<span class="oct-score">{score:.3f}</span></div>'
            '<div class="oct-rt-track">'
            f'<div class="oct-rt-fill" style="width:{score * 100:.0f}%"></div></div>'
            f'<p class="oct-rt-text">{_esc(h["text"])}</p></div>'
        )
    return "".join(cards)


def _zoom_html(info: dict) -> str:
    if not info.get("zoom_skipped"):
        background = "background:#EFF6FF;border:1px solid #BFDBFE;color:#1E3A8A"
        label = "zoomed in for a closer look"
        reason = info.get("skip_reason") or ""
    else:
        background = "background:#F0FDF4;border:1px solid #BBF7D0;color:#14532D"
        label = "zoom skipped · system was already confident"
        reason = info.get("skip_reason") or ""
    reason_html = (f'<p class="oct-muted" style="margin-bottom:0">{_esc(reason)}</p>'
                   if reason else
                   '<p class="oct-muted" style="margin-bottom:0">The system compared the full '
                   "scan with the zoomed area.</p>")
    return (
        '<div class="oct-card">'
        '<div class="oct-card-head">'
        '<span class="oct-eyebrow">ABOUT THE ZOOM</span>'
        f'<span class="oct-chip oct-chip-count" style="{background}">{_esc(label)}</span></div>'
        f'<p><b>Before zoom:</b> {_esc(info.get("original", "n/a"))}</p>'
        f'<p><b>Close-up result:</b> {_esc(info.get("crop", "n/a"))}</p>'
        f'<p><b>After combining:</b> {_esc(info.get("final", "n/a"))}</p>'
        f"{reason_html}</div>"
    )


# ═════════════════════════  Rendering helpers  ═════════════════════════

def _cam_overlay(pil_img, cam, bbox=None, alpha=0.55) -> PILImage.Image:
    """Original + jet heatmap + optional ROI rectangle."""
    cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    from matplotlib import cm
    heat = (cm.jet(cam_n) * 255).astype(np.uint8)[:, :, :3]
    heat_img = PILImage.fromarray(heat).resize(pil_img.size, PILImage.BILINEAR)
    out = PILImage.blend(pil_img.convert("RGB"), heat_img, alpha)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        w, h = out.size
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0 * w / 224, y0 * h / 224, x1 * w / 224, y1 * h / 224],
                       outline=(255, 255, 0), width=3)
    return out


def _zoom_crop_view(pil_img, bbox) -> PILImage.Image | None:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w, h = pil_img.size
    left, top = int(x0 * w / 224), int(y0 * h / 224)
    right, bottom = int(x1 * w / 224), int(y1 * h / 224)
    if right - left < 2 or bottom - top < 2:
        return None
    return pil_img.convert("RGB").crop((left, top, right, bottom)).resize((224, 224))


def _localize_view(pil, path):
    """Recompute the CAM for display (cheap, one forward+backward)."""
    from octnet.data import build_infer_tf
    img_t = build_infer_tf(224)(pil)
    cam, cls, probs = cam_fn(img_t)
    from octnet.zoom import get_cam_bbox
    bbox = get_cam_bbox(cam, 0.4)
    h, w = cam.shape
    pw, ph = pil.size
    x0, y0, x1, y1 = bbox
    pad = 0.1
    b = (max(0, int(x0 * 224 / w - 224 * pad)),
         max(0, int(y0 * 224 / h - 224 * pad)),
         min(224, int(x1 * 224 / w + 224 * pad)),
         min(224, int(y1 * 224 / h + 224 * pad)))
    return _cam_overlay(pil, cam, b)


_tmp_counter = [0]


def _tmp_file(image) -> str:
    _tmp_counter[0] += 1
    fp = Path("/tmp/opencode") / f"ui_upload_{_tmp_counter[0]}.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    return str(fp)


# ═════════════════════════  Tab 1: Automated diagnosis  ═════════════════════════

def diagnose(image, use_llm, provider, api_key, model, base_url):
    try:
        if isinstance(image, str):
            path = image
            pil = PILImage.open(path)
        else:
            gr.Info("Processing uploaded image…")
            path = _tmp_file(image)
            pil = PILImage.fromarray(image).convert("RGB")
            pil.save(path)
        cfg = resolve_llm_config(provider=provider or None,
                                 api_key=(api_key.strip() if api_key else None) or None,
                                 model=(model.strip() if model else None) or None,
                                 base_url=(base_url.strip() if base_url else None) or None)
        result = run_agent(path, tools, verbose=False, force_llm=use_llm, llm_config=cfg)
        if "error" in result and not result.get("_tool_calls"):
            raise RuntimeError(result["error"])

        overlay = _localize_view(pil, path)
        crop_view = None
        for c in (result.get("_tool_calls") or []):
            if c["tool"] == "zoom_reanalyze" and not c.get("result", {}).get("zoom_skipped"):
                bbox = c["result"].get("bbox")
                if bbox:
                    crop_view = _zoom_crop_view(pil, bbox)
                    break

        out = {k: v for k, v in result.items() if not k.startswith("_")}
        return (overlay, crop_view, _finding_card(result, cfg), _trace_html(result),
                json.dumps(out, indent=2))
    except Exception as e:
        return (None, None, _finding_card({"error": str(e)}, None),
                '<div class="oct-card oct-muted">trace unavailable</div>', str(e))


# ═════════════════════════  Tab 2: Knowledge retrieval  ═════════════════════════

def retrieve(query, top_k):
    hits = retriever.retrieve(query, top_k=top_k) if retriever else []
    rows = [
        {"rank": h["rank"], "label": h["label"], "score": round(h["score"], 4), "text": h["text"]}
        for h in hits
    ]
    return _retrieval_html(rows)


# ═════════════════════════  Tab 3: Zoom explorer  ═════════════════════════

def zoom_explore(image, variant):
    try:
        if isinstance(image, str):
            pil = PILImage.open(image)
            path = image
        else:
            path = _tmp_file(image)
            PILImage.fromarray(image).convert("RGB").save(path)
            pil = PILImage.open(path)
        kwargs = zoom_mod.VARIANTS[variant]
        r = zoom_mod.zoom_and_reanalyze(pil, model, cam_fn, config.CLASS_NAMES, **kwargs)
        overlay = _localize_view(pil, path)
        crop = _zoom_crop_view(pil, r["bbox"]) if not r["zoom_skipped"] else None
        info = {
            "original": f"{r['original_pred']} ({r['original_conf']:.2f})",
            "crop": r["crop_pred"],
            "final": f"{r['final_pred']} ({r['final_conf']:.2f})",
            "zoom_skipped": r["zoom_skipped"],
            "skip_reason": r["skip_reason"],
        }
        return overlay, crop, _zoom_html(info), info
    except Exception as e:
        return (None, None, _finding_card({"error": str(e)}, None), {"error": str(e)})


# ═════════════════════════  Gallery  ═════════════════════════

def _build_gallery():
    choices = []
    import os as _os
    import random
    rng = random.Random(42)
    for cls in config.CLASS_NAMES:
        d = config.TEST_DIR / cls
        if not d.is_dir():
            continue
        files = sorted(d.iterdir())
        for f in rng.sample(files, min(2, len(files))):
            choices.append((f"Kermany test · {cls} · {f.name[:24]}", str(f)))
    for name, root in (("OCT-C8", config.OCTC8_ROOT / "test"), ("OCTID", config.OCTID_ROOT)):
        if not root.is_dir():
            continue
        for cls in sorted(_os.listdir(root)):
            d = root / cls
            if not d.is_dir():
                continue
            files = sorted(d.iterdir())
            if not files:
                continue
            f = rng.choice(files)
            choices.append((f"{name} · {cls} · {f.name[:24]}", str(f)))
    return choices


GALLERY = _build_gallery()

PROVIDER_OPTIONS = ["auto"] + list(PROVIDERS) + ["custom"]
FREE_HINT = ("💡 **Free & private options:** the AI assistant can use a free key from Groq, "
             "Cerebras, or OpenRouter (`:free` models), or run entirely on your own computer "
             "with **Ollama** (no key at all). Keys typed here stay only in this session. "
             "They are never saved.")

# ═════════════════════════  Styles  ═════════════════════════

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }

.oct-hero { background: linear-gradient(135deg,#0B2545 0%,#13315C 48%,#134E4A 100%);
  border-radius: 18px; padding: 26px 32px; margin: 4px 0 18px;
  box-shadow: 0 12px 28px rgba(11,37,69,.28); }
.oct-hero-title { font-size: 26px; font-weight: 800; letter-spacing: -.5px; color: #F8FAFC; }
.oct-hero-sub { font-size: 14px; color: #CBD5E1; margin-top: 6px; line-height: 1.5; }
.oct-chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.oct-chip { display: inline-block; background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.28); color: #E2E8F0; font-size: 12px;
  font-weight: 600; padding: 4px 11px; border-radius: 999px; }
.oct-chip strong { color: #fff; }
.oct-chips-dark { margin: 2px 0 14px; }
.oct-chips-dark .oct-chip { background: #E2E8F0; border-color: #CBD5E1; color: #334155; }
.oct-chips-dark .oct-chip strong { color: #0F172A; }
.oct-chip-mode { display: inline-block; background: #ECFDF5; border: 1px solid #A7F3D0;
  color: #065F46; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px; }
.oct-chip-count { background: #F1F5F9; border-color: #E2E8F0; color: #334155; }
.oct-chip-count strong { color: #0F172A; }

.oct-panel { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06); }

.oct-card { background: #fff; border: 1px solid #E2E8F0; border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06); margin-bottom: 14px; }
.oct-card-head { display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px; flex-wrap: wrap; gap: 6px; }
.oct-eyebrow { font-size: 11px; font-weight: 800; letter-spacing: .12em; color: #64748B; }
.oct-badge { color: #fff; font-weight: 700; font-size: 15px; padding: 6px 14px;
  border-radius: 999px; letter-spacing: .03em; box-shadow: 0 2px 6px rgba(15,23,42,.18); }
.oct-loc { font-size: 14px; color: #475569; font-weight: 600; }
.oct-finding-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.oct-conf { margin-top: 14px; }
.oct-conf-label { font-size: 13px; color: #475569; }
.oct-conf-label b { color: #0F172A; font-size: 14px; }
.oct-conf-track { background: #E2E8F0; border-radius: 999px; height: 10px;
  overflow: hidden; margin-top: 6px; }
.oct-conf-fill { background: linear-gradient(90deg,#0E7490,#2DD4BF); height: 100%;
  border-radius: 999px; transition: width .5s ease; }
.oct-ev { margin-top: 14px; }
.oct-ev-title { font-size: 12px; font-weight: 800; letter-spacing: .06em;
  color: #64748B; text-transform: uppercase; }
.oct-ev ul { margin: 6px 0 0; padding-left: 18px; color: #334155; font-size: 14px; line-height: 1.55; }
.oct-flags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.oct-flag { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A;
  font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; }
.oct-flag-ok { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }
.oct-alert { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B;
  padding: 12px 16px; border-radius: 12px; font-size: 14px; }

.oct-step { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px dashed #E2E8F0; }
.oct-step:last-child { border-bottom: none; }
.oct-step-idx { flex: 0 0 26px; height: 26px; border-radius: 50%; background: #0E7490;
  color: #fff; font-weight: 700; font-size: 13px; display: flex; align-items: center;
  justify-content: center; }
.oct-step b { font-size: 13px; color: #0F172A; }
.oct-step code { display: block; font-size: 12px; color: #475569; background: #F1F5F9;
  border-radius: 8px; padding: 6px 10px; margin-top: 4px; white-space: pre-wrap;
  word-break: break-all; }

.oct-rt { margin-bottom: 12px; }
.oct-rt-head { display: flex; align-items: center; gap: 10px; }
.oct-rank { background: #0B2545; color: #fff; font-weight: 800; font-size: 13px;
  padding: 3px 11px; border-radius: 999px; }
.oct-score { margin-left: auto; font-weight: 700; color: #0E7490; font-size: 13px; }
.oct-rt-track { background: #E2E8F0; border-radius: 999px; height: 6px; overflow: hidden;
  margin: 10px 0; }
.oct-rt-fill { background: linear-gradient(90deg,#0E7490,#2DD4BF); height: 100%;
  border-radius: 999px; }
.oct-rt-text { color: #334155; font-size: 14px; margin: 0; line-height: 1.5; }

.oct-muted { color: #94A3B8; font-style: italic; }
.oct-friendly { font-size: 15px; font-weight: 700; color: #0F172A; line-height: 1.3; }
.oct-note { margin-top: 14px; padding: 10px 12px; background: #FFF7ED;
  border: 1px solid #FED7AA; color: #9A3412; border-radius: 10px;
  font-size: 13px; line-height: 1.45; }
.oct-disclaimer { background: #FFF7ED; border: 1px solid #FED7AA; color: #9A3412;
  text-align: center; padding: 10px 14px; border-radius: 12px;
  font-size: 13px; margin: 6px 0 2px; }
.oct-footer { color: #94A3B8; font-size: 12px; text-align: center; margin: 10px 0 4px; }
"""

# ═════════════════════════  Gradio app  ═════════════════════════

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="OCT Agentic Diagnostic Pipeline") as demo:
    gr.HTML(_hero_html())

    # ── Tab 1: Automated diagnosis ────────────────────────────────────────────
    with gr.Tab("🩺 Check my scan"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes=["oct-panel"]):
                gr.Markdown("**YOUR SCAN**")
                src = gr.Radio(["Upload my own scan", "Try a sample scan"],
                               value="Try a sample scan", label="How would you like to start?")
                upload = gr.Image(type="numpy", label="Upload your scan (OCT image)", visible=False)
                gallery = gr.Dropdown(GALLERY, label="Sample scans")
                use_llm = gr.Checkbox(value=False,
                                      label="Use the AI assistant for a more detailed answer (slower)")
                with gr.Accordion("Advanced: AI assistant settings (optional)", open=False):
                    prov = gr.Dropdown(PROVIDER_OPTIONS, value="auto", label="Provider")
                    key = gr.Textbox(type="password", label="Free AI assistant key (optional, session only)",
                                     placeholder="Paste a key, or rely on the environment …")
                    model = gr.Textbox(label="AI model (optional)",
                                       placeholder="provider default, e.g. llama-3.3-70b-versatile")
                    base_url = gr.Textbox(label="Custom server address", visible=False,
                                          placeholder="http://localhost:8000/v1")
                    gr.Markdown(FREE_HINT)
                run_btn = gr.Button("Check my scan", variant="primary")
            with gr.Column(scale=7):
                with gr.Row():
                    overlay_out = gr.Image(label="Your scan, with the important area highlighted",
                                           type="pil")
                    crop_out = gr.Image(label="Close-up of the highlighted area", type="pil")
                finding_html = gr.HTML()
                trace_html = gr.HTML()
                with gr.Accordion("Technical details (for researchers)", open=False):
                    result_code = gr.Code(label="Full structured result",
                                          language="json", interactive=False)

        def _pick(srcv, up, gal):
            if srcv == "Upload my own scan":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        src.change(_pick, [src, upload, gallery], [upload, gallery])

        DIAG_INPUTS = [gallery, use_llm, prov, key, model, base_url]
        DIAG_OUTPUTS = [overlay_out, crop_out, finding_html, trace_html, result_code]
        run_btn.click(diagnose, DIAG_INPUTS, DIAG_OUTPUTS, api_name="diagnose")
        upload.upload(diagnose, [upload, use_llm, prov, key, model, base_url], DIAG_OUTPUTS)

    # ── Tab 2: Knowledge retrieval ───────────────────────────────────────────
    with gr.Tab("📚 Eye conditions & search"):
        gr.Markdown("**Learn about what OCT scans show.** Ask in plain words. The system finds "
                    "matching reference notes written by researchers in our knowledge base.")
        gr.HTML(_corpus_chips_html())
        with gr.Row():
            q = gr.Textbox(placeholder="e.g. fluid under the retina, swelling, drusen, "
                                       "what does a normal scan look like…",
                           label="Your question", scale=5)
            k = gr.Slider(1, 5, value=3, step=1, label="Number of answers", scale=1)
            ret_btn = gr.Button("Search", variant="primary", scale=0, min_width=130)
        ret_html = gr.HTML()
        ret_btn.click(retrieve, [q, k], [ret_html], api_name="retrieve")
        q.submit(retrieve, [q, k], [ret_html])

    # ── Tab 3: Zoom explorer ─────────────────────────────────────────────────
    with gr.Tab("🔍 Zoom closer"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes=["oct-panel"]):
                gr.Markdown("**YOUR SCAN**")
                z_src = gr.Radio(["Upload my own scan", "Try a sample scan"],
                                 value="Try a sample scan", label="How would you like to start?")
                z_upload = gr.Image(type="numpy", label="Upload your scan (OCT image)", visible=False)
                z_gallery = gr.Dropdown(GALLERY, label="Sample scans")
                z_variant = gr.Dropdown(
                    [("Always zoom (research mode)", "baseline"),
                     ("Zoom only when it's safe (recommended)", "guarded"),
                     ("Zoom with extra caution", "margin")],
                    value="guarded", label="How should the system zoom?")
                gr.Markdown("**Zoom only when it's safe (recommended):** the system zooms into the "
                            "highlighted area only when it is confident enough. "
                            "**Extra caution:** adds one more safety check. "
                            "**Always zoom:** used for research. It always crops in.")
                z_btn = gr.Button("Zoom in and look closer", variant="primary")
            with gr.Column(scale=7):
                with gr.Row():
                    z_overlay = gr.Image(label="Your scan, with the important area highlighted",
                                         type="pil")
                    z_crop = gr.Image(label="Close-up of the highlighted area", type="pil")
                z_info = gr.HTML()
                with gr.Accordion("Technical details (for researchers)", open=False):
                    z_info_json = gr.JSON(label="Policy result")

        def _zpick(srcv, up, gal):
            if srcv == "Upload my own scan":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        z_src.change(_zpick, [z_src, z_upload, z_gallery], [z_upload, z_gallery])
        z_btn.click(zoom_explore, [z_gallery, z_variant],
                    [z_overlay, z_crop, z_info, z_info_json], api_name="zoom")
        z_upload.upload(zoom_explore, [z_upload, z_variant],
                        [z_overlay, z_crop, z_info, z_info_json])

    gr.HTML('<div class="oct-disclaimer">⚠️ <b>For research and education only.</b> This tool does '
            'not provide a medical diagnosis. Always discuss your eye health with a qualified '
            'eye care professional.</div>'
            '<div class="oct-footer">OCT research pipeline · web app built on <b>octnet/</b> · '
            "full technical details in README.md</div>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True,
                theme=THEME, css=CSS)