"""Tool 5 — LLM agent orchestrator.

Tools 1–4 are Python callables in a registry; an LLM (Cerebras-compatible
OpenAI-style API) decides which to call and in what order, then synthesizes a
structured diagnostic impression.

API key resolution: CEREBRAS_API_KEY environment variable only — never hardcode
keys. Without a key, run_agent() falls back to the deterministic offline
pipeline (classify + localize + guarded zoom + retrieval) so the system stays
runnable for demos and ablations of the vision components.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable

import numpy as np
import torch
from PIL import Image

from . import config, zoom as zoom_mod
from .data import build_infer_tf
from .retrieval import Retriever


def _load_env_file(path: str) -> None:
    """Minimal stdlib .env loader (KEY=VALUE, '#' comments, optional quotes).

    Never overrides variables already set in the real environment, so a
    key exported in the shell takes precedence over a local .env file.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                line = line[7:] if line.startswith("export ") else line
                key, _, value = line.partition("=")
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


# Keys may come from a gitignored .env at the project root (or cwd), or from
# the shell environment. Real shell env always wins.
_load_env_file(os.path.join(os.path.dirname(__file__), "..", ".env"))
_load_env_file(".env")


# ═══════════════════════════════ Tools ═══════════════════════════════════

def make_tools(model, cam_fn, retriever: Retriever | None,
               class_names: list[str] | None = None,
               img_size: int = 224,
               zoom_kwargs: dict | None = None) -> dict[str, Callable]:
    """Build the four tool callables bound to loaded models."""
    class_names = class_names or config.CLASS_NAMES
    infer_tf = build_infer_tf(img_size)
    zoom_kwargs = zoom_kwargs or zoom_mod.VARIANTS["guarded"]

    def coarse_classify(image_path: str) -> dict:
        """Tool 1: run OCTNet on the full image."""
        pil = Image.open(image_path)
        img_t = infer_tf(pil)
        _, cls_idx, probs = cam_fn(img_t)
        return {
            "predicted_class": class_names[cls_idx],
            "confidence": round(float(probs[cls_idx]), 4),
            "all_probs": {c: round(float(p), 4) for c, p in zip(class_names, probs)},
        }

    def localize(image_path: str) -> dict:
        """Tool 2: GradCAM++ region description."""
        pil = Image.open(image_path)
        img_t = infer_tf(pil)
        cam, cls_idx, _ = cam_fn(img_t)
        h, w = cam.shape
        peak_y, peak_x = divmod(int(cam.argmax()), w)
        frac_x, frac_y = peak_x / w, peak_y / h
        horiz = "nasal" if frac_x < 0.4 else ("temporal" if frac_x > 0.6 else "central")
        vert = "superior" if frac_y < 0.4 else ("inferior" if frac_y > 0.6 else "mid")
        region = f"{vert}-{horiz}" if (horiz != "central" or vert != "mid") else "central foveal"
        spread = float((cam > 0.4).sum()) / cam.size
        return {
            "predicted_class": class_names[cls_idx],
            "peak_activation_at": region,
            "activation_spread": round(spread, 3),
            "spread_description": "focal" if spread < 0.15 else ("moderate" if spread < 0.35 else "diffuse"),
        }

    def zoom_reanalyze(image_path: str) -> dict:
        """Tool 3: crop the activated region and re-classify (guarded)."""
        pil = Image.open(image_path)
        result = zoom_mod.zoom_and_reanalyze(pil, model, cam_fn, class_names, **zoom_kwargs)
        return {
            "full_image_pred": result["original_pred"],
            "full_image_conf": round(result["original_conf"], 4),
            "crop_pred": result["crop_pred"],
            "crop_conf": round(result["crop_conf"], 4) if result["crop_conf"] is not None else None,
            "final_pred": result["final_pred"],
            "final_conf": round(result["final_conf"], 4),
            "zoom_skipped": result["zoom_skipped"],
            "skip_reason": result["skip_reason"],
            "agreement": result["agreement"],
            "bbox": result["bbox"],
        }

    def retrieve_knowledge(query: str, top_k: int = 3) -> dict:
        """Tool 4: retrieve reference radiology text relevant to a query."""
        if retriever is None:
            return {"error": "retriever not loaded"}
        results = retriever.retrieve(query, top_k=top_k)
        return {"query": query,
                "results": [{"rank": r["rank"], "label": r["label"],
                             "text": r["text"], "score": round(r["score"], 4)}
                            for r in results]}

    return {
        "coarse_classify": coarse_classify,
        "localize": localize,
        "zoom_reanalyze": zoom_reanalyze,
        "retrieve_knowledge": retrieve_knowledge,
    }


SYSTEM_PROMPT = """
You are a diagnostic AI assistant for OCT retinal imaging.
You have access to four tools:

1. coarse_classify(image_path) -> {predicted_class, confidence, all_probs}
   Run a CNN classifier on the full image. Classes: CNV, DME, DRUSEN, NORMAL.

2. localize(image_path) -> {predicted_class, peak_activation_at, spread_description}
   Run GradCAM++ to identify which region drove the prediction.

3. zoom_reanalyze(image_path) -> {full_image_pred, crop_pred, final_pred, agreement, zoom_skipped}
   Crop the activated region and re-run the classifier on it.
   Use when confidence is below 85% or finding is ambiguous.

4. retrieve_knowledge(query, top_k=3) -> {results: [{label, text, score}]}
   Retrieve reference radiology text. E.g. query: 'CNV finding', 'subretinal fluid'.

INSTRUCTIONS:
- Always start with coarse_classify and localize.
- Call zoom_reanalyze if confidence < 85% or finding is ambiguous.
- Call retrieve_knowledge for the predicted class and runner-up if probs are within 15%.
- You may call tools multiple times in any order.
- Produce a final structured JSON with these exact keys:
  {
    "finding": "<CNV|DME|DRUSEN|NORMAL>",
    "confidence": <0.0-1.0>,
    "localization": "<where in the image>",
    "supporting_evidence": ["<clinical features that match>"],
    "justification": "<2-3 sentences referencing each tool>",
    "uncertainty_flags": ["<low confidence, disagreement, etc>"]
  }

Call tools with:
TOOL: <tool_name>
ARGS: <JSON args>

When done:
FINAL: <JSON output>
"""


# ═══════════════════════ LLM provider configuration ══════════════════════

# OpenAI-compatible endpoints. Any provider whose key is present in the
# environment (or passed by the caller) can drive the agent loop.
PROVIDERS = {
    "cerebras":   {"base_url": "https://api.cerebras.ai/v1",
                   "key_env": "CEREBRAS_API_KEY",
                   "default_model": "gpt-oss-120b"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",
                   "key_env": "GROQ_API_KEY",
                   "default_model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY",
                   "default_model": "meta-llama/llama-3.3-70b-instruct:free"},
    "ollama":     {"base_url": "http://127.0.0.1:11434/v1",
                   "key_env": None,                # local — no key required
                   "default_model": "llama3.2"},
}


def _ollama_reachable(timeout: float = 1.0) -> bool:
    """True when a local Ollama server answers on the default port."""
    try:
        import requests
        return requests.get("http://127.0.0.1:11434/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def resolve_provider(provider: str | None = None) -> str | None:
    """Pick an LLM provider, or None when nothing is usable.

    provider: explicit name ('auto', 'cerebras', 'groq', 'openrouter',
    'ollama', 'custom') or the LLM_PROVIDER env var. 'auto' → the first
    keyed provider with a key in the environment; else local Ollama if it is
    running; else None (agent falls back to the offline pipeline).
    """
    p = (provider or os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    if p in PROVIDERS or p == "custom":
        return p
    for name, spec in PROVIDERS.items():
        if spec.get("key_env") and os.getenv(spec["key_env"]):
            return name
    if _ollama_reachable():
        return "ollama"
    return None


def resolve_llm_config(provider: str | None = None, api_key: str | None = None,
                       model: str | None = None, base_url: str | None = None,
                       reasoning_effort: str | None = None) -> dict | None:
    """Resolve a full LLM config from explicit args → environment → defaults.

    Returns None when no endpoint is usable (no key, no local Ollama),
    in which case run_agent() uses the deterministic offline pipeline.
    """
    prov = resolve_provider(provider)
    if prov is None:
        return None
    fast = os.getenv("FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}

    if prov == "custom":
        b_url = base_url or os.getenv("LLM_BASE_URL")
        if not b_url:
            return None
        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("CEREBRAS_API_KEY") or "not-needed"
        model_name = model or os.getenv("LLM_MODEL") or os.getenv("CEREBRAS_MODEL") or "local-model"
        effort = reasoning_effort or os.getenv("REASONING_EFFORT", "low" if fast else "medium")
        return {"provider": "custom", "api_key": key, "base_url": b_url,
                "model": model_name, "reasoning_effort": effort}

    spec = PROVIDERS[prov]
    key = api_key or os.getenv(spec["key_env"] or "") or os.getenv("LLM_API_KEY")
    if spec.get("key_env") and not key:
        return None  # keyed provider selected but no key available
    model_name = (model or os.getenv("LLM_MODEL") or os.getenv("CEREBRAS_MODEL")
                  or spec["default_model"])
    effort = reasoning_effort or os.getenv("REASONING_EFFORT", "low" if fast else "medium")
    return {"provider": prov, "api_key": key, "base_url": spec["base_url"],
            "model": model_name, "reasoning_effort": effort}


# ═══════════════════════════ Parsing / fallback ══════════════════════════

def parse_tool_call(text: str):
    tool_m = re.search(r"TOOL:\s*(\w+)", text)
    args_m = re.search(r"ARGS:\s*(\{.*?\})", text, re.DOTALL)
    if not tool_m:
        return None
    args = json.loads(args_m.group(1)) if args_m else {}
    return tool_m.group(1).strip(), args


def parse_final(text: str):
    m = re.search(r"FINAL:\s*(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def build_offline_result(image_path: str, tools: dict) -> dict:
    """Deterministic no-LLM pipeline: classify → localize → (guarded) zoom →
    retrieval for the predicted class. Used when no API key is configured."""
    coarse = tools["coarse_classify"](image_path)
    local = tools["localize"](image_path)
    zres = tools["zoom_reanalyze"](image_path)

    probs = coarse.get("all_probs", {})
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True) if probs else []
    top_conf = ranked[0][1] if ranked else 0.0
    runner_conf = ranked[1][1] if len(ranked) > 1 else 0.0

    flags = []
    if coarse["confidence"] < 0.85:
        flags.append("low confidence")
    if probs and (top_conf - runner_conf) < 0.15:
        flags.append("close runner-up")
    if zres.get("crop_pred") and not zres.get("zoom_skipped") and not zres.get("agreement"):
        flags.append("zoom/crop disagreement")

    finding = zres.get("final_pred") or coarse["predicted_class"]
    retrieval = tools["retrieve_knowledge"](finding, top_k=3)

    support = [f"{coarse['predicted_class']} with {local['spread_description']} activation"]
    if retrieval.get("results"):
        support.append(f"retrieved {len(retrieval['results'])} reference snippets "
                       f"(top score {retrieval['results'][0]['score']:.3f})")

    return {
        "finding": finding,
        "confidence": zres.get("final_conf", coarse["confidence"]),
        "localization": local["peak_activation_at"],
        "supporting_evidence": support,
        "justification": (
            f"The CNN classifier predicted {coarse['predicted_class']} with confidence "
            f"{coarse['confidence']:.2f}. GradCAM++ localized the dominant activation to "
            f"{local['peak_activation_at']} with {local['spread_description']} spread. "
            f"Zoom-reanalyze {'was skipped (' + (zres.get('skip_reason') or '') + ')' if zres.get('zoom_skipped') else 'agreed' if zres.get('agreement') else 'disagreed and blended'}."
        ),
        "uncertainty_flags": flags,
        "_tool_calls": [
            {"tool": "coarse_classify", "args": {"image_path": image_path}, "result": coarse},
            {"tool": "localize", "args": {"image_path": image_path}, "result": local},
            {"tool": "zoom_reanalyze", "args": {"image_path": image_path}, "result": zres},
            {"tool": "retrieve_knowledge", "args": {"query": finding, "top_k": 3}, "result": retrieval},
        ],
        "_offline": True,
    }


# ═══════════════════════════ Agent loop ══════════════════════════════════

def get_client(cfg: dict | None = None):
    """OpenAI-compatible client for the resolved config; None if unusable."""
    cfg = cfg or resolve_llm_config()
    if cfg is None:
        return None
    if cfg["provider"] == "cerebras":
        try:
            from cerebras.cloud.sdk import Cerebras
            return Cerebras(api_key=cfg["api_key"])
        except ImportError:
            pass
    from openai import OpenAI
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _chat_once(client, model_name, messages, max_tokens, temperature, reasoning_effort):
    """Single chat completion, tolerant of providers that reject extra kwargs."""
    kwargs = dict(model=model_name, messages=messages,
                  max_completion_tokens=max_tokens, temperature=temperature,
                  top_p=1, stream=False)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)  # e.g. plain OpenAI-compatible endpoints
        return client.chat.completions.create(**kwargs)


def run_agent(image_path: str, tools: dict, verbose: bool = True,
              max_turns: int | None = None, force_llm: bool = False,
              llm_config: dict | None = None,
              provider: str | None = None, api_key: str | None = None,
              model: str | None = None, base_url: str | None = None) -> dict:
    """Run the agentic loop. Uses an LLM only when a usable endpoint is
    configured (env key, runtime-supplied key, or local Ollama); otherwise
    returns the deterministic offline pipeline result."""
    if llm_config is None and any((provider, api_key, model, base_url)):
        llm_config = resolve_llm_config(provider=provider, api_key=api_key,
                                        model=model, base_url=base_url)
    client = get_client(llm_config)
    if client is None:
        if verbose:
            print("No LLM endpoint configured — using offline (no-LLM) pipeline.")
        return build_offline_result(image_path, tools)
    cfg = llm_config or resolve_llm_config()

    fast = os.getenv("FAST_MODE", "true").lower() in {"1", "true", "yes", "on"}
    model_name = cfg["model"]
    max_turns = max_turns or int(os.getenv("MAX_TURNS", "4" if fast else "8"))
    max_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "300" if fast else "700"))
    temperature = float(os.getenv("TEMPERATURE", "0.0" if fast else "0.2"))
    reasoning_effort = cfg.get("reasoning_effort")

    if fast and not force_llm:
        offline = build_offline_result(image_path, tools)
        coarse = offline["_tool_calls"][0]["result"]
        probs = coarse.get("all_probs", {})
        ranked = sorted(probs.values(), reverse=True) if probs else []
        margin = (ranked[0] - ranked[1]) if len(ranked) > 1 else 1.0
        if coarse["confidence"] >= 0.90 and margin >= 0.15:
            if verbose:
                print("Fast path used; skipping LLM for this high-confidence case.")
            offline["_fast_fallback"] = True
            return offline

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Analyze this OCT image and provide a structured diagnostic impression.\n"
            f"Image path: {image_path}"},
    ]
    tool_log = []

    for turn in range(max_turns):
        resp = None
        for attempt in range(3):
            try:
                resp = _chat_once(client, model_name, messages, max_tokens,
                                  temperature, reasoning_effort)
                break
            except Exception as e:
                if any(s in str(e) for s in ("429", "queue_exceeded", "RESOURCE_EXHAUSTED")):
                    wait = 8 * (attempt + 1)
                    if verbose:
                        print(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/3)...")
                    time.sleep(wait)
                else:
                    raise
        if resp is None:
            return {"error": "Rate limit retries exhausted", "_tool_calls": tool_log}

        assistant_text = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        final = parse_final(assistant_text)
        if final:
            final["_tool_calls"] = tool_log
            if verbose:
                print(f"=== FINAL OUTPUT (after {turn + 1} turns) ===")
                print(json.dumps(final, indent=2))
            return final

        parsed = parse_tool_call(assistant_text)
        if not parsed:
            break
        tool_name, args = parsed
        if tool_name not in tools:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = tools[tool_name](**args)
            except Exception as e:
                result = {"error": str(e)}
        tool_log.append({"tool": tool_name, "args": args, "result": result})
        if verbose:
            print(f"Turn {turn + 1} | {tool_name} -> {json.dumps(result)[:120]}...")
        messages.append({"role": "user", "content":
            f"TOOL_RESULT for {tool_name}:\n{json.dumps(result, indent=2)}"})

    return {"error": "Agent did not produce final output", "_tool_calls": tool_log}


# ═══════════════════════════ Ablation ════════════════════════════════════

ABLATION_CONDITIONS = {
    "Full pipeline":            [],
    "No zoom (-Tool3)":         ["zoom_reanalyze"],
    "No retrieval (-Tool4)":    ["retrieve_knowledge"],
    "No localization (-Tool2)": ["localize"],
    "Classifier only (-2,3,4)": ["localize", "zoom_reanalyze", "retrieve_knowledge"],
}


def ablation_run(image_path: str, tools: dict, disabled: list[str],
                 class_names: list[str] | None = None) -> str:
    """Run the pipeline with the named tools disabled; return the predicted class."""
    class_names = class_names or config.CLASS_NAMES
    originals = {}
    for t in disabled:
        if t in tools:
            originals[t] = tools[t]
            tools[t] = lambda **_: {"error": "tool disabled for ablation"}
    try:
        result = run_agent(image_path, tools, verbose=False, force_llm=True)
        pred = result.get("finding") or result.get("predicted_class")
        if pred not in class_names:
            pred = tools["coarse_classify"](image_path)["predicted_class"]
    except Exception:
        try:
            pred = tools["coarse_classify"](image_path)["predicted_class"]
        except Exception:
            pred = class_names[0]
    finally:
        for t, fn in originals.items():
            tools[t] = fn
    return pred