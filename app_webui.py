"""Web server + thin API for the OCT Workbench UI.

Serves the static frontend (webui/static/*) and a small JSON API that wraps
the existing pipeline via octnet.predict.analyze_image — no changes to the ML
code, no new dependencies (stdlib http.server + email parsing).

Endpoints
    GET  /                → webui/static/index.html
    GET  /static/<path>   → static assets
    POST /api/analyze     → multipart upload (field "file") → analysis result
    GET  /api/history     → recent analyses (metadata only)
    GET  /api/result?id=  → full stored result for one id
    DELETE /api/history   → clear history

Run:  oct_env/bin/python app_webui.py   → http://127.0.0.1:7861
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "webui" / "static"
HISTORY_FILE = PROJECT_ROOT / "webui" / "history.json"

MAX_BODY = 30 * 1024 * 1024        # 30 MB upload cap
MAX_HISTORY = 20                   # entries kept on disk (full result each)
HISTORY_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


# ── History persistence (flat JSON file — no database, matching project style)

def _load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(entries: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(entries, indent=1))


def _append_history(entry: dict) -> None:
    with HISTORY_LOCK:
        entries = _load_history()
        entries.insert(0, entry)
        del entries[MAX_HISTORY:]
        _save_history(entries)


# ── Small helpers

def _pil_to_data_url(pil) -> str | None:
    if pil is None:
        return None
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _thumb(pil, side: int = 96) -> str:
    w, h = pil.size
    scale = min(1.0, side / max(w, h))
    if scale < 1.0:
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return _pil_to_data_url(pil)


def _run_analysis(filename: str, data: bytes) -> dict:
    from octnet import predict

    with INFERENCE_LOCK:                      # models are not re-entrant
        result = predict.analyze_image(data)
    images = result.pop("images")
    result["filename"] = filename
    result["thumbnail"] = _thumb(images["original"], 96)
    result["images"] = {
        "original": _pil_to_data_url(images["original"]),
        "overlay": _pil_to_data_url(images["overlay"]),
        "heatmap": _pil_to_data_url(images["heatmap"]),
        "crop": _pil_to_data_url(images["crop"]),
    }
    return result


def _parse_multipart(body: bytes, content_type: str) -> dict | None:
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        return None
    boundary = m.group(1).encode()
    parts = {}
    for chunk in body.split(b"--" + boundary):
        if chunk in (b"", b"\r\n", b"--\r\n"):
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if head.endswith(b"--"):
            continue
        cd = ""
        for line in head.decode("latin1").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                cd = line.split(":", 1)[1].strip()
                break
        name_m = re.search(r'name="([^"]*)"', cd)
        fn_m = re.search(r'filename="([^"]*)"', cd)
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        parts[name_m.group(1) if name_m else "file"] = {
            "filename": fn_m.group(1) if fn_m else None,
            "data": payload,
        }
    return parts or None


def _read_body(handler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0 or length > MAX_BODY:
        raise ValueError("request too large")
    return handler.rfile.read(length)


# ── HTTP handler

class Handler(BaseHTTPRequestHandler):
    server_version = "OCTWorkbench/1.0"

    # -- helpers ---------------------------------------------------------
    def _send_json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str | None = None):
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        data = path.read_bytes()
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _log(self, msg: str):
        print(f"[webui] {msg}")

    def _get_param(self, name: str) -> str | None:
        qs = parse_qs(urlparse(self.path).query)
        vals = qs.get(name)
        return vals[0] if vals else None

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())):
                self._send_json({"error": "forbidden"}, 403)
                return
            self._send_file(target)
        elif path == "/api/history":
            with HISTORY_LOCK:
                entries = _load_history()
            meta = [{k: e[k] for k in ("id", "filename", "analyzed_at", "prediction",
                                       "confidence", "status", "thumbnail")} for e in entries]
            self._send_json({"entries": meta})
        elif path == "/api/result":
            aid = self._get_param("id")
            with HISTORY_LOCK:
                entries = _load_history()
            found = next((e for e in entries if e.get("id") == aid), None)
            if not found:
                self._send_json({"error": "result not found"}, 404)
                return
            self._send_json({"entry": found})
        elif path == "/api/status":
            self._send_json({"ok": True, "endpoints": ["analyze", "history", "result"]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/analyze":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            body = _read_body(self)
        except ValueError as e:
            self._send_json({"error": str(e)}, 413)
            return

        ctype = self.headers.get("Content-Type", "")
        filename = "scan.png"
        data = body
        if ctype.startswith("multipart/form-data"):
            parts = _parse_multipart(body, ctype)
            if not parts:
                self._send_json({"error": "could not parse upload"}, 400)
                return
            file_part = parts.get("file") or next(iter(parts.values()))
            data = file_part["data"]
            if file_part.get("filename"):
                filename = file_part["filename"]
        if not data:
            self._send_json({"error": "empty upload"}, 400)
            return

        # validate that it is a readable image before touching the models
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
        except Exception:
            self._send_json({"error": "uploaded file is not a readable image. "
                                      "Supported: PNG, JPEG, BMP, TIFF, WebP."}, 400)
            return

        self._log(f"analyze start: {filename} ({len(data) / 1024:.0f} KB)")
        try:
            result = _run_analysis(filename, data)
        except Exception as e:                     # inference failure (not user error)
            self._log(f"analyze failed: {e!r}")
            self._send_json({"error": f"model inference failed: {e}"}, 500)
            return

        entry = {
            "id": result["id"],
            "filename": filename,
            "analyzed_at": result["analyzed_at"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "status": "ok",
            "thumbnail": result.get("thumbnail"),
            "result": result,
        }
        _append_history(entry)
        self._log(f"analyze done: {filename} -> {result['prediction']} "
                  f"({result['confidence']:.3f})")
        self._send_json({"status": "ok", "result": result})

    def do_DELETE(self):
        if self.path.split("?", 1)[0] != "/api/history":
            self._send_json({"error": "not found"}, 404)
            return
        with HISTORY_LOCK:
            _save_history([])
        self._send_json({"status": "ok"})

    def log_message(self, fmt, *args):             # quiet the default access log
        pass


def main():
    import sys

    port = int(os.getenv("PORT", "7861"))
    sys.path.insert(0, str(PROJECT_ROOT))          # ensure octnet importable
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # eager-load the runtime so the first request is fast; surface load errors early
    from octnet import predict
    predict.get_runtime()
    print(f"[webui] pipeline loaded | device={predict.config.DEVICE}")

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[webui] OCT Workbench -> http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] stopped")


if __name__ == "__main__":
    main()