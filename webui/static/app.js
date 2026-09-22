/* OCT Workbench. SPA logic (no build step, no dependencies).
   Pages: landing (#/), analyze (#/analyze), result (#/result?id=…). */
"use strict";

/* ═══════════════ helpers ═══════════════ */

const el = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, {
  dateStyle: "medium", timeStyle: "short" }) : "");

const CLASS_META = {
  CNV:     { color: "#E63946", friendly: "Wet AMD, abnormal new blood vessels (CNV)" },
  DME:     { color: "#E76F51", friendly: "Diabetic eye swelling (DME)" },
  DRUSEN:  { color: "#B45309", friendly: "Drusen, early sign of age-related changes" },
  NORMAL:  { color: "#2A9D8F", friendly: "No disease detected in this scan" },
};
const ACCEPTED = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];
const ACCEPTED_EXT = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"];

// Set when this page is hosted without the Python backend (e.g. GitHub Pages).
// The frontend never invents results: if the API is unreachable it says so.
const BACKEND_OFFLINE = "The inference backend is not connected on this hosted demo. " +
  "Analysis is not faked, so nothing runs here. Locally, start it with python app_webui.py.";
const isOfflineError = (res) => !res || res.status === 404 || res.status === 502;

const badge = (cls) => {
  const m = CLASS_META[cls] || { color: "#64748B", friendly: cls };
  return `<span class="pred-badge" style="background:${m.color}" title="${esc(m.friendly)}">${esc(cls)}</span>`;
};
const dot = (cls) => {
  const m = CLASS_META[cls] || { color: "#64748B" };
  return `<span class="prob-dot" style="background:${m.color}"></span>`;
};

/* ═══════════════ state + theme ═══════════════ */

const state = {
  theme: localStorage.getItem("oct-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
  lastResult: null,       // result of the just-run analysis
};

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  localStorage.setItem("oct-theme", state.theme);
}

/* ═══════════════ API ═══════════════ */

async function apiAnalyze(file, signal) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  let res;
  try {
    res = await fetch("api/analyze", { method: "POST", body: fd, signal });
  } catch {
    throw new Error(BACKEND_OFFLINE);
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(isOfflineError(res) ? BACKEND_OFFLINE : (json.error || `Server error (${res.status})`));
  return json.result;
}

async function apiHistory() {
  let res;
  try {
    res = await fetch("api/history");
  } catch {
    return { offline: true, entries: [] };
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) return { offline: isOfflineError(res), entries: [] };
  return { offline: false, entries: json.entries || [] };
}

async function apiResult(id) {
  let res;
  try {
    res = await fetch(`api/result?id=${encodeURIComponent(id)}`);
  } catch {
    throw new Error(BACKEND_OFFLINE);
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(isOfflineError(res) ? BACKEND_OFFLINE : (json.error || "Result not found"));
  return json.entry;
}

/* ═══════════════ router ═══════════════ */

function router() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [name, query = ""] = raw.split("?");
  const params = new URLSearchParams(query);
  const app = el("#app");
  app.classList.add("page-fade");

  for (const link of document.querySelectorAll("[data-nav]")) {
    link.classList.toggle("active", link.dataset.nav === (name || "home"));
  }

  if (name === "analyze") {
    app.innerHTML = renderAnalyze();
    setupAnalyze();
  } else if (name === "result") renderResult(app, params.get("id"));
  else {
    app.innerHTML = renderHome();
    renderRecent();
  }
  window.scrollTo({ top: 0 });
}

/* ═══════════════ landing ═══════════════ */

function renderHome() {
  return `
  <section class="wrap">
    <div class="hero">
      <div>
        <p class="hero-eyebrow">Retinal OCT research tool</p>
        <h1 class="hero-title">Understand a retinal OCT scan in seconds.</h1>
        <p class="hero-lead">
          Upload an OCT scan of the back of the eye and the workbench classifies it,
          shows how confident the model is, highlights the region that drove the
          prediction, and pulls matching reference notes from its knowledge base.
          Every output is produced by the real pipeline. Nothing is faked.
        </p>
        <div class="hero-actions">
          <a class="btn btn-primary" href="#/analyze">Analyze an OCT scan</a>
          <button class="btn btn-secondary" type="button" id="how-btn">How it works</button>
        </div>
        <p class="hero-note">Research prototype · from-scratch deep learning · no API keys needed</p>
      </div>
      <div class="oct-visual" aria-hidden="true">
        <img src="static/assets/hero-scene.svg" alt="">
        <span class="scan-line"></span>
        <p class="oct-caption">Stylized OCT B-scan illustration</p>
      </div>
    </div>
  </section>

  <section class="wrap section" id="how">
    <div class="section-head">
      <h2>What you get</h2>
      <p>Three things, built from the same models that run in the research notebooks.</p>
    </div>
    <ol class="feature-list">
      <li>
        <span class="feature-num">01</span>
        <div>
          <h3>A prediction and its confidence</h3>
          <p>The classifier outputs per-class probabilities for CNV, DME, DRUSEN and NORMAL.
          You always see the full distribution, not just the top label.</p>
        </div>
      </li>
      <li>
        <span class="feature-num">02</span>
        <div>
          <h3>Where the model looked</h3>
          <p>Grad-CAM++ highlights the region of the scan that most influenced the prediction,
          with the region drawn directly on your image. Switch between the raw scan,
          highlighted scan and the raw heatmap.</p>
        </div>
      </li>
      <li>
        <span class="feature-num">03</span>
        <div>
          <h3>Reference notes for context</h3>
          <p>Reference snippets are retrieved from a curated knowledge base so you can compare
          the scan's findings with written descriptions of how these conditions appear on OCT.</p>
        </div>
      </li>
    </ol>
  </section>

  <section class="wrap section">
    <div class="section-head">
      <h2>Recent analyses</h2>
      <p>Stored locally on this machine · newest first</p>
    </div>
    <div id="recent"></div>
  </section>`;
}

async function renderRecent() {
  const box = el("#recent");
  try {
    const { offline, entries } = await apiHistory();
    if (offline) {
      box.innerHTML = `<div class="empty"><strong>Backend not connected</strong>
        This hosted page is a static frontend only. Run <code>python app_webui.py</code>
        locally and recent analyses will appear here.</div>`;
      return;
    }
    if (!entries.length) {
      box.innerHTML = `<div class="empty"><strong>No analyses yet</strong>
        Your first scan result will appear here.</div>`;
      return;
    }
    box.innerHTML = `<div class="hist-grid">${entries.map((e) => `
      <a class="hist-card" href="#/result?id=${encodeURIComponent(e.id)}">
        ${e.thumbnail ? `<img class="hist-thumb" src="${e.thumbnail}" alt="Thumbnail of ${esc(e.filename)}" loading="lazy">` : ""}
        <div class="hist-meta">
          <p class="hist-name" title="${esc(e.filename)}">${esc(e.filename)}</p>
          <p class="hist-date">${fmtDate(e.analyzed_at)}</p>
          <div class="hist-foot">
            ${badge(e.prediction)}
            <span class="hist-conf">${pct(e.confidence)}</span>
          </div>
        </div>
      </a>`).join("")}</div>`;
  } catch {
    box.innerHTML = `<div class="empty"><strong>History unavailable</strong>
      The history endpoint did not respond.</div>`;
  }
}

/* ═══════════════ analyze page ═══════════════ */

const SWEEP_STEPS = [
  "Preparing the model…",
  "Classifying the scan…",
  "Locating the region with Grad-CAM…",
  "Re-checking the highlighted area…",
  "Pulling reference notes…",
];

function renderAnalyze() {
  return `
  <section class="wrap page-hero">
    <h1>Analyze an OCT scan</h1>
    <p>Drop a retinal OCT scan here, or choose a file from your computer.
    Supported formats: PNG, JPEG, BMP, TIFF, WebP.</p>
  </section>

  <section class="wrap analyze-grid">
    <div>
      <label class="dropzone" id="dz" for="file-input">
        <span class="dz-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>
          </svg>
        </span>
        <p class="dz-title">Drag and drop your scan here</p>
        <p class="dz-sub">or click to browse your computer</p>
        <span class="btn btn-secondary btn-sm">Choose file…</span>
        <p class="dz-formats">Accepts <code>.png</code> <code>.jpg</code> <code>.jpeg</code>
        <code>.bmp</code> <code>.tif</code> <code>.tiff</code> <code>.webp</code> · up to 30&nbsp;MB</p>
      </label>
      <input id="file-input" type="file" accept="${ACCEPTED.join(",")}" hidden>

      <div id="file-pane"></div>
      <div id="analyze-status"></div>
    </div>

    <aside class="steps-aside">
      <h3>What happens after you upload</h3>
      <ol>
        <li>The OCTNet classifier reads the full scan and produces class probabilities.</li>
        <li>Grad-CAM++ finds the region that most influenced the prediction.</li>
        <li>A guarded zoom re-checks the highlighted area when the model is unsure.</li>
        <li>Reference notes are retrieved for the predicted condition.</li>
      </ol>
      <p class="hero-note" style="margin-top:14px">Runs entirely on this machine. Your image is
      not sent to any external service.</p>
    </aside>
  </section>`;
}

function setupAnalyze() {
  const dz = el("#dz");
  const input = el("#file-input");
  const status = el("#analyze-status");
  const pane = el("#file-pane");
  let currentFile = null;

  for (const evt of ["dragenter", "dragover"]) {
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add("drag"); });
  }
  for (const evt of ["dragleave", "drop"]) {
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove("drag"); });
  }
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
  input.addEventListener("change", () => { if (input.files[0]) pick(input.files[0]); });

  function pick(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!ACCEPTED_EXT.includes(ext)) {
      setError(`"<b>${esc(file.name)}</b>" does not look like a supported image.
        Please use a PNG, JPEG, BMP, TIFF or WebP file.`);
      return;
    }
    if (file.size > 30 * 1024 * 1024) { setError("File is larger than 30 MB."); return; }
    currentFile = file;
    status.innerHTML = "";
    const url = URL.createObjectURL(file);
    pane.innerHTML = `
      <div class="file-pill" style="margin-top:16px">
        <img src="${url}" alt="Preview of ${esc(file.name)}">
        <div style="min-width:0">
          <div class="f-name">${esc(file.name)}</div>
          <div class="f-size">${(file.size / 1024).toFixed(0)} KB · ready to analyze</div>
        </div>
        <button class="btn btn-ghost btn-sm f-clear" type="button">Remove</button>
      </div>
      <div class="analyze-btns">
        <button class="btn btn-primary" id="run-btn" type="button">Analyze this scan</button>
      </div>`;
    el("#run-btn").addEventListener("click", () => run(currentFile));
    el(".f-clear").addEventListener("click", reset);
  }

  function setError(msg) {
    status.innerHTML = `<div class="banner banner-error" role="alert">${msg}</div>`;
  }

  function reset() {
    const oldSrc = pane.querySelector("img")?.src;
    currentFile = null;
    input.value = "";
    pane.innerHTML = "";
    status.innerHTML = "";
    if (oldSrc && oldSrc.startsWith("blob:")) URL.revokeObjectURL(oldSrc);
  }

  async function run(file) {
    if (!file) return;
    el("#run-btn").disabled = true;
    const ctrl = new AbortController();
    let sweepIdx = 0;
    const ticker = setInterval(() => {
      sweepIdx = (sweepIdx + 1) % SWEEP_STEPS.length;
      const t = el("#sweep-text");
      if (t) t.textContent = SWEEP_STEPS[sweepIdx];
    }, 1600);

    status.innerHTML = `
      <div class="scan-frame" style="margin-top:16px">
        <div class="scan-visual">
          <div class="scan-beams"></div>
          <div class="scan-ring"></div>
        </div>
        <div class="scan-status">
          <span class="dot"></span><span id="sweep-text">${SWEEP_STEPS[0]}</span>
        </div>
      </div>`;

    try {
      const timeout = setTimeout(() => ctrl.abort(), 180000);
      const result = await apiAnalyze(file, ctrl.signal);
      clearTimeout(timeout);
      clearInterval(ticker);
      state.lastResult = result;
      location.hash = `#/result?id=${encodeURIComponent(result.id)}`;
    } catch (err) {
      clearInterval(ticker);
      setError((err.name === "AbortError" ? "Analysis took too long and was cancelled. " : "") +
        esc(err.message || "Something went wrong."));
      el("#run-btn").disabled = false;
    }
  }
}

/* ═══════════════ result page ═══════════════ */

function renderResult(app, id) {
  const data = state.lastResult && state.lastResult.id === id
    ? { result: state.lastResult }
    : null;
  if (data) { app.innerHTML = buildResultPage(data.result); setupResult(data.result); return; }
  app.innerHTML = `<section class="wrap page-hero"><div class="scan-frame">
    <div class="scan-status"><span class="dot"></span><span id="sweep-text">Loading result…</span></div>
  </div></section>`;
  apiResult(id)
    .then((entry) => {
      state.lastResult = entry.result || null;
      app.innerHTML = buildResultPage(entry.result || entry);
      setupResult(entry.result || entry);
    })
    .catch((err) => {
      app.innerHTML = `<section class="wrap page-hero"><div class="banner banner-error" role="alert">
        ${esc(err.message)}. <a href="#/analyze">Analyze a new scan</a>.</div></section>`;
    });
}

function buildResultPage(r) {
  const probs = Object.entries(r.probabilities || {})
    .sort((a, b) => b[1] - a[1]);
  const loc = r.localization || {};
  const zoom = r.zoom || {};
  const bbox = loc.bbox || [0, 0, 1, 1];
  const [bx0, by0, bx1, by1] = bbox;
  const roiStyle = `left:${bx0 * 100}%; top:${by0 * 100}%; width:${(bx1 - bx0) * 100}%; height:${(by1 - by0) * 100}%;`;

  return `
  <section class="wrap result-head">
    <div>
      <h1>Analysis result</h1>
      <p class="sub">${esc(r.filename)} · ${fmtDate(r.analyzed_at)} · ${esc(r.model)}</p>
    </div>
    <a class="btn btn-primary" href="#/analyze">Analyze another scan</a>
  </section>

  <section class="wrap result-grid">
    <div>
      <div class="stage">
        <span class="stage-chip raw" id="stage-chip">Raw scan</span>
        <img id="stage-img" class="stage-img" src="${r.images.original}" alt="OCT scan being analyzed">
        <div class="roi-box" id="roi-box" style="${roiStyle}" hidden></div>
      </div>
      <div class="heat-legend" id="heat-legend" hidden>
        <span>weak</span>
        <span class="heat-scale" aria-hidden="true"></span>
        <span>strong</span>
      </div>
      <div class="seg" role="group" aria-label="Visualization mode">
        <button type="button" data-view="original" class="active">Scan (raw)</button>
        <button type="button" data-view="overlay">Highlighted</button>
        <button type="button" data-view="heatmap">Heatmap</button>
      </div>

      ${r.images.crop ? `
      <div class="crop-row">
        <img class="crop-thumb" src="${r.images.crop}" alt="Close-up of the highlighted region">
        <div class="crop-note">Close-up of the highlighted region, re-classified by the model
        ${zoom.crop_pred ? `as <b>${esc(zoom.crop_pred)}</b> (${pct(zoom.crop_conf)})` : ""}.</div>
      </div>` : ""}
    </div>

    <div>
      <div class="panel">
        <h3 class="panel-title">Model prediction</h3>
        ${badge(r.prediction)}
        <p class="pred-name">${esc((CLASS_META[r.prediction] || {}).friendly || r.prediction)}</p>
        <p class="pred-overview">This is what the model believes the scan shows. It is model
        output, not a diagnosis.</p>
        <div class="conf-label"><span>Confidence</span><b>${pct(r.confidence)}</b></div>
        <div class="bar"><div class="bar-fill" style="width:${pct(r.confidence)}"></div></div>
        ${probs.map(([c, p]) => `
        <div class="prob-row">
          <span class="prob-name">${dot(c)} ${esc(c)}</span>
          <div class="bar"><div class="bar-fill" style="width:${pct(p)}"></div></div>
          <span class="pct">${pct(p)}</span>
        </div>`).join("")}
      </div>

      <div class="panel">
        <h3 class="panel-title">Where the model looked</h3>
        <div class="kv"><span>Dominant region</span><span>${esc(loc.region || "n/a")}</span></div>
        <div class="kv"><span>Activation</span><span>${esc(loc.spread_description || "n/a")} (${pct(loc.spread)})</span></div>
        <div class="kv"><span>ROI on image</span><span>${pct(bx0)} to ${pct(bx1)} × ${pct(by0)} to ${pct(by1)}</span></div>
        <p class="ev-list" style="padding-left:18px;margin:10px 0 0;font-size:13.5px;color:var(--text-3)">
          Grad-CAM++ heatmap overlaid on your scan in the <b>Highlighted</b> view; yellow box marks
          the region of interest.</p>
      </div>

      <div class="panel">
        <h3 class="panel-title">What the model found</h3>
        <p class="explanation">${esc(r.explanation)}</p>
        <ul class="ev-list" style="margin-top:12px">
          ${(r.evidence || []).map((e) => `<li>${esc(e)}</li>`).join("")}
        </ul>
        <div class="flag-row" style="margin-top:14px">
          ${(r.uncertainty_flags || []).length
            ? r.uncertainty_flags.map((f) => `<span class="flag">⚠ ${esc(f)}</span>`).join("")
            : `<span class="flag flag-ok">✓ nothing unusual to double-check</span>`}
        </div>
      </div>

      <div class="panel">
        <h3 class="panel-title">Reference notes for context</h3>
        ${(r.retrieval || []).map((h) => `
        <div class="ref-note">
          <div class="ref-head">
            <span class="ref-label">${esc(h.label)}</span>
            <span class="ref-score">match ${h.score.toFixed(3)}</span>
          </div>
          <p class="ref-text">${esc(h.text)}</p>
        </div>`).join("") || `<p class="explanation">No reference notes were retrieved.</p>`}
      </div>

      <div class="panel">
        <h3 class="panel-title">Steps the model ran</h3>
        <ol class="timeline">
          ${(r.steps || []).map((s, i) => `
          <li><span class="t-ix">${i + 1}</span>
            <div><b>${esc(s.name)}</b><span>${esc(s.note)}</span></div>
          </li>`).join("")}
        </ol>
      </div>

      <div class="result-note">
        ⚠ <strong>Research output, not a diagnosis.</strong> These results are produced by a
        model trained for research purposes and have not been clinically validated.
        Please discuss your eye health with a qualified eye care professional.
      </div>
    </div>
  </section>`;
}

function setupResult(r) {
  const img = el("#stage-img");
  const chip = el("#stage-chip");
  const roi = el("#roi-box");
  const legend = el("#heat-legend");
  const M = { original: r.images.original, overlay: r.images.overlay, heatmap: r.images.heatmap };
  const a = (view) => { img.src = M[view]; };

  el(".seg").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-view]");
    if (!btn) return;
    el(".seg button").forEach((b) => b.classList.toggle("active", b === btn));
    const view = btn.dataset.view;
    a(view);
    chip.textContent = view === "original" ? "Raw scan"
      : view === "overlay" ? "Model output · Grad-CAM++ heatmap + ROI" : "Model output · Grad-CAM++ heatmap";
    chip.classList.toggle("raw", view === "original");
    chip.classList.toggle("model", view !== "original");
    roi.hidden = view !== "overlay";
    legend.hidden = view === "original";
  });
}

/* ═══════════════ boot ═══════════════ */

async function boot() {
  applyTheme();
  el("#theme-toggle").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
  });
  document.addEventListener("click", (e) => {
    const how = e.target.closest("#how-btn");
    if (how) {
      e.preventDefault();
      const target = document.getElementById("how");
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
  window.addEventListener("hashchange", router);
  router();
}
boot();