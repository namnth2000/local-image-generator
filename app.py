import os
import re
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openvino import Core
from optimum.intel import OVLatentConsistencyModelPipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path("models/LCM_Dreamshaper_v7-int8-ov")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAVICON_PATH = SCRIPT_DIR / "favicon.ico"

PIPE_CACHE: Dict[Tuple[str, str], OVLatentConsistencyModelPipeline] = {}
GEN_LOCK = threading.Lock()

PRESETS = {
    "Fast": {"size": 256, "steps": 4},
    "Balanced": {"size": 384, "steps": 4},
    "Quality": {"size": 512, "steps": 4},
    "Custom": {"size": None, "steps": None},
}

app = FastAPI(title="Local Image Generator")
app.mount("/outputs", StaticFiles(directory=str(DEFAULT_OUTPUT_DIR)), name="outputs")


class GenerateRequest(BaseModel):
    prompt: str
    preset: str = "Balanced"
    device: str = "GPU"
    size: int = 384
    steps: int = 4
    count: int = 1
    model_path: str = str(DEFAULT_MODEL_PATH)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


def slugify(text: str, max_length: int = 50) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text or "image")[:max_length].strip("-")


def get_available_devices():
    try:
        devices = list(Core().available_devices)
        return [d for d in devices if d in ("GPU", "CPU")] or devices or ["CPU"]
    except Exception:
        return ["CPU"]


def get_pipeline(device: str, model_path: str):
    key = (device, model_path)
    if key in PIPE_CACHE:
        return PIPE_CACHE[key], False

    pipe = OVLatentConsistencyModelPipeline.from_pretrained(
        Path(model_path),
        device=device,
        safety_checker=None,
        requires_safety_checker=False,
    )
    PIPE_CACHE[key] = pipe
    return pipe, True


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if not FAVICON_PATH.exists():
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
def index():
    devices = get_available_devices()
    default_device = "GPU" if "GPU" in devices else devices[0]
    presets_json = json.dumps(PRESETS)
    device_options = "".join(
        f'<option value="{d}" {"selected" if d == default_device else ""}>{d}</option>'
        for d in devices
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Generator</title>
<link rel="icon" href="/favicon.ico" type="image/x-icon">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #09100c;
  --panel: #0f1812;
  --panel-2: #131f17;
  --panel-3: #18271e;
  --border: #26392d;
  --border-strong: #35513f;
  --text: #f2f6f3;
  --muted: #93a198;
  --muted-2: #68766d;
  --accent: #177a48;
  --accent-hover: #1f9258;
  --accent-soft: #123c27;
  --danger: #d06c6c;
  --radius: 16px;
  --radius-sm: 10px;
}}

* {{
  box-sizing: border-box;
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

html, body {{
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
}}

button, input, textarea, select {{
  font: inherit;
}}

button {{
  cursor: pointer;
}}

.app {{
  width: 100%;
  height: 100vh;
  padding: 20px;
}}

.shell {{
  width: min(1440px, 100%);
  height: 100%;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 380px minmax(0, 1fr);
  gap: 18px;
}}

.left {{
  min-height: 0;
  display: grid;
  grid-template-rows: 40% minmax(0, 60%);
  gap: 16px;
}}

.panel {{
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}}

.prompt-panel {{
  padding: 18px;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 12px;
}}

.settings-panel {{
  padding: 18px;
  overflow: hidden;
}}

.results-panel {{
  padding: 18px;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) 82px auto;
  gap: 12px;
}}

.header-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}

.title {{
  margin: 0;
  font-size: 16px;
  line-height: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
}}

.muted {{
  color: var(--muted);
}}

textarea {{
  width: 100%;
  min-height: 0;
  resize: none;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 13px 14px;
  outline: none;
  font-size: 13px;
  line-height: 1.55;
}}

textarea::placeholder {{
  color: var(--muted-2);
}}

textarea:focus,
select:focus,
input:focus {{
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(23, 122, 72, 0.14);
}}

.actions {{
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 10px;
}}

.btn {{
  height: 40px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  color: var(--text);
  background: transparent;
  font-size: 12px;
  font-weight: 600;
}}

.btn:hover {{
  background: var(--panel-2);
  border-color: var(--border-strong);
}}

.btn-primary {{
  background: var(--accent);
  border-color: var(--accent);
}}

.btn-primary:hover {{
  background: var(--accent-hover);
  border-color: var(--accent-hover);
}}

.btn-small {{
  height: 32px;
  padding: 0 12px;
  font-size: 11px;
  font-weight: 500;
}}

.settings-stack {{
  display: grid;
  gap: 12px;
}}

.field {{
  min-width: 0;
}}

.field-label {{
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 500;
}}

.preset-row {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}}

.preset {{
  height: 38px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel-2);
  color: #c6d0c9;
  font-size: 11px;
  font-weight: 500;
}}

.preset:hover {{
  background: var(--panel-3);
  border-color: var(--border-strong);
}}

.preset.active {{
  color: #e7f5ec;
  background: var(--accent-soft);
  border-color: #2f7653;
}}

.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}}

select,
input[type="text"],
input[type="number"] {{
  width: 100%;
  height: 38px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  outline: none;
  padding: 0 12px;
  font-size: 12px;
}}

.advanced {{
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 0 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: #bdc8c0;
  background: var(--panel-2);
  font-size: 11px;
}}

.preview-wrap {{
  position: relative;
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: #08100b;
  scrollbar-color: #31483a #0a100c;
  scrollbar-width: thin;
}}

.preview-wrap::-webkit-scrollbar {{
  width: 10px;
  height: 10px;
}}

.preview-wrap::-webkit-scrollbar-track {{
  background: #0a100c;
}}

.preview-wrap::-webkit-scrollbar-thumb {{
  background: #31483a;
  border-radius: 999px;
  border: 2px solid #0a100c;
}}

.preview-stage {{
  width: max-content;
  min-width: 100%;
  min-height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px;
}}

.preview-wrap img {{
  display: none;
  width: auto;
  height: auto;
  max-width: none;
  max-height: none;
  object-fit: initial;
}}

.preview-empty {{
  color: var(--muted-2);
  font-size: 12px;
}}

.thumbs {{
  min-height: 0;
  display: flex;
  align-items: stretch;
  gap: 8px;
  overflow-x: auto;
  overflow-y: hidden;
}}

.thumb {{
  flex: 0 0 74px;
  height: 74px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel-2);
  overflow: hidden;
}}

.thumb.active {{
  border-color: var(--accent-hover);
  box-shadow: 0 0 0 1px var(--accent-hover);
}}

.thumb img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
}}

.meta {{
  min-height: 42px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px 16px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 11px;
  background: var(--panel-2);
  color: #aab8ae;
  font-size: 11px;
}}

.meta strong {{
  color: #e4ebe6;
  font-weight: 500;
}}

.full-viewer {{
  position: fixed;
  inset: 0;
  z-index: 30;
  display: none;
  background: rgba(0, 0, 0, 0.92);
}}

.full-viewer.open {{
  display: block;
}}

.full-toolbar {{
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 2;
  display: flex;
  gap: 8px;
}}

.full-scroll {{
  position: absolute;
  inset: 0;
  overflow: auto;
  padding: 64px 24px 24px;
  scrollbar-color: #3a4a40 #080b09;
  scrollbar-width: thin;
}}

.full-stage {{
  min-width: 100%;
  min-height: 100%;
  width: max-content;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.full-stage img {{
  display: block;
  width: auto;
  height: auto;
  max-width: none;
  max-height: none;
}}

.modal-backdrop {{
  position: fixed;
  inset: 0;
  z-index: 20;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(0, 0, 0, 0.55);
}}

.modal-backdrop.open {{
  display: flex;
}}

.modal {{
  width: min(460px, 100%);
  padding: 18px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--panel);
}}

.modal .settings-stack {{
  margin-top: 16px;
}}

.loading {{
  pointer-events: none;
  opacity: 0.7;
}}

.error {{
  color: var(--danger);
}}

@media (max-width: 980px) {{
  .app {{ padding: 12px; }}
  .shell {{ grid-template-columns: 330px minmax(0, 1fr); gap: 12px; }}
  .left {{ gap: 12px; }}
  .prompt-panel, .settings-panel, .results-panel {{ padding: 14px; }}
}}

@media (max-height: 760px) {{
  .app {{ padding: 10px; }}
  .shell {{ gap: 10px; }}
  .left {{ gap: 10px; }}
  .prompt-panel, .settings-panel, .results-panel {{ padding: 12px; }}
  .settings-stack {{ gap: 9px; }}
  .results-panel {{ grid-template-rows: auto minmax(0, 1fr) 68px auto; }}
  .thumb {{ flex-basis: 60px; height: 60px; }}
}}
</style>
</head>
<body>
<div class="app">
  <main class="shell">
    <section class="left">
      <div class="panel prompt-panel">
        <div class="header-row">
          <h2 class="title">Prompt</h2>
        </div>

        <textarea id="prompt" placeholder="Describe the image you want to generate..."></textarea>

        <div class="actions">
          <button class="btn" id="clearBtn">Clear</button>
          <button class="btn btn-primary" id="generateBtn">Generate</button>
        </div>
      </div>

      <div class="panel settings-panel">
        <div class="header-row" style="margin-bottom:14px">
          <h2 class="title">Settings</h2>
          <button class="btn btn-small" id="resetBtn">Reset</button>
        </div>

        <div class="settings-stack">
          <div class="field">
            <div class="field-label">Preset</div>
            <div class="preset-row" id="presetRow">
              <button class="preset" data-preset="Fast">Fast</button>
              <button class="preset active" data-preset="Balanced">Balanced</button>
              <button class="preset" data-preset="Quality">Quality</button>
              <button class="preset" data-preset="Custom">Custom</button>
            </div>
          </div>

          <div class="two-col">
            <div class="field">
              <div class="field-label">Device</div>
              <select id="device">{device_options}</select>
            </div>
            <div class="field">
              <div class="field-label">Images</div>
              <select id="count">
                <option>1</option><option>2</option><option>3</option>
                <option>4</option><option>5</option><option>6</option>
              </select>
            </div>
          </div>

          <div class="two-col">
            <div class="field">
              <div class="field-label">Image size</div>
              <select id="size">
                <option value="256">256 × 256</option>
                <option value="384" selected>384 × 384</option>
                <option value="512">512 × 512</option>
              </select>
            </div>
            <div class="field">
              <div class="field-label">Steps</div>
              <select id="steps">
                <option>2</option>
                <option>3</option>
                <option selected>4</option>
              </select>
            </div>
          </div>

          <button class="advanced" id="advancedBtn">
            <span>Advanced</span>
            <span>›</span>
          </button>
        </div>
      </div>
    </section>

    <section class="panel results-panel">
      <div class="header-row">
        <h2 class="title" id="resultsTitle">Results</h2>
        <div style="display:flex;gap:8px">
          <button class="btn btn-small" id="viewFullBtn" disabled>View full</button>
          <button class="btn btn-small" id="openFolderBtn">Open folder</button>
        </div>
      </div>

      <div class="preview-wrap" id="previewWrap">
        <div class="preview-stage">
          <div class="preview-empty" id="previewEmpty">Generated image will appear here</div>
          <img id="previewImage" alt="">
        </div>
      </div>

      <div class="thumbs" id="thumbs"></div>

      <div class="meta" id="meta">
        <span>Ready</span>
      </div>
    </section>
  </main>
</div>

<div class="full-viewer" id="fullViewer">
  <div class="full-toolbar">
    <button class="btn btn-small" id="closeFullBtn">Close</button>
  </div>
  <div class="full-scroll" id="fullScroll">
    <div class="full-stage">
      <img id="fullImage" alt="Full image">
    </div>
  </div>
</div>

<div class="modal-backdrop" id="modalBackdrop">
  <div class="modal">
    <div class="header-row">
      <h2 class="title">Advanced</h2>
      <button class="btn btn-small" id="closeModalBtn">Close</button>
    </div>

    <div class="settings-stack">
      <div class="field">
        <div class="field-label">Model path</div>
        <input id="modelPath" type="text" value="{DEFAULT_MODEL_PATH}">
      </div>
      <div class="field">
        <div class="field-label">Output folder</div>
        <input id="outputDir" type="text" value="{DEFAULT_OUTPUT_DIR}">
      </div>
    </div>
  </div>
</div>

<script>
const presets = {presets_json};

const state = {{
  preset: "Balanced",
  images: []
}};

const promptEl = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const clearBtn = document.getElementById("clearBtn");
const resetBtn = document.getElementById("resetBtn");
const deviceEl = document.getElementById("device");
const countEl = document.getElementById("count");
const sizeEl = document.getElementById("size");
const stepsEl = document.getElementById("steps");
const presetButtons = Array.from(document.querySelectorAll(".preset"));
const previewImage = document.getElementById("previewImage");
const previewEmpty = document.getElementById("previewEmpty");
const thumbs = document.getElementById("thumbs");
const meta = document.getElementById("meta");
const resultsTitle = document.getElementById("resultsTitle");
const openFolderBtn = document.getElementById("openFolderBtn");
const viewFullBtn = document.getElementById("viewFullBtn");
const fullViewer = document.getElementById("fullViewer");
const fullImage = document.getElementById("fullImage");
const closeFullBtn = document.getElementById("closeFullBtn");
const modalBackdrop = document.getElementById("modalBackdrop");
const advancedBtn = document.getElementById("advancedBtn");
const closeModalBtn = document.getElementById("closeModalBtn");
const modelPath = document.getElementById("modelPath");
const outputDir = document.getElementById("outputDir");

function setPreset(name) {{
  state.preset = name;
  presetButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.preset === name));

  const config = presets[name];
  if (name !== "Custom") {{
    sizeEl.value = String(config.size);
    stepsEl.value = String(config.steps);
  }}
}}

presetButtons.forEach(btn => {{
  btn.addEventListener("click", () => setPreset(btn.dataset.preset));
}});

sizeEl.addEventListener("change", () => setPreset("Custom"));
stepsEl.addEventListener("change", () => setPreset("Custom"));

clearBtn.addEventListener("click", () => {{
  promptEl.value = "";
  promptEl.focus();
}});

resetBtn.addEventListener("click", () => {{
  setPreset("Balanced");
  deviceEl.value = "{default_device}";
  countEl.value = "1";
  sizeEl.value = "384";
  stepsEl.value = "4";
}});

advancedBtn.addEventListener("click", () => modalBackdrop.classList.add("open"));
closeModalBtn.addEventListener("click", () => modalBackdrop.classList.remove("open"));
modalBackdrop.addEventListener("click", e => {{
  if (e.target === modalBackdrop) modalBackdrop.classList.remove("open");
}});

viewFullBtn.addEventListener("click", () => {{
  const url = viewFullBtn.dataset.url;
  if (!url) return;

  fullImage.src = url + "?t=" + Date.now();
  fullViewer.classList.add("open");
}});

closeFullBtn.addEventListener("click", () => {{
  fullViewer.classList.remove("open");
}});

fullViewer.addEventListener("click", (event) => {{
  if (event.target === fullViewer) {{
    fullViewer.classList.remove("open");
  }}
}});

document.addEventListener("keydown", (event) => {{
  if (event.key === "Escape") {{
    fullViewer.classList.remove("open");
    modalBackdrop.classList.remove("open");
  }}
}});

openFolderBtn.addEventListener("click", async () => {{
  try {{
    const response = await fetch("/api/open-folder", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ output_dir: outputDir.value }})
    }});

    if (!response.ok) {{
      const data = await response.json();
      throw new Error(data.detail || "Could not open folder");
    }}
  }} catch (error) {{
    meta.innerHTML = `<span class="error">${{error.message}}</span>`;
  }}
}});

function showImage(index) {{
  const image = state.images[index];
  if (!image) return;

  const url = image.url + "?t=" + Date.now();
  previewImage.src = url;
  previewImage.style.display = "block";
  previewEmpty.style.display = "none";
  viewFullBtn.disabled = false;
  viewFullBtn.dataset.url = image.url;

  Array.from(thumbs.children).forEach((el, i) => el.classList.toggle("active", i === index));
}}

function renderImages(images) {{
  state.images = images;
  thumbs.innerHTML = "";

  images.forEach((image, index) => {{
    const button = document.createElement("button");
    button.className = "thumb" + (index === 0 ? " active" : "");

    const img = document.createElement("img");
    img.src = image.url + "?t=" + Date.now();
    img.alt = "Generated " + (index + 1);

    button.appendChild(img);
    button.addEventListener("click", () => showImage(index));
    thumbs.appendChild(button);
  }});

  resultsTitle.textContent = images.length ? `Results (${{images.length}})` : "Results";

  if (images.length) showImage(0);
}}

function setLoading(loading) {{
  generateBtn.disabled = loading;
  generateBtn.textContent = loading ? "Generating..." : "Generate";
  document.querySelector(".shell").classList.toggle("loading", loading);
}}

generateBtn.addEventListener("click", async () => {{
  const prompt = promptEl.value.trim();
  if (!prompt) {{
    meta.innerHTML = '<span class="error">Enter a prompt first</span>';
    promptEl.focus();
    return;
  }}

  setLoading(true);
  meta.innerHTML = "<span>Generating...</span>";

  try {{
    const response = await fetch("/api/generate", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        prompt,
        preset: state.preset,
        device: deviceEl.value,
        size: Number(sizeEl.value),
        steps: Number(stepsEl.value),
        count: Number(countEl.value),
        model_path: modelPath.value,
        output_dir: outputDir.value
      }})
    }});

    const data = await response.json();

    if (!response.ok) {{
      throw new Error(data.detail || "Generation failed");
    }}

    renderImages(data.images);

    meta.innerHTML = `
      <span><strong>${{data.count}}</strong> image${{data.count === 1 ? "" : "s"}}</span>
      <span>${{data.device}}</span>
      <span>${{data.size}} × ${{data.size}}</span>
      <span>${{data.steps}} steps</span>
      <span>${{data.average.toFixed(2)}}s / image</span>
    `;
  }} catch (error) {{
    console.error(error);
    meta.innerHTML = `<span class="error">${{error.message}}</span>`;
  }} finally {{
    setLoading(false);
  }}
}});
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "devices": get_available_devices(),
        "model_path_exists": DEFAULT_MODEL_PATH.exists(),
        "output_dir": str(DEFAULT_OUTPUT_DIR.resolve()),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if req.device not in get_available_devices():
        raise HTTPException(status_code=400, detail=f"Device not available: {req.device}")

    if req.size not in (256, 384, 512):
        raise HTTPException(status_code=400, detail="Invalid image size.")

    if req.steps not in (2, 3, 4):
        raise HTTPException(status_code=400, detail="Invalid steps.")

    if not (1 <= req.count <= 6):
        raise HTTPException(status_code=400, detail="Count must be 1 to 6.")

    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")

    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        # To keep serving simple and safe, generation is always exposed through /outputs.
        # If a custom output folder is selected, files are still saved there and mirrored to outputs.
        mirror_to_default = True
    else:
        mirror_to_default = False

    with GEN_LOCK:
        load_start = time.perf_counter()
        pipe, loaded_now = get_pipeline(req.device, str(model_path))
        load_time = time.perf_counter() - load_start

        images = []
        times = []

        for index in range(req.count):
            start = time.perf_counter()

            image = pipe(
                prompt,
                width=req.size,
                height=req.size,
                num_inference_steps=req.steps,
            ).images[0]

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = (
                f"{timestamp}_{req.device.lower()}_{req.size}px_"
                f"{req.steps}steps_{index + 1}_{slugify(prompt)}.png"
            )

            primary_path = output_dir / filename
            image.save(primary_path)

            served_path = primary_path
            if mirror_to_default:
                served_path = DEFAULT_OUTPUT_DIR / filename
                image.save(served_path)

            images.append({
                "url": f"/outputs/{served_path.name}",
                "path": str(primary_path.resolve()),
            })

    average = sum(times) / len(times)

    return {
        "images": images,
        "count": req.count,
        "device": req.device,
        "size": req.size,
        "steps": req.steps,
        "average": average,
        "model_load": load_time if loaded_now else 0.0,
    }


class OpenFolderRequest(BaseModel):
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


@app.post("/api/open-folder")
def open_folder(req: OpenFolderRequest):
    folder = Path(req.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    resolved = folder.resolve()

    try:
        if os.name == "nt":
            os.startfile(resolved)
        elif sys.platform == "darwin":
            os.system(f'open "{resolved}"')
        else:
            os.system(f'xdg-open "{resolved}"')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
