#!/usr/bin/env python
"""
Resident HTTP server for the FireRed-Image-Edit (QwenImageEdit) Neuron port.

Loads the compiled v3_cfg pipeline ONCE onto Trainium2 (a few minutes), then
serves image edits over HTTP so you can poke it with curl repeatedly.

Endpoints:
  GET  /health                       -> {"status": "loading"|"ready", ...}
  POST /edit  (multipart or JSON)     -> returns the edited PNG bytes
       JSON  body: {"prompt": "...", "image_url": "...", "image_b64": "...",
                    "negative_prompt": "", "num_inference_steps": 40,
                    "true_cfg_scale": 4.0, "seed": 42}
       multipart: -F prompt=... -F image=@input.png

The model is fixed at the COMPILED dimensions (1024x1024, image_size 448),
because the Neuron graphs were traced for exactly those shapes.

Run:
  cd .../src && PATH=<venv/bin>:/opt/aws/neuron/bin:$PATH \
    PYTHONPATH=<src>:$PYTHONPATH QIE_MODEL_PATH=FireRedTeam/FireRed-Image-Edit-1.0 \
    NEURON_RT_NUM_CORES=8 python ../serve_firered.py --port 8000
"""
import argparse
import base64
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

# run_qwen_image_edit lives in the contrib src/ (on PYTHONPATH). Reuse its loaders.
import run_qwen_image_edit as R
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

STATE = {"status": "loading", "error": "", "loaded_at": None, "infer_count": 0}
PIPE = None
LOCK = threading.Lock()  # Neuron graphs are not re-entrant; serialize inference.

# Fixed compiled dimensions (must match what compile.sh v3_cfg produced).
HEIGHT = 1024
WIDTH = 1024
IMAGE_SIZE = 448


class _Args:
    """Mimic the argparse Namespace that run_inference / load_all_compiled_models read."""
    def __init__(self, compiled_dir):
        self.height = HEIGHT
        self.width = WIDTH
        self.image_size = IMAGE_SIZE
        self.image_h = None
        self.image_w = None
        self.patch_multiplier = 2
        self.max_sequence_length = 1024
        self.num_inference_steps = 40
        self.true_cfg_scale = 4.0
        self.seed = 42
        self.compiled_models_dir = compiled_dir
        self.vae_tile_size = 512
        # parallelism / placement flags (match the v3_cfg defaults)
        self.use_v3_cfg = True
        self.use_v3_cp = False
        self.use_v2 = False
        self.use_v1_flash = False
        self.use_v2_flash = False
        self.use_v3_language_model = True
        self.use_v3_vision_encoder = True
        self.cpu_language_model = True
        self.neuron_language_model = False
        self.cpu_vision_encoder = False
        self.neuron_vision_encoder = False
        self.vision_tp = False
        self.cpu_vae_decode = False
        self.debug_text_encoder = False
        self.warmup = False
        self.save_comparison = False
        self.negative_prompt = ""
        self.prompt = ""
        self.images = []
        self.output = None


def _apply_pipeline_patches(args):
    """Replicate run_inference()'s module-level pre-processing patches so input
    images are resized to exactly the compiled grid (else shape mismatch)."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qpm
    compiled_vae_pixels = args.height * args.width
    qpm.VAE_IMAGE_SIZE = compiled_vae_pixels
    vlm_h = args.image_h or args.image_size
    vlm_w = args.image_w or args.image_size
    qpm.CONDITION_IMAGE_SIZE = vlm_h * vlm_w
    _orig = qpm.calculate_dimensions
    _cond = vlm_h * vlm_w
    _vae = compiled_vae_pixels

    def _patched(target_area, ratio):
        if target_area == _cond:
            return vlm_w, vlm_h
        if target_area == _vae:
            return args.width, args.height
        return _orig(target_area, ratio)

    qpm.calculate_dimensions = _patched
    return vlm_h, vlm_w


def load_pipeline(compiled_dir):
    global PIPE
    t0 = time.time()
    args = _Args(compiled_dir)
    vlm_h, vlm_w = _apply_pipeline_patches(args)
    print(f"[load] from_pretrained {R.MODEL_ID} (cache={R.HUGGINGFACE_CACHE_DIR})", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        R.MODEL_ID, torch_dtype=torch.bfloat16,
        cache_dir=R.HUGGINGFACE_CACHE_DIR, local_files_only=True,
    )
    target_pixels = vlm_h * vlm_w
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels
    pipe.processor.image_processor.size = {"shortest_edge": target_pixels, "longest_edge": target_pixels}
    print("[load] loading compiled Neuron graphs onto device ...", flush=True)
    pipe = R.load_all_compiled_models(args.compiled_models_dir, pipe, args)
    PIPE = (pipe, args)
    STATE["status"] = "ready"
    STATE["loaded_at"] = time.time()
    print(f"[load] READY in {time.time() - t0:.1f}s", flush=True)


def _read_input_image(payload, raw_multipart=None):
    if raw_multipart is not None:
        return Image.open(io.BytesIO(raw_multipart)).convert("RGB")
    if payload.get("image_b64"):
        return Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")
    if payload.get("image_url"):
        return load_image(payload["image_url"]).convert("RGB")
    raise ValueError("provide image_url, image_b64, or a multipart 'image' file")


def run_edit(payload, raw_image=None):
    pipe, args = PIPE
    prompt = payload.get("prompt") or ""
    if not prompt:
        raise ValueError("prompt is required")
    img = _read_input_image(payload, raw_image).resize((WIDTH, HEIGHT))
    steps = int(payload.get("num_inference_steps", 40))
    cfg = float(payload.get("true_cfg_scale", 4.0))
    seed = int(payload.get("seed", 42))
    neg = payload.get("negative_prompt", "")
    gen = torch.Generator().manual_seed(seed)
    with LOCK:
        t0 = time.time()
        out = pipe(image=img, prompt=prompt, negative_prompt=neg,
                   height=HEIGHT, width=WIDTH, true_cfg_scale=cfg,
                   num_inference_steps=steps, generator=gen)
        dt = time.time() - t0
    STATE["infer_count"] += 1
    buf = io.BytesIO()
    out.images[0].save(buf, format="PNG")
    return buf.getvalue(), dt


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, json.dumps(STATE).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if not self.path.startswith("/edit"):
            self._send(404, b'{"error":"not found"}')
            return
        if STATE["status"] != "ready":
            self._send(503, json.dumps({"error": "model not ready", **STATE}).encode())
            return
        try:
            ctype = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            raw_image = None
            payload = {}
            if ctype.startswith("multipart/form-data"):
                payload, raw_image = _parse_multipart(body, ctype)
            else:
                payload = json.loads(body or b"{}")
            png, dt = run_edit(payload, raw_image)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Inference-Seconds", f"{dt:.2f}")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            print(f"[edit] ok prompt={payload.get('prompt','')[:50]!r} {dt:.1f}s -> {len(png)}B", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print("[edit] ERROR\n" + tb, flush=True)
            self._send(500, json.dumps({"error": str(e), "traceback": tb.splitlines()[-8:]}).encode())

    def log_message(self, *a):
        pass  # quiet default access log


def _parse_multipart(body, ctype):
    """Minimal multipart parser: extract text fields + one 'image' file."""
    boundary = ctype.split("boundary=")[-1].strip().encode()
    parts = body.split(b"--" + boundary)
    payload, raw_image = {}, None
    for p in parts:
        if b"Content-Disposition" not in p:
            continue
        header, _, data = p.partition(b"\r\n\r\n")
        data = data.rstrip(b"\r\n")
        hl = header.decode(errors="ignore")
        name = ""
        for tok in hl.split(";"):
            tok = tok.strip()
            if tok.startswith("name="):
                name = tok.split("=", 1)[1].strip('"')
        if not name:
            continue
        if "filename=" in hl:
            raw_image = data
        else:
            payload[name] = data.decode(errors="ignore")
    return payload, raw_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--compiled_models_dir", type=str, default=R.COMPILED_MODELS_DIR)
    args = ap.parse_args()
    threading.Thread(target=lambda: _safe_load(args.compiled_models_dir), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] listening on :{args.port} (model loading in background; GET /health)", flush=True)
    srv.serve_forever()


def _safe_load(d):
    try:
        load_pipeline(d)
    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)
        print("[load] FAILED\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
