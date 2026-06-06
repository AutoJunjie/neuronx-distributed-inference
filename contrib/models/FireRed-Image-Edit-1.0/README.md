# Contrib Model: FireRed-Image-Edit-1.0 (image editing, diffusion)

NeuronX Distributed Inference recipe for the image-editing diffusion model
[`FireRedTeam/FireRed-Image-Edit-1.0`](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0)
on AWS Trainium2 (Neuron), served over HTTP.

> **Different shape from the LLM contrib models.** This is a **diffusion** pipeline
> (`QwenImageEditPlusPipeline`), not a causal-LM, so the contribution is a **self-contained
> tutorial notebook + a resident HTTP server**, not a `modeling_*.py`. The notebook drives
> the whole flow end-to-end; it does **not** vendor anyone else's modeling code (it clones
> the upstream community port at run time — see below).

## Model Information

- **HuggingFace ID:** `FireRedTeam/FireRed-Image-Edit-1.0`
- **Pipeline:** `QwenImageEditPlusPipeline` — a Qwen2.5-VL text/vision encoder + a
  60-layer `QwenImageTransformer2DModel` DiT (~20B) + a 3D `AutoencoderKLQwenImage` VAE +
  `FlowMatchEulerDiscreteScheduler`.
- **Key fact that makes this work:** FireRed's `transformer/config.json` and `vae/config.json`
  are **byte-identical** to the base `Qwen/Qwen-Image-Edit-2509`, and the text encoder shares
  the same architecture. FireRed is a **fine-tuned weight variant** of the same architecture,
  so the compiled Neuron graphs fit FireRed's weights directly — you only repoint the model id.
- **License:** Check the HuggingFace model card.

## Why a notebook (and not pure `diffusers` / `optimum-neuron`)

Upstream `huggingface/diffusers` has **no Neuron backend** (CUDA/CPU/MPS only), and
`optimum-neuron` does not support the QwenImageEdit architecture. The only viable path is the
community **contrib port** ([`whn09/neuronx-distributed-inference`](https://github.com/whn09/neuronx-distributed-inference),
`contrib/diffusion-models` branch, `contrib/models/Qwen-Image-Edit`), which **reuses the
official `diffusers` model definitions / pipeline / weight loading** and only traces+compiles
the official transformer / VAE modules into Neuron NEFF graphs via `ModelBuilder`, then plugs
them back into the official pipeline. The notebook **clones that port at run time** and adapts
it to FireRed by repointing the model id — it does not copy that code into this repo.

## Contents

```
FireRed-Image-Edit-1.0/
├── firered_neuron_tutorial.ipynb   # main tutorial (13 steps + appendices), runs end-to-end
├── build_nb.py                     # regenerates the notebook (edit this, then re-run)
├── serve_firered.py                # resident HTTP server (load once, curl repeatedly)
├── assets/input.png                # sample input image
└── README.md
```

## Quick start

Use the Neuron SDK venv's Jupyter so `import torch` / `diffusers` work:

```bash
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/jupyter lab \
  --notebook-dir contrib/models/FireRed-Image-Edit-1.0 --ip 0.0.0.0 --no-browser
```

Open `firered_neuron_tutorial.ipynb`, register + switch to the **"FireRed Neuron (venv)"**
kernel (step 0), then run the cells top to bottom. The notebook: clones the contrib port →
repaths its hardcoded `/opt/dlami/nvme` to a local dir → points `MODEL_ID` at FireRed →
installs deps (official `diffusers` from git) → downloads weights → **compiles the 5 components**
→ runs an image edit → optionally starts the HTTP server.

### Serving (curl)

After compiling, run the resident server (loads the compiled graphs once):

```bash
python serve_firered.py --port 8000 --compiled_models_dir <compiled dir>
```

```bash
# health
curl -s http://localhost:8000/health

# edit a local image (multipart)
curl -s -X POST http://localhost:8000/edit \
  -F 'prompt=Add a small red hot-air balloon in the sky.' \
  -F 'image=@your.png' -o edited.png

# or JSON (image_url accepts an absolute path or http(s) URL); print timing
curl -s -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Make it night with a starry sky.","image_url":"/abs/path/input.png",
       "num_inference_steps":40,"true_cfg_scale":4.0,"seed":42}' \
  -D - -o edited2.png | grep -i X-Inference-Seconds
```

Tunable fields: `prompt` (required), `negative_prompt`, `num_inference_steps` (default 40),
`true_cfg_scale` (default 4.0), `seed` (default 42). Output is fixed 1024×1024.

## Validation Results

**Validated:** 2026-06-06
**Instance:** trn2.48xlarge | **Mode:** `v3_cfg` (CFG Parallel + NKI Flash, TP=4, DP=2)

| Check | Status |
|------|--------|
| Compile all 5 components (vae_encoder, vae_decoder, transformer_v3_cfg, language_model_v3, vision_encoder_v3) | ✅ |
| CLI inference produces a valid 1024×1024 edit | ✅ (`output_edited.png`, std ≈ 57, non-blank) |
| Resident HTTP server + `curl` round trip | ✅ (HTTP 200, `image/png`, ~44 s/image) |

Approximate per-step transformer latency ~1.07 s; ~43 s for a 40-step edit after warmup.

## Time & disk expectations

| Step | Cost |
|------|------|
| Download weights (~60 GB) | ~15–30 min |
| **Compile 5 components** | **~30–45 min** |
| Single inference (1024×1024, 40 steps) | ~45 s |

Disk: weights + compiled artifacts ≈ **120 GB**. Requires a trn2 / trn1 / inf2 instance.

## Notes

- The compiled graphs are fixed to the compile-time image size (1024×1024); to change size,
  recompile for that size.
- The notebook is idempotent — it skips the download/compile steps if their outputs already
  exist, so you can re-run it safely.
- Any `QwenImageEditPlusPipeline` model whose `transformer`/`vae` configs match
  `Qwen/Qwen-Image-Edit-2509` can use this same flow by changing the model id.

## Maintainer

Community contribution.

**Last Updated:** 2026-06-06
