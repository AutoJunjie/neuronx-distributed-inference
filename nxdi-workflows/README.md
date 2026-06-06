# NxD Inference onboarding workflows

Multi-agent orchestration scripts (Claude Code "Workflow" / dynamic-workflow format)
used to onboard HuggingFace models to AWS NxD Inference. Each script fans out
subagents across phases (profile → ground → scaffold → review → emit → verify →
repair), grounding generated code against a verified knowledge base and the real
reference modeling files, and — for the model workflow — actually compiling and
running the result on a Neuron device with a source-grounded repair loop.

These are the workflows that produced the `contrib/models/gemma-4-12B-it` port.

## Files

| File | What it does |
|------|--------------|
| `nxdi-onboard-model.js` | Onboard a dense/MoE **causal-LM** to NxDI: profile the HF config, ground against the KB + reference modeling files, scaffold `modeling_<model>.py` (config / model / task-head / weight-converter), adversarially review (base-contract / sharding / converter / MoE / imports), emit `modeling_*.py` + `ONBOARDING_PLAN.md` + `RUNBOOK.md`, then verify on hardware (download weights, load the generated file into vLLM, compile, chat) with an automatic repair loop. Pass the HF model id as `args`, or an object `{model, config_url, instance, tp_degree, fused_qkv, notes, verify, max_repair_rounds}`. |
| `nxdi-onboard-diffusion.js` | Diffusion-model counterpart (e.g. Qwen-Image-Edit / FireRed-Image-Edit): profile `model_index` + transformer config, classify adaptability vs a base contrib port, materialize + repath contrib code, compile the components, then run a real image edit and validate the output. |
| `gemma4-fix-verify.js` | Focused fix → verify → repair workflow used to bring the generated Gemma-4 text-decoder modeling file to numerical correctness on Neuron via `inference_demo` (TP=8). Seeded with grounded findings (proportional RoPE, `attention_k_eq_v`, per-layer-head_dim KV cache, `softmax_scale`, `layer_scalar`, …) so agents apply known fixes instead of rediscovering them. A worked example of the repair pattern. |
| `nxdi-onboarding-kb.md` | The verified knowledge base the workflows read each run: NxDI base-class contracts, verified vs. unverified import paths, sharding rules, weight-converter conventions, and common onboarding bugs. |

## Usage

The `.js` files are dynamic-workflow scripts. Run them with the Claude Code
`Workflow` tool, e.g.:

```
Workflow({ scriptPath: "nxdi-workflows/nxdi-onboard-model.js",
           args: { model: "google/gemma-4-12B-it", instance: "trn2.48xlarge",
                   tp_degree: 8, verify: true, max_repair_rounds: 3 } })
```

Paths inside the scripts (venv, KB location, output dir) are resolved per-machine
by a Setup phase, but review the constants near the top of each script before
running on a new host.

## Notes / known gaps

These workflows reliably handle the mechanical ~80% of an onboarding (architecture
profiling, the four-file scaffold, adversarial review, compile flags, KV-cache
subclassing, weight conversion). For a **post-training-cutoff** architecture, the
last mile — numerical correctness — still needs:

- a **golden reference**: an isolated venv with a `transformers` build that knows the
  new architecture, run on CPU to produce reference logits / per-layer norms;
- **source-grounding** every `TODO(verify)` against the real HF reference modeling
  file rather than guessing;
- distinguishing **"compiles / doesn't crash"** from **"numerically correct"** (an
  instruction-tuned model emitting `<pad>` or degenerate text on a raw prompt can be
  expected, not a bug — judge by logit parity against the golden reference).

The Gemma-4 onboarding hit all three of these; `gemma4-fix-verify.js` plus the
contrib model README capture what that took.
