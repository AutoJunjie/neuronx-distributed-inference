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
| `ONBOARDING_RETROSPECTIVE.md` | Candid write-up of the gemma-4-12B-it & FireRed-Image-Edit onboardings: full process, the bugs that actually mattered, exact token/time cost, and where the workflow did not help (the gaps that motivated the P0/P1 hardening). |

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
subclassing, weight conversion). The last mile — *numerical* correctness on a
**post-training-cutoff** architecture — needs a golden reference, source-grounding
instead of guessing, and treating "compiles" as distinct from "correct".

`nxdi-onboard-model.js` now bakes these in (informed by the Gemma-4 onboarding):

- **Golden phase** — builds an isolated CPU venv with a `transformers` build that
  *knows* the architecture, runs the real model, and captures the golden next-token
  id + top-k logits + source-quoted arch facts as the correctness oracle. Toggle with
  `golden_ref:false`.
- **Three-tier Verify** — `failed` / `compiles` / `runs` / `numerically_correct`.
  Full success requires the device's greedy next-token id to **match the golden**, not
  merely "it didn't crash". An instruction-tuned model emitting degenerate text on a
  *raw* prompt is expected — correctness is judged by logit/token parity.
- **Fail-loud** — `require_verify:true` makes the whole run return an `error` (instead of
  a polished-but-unproven scaffold) if Setup finds no device/engine or Verify never
  reaches full success. Setup now detects the device/venv by *running commands*, not by
  agent self-report (the original silently skipped Verify on a machine that had 16
  Neuron devices).
- **Engine fallback** — the numerical check prefers the engine least coupled to the
  installed `transformers` (native `inference_demo`) when the front-end can't parse a
  brand-new arch, rather than assuming vLLM.
- **Regression-guarded Repair** — numerical divergences are diffed against the golden
  arch facts (not chased as tracebacks), and the repair agent is told to make minimal,
  targeted edits and not "improve" already-correct code (a prior round had regressed a
  correct `softmax_scale`).

`gemma4-fix-verify.js` plus the `contrib/models/gemma-4-12B-it` README capture the
worked example these improvements generalize from.
