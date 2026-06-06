# Onboarding retrospective: gemma-4-12B-it & FireRed-Image-Edit-1.0

A candid write-up of two NxD Inference onboardings driven with the
`nxdi-workflows` multi-agent workflows — the full process, the bugs that
actually mattered, the time/token cost, and **where the workflow did not help**.
The gaps documented here are what motivated the P0/P1 hardening already merged
into `nxdi-onboard-model.js`.

> **Data honesty.** The gemma-4 figures are exact — taken from each subagent /
> workflow completion receipt (`subagent_tokens`, `agent_count`, `duration_ms`).
> The FireRed figures are **estimates** (that run pre-dated per-agent receipts);
> they are labeled as such. Main-thread (orchestrator) tokens are **not** counted
> in either total, so real totals are higher than the subagent sums below.

---

## 1. gemma-4-12B-it (text decoder, causal-LM) — hand-written NxDI port

### What it is
`google/gemma-4-12B-it` = `Gemma4UnifiedForConditionalGeneration` (`model_type=gemma4_unified`),
a unified text+vision+audio model **released after the assistant's training cutoff**, needing
`transformers >= 5.10.0.dev0`. We ported the **text decoder** only.

### Process (the path that actually happened)
1. **Profile + scaffold + review + emit** — `nxdi-onboard-model` workflow produced
   `modeling_*.py` + plan + runbook, with an adversarial review surfacing 3 blockers + 7 majors.
   **But it did not verify on hardware** (see §3, gap #1).
2. **Pre-flight grounding (manual)** — read the real `model.safetensors` header (key names,
   per-layer shapes), the live HF config, and confirmed the engine path: vLLM front-end couldn't
   parse the arch on the installed `transformers`, but native `inference_demo` could (json config
   + converter, transformers-independent).
3. **Fix → verify → repair on hardware** — `gemma4-fix-verify` workflow; compiled (TP=8), loaded
   weights, then failed inside token-generation attention.
4. **Token-gen shape fix** — heterogeneous global-layer KV geometry (16 vs 128 heads).
5. **Golden CPU reference** — isolated venv with transformers-main; produced golden next-token id
   + top-10 logits + source-quoted arch facts. This is what turned guessing into measurement.
6. **Numerics fix** — drove device logits to match golden (the real bug was `layer_scalar`; see §4).
7. **vLLM-on-Neuron bring-up** — got the OpenAI server serving by passing three front-end gates.

### Result
On-device (TP=8, bf16) greedy next-token **236770** (logit 19.45) matches the golden fp32 CPU
reference (236770, 19.57); chat template returns "The capital of France is Paris." and "17×24"→"408",
through **both** `inference_demo` and the vLLM-on-Neuron OpenAI server.

### Cost (exact, subagent-only)

| Phase | id | subagent tokens | agents | wall time |
|---|---|---:|---:|---:|
| profile+scaffold+review+emit | `nxdi-onboard-model` | 416,363 | 10 | ~13.9 min |
| fix→verify→repair (compile+load) | `gemma4-fix-verify` | 723,645 | 8 | ~132.7 min |
| token-gen shape fix | single agent | 202,821 | 1 | ~33.2 min |
| golden CPU reference | single agent | 52,421 | 1 | ~5.7 min |
| numerics fix (match golden) | single agent | 190,037 | 1 | ~36.9 min |
| vLLM-on-Neuron bring-up | single agent | 84,954 | 1 | ~13.2 min |
| **subagent total** | | **1,670,241** | **22** | **~235.6 min of agent wall time** |

- End-to-end wall clock from scaffold-emit to dual-engine verification: **~5 hours**, dominated by
  repeated Neuron recompiles (12B, two graphs, minutes each).
- Real total tokens (incl. uncounted main-thread orchestration) is **likely > 2M**.
- The single biggest line item — `gemma4-fix-verify` at 724k tokens / 133 min — is where the
  repair loop ran its full rounds against crash-class bugs with a recompile each round.

---

## 2. FireRed-Image-Edit-1.0 (diffusion image editing) — reuse of a community port

### What it is
`FireRedTeam/FireRed-Image-Edit-1.0` = a `QwenImageEditPlusPipeline` diffusion model. Its
`transformer`/`vae` configs are **byte-identical** to the base `Qwen/Qwen-Image-Edit-2509`, so it
is a fine-tuned **weight variant** — the compiled Neuron graphs fit by repointing the model id.

### Process
A diffusion model, so there was **no `modeling_*.py` to write**. Upstream `diffusers` has no
Neuron backend and `optimum-neuron` doesn't support QwenImageEdit; the viable path is the
community contrib port (`whn09/.../contrib/models/Qwen-Image-Edit`) which reuses official
`diffusers` modules and traces+compiles them via `ModelBuilder`. Steps: clone that port →
repath its hardcoded `/opt/dlami/nvme` → repoint `MODEL_ID` to FireRed (in **two** scripts —
one was hardcoded and didn't read the env var) → install deps → download (~60 GB) → compile 5
components → run an edit → stand up a resident HTTP server.

### Result
On trn2.48xlarge (`v3_cfg`, TP=4/DP=2): all 5 components compile, CLI inference produces a valid
1024×1024 edit, and the HTTP server returns a PNG over `curl` (~44 s/image after warmup).

### Cost (ESTIMATED — no per-agent receipts for this run)
- Ran across a diffusion-specific workflow (`nxdi-onboard-diffusion`, two branches) plus manual
  server bring-up and a tutorial-notebook build/verify pass.
- **Rough order of magnitude: ~0.5–1M subagent tokens; a few hours wall clock**, again dominated
  by the ~60 GB download + ~30–45 min compile. Treat these as estimates, not measured figures.

---

## 3. Where the workflow did NOT help (the motivating gaps)

These are specific to the workflow *as it was* during these runs. The P0/P1 items were
subsequently fixed (see §5).

1. **Verify silently didn't run (most severe).** The named-workflow run used a stale cached
   script copy with no Verify phase, and the Setup agent under-reported the present hardware
   (`neuron_device_present=false` on a host with 16 live Neuron devices), so verification was
   skipped. The workflow returned a polished scaffold + plan and read as "done" with **zero
   on-hardware proof** — the exact "looks finished but isn't" failure.

2. **"Compiles / doesn't crash" was treated as success.** Even when verify ran, its success
   signal was "startup complete + coherent chat". For a post-cutoff arch, *not crashing ≠ correct*.
   When the model emitted `<pad>`, the workflow couldn't distinguish "expected degenerate output of
   an instruction-tuned model on a raw prompt" from "a real numerical bug".

3. **No golden-reference mechanism.** The decisive debugging tool — an isolated venv with a
   transformers build that knows the arch, run on CPU for reference logits — was entirely manual.
   The workflow also assumed the local box could load the model; it couldn't.

4. **`TODO(verify)` was guessed, not grounded.** The scaffold marked uncertain arch math as
   `TODO(verify)` and then **guessed** — several guesses were wrong (see §4). There was no
   mechanism forcing each `TODO` to be grounded against the real HF reference.

5. **Engine choice was hard-wired to vLLM, no fallback.** Verify assumed "patch MODEL_TYPES →
   vLLM runs". It didn't know the vLLM front-end would reject the arch via transformers
   `AutoConfig`, nor that `inference_demo` was the right engine to verify numerics first.

6. **The repair loop introduced a regression and was expensive.** A repair agent changed a
   *correct* `softmax_scale` (1.0) to an incorrect `sqrt(head_dim)` with no regression check, and
   every round recompiled the full 12B (minutes). 724k tokens / 133 min went here.

7. **Multimodal/nested config not handled.** Flattening `text_config`/`vision_config`/`audio_config`
   into a text-decoder config and dropping the vision/audio towers was manual.

8. **vLLM-on-Neuron front-end gates not covered.** `AutoConfig.register` / `ModelRegistry`
   registration / the `patch_rope_parameters` nested-rope corruption were each found by hand.

**One-line summary:** the workflow reliably handled the mechanical ~80% (architecture profiling,
the four-file scaffold, adversarial review, compile flags, KV-cache subclassing, weight
conversion). The last 20% — *numerical correctness on a brand-new architecture* — needed three
things it lacked: a golden reference, source-grounding instead of guessing, and the judgment to
distinguish "runs" from "correct" (and to switch engines).

---

## 4. The bugs that actually mattered (gemma-4)

Most were **not** the scary-looking ones; the real killers were subtle:

- **`layer_scalar` declared as `register_buffer` (THE root cause of `<pad>`).** NxD's trace-time
  weight loader iterates only `named_parameters()`; buffers are constant-folded at their init value
  (`ones`), so the real per-layer scalars (~0.005–0.36) never loaded → residual stream exploded →
  all logits saturated the `30·tanh` softcap → first token = `<pad>`. Fix: make it an `nn.Parameter`.
- **`softmax_scale` must be 1.0**, not `sqrt(head_dim)` — and a repair round had *regressed* it.
- **`proportional` RoPE** on global layers: only the first 128 of 512 dims rotate, inv_freq
  denominator is the full head_dim (512) then zero-padded — a stock `RotaryEmbedding(128)` is wrong.
- **`attention_k_eq_v`**: global layers ship no `v_proj`; converter must synthesize `v_proj := k_proj`.
- **Heterogeneous per-layer-type geometry** (sliding kv=8/head_dim=256 vs global kv=1/head_dim=512)
  needs a per-layer-head_dim KV cache manager and per-layer kv-head counts (token-gen 16-vs-128 bug).

FireRed's one notable trap: `MODEL_ID` was hardcoded in **two** scripts; patching only the
downloader (not the inference script) caused a `LocalEntryNotFoundError`.

---

## 5. What was changed in response

`nxdi-onboard-model.js` was hardened (merged) to close the P0/P1 gaps:

- **P0 — fail-loud verify.** Setup detects device/engine by *running commands* (not self-report);
  `require_verify:true` makes the whole run error out if hardware prerequisites are missing or
  verification never reaches full success.
- **P1 — numerical correctness.** A new **Golden** phase builds the CPU reference; **Verify** is now
  three-tier (`failed`/`compiles`/`runs`/`numerically_correct`) and full success requires matching
  the golden next-token; engine selection prefers the transformers-independent `inference_demo`; a
  new `unverified-assumptions` review dimension flags guessed arch math (including the
  `register_buffer`-vs-`Parameter` load bug); and **Repair** treats numerical divergence as a
  math diff against the golden facts and is regression-guarded.

P2 items (engine candidate chain, cheap CPU-logit iteration before recompiling, automatic
multimodal-config flattening) remain open.

_Last updated: 2026-06-06._
