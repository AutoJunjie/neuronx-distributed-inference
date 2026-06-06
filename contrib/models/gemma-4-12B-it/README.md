# Contrib Model: Gemma-4-12B-it (text decoder)

NeuronX Distributed Inference implementation of the **text decoder** of
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
(`Gemma4UnifiedForConditionalGeneration`, `model_type = gemma4_unified`).

## Model Information

- **HuggingFace ID:** `google/gemma-4-12B-it`
- **Architecture:** `Gemma4UnifiedForConditionalGeneration` — a unified text + vision +
  audio model. This contribution ports the **text decoder** (`gemma4_unified_text`); the
  vision (`gemma4_unified_vision`) and audio (`gemma4_unified_audio`) towers are not ported
  and their checkpoint keys are dropped by the weight converter.
- **Closest shipped ancestor:** `gemma3`.
- **License:** Check the HuggingFace model card.
- **Requires:** `transformers` with `gemma4_unified` support (>= 5.10.0.dev0, i.e. a recent
  `transformers` build). The modeling code registers a config stand-in via
  `AutoConfig.register` so it loads on older `transformers` too — see
  [Running on transformers < 5.10](#running-on-transformers--510) below.

## Architecture Details

Dense decoder (`enable_moe_block=false`), 48 layers, hidden 3840, FFN 15360,
vocab 262144, tied embeddings. The notable Gemma-4 specifics, all handled in
`src/modeling_gemma4.py`:

| Feature | Detail |
|---|---|
| Heterogeneous attention | 5 sliding : 1 full, repeating. Global (full) layers at indices 5,11,…,47. |
| Sliding layers | `num_key_value_heads=8`, `head_dim=256`, `sliding_window=1024`, RoPE `default` θ=1e4 (full rotary). |
| Global layers | `num_global_key_value_heads=1` (MQA), `global_head_dim=512`, no window, RoPE `proportional` θ=1e6, `partial_rotary_factor=0.25`, `attention_k_eq_v=true` (V := K). |
| head_dim | **Decoupled** from hidden/heads (16×256=4096 ≠ 3840); KV cache sized per-layer. |
| `proportional` RoPE | Only the first 128 of the 512 global head dims rotate (denominator = full head_dim, zero-padded); the rest are NoPE. |
| qk/v norm | per-head q/k RMSNorm applied pre-RoPE; v_norm (`with_scale=False`) on V. |
| Attention scaling | hardcoded `1.0` (not `1/sqrt(head_dim)`) → `softmax_scale=1.0`. |
| `layer_scalar` | learned per-layer scalar applied to each layer output (loaded as a Parameter). |
| Logits | `final_logit_softcapping=30.0` (tanh cap); embeddings scaled by `sqrt(hidden_size)`. |
| RMSNorm | multiplies by the raw weight (no `1 + w` offset). |

## Validation Results

**Validated:** 2026-06-06
**Configuration:** TP=8, batch_size=1, seq_len=512, bfloat16
**Instance:** trn2.48xlarge

### Numerical parity vs. a transformers-main fp32 CPU reference

For the raw prompt `"The capital of France is"` (input_ids `[818, 5279, 529, 7001, 563]`):

| Check | Golden (CPU fp32, transformers main) | On-device (TP=8, bf16) | Status |
|---|---|---|---|
| Greedy next-token id | `236770` (`'1'`) | `236770` (`'1'`) | ✅ match |
| Next-token logit | 19.57 | 19.45 | ✅ (bf16 tol) |
| Top-10 token set | — | matches golden (modulo near-tie swaps) | ✅ |

> Note: `gemma-4-12B-it` is instruction-tuned; on a **raw** prompt both the HF reference
> and this port emit the same intentionally-degenerate continuation (`"111…"`). Parity is
> judged on the logit distribution, not on prose. Use the chat template for coherent output.

### End-to-end (chat template)

| Prompt | Reply |
|---|---|
| "What is the capital of France? Answer in one short sentence." | "The capital of France is Paris." |
| "What is 17 * 24? Reply with just the number." | "408" |

Verified through **both** the native `inference_demo` engine and the
**vLLM-on-Neuron OpenAI server** (`/v1/chat/completions`).

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import the model classes from src/ (the module also registers an AutoConfig
# stand-in for the custom model_type on import).
from modeling_gemma4 import NeuronGemma4ForCausalLM, Gemma4InferenceConfig

model_path = "/home/ubuntu/models/gemma-4-12B-it/"
compiled_model_path = "/home/ubuntu/neuron_models/gemma-4-12B-it/"

neuron_config = NeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)
config = Gemma4InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGemma4ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# See test/integration/test_model.py for a full generate example.
```

### Serving with vLLM on Neuron

`vllm serve google/gemma-4-12B-it` does not work out of the box on Neuron because vLLM's
front-end does not yet know the `gemma4_unified` architecture. Register this class through
a `sitecustomize.py` (kept on `PYTHONPATH`) that, without importing NxDI/torch_xla at top
level (to avoid a fork bomb in the `libneuronpjrt-path` helper), does the following:

1. `AutoConfig.register("<routekey>", <Gemma3-derived config with model_type=routekey>)`
   so vLLM's `ModelConfig` accepts the checkpoint;
2. `ModelRegistry.register_model("<Arch>", "vllm.model_executor.models.gemma3:Gemma3ForCausalLM")`
   as a front-end metadata alias (real execution goes through the Neuron path);
3. clear the scalar rope attrs that vLLM's `patch_rope_parameters` would otherwise inject
   into the nested `rope_parameters` dict;
4. inject `NeuronGemma4ForCausalLM` into `neuronx_distributed_inference.utils.constants.MODEL_TYPES`
   under the key the Neuron loader derives from `architectures[0]`.

Then launch (TP=8):

```bash
PATH="$VLLM_VENV/bin:/opt/aws/neuron/bin:$PATH" \
PYTHONPATH="<dir with sitecustomize.py + modeling_gemma4.py>:$PYTHONPATH" \
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_RT_NUM_CORES=8 \
python -m vllm.entrypoints.openai.api_server \
  --model="$CKPT" --served-model-name gemma4 \
  --tensor-parallel-size 8 --max-num-seqs 4 --max-model-len 4096 \
  --no-enable-prefix-caching \
  --additional-config '{"override_neuron_config":{"enable_bucketing":false}}'
```

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":40,"temperature":0}'
# -> "The capital of France is Paris."
```

> `--device` was removed in vllm 0.16 (the neuron platform plugin auto-activates),
> `--no-enable-prefix-caching` is required on Neuron, and `override_neuron_config` must be
> passed inside `--additional-config` (there is no `--override-neuron-config` flag).

### Running on transformers < 5.10

`src/modeling_gemma4.py` calls `AutoConfig.register(...)` on import with a `Gemma3Config`
subclass whose `model_type` matches the checkpoint, so `AutoConfig.from_pretrained` (used by
NxDI's `load_pretrained_config` and by vLLM) succeeds even on a `transformers` build that
predates `gemma4_unified`. The text-decoder geometry is read from the top-level config
attributes (`global_head_dim`, `num_global_key_value_heads`, `layer_types`,
`rope_parameters`, …), which `from_dict` preserves.

## Compatibility Matrix

| Instance / SDK | Status |
|----------------|--------|
| Trn2 (2.25+)   | ✅ Validated (TP=8, bf16) |
| Trn1           | Not tested |
| Inf2           | Not tested |

## Example Checkpoints

* [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)

## Testing

```bash
# point MODEL_PATH / COMPILED_MODEL_PATH in test_model.py at your checkpoint + compile dir
python3 contrib/models/gemma-4-12B-it/test/integration/test_model.py
# or
pytest contrib/models/gemma-4-12B-it/test/integration/test_model.py --capture=tee-sys
```

The integration test compiles (TP=8) if needed, loads, and asserts greedy logit/token
parity against the golden next-token id for the raw prompt, plus a chat-template smoke test.

## Maintainer

Community contribution.

**Last Updated:** 2026-06-06
