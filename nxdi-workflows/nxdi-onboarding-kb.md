# Onboarding Models to Run on NxD Inference — Verified Knowledge Base

> Authoritative reference for scaffolding new model support in AWS NxD Inference
> (`neuronx-distributed-inference`, "NxDI") and its core dependency `neuronx-distributed` ("NxD").
> Used as ground-truth by the `nxdi-onboard-model` workflow.
>
> **Provenance:** All base-class signatures and import paths were confirmed against the public
> upstream GitHub mirrors (`aws-neuron/neuronx-distributed-inference` and
> `aws-neuron/neuronx-distributed`, `main`), which ship the same code as the internal package.
> See the [Import path confidence table](#11-import-path-confidence-table) for per-symbol verdicts.
> Entries marked UNVERIFIED must be resolved against the installed package or the
> `modeling_llama.py` / `modeling_qwen3_moe.py` reference files before being trusted in generated code.

---

## 1. Onboarding pipeline overview

Onboarding a new architecture means writing one file,
`src/neuronx_distributed_inference/models/<model>/modeling_<model>.py`, that subclasses a small
set of NxDI base classes. The canonical reference to copy is `models/llama/modeling_llama.py`
(full from-scratch model) or `contrib/models/Llama-2-7b-hf/src/modeling_llama2.py`
(minimal subclass-of-existing pattern).

**Ordered steps (dense model):**

1. **Config classes** — Subclass `InferenceConfig` (architecture/HF fields) and choose a
   `NeuronConfig` (runtime/optimization flags). Declare `get_required_attributes()` and
   `get_neuron_config_cls()`.
2. **Neuron model** — Subclass `NeuronBaseModel` implementing `setup_attr_for_model()` +
   `init_model()`; build the attention module on `NeuronAttentionBase`, MLP/decoder layers
   from parallel layers.
3. **Task head** — Subclass `NeuronBaseForCausalLM`: set `_model_cls`, implement
   `get_config_cls()`, `load_hf_model()`, `convert_hf_to_neuron_state_dict()`, optionally
   `update_state_dict_for_tied_weights()` and `get_compiler_args()`.
4. **Weight conversion + testing** — Implement the HF→Neuron state-dict remap (fused qkv,
   gate-up, parallel-layer names), validate accuracy, and benchmark.

**MoE model** follows the same four steps with these deltas:
- Step 1: `get_neuron_config_cls()` returns **`MoENeuronConfig`** instead of `NeuronConfig`;
  normalize HF expert fields (`num_local_experts`/`num_experts_per_tok`) in the config.
- Step 2: the decoder layer's `self.mlp` becomes a MoE block via `initialize_moe_module(...)`;
  `self.mlp(...)` returns a **tuple** (use index `[0]`).
- Step 4: the converter must rename the router gate and **stack per-expert FFN weights** into
  `[num_experts, ...]` tensors.

**Run lifecycle (driven by the base classes, not you):**
`__init__(model_path, config)` → `get_config_cls().load(model_path)` reads `config.json` →
`compile()` traces sub-models via `ModelBuilder` and saves → `load()` restores with
`torch.jit.load` and places weights on device → `generate()`.

---

## 2. Step 1 — Config classes

Two distinct config layers; confusing them is the most common onboarding bug:

| Layer | Class | Holds | How model code reads it |
|---|---|---|---|
| Runtime / optimization | `NeuronConfig` (or `MoENeuronConfig`) | `tp_degree`, `batch_size`, `torch_dtype`, `buckets`, `on_device_sampling_config`, feature flags | `config.neuron_config.tp_degree` |
| Model architecture | `InferenceConfig` subclass | HF config fields (`hidden_size`, `num_attention_heads`, …) | `config.hidden_size` |

`InferenceConfig` **wraps** a `NeuronConfig` (first positional arg `neuron_config`) and copies
the remaining HF `**kwargs` onto itself as attributes.

### `NeuronConfig` — VERIFIED `neuronx_distributed_inference.models.config.NeuronConfig`
- Constructor is `**kwargs`-driven: `def __init__(self, **kwargs)` with 100+ `kwargs.pop(...)`.
  `torch_dtype` defaults to `torch.bfloat16`.
- Confirmed flags: `tp_degree`, `pp_degree`, `ep_degree`, `world_size`, `fused_qkv`,
  `sequence_parallel_enabled`, `on_device_sampling_config`, `enable_bucketing`, `async_mode`,
  `logical_nc_config` (via `_get_lnc()`), `batch_size`, `max_batch_size`, `buckets`, `seq_len`,
  `n_active_tokens`, `n_positions`, `flash_decoding_enabled`, quantization (`quantized`,
  `quantization_type`), prefix caching (`is_prefix_caching`), speculation (`speculation_length`,
  `enable_fused_speculation`, `enable_eagle_speculation`, `is_medusa`).
- To extend: subclass, `pop` your own keys first, then `super().__init__(**kwargs)`.
- **Do NOT assume these exist** (not found): `attention_dp_degree`, `is_chunked_prefill`, `cast_type`.

### `InferenceConfig` — VERIFIED path `neuronx_distributed_inference.models.config.InferenceConfig`
- `__init__(self, neuron_config, fused_spec_config=None, load_config=None, metadata=None, **kwargs)`
  copies the HF `**kwargs` onto `self`.
- Class attr `attribute_map: Dict[str,str] = {}` — map HF attribute names here when they differ.
- `get_required_attributes(self) -> List[str]` — base returns `[]`; override to enforce HF attrs.
- `@classmethod get_neuron_config_cls(cls) -> Type[NeuronConfig]` — `NeuronConfig` (dense) or
  `MoENeuronConfig` (MoE).
- `add_derived_config(self)` — optional hook for derived fields (e.g. `num_cores_per_group`).
- `save(model_path)` / `@classmethod load(model_path, **kwargs)` — provided; rarely overridden.

```python
from typing import List, Type
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
# from neuronx_distributed_inference.utils.distributed import calculate_num_cores_per_group

class MyInferenceConfig(InferenceConfig):
    # attribute_map = {"n_embd": "hidden_size"}   # only if HF names differ

    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                self.num_attention_heads, self.num_key_value_heads, self.neuron_config.tp_degree)

    def get_required_attributes(self) -> List[str]:
        return ["hidden_size", "num_attention_heads", "num_hidden_layers",
                "num_key_value_heads", "pad_token_id", "vocab_size",
                "max_position_embeddings", "rope_theta", "rms_norm_eps", "hidden_act"]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig  # -> MoENeuronConfig for MoE models
```

- Standard HF fields map 1:1 (`config.hidden_size`). Different names → `attribute_map` or normalize in `__init__`.
- `head_dim` usually derived: `getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)`.

---

## 3. Step 2 — Neuron model (`NeuronBaseModel`)

VERIFIED `neuronx_distributed_inference.models.model_base.NeuronBaseModel`.
- `class NeuronBaseModel(nn.Module)`, `__init__(self, config, optimize_inference=True)`; base
  `__init__` calls `setup_attr_for_model(config)` then `init_model(config)`. Both abstract.
- **Do not** override `__init__`/`forward` unless necessary — the base implements the decoder
  loop, KV cache, sampling wiring.

```python
def setup_attr_for_model(self, config):
    # MUST set: on_device_sampling, tp_degree, hidden_size, num_attention_heads,
    #           num_key_value_heads, max_batch_size, buckets
    ...
def init_model(self, config):
    # MUST define: embed_tokens, layers, norm, lm_head
    ...
```

### Parallel layers — VERIFIED `neuronx_distributed.parallel_layers.layers` (NxD core pkg)

| Layer | Role | Key kwargs |
|---|---|---|
| `ColumnParallelLinear` | Shards **output** dim; all-gather. q/k/v, gate/up, `lm_head`. | `gather_output` (False when followed by RowParallel or on-device sampling), `bias`, `dtype`, `pad`, `tensor_model_parallel_group`, `sequence_parallel_enabled` |
| `RowParallelLinear` | Shards **input** dim; all-reduce. `o_proj`, MLP down. | `input_is_parallel=True` when fed by ColumnParallel, `reduce_output`, `bias`, `dtype` |
| `ParallelEmbedding` | TP embedding. `embed_tokens`. | `shard_across_embedding` (True = embedding-dim shard); vocab-parallel uses `use_spmd_rank=True` + `shard_across_embedding=False` |

Wiring rules:
- `ColumnParallelLinear(gather_output=False)` → `RowParallelLinear(input_is_parallel=True)` for QKV→O and gate/up→down.
- `lm_head`: `gather_output = not on_device_sampling`.
- All take `tensor_model_parallel_group` (via a `get_tp_group(config)` helper) and `dtype=config.neuron_config.torch_dtype`.
- **Guard** all parallel-layer construction with `parallel_state.model_parallel_is_initialized()`
  and fall back to `nn.Embedding`/`nn.Linear` for CPU debugging.
- Norm: `CustomRMSNorm` / `get_rmsnorm_cls(...)` helper (UNVERIFIED exact import — resolve at runtime).

### `NeuronAttentionBase` (UNVERIFIED exact import: `neuronx_distributed_inference.modules.attention.attention_base`)
- Reusable attention `nn.Module`; subclass supplies dims + a `rotary_emb`. Base builds
  `qkv_proj`/`o_proj`, handles **GQA repeat / TP sharding** internally
  (`init_gqa_properties()`/`repeat_kv()`), RoPE, KV cache, flash decoding, sliding window.
- `__init__` is **keyword-only after `config`** (note the `*`) — pass everything by keyword.
- Pass **full (unsharded)** head counts; base divides by `tp_degree`.
- Base calls rotary as `cos, sin = self.rotary_emb(V, position_ids)`.

```python
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding

class MyAttention(NeuronAttentionBase):
    def __init__(self, config, tensor_model_parallel_group=None):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        super().__init__(
            config=config, tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=self.get_rope(config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=getattr(config, "sliding_window", None))
    def get_rope(self, config):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return RotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings,
                               base=config.rope_theta)
```

```python
# MLP (dense)
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from transformers.activations import ACT2FN

class MyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dt = config.neuron_config.torch_dtype
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False, dtype=dt)
            self.up_proj   = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False, dtype=dt)
            self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, input_is_parallel=True, dtype=dt)
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

```python
# Decoder layer
class MyDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MyAttention(config, tensor_model_parallel_group=get_tp_group(config))
        self.mlp = MyMLP(config)                  # MoE: initialize_moe_module(config=config)
        self.input_layernorm = get_rmsnorm_cls(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kw):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, **kw)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)   # MoE: self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        return hidden_states, present
```

```python
# Model
import torch.nn as nn
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers import parallel_state

class MyModel(NeuronBaseModel):
    def setup_attr_for_model(self, config):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    def init_model(self, config):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                pad=True, tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel)
            self.lm_head = ColumnParallelLinear(
                config.hidden_size, config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype, bias=False, pad=True,
                tensor_model_parallel_group=get_tp_group(config))
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleList([MyDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = get_rmsnorm_cls(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
```

---

## 4. Step 3 — Task head (`NeuronBaseForCausalLM`)

VERIFIED `neuronx_distributed_inference.models.model_base.NeuronBaseForCausalLM`
(`class NeuronBaseForCausalLM(NeuronApplicationBase)`, class attr `_model_cls = None`).
Parent `NeuronApplicationBase` (VERIFIED `neuronx_distributed_inference.models.application_base`)
provides `__init__(self, model_path, config=None, neuron_config=None)`, `compile()`, `load()`,
`load_weights()`, `get_builder()` (constructs `ModelBuilder`). Class attrs
`_STATE_DICT_MODEL_PREFIX = "model."` → `_NEW_STATE_DICT_MODEL_PREFIX = ""`.

> Both `NeuronBaseModel` and `NeuronBaseForCausalLM` live in `models/model_base.py`. Only the
> parent `NeuronApplicationBase` lives in `models/application_base.py`.

| Member | Kind | Purpose |
|---|---|---|
| `_model_cls = MyModel` | class attr | Points at your `NeuronBaseModel` subclass. |
| `get_config_cls(cls)` | `@classmethod` | Return your `InferenceConfig` subclass. |
| `load_hf_model(model_path, **kwargs)` | `@staticmethod` | `AutoModelForCausalLM.from_pretrained(...)`. |
| `convert_hf_to_neuron_state_dict(state_dict, config)` | `@staticmethod` | Remap HF names → parallel-layer names. Default returns unchanged. |
| `update_state_dict_for_tied_weights(state_dict)` | `@staticmethod` | Tie `lm_head.weight` to `embed_tokens.weight` (only if tied). |
| `get_compiler_args(self)` | method (optional) | Compiler flag string; base returns `None`. |

> Match decorators exactly (`@staticmethod`/`@classmethod`) or base dispatch breaks. You never
> call `ModelBuilder` yourself.

```python
from typing import Type
from transformers import AutoModelForCausalLM
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel

class MyForCausalLM(NeuronBaseForCausalLM):
    _model_cls = MyModel

    @classmethod
    def get_config_cls(cls) -> Type[InferenceConfig]:
        return MyInferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        ...   # see converter example
        return state_dict

    def get_compiler_args(self) -> str:
        return None
```

```python
# State-dict conversion (fused QKV / gate-up)
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict, config):
    import torch
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}."
        if config.neuron_config.fused_qkv:
            q = state_dict.pop(p + "self_attn.q_proj.weight")
            k = state_dict.pop(p + "self_attn.k_proj.weight")
            v = state_dict.pop(p + "self_attn.v_proj.weight")
            state_dict[p + "self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
        gate = state_dict.pop(p + "mlp.gate_proj.weight")
        up   = state_dict.pop(p + "mlp.up_proj.weight")
        state_dict[p + "mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
    return state_dict
```

---

## 5. Step 4 — Weight conversion + formats

The converter bridges HF key names/layouts to your parallel-layer module names. Base default
returns the dict unchanged, so override unless names already match exactly.

Typical converter work: rename HF keys (`model.layers.*` → `layers.*`, the `model.` prefix strip
is handled by `_STATE_DICT_MODEL_PREFIX`); fuse q/k/v into `qkv_proj.weight` when `fused_qkv`;
fuse gate/up into `gate_up_proj.weight`; pad to TP-divisible shapes (vocab, heads); dequantize
FP8/requantize; tie weights via `update_state_dict_for_tied_weights`; for quantized models save
the quantized state dict **before** compile (`save_quantized_state_dict`).

| Tensor | HF layout | Neuron (dense) | Neuron (fused) |
|---|---|---|---|
| q/k/v proj | separate `q_proj`/`k_proj`/`v_proj` `[out, hidden]` | separate, sharded by `ColumnParallelLinear` | `qkv_proj.weight = cat([q,k,v], dim=0)` |
| o proj | `o_proj` `[hidden, out]` | `RowParallelLinear` (`input_is_parallel=True`) | same |
| gate/up | separate `gate_proj`/`up_proj` | separate `ColumnParallelLinear` | `gate_up_proj.weight = cat([gate,up], dim=0)` |
| down | `down_proj` | `RowParallelLinear` | same |
| embed | `embed_tokens.weight` | `ParallelEmbedding` (sharded) | same |
| lm_head | `lm_head.weight` (or tied) | `ColumnParallelLinear`; tie via `update_state_dict_for_tied_weights` | same |
| MoE router | `...gate.weight` | `...mlp.router.linear_router.weight` | — |
| MoE experts | per-expert `wi/wo` | `...expert_mlps.mlp_op.gate_up_proj.weight [E,H,2I]`, `...down_proj.weight [E,I,H]` | — |

Checkpoint formats: safetensors (`model.safetensors` / `model.safetensors.index.json` sharded),
pickle (`pytorch_model.bin` / `pytorch_model.bin.index.json` sharded).

---

## 6. MoE specifics

### `MoENeuronConfig` — VERIFIED path `neuronx_distributed_inference.models.config.MoENeuronConfig`
`NeuronConfig` subclass: `def __init__(self, capacity_factor=None, glu_mlp=True, **kwargs)` then
`super().__init__(**kwargs)`. Extra knobs: `glu_type`, `hidden_act_scaling_factor`,
`hidden_act_bias`, gate/up clamp limits, `use_index_calc_kernel`, `moe_mask_padded_tokens`,
`moe_tp_degree`, `moe_ep_degree`, `fused_shared_experts`, `transpose_shared_experts_weights`,
`shared_experts_sequence_parallel_enabled`, `normalize_top_k_affinities`, `return_expert_index`,
`return_router_logits`, `hybrid_sharding_config`, `blockwise_matmul_config`, `router_config`.

> `ep_degree`/`world_size` stay on base `NeuronConfig`. `moe_tp_degree`/`moe_ep_degree` are
> MoE-layer overrides. **`num_experts`/`top_k` are NOT on `MoENeuronConfig`** — they are
> HF/`InferenceConfig` attrs (`num_local_experts`/`num_experts_per_tok`); list them in
> `get_required_attributes()` and normalize differing HF names (e.g. Qwen3-MoE: `num_experts` →
> `num_local_experts`, `n_shared_experts = 0`).

```python
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module   # UNVERIFIED import
# v2: self.mlp = initialize_moe_module(config=config)   # adds SharedExperts/EP/fused TKG
# v1: self.mlp = initialize_moe_module(config, num_experts, top_k, hidden_size, intermediate_size, hidden_act)
hidden_states = self.mlp(hidden_states, ...)[0]   # returns a TUPLE; take index 0
```

- v1 (`modules.moe`): explicit args (Mixtral, DBRX); no shared experts / EP / fused TKG.
- v2 (`modules.moe_v2`): reads `config`, adds `SharedExperts`, EP, fused TKG (Qwen3-MoE, Llama4);
  also exposes `initialize_moe_process_group(config, enabled_hybrid_sharding)`.
- Fused TKG: when `moe_fused_nki_kernel_enabled`, pass `rmsnorm=post_attention_layernorm`,
  `init_tkg_module=True`, skip explicit norm in `forward`.

Expert modules (`neuronx_distributed.modules.moe`, NxD core — UNVERIFIED): `RouterTopK`
(weight key `router.linear_router.weight`), `ExpertMLPs`/`ExpertMLPsV2` (`gate_up_proj [E,H,2I]`,
`down_proj [E,I,H]`), `SharedExperts` (when `n_shared_experts > 0`), `MoE` container (tuple out).
Config wrappers in `...moe.moe_configs`: `RouterConfig`, `BlockwiseMatmulConfig`,
`RoutedExpertsMLPOpsConfig`, `MoEFusedTKGConfig`. NxDI-side `HybridShardingConfig` lives in
`neuronx_distributed_inference.models.config`.

EP: controlled by `moe_ep_degree`/`moe_tp_degree`. `moe_v2.initialize_moe_process_group`: if
`moe_ep_degree > 1`, calls `init_tensor_expert_parallel_moe_process_groups`, uses
`get_moe_tp_ep_group`/`get_moe_ep_group` (prefill CTE, decode TKG); else `parallel_state` groups.
`HybridShardingConfig` allows different prefill (CTE) vs decode (TKG) TP/EP layout. EP > 1 raises
TKG optimization to `O3`.

```python
# MoE converter sketch
sd["...mlp.router.linear_router.weight"] = sd.pop("...gate.weight")
sd["...mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj  # [num_experts, hidden, 2*intermediate]
sd["...mlp.expert_mlps.mlp_op.down_proj.weight"]    = down_proj     # [num_experts, intermediate, hidden]
# Reference: convert_qwen3_moe_hf_to_neuron_state_dict (FP8 dequant, padding, fused QKV)
```

---

## 7. Async mode decision checklist

`async_mode` is a verified `NeuronConfig` flag.
- [ ] Goal is throughput / overlapping host+device work? → consider `async_mode=True`.
- [ ] On-device sampling enabled? Async pairs naturally with it (logits/tokens stay on device).
- [ ] Doing logit matching? Logits must come back to CPU; prefer async off (and on-device
      sampling off), or ensure `output_logits=True`.
- [ ] Debugging numerics on CPU? Disable async for deterministic, inspectable runs.
- [ ] Serving via vLLM? vLLM drives its own loop + on-device sampling by default; configure via
      `override_neuron_config`.

Rule of thumb: **enable for production serving/throughput; disable for accuracy validation and
low-level debugging.**

---

## 8. Evaluation

Accuracy helpers: `neuronx_distributed_inference.utils.accuracy`. Benchmark:
`neuronx_distributed_inference.utils.benchmark`. Unit-test helper:
`neuronx_distributed_inference.utils.testing`. Do not cross these up.

```python
from neuronx_distributed_inference.utils.accuracy import (
    generate_expected_logits, check_accuracy_logits_v2,
    check_accuracy_logits,   # DEPRECATED in favor of v2
    check_accuracy)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.testing import validate_accuracy
```

**1. Logit matching (preferred)** — VERIFIED
```python
expected = generate_expected_logits(neuron_model, input_ids, inputs_attention_mask,
                                    generation_config, num_tokens=None)
check_accuracy_logits_v2(neuron_model, expected, inputs_input_ids, inputs_attention_mask,
                         generation_config, divergence_difference_tol=0.001,
                         tol_map=None, num_tokens_to_check=None)
```
Verified signatures:
- `generate_expected_logits(neuron_model, input_ids, inputs_attention_mask, generation_config, num_tokens=None, additional_input_args=None, tokenizer=None) -> torch.Tensor`
- `check_accuracy_logits_v2(neuron_model, expected_logits, inputs_input_ids, inputs_attention_mask, generation_config, divergence_difference_tol=0.001, tol_map=None, num_tokens_to_check=None, input_start_offsets=None, additional_input_args=None, tokenizer=None)`
- `check_accuracy_logits(...)` — DEPRECATED; use v2.

Default tolerances: divergence-difference `0.001`; absolute (all top-k) `1e-5`; relative scales
`0.01` (top k=5) → `0.05` (top k=None). Tightening may cause spurious failures on large models.

> **Correction:** logit matching is NOT categorically prohibited with on-device sampling.
> `check_accuracy_logits` asserts `neuron_config.output_logits is True` when
> `on_device_sampling_config is not None`. So you CAN logit-match with on-device sampling IF
> `output_logits=True`; otherwise disable on-device sampling for this check.

**2. Token matching** — VERIFIED — use when logits aren't on CPU.
```python
check_accuracy(neuron_model, tokenizer, generation_config=None, expected_token_ids=None,
               num_tokens_to_check=None, do_sample=False, draft_model=None, prompt=None,
               input_start_offsets=None, execution_mode="config")
```
Compares generated token IDs against HF reference. Under spec decoding tolerates `(spec_len-1)`
token-count difference. Treat mismatches as informational for large models / long sequences.

**3. `validate_accuracy` (unit-test helper)** — VERIFIED
```python
validate_accuracy(neuron_model, inputs, expected_outputs=None, cpu_callable=None, assert_close_kwargs={})
```
Uses `torch_neuronx.testing.assert_close`. Supply `expected_outputs` or a `cpu_callable` baseline
(else `ValueError`). Pair with `build_module`/`build_function` from `utils.testing`.

**Benchmarking** — VERIFIED
```python
benchmark_sampling(model, draft_model=None, generation_config=None, target=None,
                   image=False, num_runs=20, benchmark_report_path=BENCHMARK_REPORT_PATH)
```
Reports per submodel (`e2e_model`, `context_encoding_model`, `token_generation_model`, plus
speculation/medusa): `latency_ms_p50/p90/p95/p99/p100`, `latency_ms_avg`, `throughput`.

**`inference_demo` CLI + `MODEL_TYPES`** — VERIFIED
Console script `inference_demo = neuronx_distributed_inference.inference_demo:main`.
Prerequisite: register the model in `MODEL_TYPES`.
```python
MODEL_TYPES["my_model"] = {"causal-lm": MyForCausalLM, ...}
```
```bash
inference_demo --model-type my_model --task-type causal-lm run \
  --check-accuracy-mode logit-matching \   # token-matching | logit-matching | skip-accuracy-check
  --num-tokens-to-check 64 \
  --expected-outputs-path /path/to/expected_outputs.pt \   # torch.save() golden file
  --benchmark \
  --on-cpu                                 # CPU run; use torchrun to simulate TP
```

---

## 9. vLLM integration

Docs: `developer_guides/vllm-user-guide.html`. Supported architectures:
`developer_guides/model-reference.html#nxdi-supported-model-architectures`.

Prereqs: (1) class named `Neuron<ModelName>ForCausalLM` extending `NeuronBaseForCausalLM`;
(2) local checkpoint dir as `model=` with weights + a `config.json` compatible with your
`InferenceConfig` (the `architectures` field drives loading); (3) set
`VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference` or vLLM won't route through NxDI.

```python
import os
os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_seqs=4, max_model_len=128,
          override_neuron_config={"enable_bucketing": False}, device="neuron",
          tensor_parallel_size=32)
outputs = llm.generate(prompts, SamplingParams(top_k=1))
```
```bash
VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' python -m vllm.entrypoints.openai.api_server \
  --model="meta-llama/Llama-3.1-8B-Instruct" --max-num-seqs=4 --device "neuron" \
  --override-neuron-config "{\"enable_bucketing\":false}"
```
Gotchas: on-device sampling is on by default in vLLM (logit matching N/A unless `output_logits=True`);
pass custom `NeuronConfig` via `override_neuron_config`; for quantization do NOT set vLLM
`--quantization neuron_quant` — configure quantization through the Neuron config instead.

---

## 10. Gotchas + per-step verification checklist

### Gotchas
- **TP divides head counts.** `tp_degree` must divide `num_attention_heads` and (the limiter for
  GQA) `num_key_value_heads`. `intermediate_size`, `hidden_size`, `vocab_size` are sharded too.
- **Pass unsharded head counts to `NeuronAttentionBase`** — it divides by `tp_degree` internally.
- **`NeuronAttentionBase.__init__` is keyword-only after `config`.**
- **`self.mlp` returns a tuple in MoE** — index `[0]`.
- **Two config layers** — architecture on `InferenceConfig`, runtime on `config.neuron_config`.
- **Decorators matter** — `convert_hf_to_neuron_state_dict`/`load_hf_model` are `@staticmethod`;
  `get_config_cls` is `@classmethod`; `update_state_dict_for_tied_weights` is `@staticmethod`.
- **Guard parallel layers** with `parallel_state.model_parallel_is_initialized()` + CPU fallback.
- **`kv_cache_quant` needs** `XLA_HANDLE_SPECIAL_SCALAR=1`.
- **Import-path traps:** `benchmark_sampling` ∈ `utils.benchmark` (NOT `utils.accuracy`);
  `check_accuracy*` ∈ `utils.accuracy` (NOT `utils.testing`).
- **Package `__init__` files are empty** — import concrete classes from the fully-qualified
  `modeling_<model>` module, not the top-level package.
- **Save quantized state dict before compile** (`save_quantized_state_dict`).
- **Debugging:** `logging.getLogger().setLevel(logging.DEBUG)`; cannot use print/log in compiled
  forward; CPU debugging support is limited.

### Per-step checklist
**Step 1 — Config:** `MyInferenceConfig(InferenceConfig)` defined; `get_required_attributes()`
lists every HF attr read; `get_neuron_config_cls()` returns the right class; HF name mismatches
handled; checkpoint `config.json` matches.
**Step 2 — Model:** `setup_attr_for_model` sets all 7 attrs; `init_model` defines
`embed_tokens`/`layers`/`norm`/`lm_head`; parallel layers guarded + CPU fallback; attention passes
unsharded head counts + a `rotary_emb` by keyword; `tp_degree | num_key_value_heads`;
`lm_head` uses `gather_output = not on_device_sampling`.
**Step 3 — Task head:** `_model_cls` set; `get_config_cls`/`load_hf_model`/
`convert_hf_to_neuron_state_dict` overridden with correct decorators; tied weights handled.
**Step 4 — Weights:** every HF key maps to an existing module name (no leftover/missing keys);
fused qkv/gate-up matches `fused_qkv`; MoE experts stacked `[E,H,2I]`/`[E,I,H]`, router renamed.
**Eval:** CPU baseline / golden logits first; logit matching only with on-device sampling off or
`output_logits=True` else token matching; `benchmark_sampling` recorded per submodel.
**vLLM:** class named `Neuron<ModelName>ForCausalLM`; local dir has weights + compatible
`config.json`; `VLLM_NEURON_FRAMEWORK` set; custom config via `override_neuron_config`.

---

## 11. Import path confidence table

| Symbol | Import path | Status |
|---|---|---|
| `NeuronConfig` | `neuronx_distributed_inference.models.config.NeuronConfig` | VERIFIED |
| `MoENeuronConfig` | `neuronx_distributed_inference.models.config.MoENeuronConfig` | VERIFIED (path) |
| `InferenceConfig` | `neuronx_distributed_inference.models.config.InferenceConfig` | VERIFIED |
| `HybridShardingConfig` | `neuronx_distributed_inference.models.config.HybridShardingConfig` | UNVERIFIED |
| `NeuronBaseModel` | `neuronx_distributed_inference.models.model_base.NeuronBaseModel` | VERIFIED |
| `NeuronBaseForCausalLM` | `neuronx_distributed_inference.models.model_base.NeuronBaseForCausalLM` | VERIFIED |
| `NeuronApplicationBase` | `neuronx_distributed_inference.models.application_base.NeuronApplicationBase` | VERIFIED |
| `NeuronAttentionBase` | `neuronx_distributed_inference.modules.attention.attention_base.NeuronAttentionBase` | UNVERIFIED |
| `ColumnParallelLinear` | `neuronx_distributed.parallel_layers.layers.ColumnParallelLinear` | VERIFIED (NxD core) |
| `RowParallelLinear` | `neuronx_distributed.parallel_layers.layers.RowParallelLinear` | VERIFIED (NxD core) |
| `ParallelEmbedding` | `neuronx_distributed.parallel_layers.layers.ParallelEmbedding` | VERIFIED (NxD core) |
| `CustomRMSNorm` / `get_rmsnorm_cls` | NxDI modules (exact path) | UNVERIFIED |
| `RotaryEmbedding` | NxDI attention modules (exact path) | UNVERIFIED |
| `initialize_moe_module` (v1) | `neuronx_distributed_inference.modules.moe` | UNVERIFIED |
| `initialize_moe_module` (v2) | `neuronx_distributed_inference.modules.moe_v2` | UNVERIFIED |
| `RouterTopK`/`ExpertMLPs(V2)`/`SharedExperts`/`MoE` | `neuronx_distributed.modules.moe` (NxD core) | UNVERIFIED |
| `RouterConfig`/`BlockwiseMatmulConfig`/`RoutedExpertsMLPOpsConfig`/`MoEFusedTKGConfig` | `neuronx_distributed.modules.moe.moe_configs` | UNVERIFIED |
| `check_accuracy_logits_v2` | `neuronx_distributed_inference.utils.accuracy.check_accuracy_logits_v2` | VERIFIED |
| `check_accuracy_logits` | `neuronx_distributed_inference.utils.accuracy.check_accuracy_logits` | VERIFIED (DEPRECATED) |
| `generate_expected_logits` | `neuronx_distributed_inference.utils.accuracy.generate_expected_logits` | VERIFIED |
| `check_accuracy` | `neuronx_distributed_inference.utils.accuracy.check_accuracy` | VERIFIED |
| `benchmark_sampling` | `neuronx_distributed_inference.utils.benchmark.benchmark_sampling` | VERIFIED |
| `validate_accuracy` | `neuronx_distributed_inference.utils.testing.validate_accuracy` | VERIFIED |
| `build_module`/`build_function` | `neuronx_distributed_inference.utils.testing` | VERIFIED |
| `inference_demo` (console script) | `neuronx_distributed_inference.inference_demo:main` | VERIFIED |
| `MODEL_TYPES` | `neuronx_distributed_inference.inference_demo` | UNVERIFIED (registration requirement confirmed) |
| `Neuron<ModelName>ForCausalLM` | `neuronx_distributed_inference.models.<model>.modeling_<model>` | VERIFIED (import from fully-qualified module, NOT top-level package) |
| `vllm.LLM` | `from vllm import LLM, SamplingParams` | VERIFIED (vLLM pkg; route via `VLLM_NEURON_FRAMEWORK`) |
