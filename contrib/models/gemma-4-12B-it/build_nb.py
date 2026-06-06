#!/usr/bin/env python
"""Generate the Gemma-4-12B-it on Neuron tutorial notebook.

Mirrors the FireRed-Image-Edit tutorial style: a self-contained, runnable
walkthrough of porting google/gemma-4-12B-it (text decoder) to AWS Neuron via
the modeling_gemma4.py in this folder, verifying numerical correctness against a
golden CPU reference, and serving it through vLLM-on-Neuron.

Run:  python build_nb.py   ->  writes gemma4_neuron_tutorial.ipynb
(requires nbformat: pip install nbformat)
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ---------------------------------------------------------------------------
md(r"""
# 在 AWS Trainium2 (Neuron) 上从零运行 google/gemma-4-12B-it（文本解码器）

本 notebook 带你**自己从头到尾跑一遍**：把
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) 的**文本解码器**
适配到 AWS NxD Inference，在 Trainium2 上编译运行，并用 **golden 参考验证数值正确性**，
最后通过 **vLLM 的 OpenAI 接口 + `curl`** 提供服务。

## 你会学到什么

1. **为什么这是个"适配截止日期之后"的模型**：`gemma-4-12B-it` 是 `Gemma4UnifiedForConditionalGeneration`
   （`model_type=gemma4_unified`），一个文本+视觉+音频统一模型；它需要较新的 `transformers`
   （>= 5.10.0.dev0）才认识。我们只移植**文本解码器**。
2. **NxDI 移植的真实形态**：`modeling_gemma4.py`（本目录的 `src/`）继承 NxDI 的基类，
   实现 config / model / task-head / 权重转换器，处理 Gemma-4 特有的那些坑（见下表）。
3. **"能编译 ≠ 正确"**：用一个隔离 venv（装认识该架构的 transformers）在 CPU 上跑真实模型，
   拿到 **golden 下一个 token 的 logits**，再和设备上的输出逐一对齐——这是定位数值 bug 的决定性手段。
4. **两种推理引擎**：原生 `inference_demo` 与 **vLLM-on-Neuron**（前端需要绕过三道门，本教程给出做法）。

## Gemma-4 文本解码器的关键特性（移植里都处理了）

| 特性 | 细节 |
|---|---|
| 异构注意力 | 5 层 sliding : 1 层 full，循环；全局层在 5,11,…,47 |
| sliding 层 | kv_heads=8, head_dim=256, window=1024, RoPE default θ=1e4 |
| global 层 | kv_heads=1 (MQA), head_dim=512, RoPE **proportional** θ=1e6, 只转前 128 维; `attention_k_eq_v`(V=K) |
| head_dim | 与 hidden/heads **解耦**（16×256≠3840）；KV cache 按层定尺寸 |
| q/k/v norm | 每头 RMSNorm，pre-RoPE；v_norm 无可学权重 |
| 注意力缩放 | 硬编码 **1.0**（不是 1/√d）→ `softmax_scale=1.0` |
| `layer_scalar` | 每层一个可学标量，**必须声明成 nn.Parameter** 否则 trace 加载器会按 ones 常量折叠 |
| logits | `final_logit_softcapping=30.0`；embedding 乘 √hidden；RMSNorm 乘原始权重（无 1+w） |

## ⏱️ 时间与资源预期

- **下载权重**：~24 GB
- **编译（TP=8，prefill + token-gen 两张图）**：约 **几分钟～十几分钟**
- **golden 参考**（CPU fp32 跑一遍 12B）：几分钟（本机内存要够，建议 ≥64GB）
- **单次推理**：~1 秒/步级别

## 前置条件

- 一台 **trn2 / trn1 / inf2** 实例（本教程在 trn2.48xlarge、TP=8 验证）
- Neuron SDK venv（含 vllm + neuronx_distributed_inference），例如 `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
- 能访问 HuggingFace 和 GitHub
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. 用正确的 Jupyter kernel 运行

本 notebook 必须用 **Neuron venv 里的 Python** 跑。先把那个 venv 注册成 kernel，
然后在右上角切换到 **"Gemma4 Neuron (venv)"**，再继续。
""")

code(r"""
import subprocess
VENV = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"   # 改成你机器上的 vllm+nxdi venv
subprocess.run([f"{VENV}/bin/python", "-m", "ipykernel", "install", "--user",
                "--name", "gemma4-neuron",
                "--display-name", "Gemma4 Neuron (venv)"], check=True)
print("kernel 已注册：gemma4-neuron")
print("现在到 Kernel → Change Kernel 选 'Gemma4 Neuron (venv)'，然后重跑后续 cell。")
""")

code(r"""
import sys
print("当前 Python:", sys.executable)
assert "vllm" in sys.executable or "nxd_inference" in sys.executable, \
    "❌ kernel 不对！切换到 Neuron venv kernel 后重跑。"
print("✅ kernel 正确")
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. 配置路径与参数

所有产物集中在一个工作目录 `WORK` 下。`SRC` 指向本目录的 `src/`（含 `modeling_gemma4.py`）。
""")

code(r"""
import os, pathlib

VENV  = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"
PY    = f"{VENV}/bin/python"
BIN   = f"{VENV}/bin"

MODEL = "google/gemma-4-12B-it"
TP    = 8

# 本教程文件夹（含 src/modeling_gemma4.py）。默认取 notebook 所在目录。
HERE  = pathlib.Path.cwd()
SRC   = str(HERE / "src")
assert os.path.exists(f"{SRC}/modeling_gemma4.py"), \
    f"找不到 {SRC}/modeling_gemma4.py —— 请在本 contrib 目录下启动 notebook"

WORK     = os.path.expanduser("~/gemma4-neuron-run")
CKPT     = f"{WORK}/model"          # 扁平化后的文本解码器 checkpoint
COMPILED = f"{WORK}/compiled_tp8"   # 编译产物
GOLDEN   = f"{WORK}/golden"         # golden 参考
os.makedirs(WORK, exist_ok=True)

# 跑 Neuron 子进程统一的环境前缀（PATH 必须含 venv/bin 否则报 libneuronpjrt-path）
ENV = (f'PATH="{BIN}:/opt/aws/neuron/bin:$PATH" PYTHONPATH="{SRC}" '
       f'PYTHONUNBUFFERED=1 NEURON_RT_NUM_CORES={TP}')

for k in ["MODEL","SRC","WORK","CKPT","COMPILED","GOLDEN","TP"]:
    print(f"{k:9}= {globals()[k]}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. 检查硬件与环境

确认 Neuron 设备在、venv 能 import torch / vllm / neuronx_distributed_inference。
""")

code(r"""
!ls /dev/neuron* 2>/dev/null | head; echo "---"
!(/opt/aws/neuron/bin/neuron-ls 2>/dev/null || neuron-ls) | head -20
""")

code(r"""
!{ENV} {PY} -c "import torch, neuronx_distributed_inference as m, os; \
print('torch', torch.__version__); \
print('nxdi', os.path.dirname(m.__file__))"
# vllm 可选（用于第 9 步 serving）
!{ENV} {PY} -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "(无 vllm —— inference_demo 路径仍可用)"
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. 下载权重 & 扁平化为文本解码器 config（~24 GB）

`gemma-4-12B-it` 的 `config.json` 是**嵌套多模态**配置（`text_config/vision_config/audio_config`）。
NxDI 的文本路径需要一个**顶层是文本字段**的 config，且 `architectures` 指向我们的类。
这一步下载权重并生成扁平化 config（保留原始多模态 config 备份）。
""")

code(r"""
import os, json, shutil, subprocess
# 下载（已存在则跳过分片）
if not os.path.exists(f"{CKPT}/config.json"):
    os.makedirs(CKPT, exist_ok=True)
    subprocess.run(f'{BIN}/huggingface-cli download {MODEL} --local-dir {CKPT}',
                   shell=True, check=True)

raw = json.load(open(f"{CKPT}/config.json"))
if "text_config" in raw:   # 还没扁平化
    json.dump(raw, open(f"{CKPT}/config.multimodal.json","w"), indent=2)  # 备份
    flat = dict(raw["text_config"])
    flat["model_type"]    = "gemma4"
    flat["architectures"] = ["NeuronGemma4ForCausalLM"]
    flat["torch_dtype"]   = "bfloat16"
    flat.setdefault("head_dim", 256)
    json.dump(flat, open(f"{CKPT}/config.json","w"), indent=2)
    print("✅ 已扁平化 config.json -> 文本解码器")
else:
    print("config.json 已是扁平文本配置（或无 text_config），跳过")
print("layers:", json.load(open(f'{CKPT}/config.json'))["num_hidden_layers"])
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. 看一眼移植代码 `modeling_gemma4.py`

它继承 NxDI 基类，处理上面表里那些 Gemma-4 特有点。这里只打印关键结构，完整文件在 `src/`。
""")

code(r"""
import pathlib
text = pathlib.Path(f"{SRC}/modeling_gemma4.py").read_text()
print("文件行数:", len(text.splitlines()))
for ln in text.splitlines():
    s = ln.strip()
    if s.startswith("class ") or "def get_config_cls" in s or "ProportionalRotaryEmbedding" in s \
       or "PerLayerHeadDimKVCacheManager" in s or "layer_scalar" in s and "register" in s:
        print("  ", ln[:100])
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. 建立 golden CPU 参考（数值正确性的标尺）🔑

本机装的 `transformers` 可能**不认识** `gemma4_unified`，无法当参考。
我们建一个**隔离 venv**，装认识该架构的 `transformers`，在 CPU fp32 上跑真实模型，
拿到一个固定 prompt 的 **golden 下一个 token + top-10 logits**。
后面用它判断设备输出对不对。

> 这一步会装 CPU 版 torch + transformers，并加载 12B 模型到内存（fp32 ~48GB，确保内存够）。
""")

code(r"""
import os, subprocess, textwrap
GV = f"{GOLDEN}/venv"
if not os.path.exists(f"{GV}/bin/python"):
    subprocess.run(f"python3 -m venv {GV}", shell=True, check=True)
    subprocess.run(f"{GV}/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu",
                   shell=True, check=True)
    # 装认识 gemma4_unified 的 transformers（若已发布版支持可换成 pip install transformers）
    subprocess.run(f"{GV}/bin/pip -q install 'git+https://github.com/huggingface/transformers.git' "
                   f"accelerate safetensors sentencepiece protobuf", shell=True, check=True)
print("golden venv:", GV)
print(subprocess.run(f'{GV}/bin/python -c "from transformers.models.auto.configuration_auto '
                     f'import CONFIG_MAPPING; print(\'gemma4_unified in CONFIG_MAPPING:\', '
                     f'\'gemma4_unified\' in CONFIG_MAPPING)"', shell=True,
                     capture_output=True, text=True).stdout)
""")

code(r"""
# 用原始多模态 config 让 HF 正确建模，跑 CPU fp32 forward 拿 golden
import os, json, textwrap
PROBE = "The capital of France is"
script = textwrap.dedent(f'''
    import torch, json, os
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    ckpt = "{CKPT}"
    # 用多模态 config 加载完整模型（取文本解码器）
    if os.path.exists(ckpt + "/config.multimodal.json"):
        import shutil
        os.makedirs("{GOLDEN}/ckpt", exist_ok=True)
        for f in os.listdir(ckpt):
            if f.endswith(".json") and f != "config.json" or f.endswith(".safetensors") \
               or "token" in f or f.endswith(".model") or f.endswith(".jinja"):
                src = os.path.join(ckpt, f); dst = os.path.join("{GOLDEN}/ckpt", f)
                if not os.path.exists(dst):
                    try: os.symlink(src, dst)
                    except FileExistsError: pass
        shutil.copy(ckpt + "/config.multimodal.json", "{GOLDEN}/ckpt/config.json")
        refdir = "{GOLDEN}/ckpt"
    else:
        refdir = ckpt
    tok = AutoTokenizer.from_pretrained(refdir)
    ids = tok("{PROBE}", return_tensors="pt", add_special_tokens=True).input_ids
    m = AutoModelForImageTextToText.from_pretrained(refdir, dtype=torch.float32, device_map="cpu")
    m.eval()
    with torch.no_grad():
        out = m(input_ids=ids)
    logits = out.logits[0, -1].float()
    top = torch.topk(logits, 10)
    golden = {{"input_ids": ids[0].tolist(),
               "next_token_id": int(logits.argmax()),
               "top10": [[int(i), float(v)] for i, v in zip(top.indices, top.values)]}}
    json.dump(golden, open("{GOLDEN}/golden.json", "w"), indent=2)
    print("golden next_token_id:", golden["next_token_id"], tok.decode([golden["next_token_id"]]))
    print("top10:", golden["top10"][:5], "...")
''')
open(f"{GOLDEN}/run_golden.py", "w").write(script)
!{GOLDEN}/venv/bin/python {GOLDEN}/run_golden.py
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. 编译到 Neuron（TP=8）🔥

把 `modeling_gemma4.py` 注册进 NxDI 的 `MODEL_TYPES`，用 `inference_demo` 编译。
注册用一个 `sitecustomize.py`（**不在顶层 import NxDI/torch_xla**，否则 libneuronpjrt-path
辅助进程会触发 fork bomb）。这一步会编 prefill + token-gen 两张图，约几分钟。
""")

code(r'''
# 写一个最小 sitecustomize：把我们的类注入 MODEL_TYPES（lazy，避免 fork bomb）
sc = f"""
import importlib.abc, importlib.machinery, importlib.util, sys
_SRC = {SRC!r}
_KEYS = ("gemma4", "modelgemma4")  # inference_demo / vLLM 可能用到的 routing key
class _L(importlib.abc.Loader):
    def __init__(s, o): s._o = o
    def create_module(s, spec): return s._o.create_module(spec) if hasattr(s._o,'create_module') else None
    def exec_module(s, mod):
        s._o.exec_module(mod)
        if hasattr(mod, "MODEL_TYPES"):
            sys.path.insert(0, _SRC)
            from modeling_gemma4 import NeuronGemma4ForCausalLM as G
            for k in _KEYS: mod.MODEL_TYPES[k] = {{"causal-lm": G}}
            sys.stderr.write("[sitecustomize] registered gemma4\\n")
class _F(importlib.abc.MetaPathFinder):
    def find_spec(s, name, path=None, target=None):
        if name != "neuronx_distributed_inference.utils.constants": return None
        if s in sys.meta_path: sys.meta_path.remove(s)
        try: spec = importlib.util.find_spec(name)
        finally:
            if s not in sys.meta_path: sys.meta_path.insert(0, s)
        if spec and spec.loader: spec.loader = _L(spec.loader); return spec
        return None
if not any(isinstance(f, _F) for f in sys.meta_path): sys.meta_path.insert(0, _F())
"""
import os
os.makedirs(COMPILED, exist_ok=True)
open(f"{WORK}/sitecustomize.py", "w").write(sc)
print("wrote", f"{WORK}/sitecustomize.py")
''')

code(r"""
import os
# 幂等：已编译就跳过
if os.path.exists(f"{COMPILED}/model.pt"):
    print("✅ 已检测到编译产物，跳过。强制重编译请删除", COMPILED)
else:
    # PYTHONPATH 同时含 WORK(sitecustomize) 和 SRC(modeling)
    cmd = (f'PATH="{BIN}:/opt/aws/neuron/bin:$PATH" '
           f'PYTHONPATH="{WORK}:{SRC}" PYTHONUNBUFFERED=1 NEURON_RT_NUM_CORES={TP} '
           f'{BIN}/inference_demo --model-type gemma4 --task-type causal-lm run '
           f'--model-path {CKPT} --compiled-model-path {COMPILED} --tp-degree {TP} '
           f'--prompt "The capital of France is" --top-k 1 --pad-token-id 0 '
           f'--max-context-length 256 --seq-len 512 --check-accuracy-mode skip-accuracy-check')
    get_ipython().system(cmd)
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. 数值正确性校验（设备 vs golden）✅

把 golden 的 **完全相同的 input_ids** 喂给设备模型，取贪心下一个 token，和 golden 比。
相等即数值正确（小的 bf16 vs fp32 near-tie 重排可接受）。

> 注意：`gemma-4-12B-it` 是指令模型，在 **raw prompt** 上输出退化（"111…"）是**正常**的，
> 判据是 logits/token 对齐,而非生成的文本好不好看。
""")

code(r"""
import json, textwrap
g = json.load(open(f"{GOLDEN}/golden.json"))
probe_ids = g["input_ids"]; golden_id = g["next_token_id"]
script = textwrap.dedent(f'''
    import sys, os, torch
    os.chdir({WORK!r}); sys.path.insert(0, {SRC!r})
    from modeling_gemma4 import NeuronGemma4ForCausalLM as M
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from neuronx_distributed_inference.utils.accuracy import get_generate_outputs_from_token_ids
    from transformers import AutoTokenizer, GenerationConfig
    nc = NeuronConfig(tp_degree={TP}, batch_size=1, max_context_length=256, seq_len=512,
                      pad_token_id=0, on_device_sampling_config=None)
    cfg = M.get_config_cls()(nc, load_config=load_pretrained_config({CKPT!r}))
    model = M({CKPT!r}, cfg); model.load({COMPILED!r})
    tok = AutoTokenizer.from_pretrained({CKPT!r}, padding_side="right"); tok.pad_token = tok.eos_token
    ids = torch.tensor({probe_ids}, dtype=torch.long)
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=1, pad_token_id=0)
    out, _ = get_generate_outputs_from_token_ids(model, [ids], tok, is_hf=False, generation_config=gc)
    seq = out[0] if not hasattr(out, "sequences") else out.sequences[0]
    dev_id = seq.tolist()[len({probe_ids})]
    print("device next-token id:", dev_id, "| golden:", {golden_id},
          "| MATCH:", dev_id == {golden_id})
    assert dev_id == {golden_id}, "数值不匹配 —— 检查移植里的 RoPE/scaling/layer_scalar 等"
    print("✅ 数值正确")
''')
open(f"{WORK}/verify_numeric.py","w").write(script)
!{ENV.replace(SRC, WORK+':'+SRC)} {PY} {WORK}/verify_numeric.py
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. 端到端聊天（人类可读证明）

用 chat 模板问问题，贪心生成。golden 说这条路径会答出 "Paris."。
""")

code(r"""
import textwrap
script = textwrap.dedent(f'''
    import sys, os, torch
    os.chdir({WORK!r}); sys.path.insert(0, {SRC!r})
    from modeling_gemma4 import NeuronGemma4ForCausalLM as M
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from neuronx_distributed_inference.utils.accuracy import get_generate_outputs_from_token_ids
    from transformers import AutoTokenizer, GenerationConfig
    nc = NeuronConfig(tp_degree={TP}, batch_size=1, max_context_length=256, seq_len=512,
                      pad_token_id=0, on_device_sampling_config=None)
    cfg = M.get_config_cls()(nc, load_config=load_pretrained_config({CKPT!r}))
    model = M({CKPT!r}, cfg); model.load({COMPILED!r})
    tok = AutoTokenizer.from_pretrained({CKPT!r}, padding_side="right"); tok.pad_token = tok.eos_token
    for q in ["What is the capital of France? Answer in one short sentence.",
              "What is 17 * 24? Reply with just the number."]:
        ids = tok.apply_chat_template([{{"role":"user","content":q}}],
                                      add_generation_prompt=True, return_tensors="pt")[0].to(torch.long)
        gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=40,
                              pad_token_id=0, eos_token_id=[1,106])
        out, _ = get_generate_outputs_from_token_ids(model, [ids], tok, is_hf=False, generation_config=gc)
        seq = out[0] if not hasattr(out,"sequences") else out.sequences[0]
        print("Q:", q)
        print("A:", repr(tok.decode(seq.tolist()[ids.shape[0]:], skip_special_tokens=True)))
''')
open(f"{WORK}/chat.py","w").write(script)
!{ENV.replace(SRC, WORK+':'+SRC)} {PY} {WORK}/chat.py
""")

# ---------------------------------------------------------------------------
md(r"""
## 9.（可选）用 vLLM-on-Neuron 起 OpenAI 服务

`vllm serve google/gemma-4-12B-it` 在 Neuron 上**不能直接跑**：vLLM 前端不认识 `gemma4_unified`。
需要在 `sitecustomize.py` 里过三道门（都不在顶层 import NxDI）：
1. `AutoConfig.register(...)` 让前端 config 解析通过；
2. `ModelRegistry.register_model(...)` 注册架构别名（仅前端元数据，实际走 Neuron 路径）；
3. 清掉 vLLM `patch_rope_parameters` 会注入进嵌套 `rope_parameters` 的 scalar rope 字段；
4. 把我们的类注入 NxDI `MODEL_TYPES`（key 由 `architectures[0]` 派生）。

完整 sitecustomize 见本仓库 `nxdi-workflows` 的说明 / contrib README。启动命令：

```bash
PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH" PYTHONPATH="<dir with sitecustomize + src>:$PYTHONPATH" \
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_RT_NUM_CORES=8 \
python -m vllm.entrypoints.openai.api_server \
  --model="$CKPT" --served-model-name gemma4 \
  --tensor-parallel-size 8 --max-num-seqs 4 --max-model-len 4096 \
  --no-enable-prefix-caching \
  --additional-config '{"override_neuron_config":{"enable_bucketing":false}}'
```

服务起来后用 curl 测：
""")

md(r"""
```bash
# 健康检查 / 模型列表
curl -s http://localhost:8000/v1/models

# 聊天
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":40,"temperature":0}'
# -> "The capital of France is Paris."
```

> 关键 flag：vllm 0.16 没有 `--device`（neuron 插件自动激活）；`--no-enable-prefix-caching` 必需；
> `override_neuron_config` 要放在 `--additional-config` 里（没有 `--override-neuron-config` 这个 flag）。
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. 清理（释放 Neuron 设备）
""")

code(r"""
import subprocess
for pat in ["vllm.entrypoints", "inference_demo", "libneuronpjrt-path"]:
    subprocess.run(["pkill", "-9", "-f", pat])
print("已尝试清理。设备占用：")
!(/opt/aws/neuron/bin/neuron-ls 2>/dev/null || neuron-ls) | grep -iE "python|EngineCore" || echo "设备空闲 ✅"
""")

# ---------------------------------------------------------------------------
md(r"""
## 附录：移植中真正踩到的坑（数值 bug，不是崩溃）

这些"能编译、能跑、但输出全是 `<pad>`"的 bug，靠 golden 参考逐项对齐才定位到：

- **`layer_scalar` 用 `register_buffer` → 真凶**：NxD trace 加载器只加载 `named_parameters()`，
  buffer 被按初值 `ones` 常量折叠，真权重(~0.005–0.36)没加载 → 残差爆炸 → logits 饱和 30·tanh → `<pad>`。
  改成 `nn.Parameter(requires_grad=False)` 解决。
- **`softmax_scale`**：Gemma-4 硬编码 1.0（不是 1/√d），一度被"修"成 √d 反而是倒退。
- **proportional RoPE**：global 层只转前 128 维，且 inv_freq 分母是 head_dim=512 后零填充——
  stock `RotaryEmbedding(128)` 是错的。
- **`attention_k_eq_v`**：global 层无 v_proj，转换器需合成 `v_proj := k_proj`。

教训：对适配截止日期之后的新架构,**"不崩" ≠ "对"**;必须有 golden 参考做数值标尺。
""")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"name": "gemma4-neuron",
                          "display_name": "Gemma4 Neuron (venv)", "language": "python"}
nb.metadata.language_info = {"name": "python"}

out = "gemma4_neuron_tutorial.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
