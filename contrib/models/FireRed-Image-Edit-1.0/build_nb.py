#!/usr/bin/env python
"""Generate the FireRed-Image-Edit on Neuron tutorial notebook."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------------
md(r"""
# 在 AWS Trainium2 (Neuron) 上从零编译并运行 FireRed-Image-Edit

本 notebook 带你**自己从头到尾跑一遍**：把图像编辑扩散模型
[`FireRedTeam/FireRed-Image-Edit-1.0`](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0)
编译并运行在 AWS Trainium2（Neuron）上，最后用 HTTP + `curl` 调用它做图。

## 你会学到什么

1. **为什么不能直接 `pip install diffusers` 就在 Neuron 上跑** —— 上游 diffusers 只支持 CUDA/CPU/MPS，没有 Neuron 后端；官方 `optimum-neuron` 也不支持 QwenImageEdit 架构。
2. **真正可行的路径**：用社区 contrib 移植代码，它**复用官方 diffusers 的模型定义/pipeline/权重加载**，只把官方的 transformer / VAE 模块用 Neuron 的 `ModelBuilder` **trace + 编译**成计算图（NEFF），再塞回官方 pipeline。
3. **完整动手流程**：取代码 → 重定向硬编码路径 → 指向 FireRed 权重 → 装依赖 → 下权重 → **编译 5 个组件** → 推理出图 → 起服务用 curl 调。

## 这个模型的架构（4 个组件，都要各自编译）

| 组件 | 模型 | Neuron 并行 |
|---|---|---|
| 文本/视觉编码器 | Qwen2.5-VL（ViT 32 层 + LM 28 层） | TP=4 |
| 扩散主干 | `QwenImageTransformer2DModel`（60 层 DiT，~20B） | TP=4, DP=2（v3_cfg） |
| VAE | 3D AutoencoderKL（causal） | 单设备，分块 |
| 调度器 | FlowMatchEuler | CPU |

## ⏱️ 时间与资源预期（请先有心理准备）

- **下载权重**：~60 GB，十几分钟～半小时（取决于网络）
- **编译 5 个组件**：约 **30–45 分钟**（这是 Neuron 编译器把图编成 NEFF，最耗时的一步）
- **单次推理**：约 **45 秒 / 张**（1024×1024，40 步）
- **磁盘**：权重 + 编译产物合计约 **120 GB**，确保工作目录所在盘有足够空间

## 前置条件

- 一台 **trn2 / trn1 / inf2** 实例（本教程在 `trn2.48xlarge`、64 NeuronCore 上验证）
- 已安装 Neuron SDK 的 venv：`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- 能访问 HuggingFace（下载权重）和 GitHub（取 contrib 代码）
""")

# ----------------------------------------------------------------------------
md(r"""
## 0. 用正确的 Jupyter kernel 运行本 notebook

本 notebook 必须用 **Neuron venv 里的 Python** 来跑（否则 `import torch` / `diffusers` 会失败）。
先把那个 venv 注册成一个 Jupyter kernel，然后在 Jupyter 右上角把 kernel 切换为
**“FireRed Neuron (venv)”**，再继续往下。

> 下面这个 cell 用 venv 的 python 注册 kernel。注册完，到菜单 *Kernel → Change Kernel* 选它。
""")

code(r"""
import subprocess, sys
VENV = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
# 注册 kernel（幂等，重复运行无害）
subprocess.run([f"{VENV}/bin/python", "-m", "ipykernel", "install", "--user",
                "--name", "firered-neuron",
                "--display-name", "FireRed Neuron (venv)"], check=True)
print("kernel 已注册：firered-neuron")
print("现在请到 Kernel → Change Kernel 选 'FireRed Neuron (venv)'，然后重跑后续 cell。")
""")

code(r"""
# 检查当前 kernel 是否就是 Neuron venv。如果这里打印的不是 .../nxd_inference/bin/python，
# 说明你还没切 kernel —— 回到上一步切换后再继续。
import sys
print("当前 Python:", sys.executable)
assert "nxd_inference" in sys.executable, \
    "❌ kernel 不对！请切换到 'FireRed Neuron (venv)' kernel 后重跑。"
print("✅ kernel 正确")
""")

# ----------------------------------------------------------------------------
md(r"""
## 1. 配置路径与参数

所有产物都集中在一个工作目录 `WORK` 下，方便你管理和清理。

- `MODEL`：我们要适配的目标模型（FireRed）。
- `BASE`：contrib 移植代码原本针对的基准模型（`Qwen/Qwen-Image-Edit-2509`）。FireRed 与它**同架构、只是微调权重**，所以同一套编译流程可直接复用。
- `REPO/REF/SUBDIR`：社区 contrib 代码的位置（`whn09` 的 `contrib/diffusion-models` 分支，含编译器加速补丁 PR #4）。
- `NVME`：contrib 代码把大文件路径硬编码成了 `/opt/dlami/nvme`；本机没有这个盘，我们用 `WORK/nvme` 替代。
""")

code(r"""
import os, pathlib

VENV   = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PY     = f"{VENV}/bin/python"

MODEL  = "FireRedTeam/FireRed-Image-Edit-1.0"   # 目标模型
BASE   = "Qwen/Qwen-Image-Edit-2509"            # contrib 基准（同架构）

REPO   = "whn09/neuronx-distributed-inference"
REF    = "contrib/diffusion-models"             # 含 PR#4 编译器加速 flag 的分支
SUBDIR = "contrib/models/Qwen-Image-Edit"

WORK   = os.path.expanduser("~/firered-neuron-tutorial/run")  # 一切产物的根目录
SRC    = f"{WORK}/src"                           # contrib 源码
NVME   = f"{WORK}/nvme"                           # 替代 /opt/dlami/nvme
COMPILED = f"{NVME}/compiled_models_qwen_image_edit"
HF_CACHE = f"{NVME}/qwen_image_edit_hf_cache_dir"
MODE   = "v3_cfg"                                 # 推荐路径：CFG 并行 + NKI Flash，最快

os.makedirs(WORK, exist_ok=True)
os.makedirs(NVME, exist_ok=True)

# 运行 Neuron 子进程时统一用的环境前缀（注意：$PATH 由 shell 展开，不要写成 ${...}）
ENV = (f'PATH="{VENV}/bin:/opt/aws/neuron/bin:$PATH" '
       f'PYTHONPATH="{SRC}" '
       f'QIE_MODEL_PATH="{MODEL}" '
       f'NEURON_RT_NUM_CORES=8 PYTHONUNBUFFERED=1')

def _safe_contains(path, needle):
    try:
        return needle in path.read_text()
    except (UnicodeDecodeError, IsADirectoryError, OSError):
        return False

for k in ["WORK","SRC","NVME","COMPILED","HF_CACHE"]:
    print(f"{k:9}= {globals()[k]}")
""")

# ----------------------------------------------------------------------------
md(r"""
## 2. 检查 Neuron 硬件与环境

确认设备在、venv 能 `import torch`、磁盘空间够。
""")

code(r"""
# Neuron 设备列表（应能看到多张 NEURON DEVICE）
!neuron-ls 2>/dev/null || /opt/aws/neuron/bin/neuron-ls
""")

code(r"""
# 磁盘空间（确保 WORK 所在盘有 >120GB 空闲）
!df -h {WORK} /opt/dlami/nvme 2>/dev/null
# venv 能否 import torch
!{ENV} {PY} -c "import torch; print('torch', torch.__version__)"
""")

# ----------------------------------------------------------------------------
md(r"""
## 3. 释放被占用的 Neuron 设备

Neuron 设备是独占的。如果之前有 vLLM、别的编译任务、或本教程起过的服务在跑，会**占着设备导致编译失败**。
这一步把它们清掉（如果没有，命令无害）。
""")

code(r"""
import subprocess
# 杀掉常见的占用进程
for pat in ["vllm.entrypoints", "serve_firered.py", "run_qwen_image_edit", "compile.sh"]:
    subprocess.run(["pkill", "-9", "-f", pat])
print("已尝试清理占用进程。下面复查设备占用：")
!(neuron-ls 2>/dev/null || /opt/aws/neuron/bin/neuron-ls) | grep -E "EngineCore|python|PID" || echo "无占用，设备空闲 ✅"
""")

# ----------------------------------------------------------------------------
md(r"""
## 4. 获取社区 contrib 移植代码

这一步把 `whn09/neuronx-distributed-inference` 仓库 `contrib/diffusion-models` 分支里的
`Qwen-Image-Edit` 子目录取下来。用 **sparse checkout** 只拉这一个子目录，避免下整个大仓库。

> 关键认知：打开 `src/compile_transformer_v3_cfg.py` 你会看到
> `from diffusers import QwenImageEditPlusPipeline` 和
> `import diffusers.models.transformers.transformer_qwenimage` ——
> **模型定义来自官方 diffusers**，contrib 只是把它用 Neuron `ModelBuilder` 编译。这就是“在 Neuron 上用官方 diffusers”的真实形态。
""")

code(r"""
import shutil, os
# 干净重来
shutil.rmtree(f"{WORK}/_repo", ignore_errors=True)
!cd {WORK} && git clone --depth 1 --filter=blob:none --sparse --branch {REF} https://github.com/{REPO}.git _repo
!cd {WORK}/_repo && git sparse-checkout set {SUBDIR}
# 把子目录内容（src/、requirements.txt、assets/ 等）平铺到 WORK
!cp -r {WORK}/_repo/{SUBDIR}/* {WORK}/
print("\nsrc/ 内容：")
!ls {SRC}
""")

# ----------------------------------------------------------------------------
md(r"""
## 5. 把硬编码的 `/opt/dlami/nvme` 重定向到本机

contrib 脚本里把编译产物 / 权重缓存路径写死成了 `/opt/dlami/nvme/...`。
本机没有这个盘，我们用 `sed` 把所有出现替换成我们的 `WORK/nvme`。
""")

code(r"""
# 用纯 Python 做文本替换（比 shell 的 grep|sed 管道更稳，不受 !-magic 插值影响）
import pathlib
HARDCODED = "/opt/dlami/nvme"
changed = 0
for p in pathlib.Path(SRC).rglob("*"):
    if not p.is_file():
        continue
    try:
        txt = p.read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        continue  # 跳过二进制文件
    if HARDCODED in txt:
        p.write_text(txt.replace(HARDCODED, NVME))
        changed += 1
print(f"重定向了 {changed} 个文件：{HARDCODED} -> {NVME}")
# 校验：应无残留
remain = [str(p) for p in pathlib.Path(SRC).rglob("*")
          if p.is_file() and _safe_contains(p, HARDCODED)]
print("残留：", remain if remain else "✅ 无，已全部重定向")
assert not remain, f"仍有文件含 {HARDCODED}: {remain}"
""")

# ----------------------------------------------------------------------------
md(r"""
## 6. 指向 FireRed 权重

两个脚本里都把模型 id **写死**成了基准模型 `Qwen/Qwen-Image-Edit-2509`，
需要改成 FireRed：

- `cache_hf_model.py`（下载脚本）的 `MODEL_ID`
- `run_qwen_image_edit.py`（推理脚本）的 `MODEL_ID`（⚠️ 这个分支的版本**不读** `QIE_MODEL_PATH` 环境变量，顶层 `MODEL_ID` 是硬编码，必须直接改）

为什么 FireRed 能直接套用 Qwen-Image-Edit 的编译流程？因为它俩的
`transformer/config.json`、`vae/config.json` **逐字节相同**，text_encoder 同架构 ——
FireRed 只是同架构的**微调权重变体**，编译出的计算图形状完全适用。
""")

code(r"""
# 把两个脚本里硬编码的 MODEL_ID 都改成 FireRed（纯 Python，逐行替换更稳）
import pathlib, re
def set_model_id(filename, model_id):
    p = pathlib.Path(SRC) / filename
    lines = p.read_text().splitlines()
    hit = False
    for i, ln in enumerate(lines):
        # 匹配 MODEL_ID = "..."（允许行内有 os.environ.get(..., "默认") 的写法也覆盖默认值所在行）
        if re.match(r"^\s*MODEL_ID\s*=", ln):
            lines[i] = f'MODEL_ID = "{model_id}"'
            hit = True
    p.write_text("\n".join(lines) + "\n")
    assert hit, f"{filename} 里没找到 MODEL_ID 赋值行"
    cur = [ln for ln in p.read_text().splitlines() if "MODEL_ID" in ln][0]
    print(f"  {filename}: {cur.strip()}")

print("把 MODEL_ID 指向", MODEL)
set_model_id("cache_hf_model.py", MODEL)        # 下载脚本
set_model_id("run_qwen_image_edit.py", MODEL)   # 推理脚本（关键：本分支不读 QIE_MODEL_PATH）
print("✅ 两个脚本都已指向 FireRed")
""")

# ----------------------------------------------------------------------------
md(r"""
## 7. 安装依赖（官方 diffusers，从 git 装）

`requirements.txt` 里 diffusers 是从 GitHub 主分支装的 —— 因为 `QwenImageEditPlusPipeline`
是较新的 pipeline，旧版 PyPI diffusers 没有。这一步几分钟。
""")

code(r"""
!{ENV} {PY} -m pip install -q -r {WORK}/requirements.txt
# 验证官方 pipeline 类能导入（这就是“官方 diffusers”可用的证明）
!{ENV} {PY} -c "import diffusers; from diffusers import QwenImageEditPlusPipeline; print('✅ diffusers', diffusers.__version__, '+ QwenImageEditPlusPipeline OK')"
""")

# ----------------------------------------------------------------------------
md(r"""
## 8. 下载 FireRed 权重（~60 GB）

用官方 `QwenImageEditPlusPipeline.from_pretrained` 把权重缓存到本地 `HF_CACHE`。
**这步会下 ~60GB，耐心等。** 进度会实时打印。

> 💡 省流提示：如果你之前已经下过这个模型的 HF 缓存，可以把已有缓存目录 symlink 到 `HF_CACHE`
> 跳过下载，例如：
> `!ln -sfn /path/to/existing/qwen_image_edit_hf_cache_dir {HF_CACHE}`
""")

code(r"""
# 下载（已存在则会跳过已有分片）
!cd {SRC} && {ENV} {PY} {SRC}/cache_hf_model.py
print("\n缓存目录内容：")
!ls {HF_CACHE} 2>/dev/null
""")

# ----------------------------------------------------------------------------
md(r"""
## 9. 编译所有组件到 Neuron 🔥（核心步骤，~30–45 分钟）

`compile.sh v3_cfg` 会依次编译 5 个组件：`vae_encoder`、`vae_decoder`、
`transformer_v3_cfg`、`language_model_v3`、`vision_encoder_v3`。

底层做的事：用官方 diffusers 的模块构造模型 → `ModelBuilder.trace()` 描出计算图 →
`ModelBuilder.compile()` 调 Neuron 编译器生成 NEFF（带 `--lnc=2`、ccop compute-overlap
等加速 flag，来自 PR #4）→ 保存成 `nxd_model.pt`。

**这个 cell 会运行很久，输出很多，请让它跑完不要打断。** 跑完后下一步验证产物。
""")

code(r"""
import os
# 幂等守卫：5 个组件都已存在就跳过重编译（省时；想强制重编译就删掉 COMPILED 目录）
_need = ["transformer_v3_cfg","language_model_v3","vision_encoder_v3","vae_encoder","vae_decoder"]
_have = all(os.path.isdir(f"{COMPILED}/{c}") for c in _need)
if _have:
    print("✅ 检测到已编译产物，跳过重编译。若要从头编译，先删除：", COMPILED)
else:
    # 长任务：约 30–45 分钟。输出会实时流式显示。
    get_ipython().system('cd {SRC} && {ENV} bash {SRC}/compile.sh {MODE}')
""")

# ----------------------------------------------------------------------------
md(r"""
## 10. 验证编译产物

确认 5 个组件子目录都生成了，且每个里有编译好的 `nxd_model.pt`（或 `model.pt`）。
""")

code(r"""
print("编译产物目录：")
!ls -la {COMPILED}
print("\n关键组件检查：")
import os
for c in ["transformer_v3_cfg", "language_model_v3", "vision_encoder_v3", "vae_encoder", "vae_decoder"]:
    p = f"{COMPILED}/{c}"
    ok = os.path.isdir(p)
    print(f"  {'✅' if ok else '❌'} {c}", "->", (os.listdir(p) if ok else "缺失"))
""")

# ----------------------------------------------------------------------------
md(r"""
## 11. 跑一次推理（命令行方式）

用 contrib 自带的 `run_qwen_image_edit.py` 跑一张图。它会：
加载官方 pipeline → 把上一步编译好的各组件装载到 Neuron → 跑 40 步去噪 → 存图。

第一次会花点时间加载编译图到设备；之后单纯推理约 45 秒。
""")

code(r"""
# 准备一张输入图（用教程自带的样图；你也可以换成自己的任意图片路径）
import shutil, os
SAMPLE = os.path.expanduser("~/firered-neuron-tutorial/assets/input.png")
INPUT  = f"{WORK}/input.png"
if os.path.exists(SAMPLE):
    shutil.copy(SAMPLE, INPUT)
else:
    # 没有样图就生成一张占位图
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1024,1024), (135,206,235))
    ImageDraw.Draw(im).rectangle([300,600,724,900], fill=(60,140,60))
    im.save(INPUT)
print("输入图：", INPUT)
""")

code(r"""
import os
PROMPT = "Add a small red hot-air balloon in the sky."
OUTPUT = f"{WORK}/output_edited.png"
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)   # 删掉旧图，确保下面断言验证的是本次产出
!cd {SRC} && {ENV} {PY} {SRC}/run_qwen_image_edit.py \
    --images {INPUT} \
    --prompt "{PROMPT}" \
    --use_v3_cfg \
    --num_inference_steps 40 \
    --true_cfg_scale 4.0 \
    --output {OUTPUT}
# 断言真出图了（!-magic 不会因脚本失败而报错，这里显式校验）
assert os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) > 10000, \
    "❌ 推理未产出有效图片，请往上看脚本输出里的报错"
print("\n✅ 输出图：", OUTPUT, os.path.getsize(OUTPUT), "bytes")
""")

code(r"""
# 并排看输入 vs 输出
from PIL import Image
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].imshow(Image.open(INPUT));  ax[0].set_title("输入");  ax[0].axis("off")
ax[1].imshow(Image.open(OUTPUT)); ax[1].set_title("编辑结果"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
md(r"""
## 12. （可选）起一个常驻 HTTP 服务，用 `curl` 调用

每次跑 `run_qwen_image_edit.py` 都要重新把编译图加载到设备（几分钟）。
更实用的方式：起一个**常驻服务**，加载一次，之后用 `curl` 反复发请求。

`serve_firered.py` 已放在教程文件夹里，它复用上面所有加载逻辑，包成一个 HTTP 服务。
下面用 `subprocess.Popen` 在后台启动它（kernel 不退出，进程就一直活着）。
""")

code(r"""
import subprocess, os, time, json, urllib.request

SERVE = os.path.expanduser("~/firered-neuron-tutorial/serve_firered.py")
env = dict(os.environ)
env["PATH"] = f"{VENV}/bin:/opt/aws/neuron/bin:" + env.get("PATH","")
env["PYTHONPATH"] = SRC
env["QIE_MODEL_PATH"] = MODEL
env["NEURON_RT_NUM_CORES"] = "8"
env["PYTHONUNBUFFERED"] = "1"

logf = open(f"{WORK}/serve.log", "w")
proc = subprocess.Popen([PY, SERVE, "--port", "8000", "--compiled_models_dir", COMPILED],
                        cwd=SRC, env=env, stdout=logf, stderr=subprocess.STDOUT)
print(f"server 启动中 (pid={proc.pid})，正在把编译图加载到设备（约 1 分钟）...")
""")

code(r"""
# 轮询 /health 直到 ready
import time, json, urllib.request
for i in range(30):
    time.sleep(10)
    try:
        h = json.load(urllib.request.urlopen("http://localhost:8000/health", timeout=5))
    except Exception:
        h = {"status": "starting"}
    print(f"[{(i+1)*10}s] {h}")
    if h.get("status") == "ready":
        print("✅ 服务就绪"); break
    if h.get("status") == "error":
        print("❌ 加载失败，看日志：", f"{WORK}/serve.log"); break
""")

md(r"""
### 在终端里用 curl 调用

服务就绪后，**打开一个终端**复制下面的命令（把 `your.png` 换成你的图）。
输出固定 1024×1024，每次约 45 秒。

```bash
# 健康检查
curl -s http://localhost:8000/health

# 方式一：上传本地图片（multipart）
curl -s -X POST http://localhost:8000/edit \
  -F 'prompt=Add a small red hot-air balloon in the sky.' \
  -F 'image=@your.png' \
  -o edited.png

# 方式二：JSON（image_url 支持本地绝对路径或 http(s) URL），并打印耗时
curl -s -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Make it night with a starry sky.",
       "image_url":"REPLACE_WITH_ABS_PATH/input.png",
       "num_inference_steps":40, "true_cfg_scale":4.0, "seed":42}' \
  -D - -o edited2.png | grep -i X-Inference-Seconds
```

可调字段：`prompt`(必填)、`negative_prompt`、`num_inference_steps`(默认40)、
`true_cfg_scale`(默认4.0)、`seed`(默认42)。
""")

code(r"""
# 也可以直接在 notebook 里用 Python 调服务做一张图，验证 HTTP 链路
import urllib.request, json
body = json.dumps({"prompt":"Add a small red hot-air balloon in the sky.",
                   "image_url": INPUT, "num_inference_steps":40,
                   "true_cfg_scale":4.0, "seed":42}).encode()
req = urllib.request.Request("http://localhost:8000/edit", data=body,
                             headers={"Content-Type":"application/json"})
png = urllib.request.urlopen(req, timeout=600).read()
open(f"{WORK}/edited_via_http.png","wb").write(png)
print("HTTP 出图：", f"{WORK}/edited_via_http.png", len(png), "bytes")
from PIL import Image
display(Image.open(f"{WORK}/edited_via_http.png"))
""")

# ----------------------------------------------------------------------------
md(r"""
## 13. 清理（释放 Neuron 设备）

用完记得停掉服务，释放设备给别的任务。
""")

code(r"""
import subprocess
try:
    proc.terminate()
except Exception:
    pass
subprocess.run(["pkill", "-9", "-f", "serve_firered.py"])
print("已停止服务，释放设备。")
!(neuron-ls 2>/dev/null || /opt/aws/neuron/bin/neuron-ls) | grep -E "python|EngineCore" || echo "设备已空闲 ✅"
""")

# ----------------------------------------------------------------------------
md(r"""
## 附录 A：文件清单

```
~/firered-neuron-tutorial/
├── firered_neuron_tutorial.ipynb   # 本 notebook
├── serve_firered.py                # 常驻 HTTP 服务（复用 contrib 加载逻辑）
├── assets/input.png                # 示例输入图
├── build_nb.py                     # 生成本 notebook 的脚本
└── run/                            # ← 你跑出来的所有产物
    ├── src/                        # contrib 源码（已重定向路径、指向 FireRed）
    ├── requirements.txt
    ├── input.png / output_edited.png / edited_via_http.png
    └── nvme/
        ├── qwen_image_edit_hf_cache_dir/        # ~60GB 权重
        └── compiled_models_qwen_image_edit/     # ~59GB 编译产物（NEFF）
            ├── transformer_v3_cfg/  language_model_v3/  vision_encoder_v3/
            └── vae_encoder/  vae_decoder/  quant_conv/  post_quant_conv/
```

## 附录 B：常见问题 / 排错

- **`FileNotFoundError: 'libneuronpjrt-path'`** → PATH 没带 venv 的 bin。确保命令前缀里有
  `PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"`（本 notebook 的 `ENV` 已包含）。
- **编译/推理报设备被占用** → 回到第 3 步清理占用进程；Neuron 设备是独占的。
- **`QwenImageEditPlusPipeline` 导入失败** → diffusers 版本太旧。第 7 步必须从 git 装最新 diffusers。
- **改了 `height/width` 后形状报错** → 计算图是按编译时的尺寸（1024×1024）固化的；要换尺寸必须用对应尺寸**重新编译**。
- **磁盘写满** → 权重 + 编译产物约 120GB；换一个大盘做 `WORK`。
- **为什么不用纯官方 diffusers / optimum-neuron** → 官方 diffusers 无 Neuron 后端；optimum-neuron 不支持 QwenImageEdit 架构。contrib 这条“官方 diffusers 模块 + Neuron ModelBuilder 编译”是当前唯一可行路径。

## 附录 C：换一个同架构模型？

任何 `QwenImageEditPlusPipeline` 架构的模型（transformer/vae config 与
`Qwen-Image-Edit-2509` 一致）都能用同样流程：只需把第 1 步的 `MODEL` 换掉、
第 6 步的 `cache_hf_model.py` MODEL_ID 同步改，再从第 7 步往下重跑即可。
""")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"name": "firered-neuron",
                          "display_name": "FireRed Neuron (venv)",
                          "language": "python"}
nb.metadata.language_info = {"name": "python"}

out = "/home/ubuntu/firered-neuron-tutorial/firered_neuron_tutorial.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
