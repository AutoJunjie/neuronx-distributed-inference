export const meta = {
  name: 'nxdi-onboard-diffusion',
  description: 'Onboard a diffusion image-edit pipeline (QwenImageEdit-family: a Qwen2.5-VL text encoder + QwenImageTransformer2DModel DiT + AutoencoderKLQwenImage VAE) to run on AWS Neuron, by adapting an already-validated community contrib port: profile the target pipeline, diff its component configs against the contrib base to classify adaptability (weight-variant vs needs-code), materialize + repath the contrib src for THIS host, compile every component on Neuron, then ACTUALLY RUN an image edit and validate the produced PNG — with a source-grounded repair loop.',
  whenToUse: 'When you want to port a diffusers image-edit pipeline (e.g. FireRedTeam/FireRed-Image-Edit-1.0, Qwen/Qwen-Image-Edit-2509) to Trn2/Trn1 Neuron. Pass the HF model id as args, or an object {model, base_contrib, contrib_repo, contrib_ref, contrib_subdir, mode, prompt, image_url, work_root, tp, max_repair_rounds, verify, free_devices}. This is the diffusion counterpart to nxdi-onboard-model (which is causal-LM/vLLM only). Set verify:false to stop after compile.',
  phases: [
    { title: 'Setup', detail: 'resolve venv/paths, detect Neuron devices, pick a writable work root (the contrib hardcodes /opt/dlami/nvme), and free any process holding the devices' },
    { title: 'Profile', detail: 'read the target pipeline model_index.json + component configs; classify as QwenImageEdit-family or not' },
    { title: 'Ground', detail: 'fetch the contrib port, diff target vs base component configs, classify adaptability + required patches', model: 'opus' },
    { title: 'Materialize', detail: 'copy the contrib src locally, repath /opt/dlami/nvme -> work root, point at the target weights, pip install into the venv' },
    { title: 'Compile', detail: 'run the contrib compile pipeline (text encoder + DiT transformer + VAE) on Neuron; long-running, polled', model: 'opus' },
    { title: 'Verify', detail: 'run an actual image edit (input image + prompt) and validate the produced PNG is a real non-trivial image', model: 'opus' },
    { title: 'Repair', detail: 'on failure, read the contrib source + real traceback, patch the local copy, and re-run (loop)', model: 'opus' },
  ],
}

// ===== Reference: the community contrib port we adapt =====
// A validated NxDI/Neuron port of the QwenImageEdit diffusion pipeline.
const DEF_REPO    = 'qingzwang/neuronx-distributed-inference'
const DEF_REF     = 'contrib/Qwen-Image-Edit-Optimize'
const DEF_SUBDIR  = 'contrib/models/Qwen-Image-Edit'
const DEF_BASE    = 'Qwen/Qwen-Image-Edit-2509'   // the model the contrib port was built+validated on
// A neutral sample input photo for the edit smoke-test if the user gives none.
const DEF_IMAGE   = 'https://raw.githubusercontent.com/qingzwang/neuronx-distributed-inference/contrib/Qwen-Image-Edit-Optimize/contrib/models/Qwen-Image-Edit/assets/image1.png'
const DEF_PROMPT  = 'Add a small red hot-air balloon in the sky.'

// ===== Normalize args =====
const spec = (typeof args === 'string') ? { model: args }
  : (args && typeof args === 'object') ? args : {}
if (!spec.model) { log('No target model id given — pass args="<hf/model-id>" or {model:...}.'); return { error: 'no model' } }
const MODEL        = spec.model
const BASE         = spec.base_contrib || DEF_BASE
const CONTRIB_REPO = spec.contrib_repo || DEF_REPO
const CONTRIB_REF  = spec.contrib_ref  || DEF_REF
const CONTRIB_SUB  = spec.contrib_subdir || DEF_SUBDIR
const MODE         = spec.mode   || 'v3_cfg'           // contrib's recommended/fastest path
const PROMPT       = spec.prompt || DEF_PROMPT
const IMAGE_URL    = spec.image_url || DEF_IMAGE
const TP           = Number.isInteger(spec.tp) ? spec.tp : 0  // 0 = let the contrib mode decide (v3 uses TP=4/world=8)
const WANT_VERIFY  = spec.verify !== false
const FREE_DEVICES = spec.free_devices !== false       // default: reclaim the Neuron devices for the compile
const MAX_REPAIR   = Number.isInteger(spec.max_repair_rounds) ? spec.max_repair_rounds : 2
const slug = String(MODEL).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48) || 'diffusion'

const CONTRIB_RAW = `https://raw.githubusercontent.com/${CONTRIB_REPO}/${CONTRIB_REF}/${CONTRIB_SUB}`
const CONTRIB_API = `https://api.github.com/repos/${CONTRIB_REPO}/contents/${CONTRIB_SUB}?ref=${CONTRIB_REF}`
const HF = (id, path) => `https://huggingface.co/${id}/resolve/main/${path}`

log(`Diffusion onboarding: ${MODEL}  (adapting contrib ${CONTRIB_REPO}@${CONTRIB_REF}/${CONTRIB_SUB}, base=${BASE}, mode=${MODE}${WANT_VERIFY ? ', verify-on-hardware' : ', compile-only'})`)

// =========================================================================
// Phase 0: Setup — venv, devices, writable work root, free the hardware
// =========================================================================
phase('Setup')

const SETUP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['home', 'venv_python', 'venv_bin', 'neuron_device_present', 'num_devices', 'device_holder_pids', 'freed', 'work_root', 'nvme_replacement', 'git_ok', 'notes'],
  properties: {
    home: { type: 'string' },
    venv_python: { type: 'string', description: 'python in the nxd_inference (pytorch) venv used by the contrib port, or "" if not found' },
    venv_bin: { type: 'string', description: 'that venv bin dir (must be on PATH; holds libneuronpjrt-path + huggingface-cli), or ""' },
    neuron_device_present: { type: 'boolean' },
    num_devices: { type: ['number', 'string'], description: 'count of /dev/neuron* devices' },
    device_holder_pids: { type: 'array', items: { type: ['number', 'string'] }, description: 'PIDs currently holding any Neuron device (excluding this agent), before freeing' },
    freed: { type: 'boolean', description: 'true if those holders were killed and the devices are now free' },
    work_root: { type: 'string', description: 'absolute writable work root for this run (big: weights + compiled artifacts)' },
    nvme_replacement: { type: 'string', description: 'absolute dir to substitute for the contrib-hardcoded /opt/dlami/nvme' },
    git_ok: { type: 'boolean', description: 'true if `git` is available for cloning the contrib repo' },
    notes: { type: 'string' },
  },
}

const setup = await agent(
  `Prepare THIS host for compiling a Neuron diffusion image-edit port. Load tools: ToolSearch "select:Bash".

1. \`echo "$HOME"\` -> home (verbatim).
2. Find the pytorch nxd-inference venv the contrib port expects: try \`ls -d /opt/aws_neuronx_venv_pytorch*nxd_inference/bin/python 2>/dev/null\`. Pick the first whose \`PATH="<venv>/bin:/opt/aws/neuron/bin:$PATH" <python> -c "import torch"\` succeeds. Set venv_python + venv_bin (the venv's bin dir). If none, set "".
3. Devices: count \`ls /dev/neuron* 2>/dev/null | wc -l\` -> num_devices; neuron_device_present = (count>0).
4. Find what's holding the devices (a stale vLLM/EngineCore from a previous run will hold ALL of them and block compilation): run \`(neuron-ls 2>/dev/null || /opt/aws/neuron/bin/neuron-ls 2>/dev/null)\` and collect the distinct PIDs in its PID column; also \`sudo fuser /dev/neuron* 2>/dev/null\` if available. List them in device_holder_pids (exclude your own shell). ${FREE_DEVICES ? `Then FREE the devices: for each holder PID run \`kill -9 <pid>\`, also \`pkill -9 -f vllm.entrypoints; pkill -9 -f libneuronpjrt-path\`, wait ~5s, and re-check neuron-ls shows no COMMAND/PID rows. Set freed=true only if the devices are now idle.` : 'Do NOT kill anything (free_devices is false). Set freed=false.'}
5. Choose a work_root with lots of free space (weights are tens of GB, compiled artifacts more). Prefer the largest writable mount: check \`df -h /opt/dlami/nvme /home "$HOME" / 2>/dev/null\`. ${spec.work_root ? `Use "${spec.work_root}".` : `Default to "$HOME/nxdi-diffusion/${slug}" unless another mount has much more free space.`} mkdir -p it.
6. The contrib code hardcodes the path "/opt/dlami/nvme". Check if it exists+writable (\`test -w /opt/dlami/nvme\`). If yes, nvme_replacement="/opt/dlami/nvme". If NOT, set nvme_replacement="<work_root>/nvme" and mkdir -p it — we will sed-replace /opt/dlami/nvme with this in the copied scripts. Do NOT attempt to create /opt/dlami/nvme (needs root).
7. git_ok: \`git --version\` succeeds?
Return the schema with absolute paths. "" / [] for anything absent. Be truthful about freed.`,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA }
)

if (!setup) { log('Setup failed.'); return { error: 'setup failed' } }
if (!setup.venv_python) { log('No usable nxd-inference venv python found — cannot proceed.'); return { error: 'no venv', setup } }
if (!setup.neuron_device_present) log('WARNING: no Neuron device detected — compile/verify will be skipped.')
log(`Setup: venv=${setup.venv_python}, devices=${setup.num_devices}, freed=${setup.freed} (held by [${(setup.device_holder_pids||[]).join(',')}]), work_root=${setup.work_root}, nvme->${setup.nvme_replacement}`)

// =========================================================================
// Phase 1: Profile the target pipeline
// =========================================================================
phase('Profile')

const PROFILE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pipeline_class', 'is_qwen_image_edit_family', 'components', 'transformer_dims', 'confidence', 'notes'],
  properties: {
    pipeline_class: { type: 'string', description: 'model_index.json _class_name' },
    is_qwen_image_edit_family: { type: 'boolean', description: 'true if the pipeline is QwenImageEdit(Plus)Pipeline with a QwenImageTransformer2DModel + AutoencoderKLQwenImage + Qwen2_5_VL text encoder' },
    components: {
      type: 'object', additionalProperties: false,
      required: ['text_encoder_class', 'transformer_class', 'vae_class', 'scheduler_class'],
      properties: {
        text_encoder_class: { type: 'string' },
        transformer_class: { type: 'string' },
        vae_class: { type: 'string' },
        scheduler_class: { type: 'string' },
      },
    },
    transformer_dims: {
      type: 'object', additionalProperties: true,
      description: 'num_layers, num_attention_heads, attention_head_dim, axes_dims_rope, joint_attention_dim, in_channels, out_channels, patch_size from transformer/config.json',
    },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string' },
  },
}

const profile = await agent(
  `Classify the diffusion pipeline "${MODEL}" so we can port it to Neuron. Load tools: ToolSearch "select:WebFetch".
1. WebFetch ${HF(MODEL, 'model_index.json')} — read _class_name and the (lib,class) pair for text_encoder, transformer, vae, scheduler, processor.
2. WebFetch ${HF(MODEL, 'transformer/config.json')} — read num_layers, num_attention_heads, attention_head_dim, axes_dims_rope, joint_attention_dim, in_channels, out_channels, patch_size into transformer_dims.
3. Set is_qwen_image_edit_family=true iff pipeline is QwenImageEdit(Plus)Pipeline AND transformer is QwenImageTransformer2DModel AND vae is AutoencoderKLQwenImage AND text_encoder is a Qwen2_5_VL* class. This is the architecture the contrib port supports.
Return the schema. Be honest in confidence if any file 404s.`,
  { label: 'profile', phase: 'Profile', schema: PROFILE_SCHEMA }
)

if (!profile) { log('Profile failed.'); return { error: 'profile failed' } }
log(`Profile: ${profile.pipeline_class} — qwen-image-edit-family=${profile.is_qwen_image_edit_family} (transformer ${JSON.stringify(profile.transformer_dims)})`)

// =========================================================================
// Phase 2: Ground — diff target vs contrib base, classify adaptability
// =========================================================================
phase('Ground')

const GROUND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['contrib_dir_listing', 'compile_script', 'compile_modes', 'run_entrypoint', 'cache_script', 'base_model_id', 'config_deltas', 'adaptability', 'required_patches', 'hardcoded_paths', 'validation_from_readme', 'notes'],
  properties: {
    contrib_dir_listing: { type: 'array', items: { type: 'string' }, description: 'src/ filenames in the contrib port' },
    compile_script: { type: 'string', description: 'the master compile script name (e.g. src/compile.sh)' },
    compile_modes: { type: 'array', items: { type: 'string' }, description: 'accepted compile.sh modes' },
    run_entrypoint: { type: 'string', description: 'the inference driver (e.g. src/run_qwen_image_edit.py) and how it selects the model id (env var / arg)' },
    cache_script: { type: 'string', description: 'the model-download script (e.g. src/cache_hf_model.py) and the MODEL_ID / cache_dir it uses' },
    base_model_id: { type: 'string', description: 'the HF id the contrib port was built/validated on' },
    config_deltas: {
      type: 'array', description: 'meaningful differences between TARGET and BASE component configs (transformer, vae, text_encoder)',
      items: {
        type: 'object', additionalProperties: false,
        required: ['component', 'field', 'target', 'base', 'significant'],
        properties: {
          component: { type: 'string' }, field: { type: 'string' },
          target: { type: 'string' }, base: { type: 'string' },
          significant: { type: 'boolean', description: 'true if it changes tensor shapes / module structure (needs code), false if cosmetic metadata' },
        },
      },
    },
    adaptability: { type: 'string', enum: ['weight-variant', 'minor-code', 'major-code', 'incompatible'], description: 'weight-variant = same arch, just repoint weights; minor-code = a few param/path edits; major-code = structural; incompatible = different arch' },
    required_patches: { type: 'array', items: { type: 'string' }, description: 'concrete edits needed to run the contrib port on the TARGET (e.g. set QIE_MODEL_PATH, change image_size, none)' },
    hardcoded_paths: { type: 'array', items: { type: 'string' }, description: 'absolute paths hardcoded in the contrib scripts that must be repathed for this host (e.g. /opt/dlami/nvme/...)' },
    validation_from_readme: { type: 'string', description: 'validation results quoted from the contrib README (output PNG MD5 / accuracy / throughput / TP)' },
    notes: { type: 'string' },
  },
}

const grounded = await agent(
  `Determine how to run the community contrib Neuron port at ${CONTRIB_REPO}@${CONTRIB_REF}/${CONTRIB_SUB} against the TARGET model "${MODEL}", instead of its base "${BASE}". Load tools: ToolSearch "select:WebFetch".

The contrib port is an already-VALIDATED Neuron port of the QwenImageEdit pipeline. Our target is, per profiling, the same pipeline architecture (${profile.pipeline_class}). Your job: confirm same-arch and list exactly what must change.

1. List the contrib src/: WebFetch ${CONTRIB_API.replace(CONTRIB_SUB, CONTRIB_SUB + '/src')} (JSON array of {name}). Record filenames in contrib_dir_listing.
2. WebFetch ${CONTRIB_RAW}/README.md — copy validation results into validation_from_readme; note the recommended compile mode + run command.
3. WebFetch ${CONTRIB_RAW}/src/compile.sh — record compile_script, compile_modes, and every absolute path it hardcodes (look for /opt/dlami/nvme...) into hardcoded_paths.
4. WebFetch ${CONTRIB_RAW}/src/cache_hf_model.py — record its MODEL_ID + cache_dir into cache_script.
5. WebFetch the head of ${CONTRIB_RAW}/src/run_qwen_image_edit.py (the module constants near the top + the argparse) — record run_entrypoint INCLUDING how the model id is chosen (it reads an env override like QIE_MODEL_PATH) and the COMPILED_MODELS_DIR / HUGGINGFACE_CACHE_DIR constants (add them to hardcoded_paths).
6. DIFF the component configs TARGET vs BASE — WebFetch all six and compare field by field:
   - transformer: ${HF(MODEL, 'transformer/config.json')}  vs  ${HF(BASE, 'transformer/config.json')}
   - vae: ${HF(MODEL, 'vae/config.json')}  vs  ${HF(BASE, 'vae/config.json')}
   - text_encoder: ${HF(MODEL, 'text_encoder/config.json')}  vs  ${HF(BASE, 'text_encoder/config.json')}
   For every difference, add a config_deltas row and judge significant: true ONLY if it changes tensor shapes / layer counts / module structure (e.g. num_layers, num_attention_heads, head_dim, hidden_size, num_experts). Cosmetic diffs (dtype field spelling, transformers_version, *_token_id metadata) are significant=false.
7. adaptability: 'weight-variant' if NO significant deltas (same shapes everywhere → the contrib compiled graphs fit the target weights, just repoint the model id). 'minor-code' if only a couple of param/arg changes (e.g. image_size). 'major-code' if structural. 'incompatible' if not the same pipeline arch.
8. required_patches: the concrete, minimal changes to run on the target — e.g. "export QIE_MODEL_PATH=${MODEL}", "set cache_hf_model.py MODEL_ID=${MODEL}", "sed /opt/dlami/nvme -> <work_root>/nvme", or "none beyond repath+model-id".
Return the schema. Quote real values you read; do not invent.`,
  { label: 'ground', phase: 'Ground', schema: GROUND_SCHEMA, model: 'opus' }
)

if (!grounded) { log('Ground failed.'); return { error: 'ground failed', profile } }
const sigDeltas = (grounded.config_deltas || []).filter(d => d.significant)
log(`Ground: adaptability=${grounded.adaptability}, ${sigDeltas.length} significant config delta(s), patches=[${(grounded.required_patches||[]).join(' | ')}]`)
if (grounded.adaptability === 'incompatible') {
  log('Target is NOT the same architecture as the contrib port — aborting before spending compile time.')
  return { error: 'incompatible architecture', profile, grounded }
}

// =========================================================================
// Phase 3: Materialize — copy contrib src locally, repath, point at target, pip install
// =========================================================================
phase('Materialize')

const MAT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'src_dir', 'nvme_dir', 'model_id_wired', 'repathed', 'pip_ok', 'requirements_installed', 'error_summary', 'notes'],
  properties: {
    status: { type: 'string', enum: ['ok', 'failed'] },
    src_dir: { type: 'string', description: 'absolute dir holding the local copy of the contrib src/' },
    nvme_dir: { type: 'string', description: 'the work-root dir substituted for /opt/dlami/nvme' },
    model_id_wired: { type: 'boolean', description: 'true if the target model id is wired in (env QIE_MODEL_PATH and/or cache_hf_model.py MODEL_ID edited)' },
    repathed: { type: 'boolean', description: 'true if all /opt/dlami/nvme occurrences were sed-replaced in the copied scripts' },
    pip_ok: { type: 'boolean' },
    requirements_installed: { type: 'boolean' },
    error_summary: { type: 'string' },
    notes: { type: 'string' },
  },
}

const SRC_DIR = `${setup.work_root}/src`
const NVME = setup.nvme_replacement
const materialize = await agent(
  `Materialize the contrib Neuron port locally and wire it to the TARGET model + THIS host's paths. Load tools: ToolSearch "select:Bash,Edit,Read".

INPUTS:
- venv python: ${setup.venv_python} ; venv bin (prepend to PATH): ${setup.venv_bin}
- work_root: ${setup.work_root} ; nvme replacement dir: ${NVME}
- contrib: repo ${CONTRIB_REPO}, ref ${CONTRIB_REF}, subdir ${CONTRIB_SUB}
- target model id: ${MODEL}
- required patches from grounding: ${JSON.stringify(grounded.required_patches)}
- hardcoded paths to repath: ${JSON.stringify(grounded.hardcoded_paths)}

STEPS:
1. Obtain the contrib subdir into ${setup.work_root}. ${setup.git_ok ? `Prefer git sparse clone:
   \`cd ${setup.work_root} && rm -rf _repo && git clone --depth 1 --filter=blob:none --sparse --branch ${CONTRIB_REF} https://github.com/${CONTRIB_REPO}.git _repo && cd _repo && git sparse-checkout set ${CONTRIB_SUB}\`
   then \`cp -r ${setup.work_root}/_repo/${CONTRIB_SUB}/* ${setup.work_root}/\` so that ${SRC_DIR} exists (the src/ dir, plus README, requirements.txt, assets/, test/).` : `git is unavailable — download each file via the GitHub API: WebFetch ${CONTRIB_API.replace(CONTRIB_SUB, CONTRIB_SUB + '/src')} to list src/, then curl each raw file https://raw.githubusercontent.com/${CONTRIB_REPO}/${CONTRIB_REF}/${CONTRIB_SUB}/src/<name> into ${SRC_DIR}/<name>. Also fetch requirements.txt and assets/.`}
   Confirm \`ls ${SRC_DIR}\` shows compile.sh + run_qwen_image_edit.py + the compile_*.py files. (src_dir = ${SRC_DIR})
2. mkdir -p ${NVME}. Repath: in EVERY file under ${SRC_DIR} replace the literal "/opt/dlami/nvme" with "${NVME}":
   \`grep -rl "/opt/dlami/nvme" ${SRC_DIR} | while read f; do sed -i "s#/opt/dlami/nvme#${NVME}#g" "$f"; done\`
   Verify zero remain: \`grep -rn "/opt/dlami/nvme" ${SRC_DIR} | head\` (should be empty). Set repathed accordingly.
3. Wire the target model id ${MODEL}:
   - The run driver reads env QIE_MODEL_PATH (default ${BASE}); we will export it at run time, but ALSO edit the download script so it caches the TARGET: in ${SRC_DIR}/cache_hf_model.py set MODEL_ID = "${MODEL}" (Edit). If the grounding said a different mechanism, follow grounded.required_patches.
   - Set model_id_wired=true once both the download script points at ${MODEL} and you have confirmed QIE_MODEL_PATH is the run-time override.
4. pip install requirements INTO the venv (this can take a few minutes; diffusers installs from git):
   \`PATH="${setup.venv_bin}:/opt/aws/neuron/bin:$PATH" ${setup.venv_python} -m pip install -r ${setup.work_root}/requirements.txt\` (run with a long timeout; if requirements.txt is under ${SRC_DIR}, use that path). Capture failures. Then sanity check: \`${setup.venv_python} -c "import diffusers, transformers; from diffusers import QwenImageEditPlusPipeline; print('diffusers', diffusers.__version__)"\` — set pip_ok/requirements_installed from whether that import line SUCCEEDS (QwenImageEditPlusPipeline must import; if diffusers is too old it won't — then upgrade diffusers from git as requirements specifies).
5. Apply any remaining minor-code patches grounding listed (e.g. image_size), editing files under ${SRC_DIR}. Keep edits minimal and report them in notes.
Return the schema. status='ok' only if src is present, repathed, model wired, and the QwenImageEditPlusPipeline import works.`,
  { label: 'materialize', phase: 'Materialize', schema: MAT_SCHEMA }
)

if (!materialize || materialize.status !== 'ok') {
  log(`Materialize failed: ${materialize ? materialize.error_summary : 'null'}`)
  return { error: 'materialize failed', profile, grounded, materialize }
}
log(`Materialize ok: src=${materialize.src_dir}, nvme=${materialize.nvme_dir}, deps installed=${materialize.requirements_installed}`)

// Gate the hardware phases.
const canRunHw = setup.neuron_device_present && (FREE_DEVICES ? setup.freed : true)
if (!canRunHw) {
  log(`Stopping before compile: ${!setup.neuron_device_present ? 'no Neuron device' : 'devices still busy (free_devices off or free failed)'}. Local port is materialized at ${SRC_DIR}.`)
  return { model: MODEL, mode: MODE, adaptability: grounded.adaptability, work_root: setup.work_root, src_dir: SRC_DIR, compiled: false, reason: 'hardware unavailable', profile, grounded }
}

// =========================================================================
// Phase 4 + 5 + 6: Compile -> Verify (run a real edit) -> Repair loop
// =========================================================================
const COMPILED_DIR = `${NVME}/compiled_models_qwen_image_edit`
const ENV_PREFIX = `PATH="${setup.venv_bin}:/opt/aws/neuron/bin:$PATH" PYTHONPATH="${SRC_DIR}:$PYTHONPATH" QIE_MODEL_PATH="${MODEL}" PYTHONUNBUFFERED=1`

const COMPILE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'mode', 'components_compiled', 'compiled_models_dir', 'error_summary', 'traceback_tail', 'culprit_source', 'log_path'],
  properties: {
    status: { type: 'string', enum: ['success', 'failure'] },
    mode: { type: 'string' },
    components_compiled: { type: 'array', items: { type: 'string' }, description: 'component subdirs produced under compiled_models_dir (e.g. vae_encoder, vae_decoder, transformer_v3_cfg, language_model_v3, vision_encoder_v3)' },
    compiled_models_dir: { type: 'string' },
    error_summary: { type: 'string' },
    traceback_tail: { type: 'string', description: 'last ~40 lines of the real error if failed, else ""' },
    culprit_source: { type: 'string', description: 'relevant snippet (file:line) from the contrib src that defines the failing contract, or ""' },
    log_path: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'output_image_path', 'image_bytes', 'image_dims', 'looks_valid', 'prompt_used', 'input_image', 'error_summary', 'traceback_tail', 'culprit_source', 'log_path'],
  properties: {
    status: { type: 'string', enum: ['success', 'failure'] },
    output_image_path: { type: 'string' },
    image_bytes: { type: ['number', 'string'] },
    image_dims: { type: 'string', description: 'WxH from PIL, or ""' },
    looks_valid: { type: 'boolean', description: 'true if the PNG opens, is the expected size, and is not blank/uniform' },
    prompt_used: { type: 'string' },
    input_image: { type: 'string' },
    error_summary: { type: 'string' },
    traceback_tail: { type: 'string' },
    culprit_source: { type: 'string' },
    log_path: { type: 'string' },
  },
}

const compilePrompt = `Compile the QwenImageEdit Neuron port (target weights ${MODEL}) on this Trn2 host. This is LONG-RUNNING (downloads tens of GB then compiles VAE + DiT transformer + Qwen2.5-VL encoder). Load tools: ToolSearch "select:Bash,Read".

ENV (prepend to every command): ${ENV_PREFIX}
SRC: ${SRC_DIR}   MODE: ${MODE}   COMPILED_DIR (expected): ${COMPILED_DIR}

STEPS:
1. Download the TARGET weights once: \`${ENV_PREFIX} ${setup.venv_python} ${SRC_DIR}/cache_hf_model.py\` (it now points at ${MODEL}). Run in BACKGROUND with nohup to a log and poll, since it's large:
   \`cd ${SRC_DIR} && nohup bash -c '${ENV_PREFIX} ${setup.venv_python} ${SRC_DIR}/cache_hf_model.py' > ${setup.work_root}/download.log 2>&1 &\`
   Poll download.log until "downloaded successfully" or an error. (If it errors on auth/gating, report failure stage download.)
2. Compile, in BACKGROUND (this far exceeds a single command timeout — do NOT run it foreground):
   \`cd ${SRC_DIR} && NEURON_RT_NUM_CORES=8 nohup bash -c '${ENV_PREFIX} bash ${SRC_DIR}/compile.sh ${MODE}' > ${setup.work_root}/compile.log 2>&1 &\`
   Record the log_path = ${setup.work_root}/compile.log.
3. Poll compile.log in ~60-120s steps, up to ~60 min total. Success = the script exits 0 AND the expected component subdirs appear under ${COMPILED_DIR} (ls it). For mode v3_cfg expect roughly: vae_encoder, vae_decoder, transformer_v3_cfg, language_model_v3, vision_encoder_v3 (use whatever the contrib README/compile.sh says for ${MODE}). Watch for: Python "Traceback", "RuntimeError", "Neuron compilation failed", "out of memory", "No space left".
4. When the compile process is done, set status. components_compiled = the subdirs actually present under ${COMPILED_DIR}.
5. On FAILURE: put the last ~40 lines of the real error into traceback_tail, a one-line error_summary, and READ the relevant contrib source under ${SRC_DIR} (the compile_*.py for the failing component, or neuron_commons.py) to capture the failing contract into culprit_source (file:line). Do NOT claim success you didn't observe (the compiled subdirs must really exist).
Return the schema with the TRUE result.`

const verifyPrompt = `Run a REAL image edit with the compiled Neuron port and validate the output PNG. Load tools: ToolSearch "select:Bash,Read".

ENV: ${ENV_PREFIX}
SRC: ${SRC_DIR}   COMPILED_DIR: ${COMPILED_DIR}   MODE: ${MODE}

STEPS:
1. Get an input image: download ${IMAGE_URL} to ${setup.work_root}/input.png (curl -L). If that 404s, use any *.png under ${SRC_DIR}/assets (ls it) or generate a simple 1024x1024 test image with PIL.
2. Run the contrib inference driver in BACKGROUND (denoising 50 steps on Neuron takes minutes), logging to ${setup.work_root}/run.log. Use the mode-appropriate flags from the contrib README (for v3_cfg the README run command uses --use_v3_cfg). Base command:
   \`cd ${SRC_DIR} && NEURON_RT_NUM_CORES=8 nohup bash -c '${ENV_PREFIX} ${setup.venv_python} ${SRC_DIR}/run_qwen_image_edit.py --images ${setup.work_root}/input.png --prompt "${PROMPT}" --output ${setup.work_root}/output_edited.png${MODE === 'v3_cfg' ? ' --use_v3_cfg' : MODE === 'v3_cp' ? ' --use_v3_cp' : ''} --compiled_models_dir ${COMPILED_DIR}' > ${setup.work_root}/run.log 2>&1 &\`
   (If the driver names the compiled-dir flag differently, read its argparse via \`grep -n add_argument ${SRC_DIR}/run_qwen_image_edit.py\` and adjust.)
3. Poll run.log up to ~30 min. Watch for "Traceback"/"RuntimeError"/"missing"/"shape".
4. Validate the output: confirm ${setup.work_root}/output_edited.png exists with non-trivial size (image_bytes via \`stat -c%s\`), then open it with PIL to get image_dims and check it is NOT blank/uniform:
   \`${setup.venv_python} -c "from PIL import Image; import numpy as np; im=Image.open('${setup.work_root}/output_edited.png').convert('RGB'); a=np.asarray(im); print(im.size, a.std())"\`
   looks_valid=true iff it opens, dims are ~the requested size, and pixel std-dev is clearly non-zero (a real edited image, not a gray/black frame). status=success iff looks_valid.
5. On FAILURE: traceback_tail (last ~40 lines), one-line error_summary, and read the relevant ${SRC_DIR} source into culprit_source (file:line).
Return the schema truthfully — NEVER report success without a validated non-blank PNG.`

const repairPrompt = (round, stage, res) => `The Neuron diffusion port FAILED at the ${stage} stage. Fix the LOCAL contrib copy so it ${stage === 'compile' ? 'compiles' : 'runs and produces a valid edited image'}, grounding every change in the actual contrib source (not guesses). Load tools: ToolSearch "select:Read,Edit,Bash,WebFetch".

LOCAL SRC TO FIX (edit in place): ${SRC_DIR}
Target model: ${MODEL}   Base it was validated on: ${BASE}   Mode: ${MODE}
Upstream contrib (compare against the pristine source if needed): ${CONTRIB_RAW}/src/
Config deltas target-vs-base (from grounding): ${JSON.stringify(grounded.config_deltas)}

REAL FAILURE (round ${round + 1}):
- root cause: ${res.error_summary}
- traceback tail:
${res.traceback_tail}
- contract snippet:
${res.culprit_source}

GUIDANCE:
- Since the target is a "${grounded.adaptability}" of the contrib base, most failures are path/version/shape-config mismatches, NOT deep logic. Check first: leftover /opt/dlami/nvme paths (\`grep -rn /opt/dlami/nvme ${SRC_DIR}\`), QIE_MODEL_PATH not exported, a diffusers/transformers version that doesn't expose the expected class, an image_size/sequence-length default that doesn't match a compiled graph.
- If a config delta IS significant (shape change), the corresponding compiled graph must be recompiled with the target's dimension — adjust the compile_*.py arg, not the weights.
- Make the MINIMAL set of edits. After editing any .py, \`${setup.venv_python} -m py_compile <file>\` (PATH including ${setup.venv_bin}) to confirm it parses.
Report exactly what you changed and the source file:line that justifies each change.`

let compileResult = null, verifyResult = null, repairRounds = 0

for (let round = 0; round <= MAX_REPAIR; round++) {
  repairRounds = round

  // ---- Compile ----
  phase('Compile')
  compileResult = await agent(compilePrompt, { label: `compile:round${round + 1}`, phase: 'Compile', schema: COMPILE_SCHEMA, model: 'opus' })
  if (!compileResult) { log(`Compile round ${round + 1}: null — stopping.`); break }
  log(`Compile round ${round + 1}: ${compileResult.status} — components=[${(compileResult.components_compiled||[]).join(',')}]${compileResult.status !== 'success' ? ` — ${compileResult.error_summary}` : ''}`)

  if (compileResult.status !== 'success') {
    if (round === MAX_REPAIR) { log(`Max repair rounds reached during compile.`); break }
    phase('Repair')
    const rep = await agent(repairPrompt(round, 'compile', compileResult), { label: `repair-compile:round${round + 1}`, phase: 'Repair', schema: { type: 'object', additionalProperties: false, required: ['changes', 'justification'], properties: { changes: { type: 'array', items: { type: 'string' } }, justification: { type: 'string' } } }, model: 'opus' })
    if (!rep) { log('Repair returned null — stopping.'); break }
    log(`Repair (compile) round ${round + 1}: ${rep.changes.length} change(s).`)
    continue
  }

  if (!WANT_VERIFY) { log('Compile succeeded; verify disabled.'); break }

  // ---- Verify (run a real edit) ----
  phase('Verify')
  verifyResult = await agent(verifyPrompt, { label: `verify:round${round + 1}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'opus' })
  if (!verifyResult) { log(`Verify round ${round + 1}: null — stopping.`); break }
  log(`Verify round ${round + 1}: ${verifyResult.status}${verifyResult.status === 'success' ? ` — ${verifyResult.image_dims}, ${verifyResult.image_bytes}B -> ${verifyResult.output_image_path}` : ` — ${verifyResult.error_summary}`}`)

  if (verifyResult.status === 'success') break
  if (round === MAX_REPAIR) { log(`Max repair rounds reached during verify.`); break }

  // ---- Repair (verify failure) ----
  phase('Repair')
  const rep = await agent(repairPrompt(round, 'verify', verifyResult), { label: `repair-verify:round${round + 1}`, phase: 'Repair', schema: { type: 'object', additionalProperties: false, required: ['changes', 'justification'], properties: { changes: { type: 'array', items: { type: 'string' } }, justification: { type: 'string' } } }, model: 'opus' })
  if (!rep) { log('Repair returned null — stopping.'); break }
  log(`Repair (verify) round ${round + 1}: ${rep.changes.length} change(s).`)
}

return {
  model: MODEL,
  base_contrib: BASE,
  adaptability: grounded.adaptability,
  mode: MODE,
  work_root: setup.work_root,
  src_dir: SRC_DIR,
  compiled_models_dir: COMPILED_DIR,
  significant_config_deltas: sigDeltas,
  required_patches: grounded.required_patches,
  validation_reference: grounded.validation_from_readme,
  repair_rounds_used: repairRounds,
  compile: compileResult ? { status: compileResult.status, components: compileResult.components_compiled, error: compileResult.error_summary } : null,
  verify: WANT_VERIFY ? (verifyResult ? { status: verifyResult.status, output_image: verifyResult.output_image_path, dims: verifyResult.image_dims, looks_valid: verifyResult.looks_valid, error: verifyResult.error_summary } : { status: 'not-run' }) : { status: 'disabled' },
}
