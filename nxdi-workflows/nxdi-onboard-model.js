export const meta = {
  name: 'nxdi-onboard-model',
  description: 'Onboard a model to run on NxD Inference: profile the architecture, ground against the verified KB + reference code, scaffold modeling_<model>.py (config/model/task-head/weight-converter), adversarially review, emit a tailored plan + runbook, then ACTUALLY VERIFY on hardware (download weights, load the generated file into vLLM, compile, chat) with an automatic source-grounded repair loop.',
  whenToUse: 'When you want to port a HuggingFace model to AWS NxD Inference (Trn1/Trn2/Inf2). Pass the HF model id as args (e.g. "Qwen/Qwen3-30B-A3B"), or an object {model, config_url, instance, tp_degree, fused_qkv, notes, verify, max_repair_rounds}. Set verify:false to skip the on-hardware verify+repair phases (e.g. no Neuron device). Omit args for a generic Llama-reference playbook + scaffold.',
  phases: [
    { title: 'Setup', detail: 'resolve portable absolute paths ($HOME, KB location, output dir) + detect Neuron device / vLLM venv for this machine' },
    { title: 'Profile', detail: 'read the HF config to classify the architecture (dense vs MoE, dims, quirks)' },
    { title: 'Ground', detail: 'read the verified KB + reference modeling files; resolve UNVERIFIED import paths' },
    { title: 'Scaffold', detail: 'generate the full modeling_<model>.py tailored to the architecture', model: 'opus' },
    { title: 'Review', detail: 'parallel adversarial reviewers across base-contract / sharding / converter / MoE / imports' },
    { title: 'Finalize', detail: 'merge fixes into final code + tailored onboarding plan + eval/vLLM runbook', model: 'opus' },
    { title: 'Emit', detail: 'write modeling_<model>.py, ONBOARDING_PLAN.md, RUNBOOK.md to disk' },
    { title: 'Verify', detail: 'download weights, inject the GENERATED file into vLLM MODEL_TYPES, compile on Neuron, and chat — proving the produced code actually runs', model: 'opus' },
    { title: 'Repair', detail: 'on failure, read the INSTALLED package source + the real traceback, patch the generated file, and re-verify (loop)', model: 'opus' },
  ],
}

// ----- Reference material the agents read each run -----
const KB_FILENAME = 'nxdi-onboarding-kb.md'
const ONBOARDING_DOC = 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html'
const GH = 'https://github.com/aws-neuron/neuronx-distributed-inference'
const GH_RAW = 'https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-inference/main/src/neuronx_distributed_inference'
const REF_DENSE = `${GH_RAW}/models/llama/modeling_llama.py`
const REF_MOE = `${GH_RAW}/models/qwen3_moe/modeling_qwen3_moe.py`
// Community contrib library: 60+ already-implemented & validated NxDI ports
// (many architectures the official package does NOT ship: gemma3, phi-3.5,
// glm-4, internlm3, various VL/MoE). Each model dir has src/modeling_*.py +
// a README with validation results (token-match %, throughput, TP config).
const CONTRIB_REF = 'contrib/MiMo-V2.5'  // branch/ref
const CONTRIB_REPO = 'whn09/neuronx-distributed-inference'
const CONTRIB_API = `https://api.github.com/repos/${CONTRIB_REPO}/contents/contrib/models?ref=${CONTRIB_REF}`
const CONTRIB_RAW = `https://raw.githubusercontent.com/${CONTRIB_REPO}/${CONTRIB_REF}/contrib/models`

// ----- Normalize args -----
const spec = (typeof args === 'string') ? { model: args }
  : (args && typeof args === 'object') ? args
  : {}
const GENERIC = !spec.model
const MODEL = spec.model || 'Llama-3.1-8B (generic reference)'
const CONFIG_URL = spec.config_url
  || (spec.model && spec.model.includes('/') ? `https://huggingface.co/${spec.model}/resolve/main/config.json` : null)
const slug = String(MODEL).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48) || 'model'

// Verify-on-hardware controls. Default ON for a real (non-generic) target; the
// Setup agent still gates on an actually-present Neuron device + vLLM venv, so
// this is safe to leave on. Set verify:false to force skip.
const WANT_VERIFY = spec.verify !== false && !GENERIC
const MAX_REPAIR_ROUNDS = Number.isInteger(spec.max_repair_rounds) ? spec.max_repair_rounds : 3

log(`Onboarding target: ${MODEL}${GENERIC ? ' [generic playbook mode]' : ''}${WANT_VERIFY ? ` [verify-on-hardware, up to ${MAX_REPAIR_ROUNDS} repair rounds]` : ''}`)

// =========================================================================
// Phase 0: Setup — resolve portable absolute paths for THIS machine.
// The JS sandbox has no filesystem/env access, so an agent resolves $HOME,
// locates the KB, and computes the output dir. Optional args override:
//   args.kb_path   — explicit path to nxdi-onboarding-kb.md
//   args.out_root  — explicit root for outputs (default <home>/nxdi-onboarding)
// =========================================================================
phase('Setup')

const PATHS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['home', 'kb_path', 'kb_found', 'out_dir', 'neuron_device_present', 'vllm_python', 'nxdi_pkg_dir', 'neuron_bin_dir', 'notes'],
  properties: {
    home: { type: 'string', description: 'absolute home directory of the current user' },
    kb_path: { type: 'string', description: `absolute path to ${KB_FILENAME}` },
    kb_found: { type: 'boolean', description: 'true only if the KB file actually exists at kb_path' },
    out_dir: { type: 'string', description: 'absolute output directory for this run' },
    neuron_device_present: { type: 'boolean', description: 'true if /dev/neuron0 (or similar) exists' },
    vllm_python: { type: 'string', description: 'absolute path to the python in the vLLM+NxDI venv (e.g. /opt/aws_neuronx_venv_*vllm*/bin/python), or "" if not found' },
    nxdi_pkg_dir: { type: 'string', description: 'absolute dir of the INSTALLED neuronx_distributed_inference package in that venv, or "" if not found' },
    neuron_bin_dir: { type: 'string', description: 'absolute dir holding libneuronpjrt-path (usually the vLLM venv bin dir), or "" if not found' },
    notes: { type: 'string' },
  },
}

const paths = await agent(
  `Resolve portable absolute paths AND detect the Neuron runtime for an onboarding run on THIS machine. Load tools: ToolSearch "select:Bash".

1. Get the home dir: \`echo "$HOME"\` (Bash). Use it verbatim — do NOT hardcode /Users/... or /home/....
2. Locate the knowledge base file "${KB_FILENAME}".
   ${spec.kb_path ? `An explicit path was provided: "${spec.kb_path}" — verify it exists with \`ls -la\`.` : `Search the likely locations in order and use the first that exists:
     - "$HOME/.claude/workflows/${KB_FILENAME}"
     - "$HOME/${KB_FILENAME}"
     - any "${KB_FILENAME}" found via: \`find "$HOME" -maxdepth 4 -name ${KB_FILENAME} 2>/dev/null | head -5\``}
   Set kb_found=true ONLY if the chosen kb_path exists (confirm with ls). If none found, set kb_found=false and put your best-guess path in kb_path.
3. Compute out_dir = ${spec.out_root ? `"${spec.out_root}/${slug}"` : `"$HOME/nxdi-onboarding/${slug}"`} with $HOME expanded to the real absolute path (no literal "$HOME" or "~").
4. Detect the Neuron device: \`ls /dev/neuron0 2>/dev/null\` — set neuron_device_present true iff it exists.
5. Find the vLLM+NxDI venv python. Try: \`ls -d /opt/aws_neuronx_venv*vllm*/bin/python 2>/dev/null\` and any "*venv*vllm*" under $HOME. Pick the first whose \`<python> -c "import vllm, neuronx_distributed_inference"\` succeeds (you MUST prepend that venv's bin dir to PATH for the import to work: \`PATH="<venv>/bin:/opt/aws/neuron/bin:$PATH" <python> -c ...\`). Set vllm_python to that python path, or "" if none works.
6. If vllm_python found: get the installed package dir via \`PATH=... <vllm_python> -c "import neuronx_distributed_inference as m,os;print(os.path.dirname(m.__file__))"\` -> nxdi_pkg_dir. And confirm \`ls <venv>/bin/libneuronpjrt-path\` exists -> neuron_bin_dir = that venv's bin dir (this is the dir that MUST be on PATH to avoid a libneuronpjrt-path fork bomb / FileNotFoundError).
7. notes: summarize what you found (device yes/no, venv path, any import failures).
Return the schema with fully-expanded absolute paths. Empty string "" for anything not found — do NOT guess.`,
  { label: 'setup:paths', phase: 'Setup', schema: PATHS_SCHEMA }
)

if (!paths) { log('Setup failed — could not resolve paths.'); return { error: 'setup failed' } }
const KB_PATH = paths.kb_path
const OUT_DIR = paths.out_dir
if (!paths.kb_found) log(`WARNING: KB not found at ${KB_PATH} — agents will fall back to the onboarding doc + reference code.`)
log(`Paths resolved: KB=${KB_PATH} (${paths.kb_found ? 'found' : 'MISSING'}), out=${OUT_DIR}`)

// =========================================================================
// Schemas
// =========================================================================
const PROFILE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['model_name', 'architecture_family', 'is_moe', 'dims', 'features', 'hf_name_quirks', 'recommended_tp_degrees', 'confidence', 'notes'],
  properties: {
    model_name: { type: 'string' },
    architecture_family: { type: 'string', description: 'e.g. llama, qwen3, qwen3_moe, mixtral, gpt-neox' },
    is_moe: { type: 'boolean' },
    dims: {
      type: 'object', additionalProperties: false,
      required: ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'num_hidden_layers', 'intermediate_size', 'vocab_size', 'head_dim', 'max_position_embeddings', 'rope_theta', 'rms_norm_eps', 'hidden_act'],
      properties: {
        hidden_size: { type: ['number', 'string'] },
        num_attention_heads: { type: ['number', 'string'] },
        num_key_value_heads: { type: ['number', 'string'] },
        num_hidden_layers: { type: ['number', 'string'] },
        intermediate_size: { type: ['number', 'string'] },
        vocab_size: { type: ['number', 'string'] },
        head_dim: { type: ['number', 'string'] },
        max_position_embeddings: { type: ['number', 'string'] },
        rope_theta: { type: ['number', 'string'] },
        rms_norm_eps: { type: ['number', 'string'] },
        hidden_act: { type: 'string' },
      },
    },
    moe: {
      type: 'object', additionalProperties: true,
      description: 'MoE fields if is_moe, else null/empty. num_experts/top_k/n_shared_experts/moe_intermediate_size + the HF attr names used.',
    },
    features: {
      type: 'object', additionalProperties: false,
      required: ['tied_word_embeddings', 'attention_bias', 'qk_norm', 'sliding_window', 'fused_qkv_recommended'],
      properties: {
        tied_word_embeddings: { type: 'boolean' },
        attention_bias: { type: 'boolean' },
        qk_norm: { type: 'boolean', description: 'per-head q/k RMSNorm (qwen3-style)' },
        sliding_window: { type: ['number', 'string', 'null'] },
        fused_qkv_recommended: { type: 'boolean' },
      },
    },
    hf_name_quirks: {
      type: 'array', description: 'HF config field names that differ from canonical NxDI names',
      items: {
        type: 'object', additionalProperties: false,
        required: ['hf_name', 'canonical_name'],
        properties: { hf_name: { type: 'string' }, canonical_name: { type: 'string' } },
      },
    },
    recommended_tp_degrees: { type: 'array', items: { type: 'number' }, description: 'TP degrees that divide num_key_value_heads (GQA limiter) and num_attention_heads' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string', description: 'gating, missing fields, architectural oddities, instance fit' },
  },
}

const RESOLVED_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['imports', 'attention_base_signature', 'converter_pattern', 'moe_module_pattern', 'reference_used', 'contrib_match', 'notes'],
  properties: {
    contrib_match: {
      type: 'object', additionalProperties: false,
      description: 'Best-matching already-implemented model found in the community contrib library, or null-ish empty strings if none applies',
      required: ['model_dir', 'raw_url', 'is_same_arch', 'validation', 'key_patterns'],
      properties: {
        model_dir: { type: 'string', description: 'contrib/models/<dir> that best matches this target architecture, or "" if none' },
        raw_url: { type: 'string', description: 'raw URL of that dir\'s src/modeling_*.py that was actually fetched, or ""' },
        is_same_arch: { type: 'boolean', description: 'true if the contrib model shares this target\'s architecture family (directly adaptable), false if only loosely related' },
        validation: { type: 'string', description: 'validation status copied from that model\'s README (e.g. "token-match 100%, 196 tok/s, TP=8"), or "unknown"' },
        key_patterns: { type: 'string', description: 'concrete patterns to copy from the contrib impl: exact imports, attention signature, converter renames, any arch-specific quirk it solved. "" if no usable match.' },
      },
    },
    imports: {
      type: 'array', description: 'Resolved real import paths for symbols the KB marked UNVERIFIED',
      items: {
        type: 'object', additionalProperties: false,
        required: ['symbol', 'import_path', 'status'],
        properties: {
          symbol: { type: 'string' },
          import_path: { type: 'string', description: 'exact "from X import Y" resolved from reference code' },
          status: { type: 'string', enum: ['resolved', 'still-unverified'] },
        },
      },
    },
    attention_base_signature: { type: 'string', description: 'the real NeuronAttentionBase.__init__ kwargs the reference passes, verbatim if possible' },
    converter_pattern: { type: 'string', description: 'how the reference convert_hf_to_neuron_state_dict remaps weights (fused qkv, gate-up, key renames)' },
    moe_module_pattern: { type: 'string', description: 'how the reference MoE decoder layer initializes/uses the expert module (v1 vs v2, tuple return), or "n/a (dense)"' },
    reference_used: { type: 'string' },
    notes: { type: 'string' },
  },
}

const SCAFFOLD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['filename', 'code', 'config_json_notes', 'model_types_registration', 'open_questions'],
  properties: {
    filename: { type: 'string' },
    code: { type: 'string', description: 'the complete modeling_<model>.py contents' },
    config_json_notes: { type: 'string', description: 'what the checkpoint config.json needs (architectures field, neuron_config keys)' },
    model_types_registration: { type: 'string', description: 'the MODEL_TYPES dict entry + where to add it' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'findings'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'location', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          location: { type: 'string', description: 'class/method/line region' },
          problem: { type: 'string' },
          fix: { type: 'string', description: 'concrete corrected code or instruction' },
        },
      },
    },
  },
}

// =========================================================================
// Phase 1+2: Profile (HF config) and Ground (KB + reference) — independent
// =========================================================================
phase('Profile')
phase('Ground')

const [profile, grounded] = await parallel([
  // --- Profile ---
  () => agent(
    `Classify a model's architecture so we can scaffold its NxD Inference port.
Target: ${MODEL}
${CONFIG_URL ? `HF config.json URL: ${CONFIG_URL}` : 'No config URL available.'}
${GENERIC ? 'GENERIC MODE: no specific model given. Profile Llama-3.1-8B as the canonical dense reference (hidden_size 4096, 32 heads, 8 kv heads, 32 layers, intermediate 14336, vocab 128256, head_dim 128, rope_theta 500000, rms_norm_eps 1e-5, silu, tied_word_embeddings false). Set confidence "high" and note this is the generic reference.' : ''}

Steps:
1. Load tools: ToolSearch with query "select:WebFetch".
${CONFIG_URL ? `2. WebFetch ${CONFIG_URL} and read every field. If it 404s/403s (gated/private), say so in notes and fall back to known values for the "${MODEL}" architecture family from your knowledge + the HF model card (WebFetch https://huggingface.co/${spec.model || ''}).` : '2. Use known values for this architecture family.'}
3. Decide dense vs MoE: MoE if the config has num_local_experts / num_experts / n_routed_experts / moe_intermediate_size.
4. For MoE, fill the moe object: num_experts, top_k (num_experts_per_tok), n_shared_experts, moe_intermediate_size, AND the exact HF field names used.
5. hf_name_quirks: list HF field names that differ from canonical NxDI names (e.g. num_experts->num_local_experts, n_head->num_attention_heads). Empty array if none.
6. recommended_tp_degrees: powers of 2 (and Trn2-relevant 32/64) that EVENLY DIVIDE num_key_value_heads (the GQA limiter) and num_attention_heads. This is critical — TP must divide the kv head count.
7. features: tied_word_embeddings, attention_bias (q/k/v bias), qk_norm (per-head q/k RMSNorm, qwen3-style), sliding_window, fused_qkv_recommended (true unless the arch has per-head qk_norm or other reason not to fuse).
Return the schema. Be honest in confidence/notes about anything you could not read directly.`,
    { label: `profile:${slug}`, phase: 'Profile', schema: PROFILE_SCHEMA }
  ),

  // --- Ground ---
  () => agent(
    `Resolve the import paths and code patterns the NxD Inference onboarding KB left UNVERIFIED, using the reference modeling files. This makes the generated scaffold use REAL imports.

Steps:
1. Load tools: ToolSearch with query "select:Read,WebFetch".
2. Read the verified KB at ${KB_PATH} (Read tool). Pay attention to section 11 (import path confidence table) — every row marked UNVERIFIED is your target.
3. WebFetch the dense reference: ${REF_DENSE} (if it 404s, browse ${GH}/tree/main/src/neuronx_distributed_inference/models to find the llama modeling file and fetch its raw URL).
4. ${GENERIC ? 'Also' : (spec.model ? 'If the target is MoE, also' : 'Also')} WebFetch the MoE reference: ${REF_MOE} (or find the qwen3_moe / mixtral modeling file under models/).
5. From the actual reference code, resolve the EXACT import lines for these UNVERIFIED symbols: NeuronAttentionBase, get_rmsnorm_cls / CustomRMSNorm, RotaryEmbedding (and any rope-scaling helper), initialize_moe_module (v1 modules.moe AND v2 modules.moe_v2), the get_tp_group / parallel_state helper, and any GroupQueryAttention pieces. For each, set status "resolved" only if you saw the import in the reference file; else "still-unverified" with your best guess.
5b. CHECK THE COMMUNITY CONTRIB LIBRARY for an already-implemented & validated port of this architecture — these are gold because they already RAN on Neuron. This is especially important when the official package ships no builtin for the target's family.
   - List available contrib models: WebFetch ${CONTRIB_API} (a JSON array of {name,type}). The dirs are model names like "Qwen3-0.6B", "gemma-3-1b-it", "Phi-3.5-mini-instruct", "internlm3-8b-instruct", "glm-4-9b-chat-hf", "Mixtral-8x7B-Instruct-v0.1", etc.
   - Pick the dir whose architecture BEST matches the target "${MODEL}" (same family first; else the closest dense/MoE analogue).
   - WebFetch that model's implementation: ${CONTRIB_RAW}/<dir>/src/modeling_<family>.py (if you don't know the exact filename, first WebFetch ${CONTRIB_API.replace('/contents/contrib/models', '/contents/contrib/models/<dir>/src')} to list it). Also WebFetch its README.md to copy the validation results (token-match %, throughput, TP).
   - Fill contrib_match: model_dir, the raw_url you fetched, is_same_arch (true only if same family), validation (from README), and key_patterns = the concrete things worth copying (exact imports, attention super().__init__ kwargs, converter key renames, any arch quirk it solved). If NOTHING in the library is a usable match, set model_dir="" and key_patterns="" and say why in notes. Prefer a contrib impl over the generic llama reference whenever is_same_arch is true.
6. attention_base_signature: copy the exact super().__init__(...) kwargs the reference attention class passes to NeuronAttentionBase.
7. converter_pattern: summarize precisely how the reference convert_hf_to_neuron_state_dict remaps weights (the key renames, fused qkv, gate-up fusion, padding).
8. moe_module_pattern: how the MoE reference decoder layer builds and calls the expert module (initialize_moe_module args, the [0] tuple indexing, shared experts). "n/a (dense)" if you only looked at dense.
Return the schema.`,
    { label: 'ground:imports', phase: 'Ground', schema: RESOLVED_SCHEMA }
  ),
]).then(r => r)

if (!profile) { log('Profile failed — aborting.'); return { error: 'profile failed' } }
log(`Profiled ${profile.model_name}: ${profile.is_moe ? 'MoE' : 'dense'} ${profile.architecture_family}, TP candidates [${(profile.recommended_tp_degrees||[]).join(', ')}]`)
const resolvedSummary = grounded
  ? `RESOLVED IMPORTS + PATTERNS:\n${JSON.stringify(grounded, null, 2)}`
  : 'Grounding step failed — use the KB import paths and mark UNVERIFIED ones with a TODO comment in the code.'

// =========================================================================
// Phase 3: Scaffold the full modeling_<model>.py
// =========================================================================
phase('Scaffold')

const scaffold = await agent(
  `Generate a COMPLETE, well-structured \`modeling_${slug}.py\` for porting "${profile.model_name}" to AWS NxD Inference. This is a scaffold an engineer will refine — it must be coherent, import-correct, and follow the verified contract exactly.

Use the verified knowledge base (Read it): ${KB_PATH}
Also Read it to copy the exact base-class contracts, skeletons, and gotchas. Load tools first: ToolSearch "select:Read".

MODEL PROFILE:
${JSON.stringify(profile, null, 2)}

${resolvedSummary}

If the grounding above found a contrib_match with is_same_arch=true and a non-empty raw_url, that community implementation ALREADY RAN AND VALIDATED on Neuron — treat it as the PRIMARY blueprint: WebFetch its raw_url (ToolSearch "select:WebFetch") and follow its structure, imports, attention signature, and converter renames closely, adapting only the dims/quirks that differ for "${profile.model_name}". The generic llama/qwen3_moe reference is the fallback when no same-arch contrib match exists.

Requirements for the generated file:
1. Module docstring naming the model + a "GENERATED SCAFFOLD — review TODOs" banner.
2. Imports: use VERIFIED paths from the KB section 11 verbatim. For symbols the grounding step resolved, use the resolved import. For anything still unverified, import it but add a "# TODO(verify import)" comment on that line.
3. ${profile.is_moe ? 'MoENeuronConfig' : 'NeuronConfig'} subclass IF the model needs extra runtime knobs (else use the base directly and say so in a comment).
4. \`${cap(slug)}InferenceConfig(InferenceConfig)\`: get_required_attributes() listing exactly the HF attrs this arch reads (from the profile dims${profile.is_moe ? ' + MoE expert fields' : ''}); get_neuron_config_cls() returning ${profile.is_moe ? 'MoENeuronConfig' : 'NeuronConfig'}; an attribute_map or __init__ normalization for every entry in hf_name_quirks; add_derived_config() for head_dim / num_cores_per_group.
5. Attention class extending NeuronAttentionBase passing UNSHARDED head counts by keyword (head_dim, num_attention_heads=${profile.dims.num_attention_heads}, num_key_value_heads=${profile.dims.num_key_value_heads}), a get_rope(), ${profile.features.qk_norm ? 'per-head q/k RMSNorm wiring (qk_norm),' : ''} ${profile.features.attention_bias ? 'attention bias enabled,' : 'no attention bias,'} sliding_window=${JSON.stringify(profile.features.sliding_window)}.
6. ${profile.is_moe ? 'MoE decoder layer: self.mlp = initialize_moe_module(config=config); take [0] from its output in forward.' : 'Dense MLP (gate/up/down with ColumnParallel/RowParallel) + decoder layer.'} Both with the parallel_state.model_parallel_is_initialized() guard + nn.Linear CPU fallback.
7. Model class extending NeuronBaseModel: setup_attr_for_model (all 7 attrs) + init_model (embed_tokens/layers/norm/lm_head); lm_head gather_output = not on_device_sampling.
8. Task head \`Neuron${cap(slug)}ForCausalLM(NeuronBaseForCausalLM)\`: _model_cls, get_config_cls (@classmethod), load_hf_model (@staticmethod), convert_hf_to_neuron_state_dict (@staticmethod) implementing ${profile.features.fused_qkv_recommended ? 'fused qkv +' : ''} gate-up fusion ${profile.is_moe ? '+ MoE router rename + per-expert weight stacking to [E,H,2I]/[E,I,H]' : ''}${profile.features.tied_word_embeddings ? ', and update_state_dict_for_tied_weights (@staticmethod)' : ''}. Correct decorators are mandatory.
9. A get_tp_group(config) helper and a get_rmsnorm_cls usage consistent with the resolved imports.
10. At the bottom: a commented MODEL_TYPES registration snippet and a short "REMAINING TODOs" comment block.

Output the schema. The "code" field is the entire file. Keep it runnable-shaped (valid Python structure), thorough, and faithful to the verified contract — do NOT invent NeuronConfig flags that the KB says don't exist (attention_dp_degree, is_chunked_prefill, cast_type).`,
  { label: `scaffold:${slug}`, phase: 'Scaffold', schema: SCAFFOLD_SCHEMA, model: 'opus' }
)

if (!scaffold) { log('Scaffold failed — aborting.'); return { error: 'scaffold failed', profile } }

// =========================================================================
// Phase 4: Adversarial review across dimensions
// =========================================================================
phase('Review')

const DIMENSIONS = [
  { key: 'base-contract', focus: 'Base-class contract: setup_attr_for_model sets ALL 7 required attrs; init_model defines embed_tokens/layers/norm/lm_head; NeuronBaseModel __init__/forward NOT wrongly overridden; task-head decorators exactly right (@staticmethod vs @classmethod); _model_cls set.' },
  { key: 'sharding-tp', focus: 'Tensor parallelism & sharding: head counts passed UNSHARDED to NeuronAttentionBase; recommended TP divides num_key_value_heads; ColumnParallelLinear gather_output flags (False before RowParallel; lm_head = not on_device_sampling); ParallelEmbedding args; parallel_state guard + CPU fallback present everywhere.' },
  { key: 'weight-converter', focus: 'convert_hf_to_neuron_state_dict completeness: every HF key remapped, no leftover/missing keys; fused_qkv concatenation matches the fused_qkv setting; gate-up fusion correct; tied weights handled iff tied_word_embeddings; layer-prefix handling consistent with _STATE_DICT_MODEL_PREFIX.' },
  { key: 'imports-flags', focus: 'Import correctness & config flags: imports match KB section 11 (VERIFIED ones verbatim; UNVERIFIED flagged); NO invented NeuronConfig flags (attention_dp_degree/is_chunked_prefill/cast_type); accuracy helpers from utils.accuracy, benchmark from utils.benchmark; class named Neuron<Model>ForCausalLM; concrete classes imported from fully-qualified modules.' },
]
if (profile.is_moe) DIMENSIONS.push({ key: 'moe-specifics', focus: 'MoE specifics: get_neuron_config_cls returns MoENeuronConfig; self.mlp output indexed [0]; num_experts/top_k read from InferenceConfig (NOT MoENeuronConfig); router renamed to router.linear_router.weight; expert weights stacked [E,H,2I]/[E,I,H]; initialize_moe_module call shape (v1 vs v2) consistent.' })

const reviews = await parallel(DIMENSIONS.map(d => () =>
  agent(
    `Adversarially review this generated NxD Inference modeling file along ONE dimension. Be skeptical and specific — assume there are bugs. Read the verified KB for the contract: ${KB_PATH} (load Read via ToolSearch "select:Read").

DIMENSION: ${d.key}
FOCUS: ${d.focus}

MODEL PROFILE: ${JSON.stringify(profile)}

FILE UNDER REVIEW (${scaffold.filename}):
\`\`\`python
${String(scaffold.code).slice(0, 60000)}
\`\`\`

Report only real problems in this dimension with concrete fixes. severity: blocker (won't load/compile or wrong numerics), major (silent correctness/contract violation), minor, nit. If the dimension is clean, return an empty findings array.`,
    { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }
  )
)).then(r => r.filter(Boolean))

const allFindings = reviews.flatMap(r => (r.findings || []).map(f => ({ ...f, dimension: r.dimension })))
const blockers = allFindings.filter(f => f.severity === 'blocker').length
const majors = allFindings.filter(f => f.severity === 'major').length
log(`Review: ${allFindings.length} findings (${blockers} blocker, ${majors} major) across ${reviews.length} dimensions`)

// =========================================================================
// Phase 5: Finalize — corrected code + tailored plan + runbook
// =========================================================================
phase('Finalize')

const FINAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['modeling_code', 'onboarding_plan_md', 'runbook_md', 'changelog'],
  properties: {
    modeling_code: { type: 'string', description: 'the corrected, final modeling_<model>.py' },
    onboarding_plan_md: { type: 'string', description: 'tailored step-by-step onboarding plan for THIS model (markdown)' },
    runbook_md: { type: 'string', description: 'eval + vLLM + benchmark runbook with exact commands (markdown)' },
    changelog: { type: 'array', items: { type: 'string' }, description: 'what was fixed from review' },
  },
}

const final = await agent(
  `Produce the final onboarding artifacts for "${profile.model_name}" on NxD Inference. Apply ALL review fixes to the scaffold and write a tailored plan + runbook. Read the verified KB for exact APIs/commands: ${KB_PATH} (ToolSearch "select:Read").

MODEL PROFILE:
${JSON.stringify(profile, null, 2)}

GENERATED SCAFFOLD (apply fixes to this):
\`\`\`python
${scaffold.code}
\`\`\`

REVIEW FINDINGS TO RESOLVE (prioritize blocker > major):
${JSON.stringify(allFindings, null, 2)}

SCAFFOLD COMPANIONS:
config.json notes: ${scaffold.config_json_notes}
MODEL_TYPES: ${scaffold.model_types_registration}
open questions: ${JSON.stringify(scaffold.open_questions)}

Produce three artifacts:
1. modeling_code: the corrected final file. Fix every blocker and major; apply minors where clearly correct. Keep TODO comments only where an import/value genuinely cannot be resolved without the installed package.
2. onboarding_plan_md: a tailored, sequential plan for THIS model — the 4 steps mapped to concrete actions for ${profile.is_moe ? 'this MoE' : 'this dense'} ${profile.architecture_family} model, including the recommended TP degree(s) [${(profile.recommended_tp_degrees||[]).join(', ')}] and why (kv-head divisibility), the file layout (models/${slug}/modeling_${slug}.py + __init__), the config.json/checkpoint prep, the MODEL_TYPES registration, and the per-step verification checklist from the KB adapted to this model. Include async-mode guidance.
3. runbook_md: copy-paste commands for accuracy validation (generate_expected_logits + check_accuracy_logits_v2 vs check_accuracy; when to use which given on-device sampling), the inference_demo CLI invocation (--model-type ${slug} --task-type causal-lm run --check-accuracy-mode ... --benchmark --on-cpu), benchmark_sampling usage + the latency/throughput metrics it reports, and the vLLM offline + server invocation with VLLM_NEURON_FRAMEWORK and override_neuron_config. Use the model's real name and TP degree in examples.
Return the schema.`,
  { label: `finalize:${slug}`, phase: 'Finalize', schema: FINAL_SCHEMA, model: 'opus' }
)

if (!final) { log('Finalize failed.'); return { error: 'finalize failed', profile, scaffold, findings: allFindings } }

// =========================================================================
// Phase 6: Emit files to disk
// =========================================================================
phase('Emit')

const header = `Generated by the nxdi-onboard-model workflow.\nTarget: ${profile.model_name}  |  ${profile.is_moe ? 'MoE' : 'dense'} ${profile.architecture_family}\nVerified against the NxD Inference onboarding KB. Review TODOs before use.\n`

const emit = await agent(
  `Write three files to disk. Load tools: ToolSearch "select:Bash,Write".
1. Create the directory ${OUT_DIR} (Bash: mkdir -p ${OUT_DIR}).
2. Write ${OUT_DIR}/modeling_${slug}.py with EXACTLY this content (do not edit it):
<<<FILE:modeling>>>
${final.modeling_code}
<<<END>>>
3. Write ${OUT_DIR}/ONBOARDING_PLAN.md with a one-line comment header then this content:
<<<FILE:plan>>>
${final.onboarding_plan_md}
<<<END>>>
4. Write ${OUT_DIR}/RUNBOOK.md with this content:
<<<FILE:runbook>>>
${final.runbook_md}
<<<END>>>
After writing, run \`ls -la ${OUT_DIR}\` and report the absolute paths + byte sizes of the three files. Do NOT summarize the file contents — just confirm they were written.`,
  { label: 'emit:files', phase: 'Emit' }
)

// =========================================================================
// Phase 7+8: Verify on hardware + Repair loop
//
// This is the part that makes "onboarding" mean RUNS, not just COMPILES-on-
// paper. We download the real weights, force vLLM to load the GENERATED file
// (not any builtin implementation) by patching NxDI's MODEL_TYPES via a lazy
// sitecustomize import hook, compile on the Neuron device, and send a chat
// request. If anything fails, the Repair agent reads the INSTALLED package
// source + the real traceback, patches the generated file on disk, and we
// re-verify — up to MAX_REPAIR_ROUNDS times.
// =========================================================================
const GEN_CLASS = `Neuron${cap(slug)}ForCausalLM`
const MODEL_FILE = `${OUT_DIR}/modeling_${slug}.py`
const tpDegree = spec.tp_degree || (profile.recommended_tp_degrees && profile.recommended_tp_degrees.includes(2) ? 2 : (profile.recommended_tp_degrees || [1])[0] || 1)

let verifyResult = null
let repairRounds = 0

const canVerify = WANT_VERIFY && paths.neuron_device_present && paths.vllm_python && paths.nxdi_pkg_dir
if (WANT_VERIFY && !canVerify) {
  log(`Verify skipped: ${!paths.neuron_device_present ? 'no Neuron device' : !paths.vllm_python ? 'no vLLM venv' : 'no installed NxDI pkg'} (${paths.notes || ''}).`)
}

if (canVerify) {
  const VERIFY_SCHEMA = {
    type: 'object', additionalProperties: false,
    required: ['status', 'stage', 'chat_reply', 'error_summary', 'traceback_tail', 'culprit_source', 'log_path'],
    properties: {
      status: { type: 'string', enum: ['success', 'failure'] },
      stage: { type: 'string', enum: ['download', 'patch', 'launch', 'compile', 'load', 'startup', 'chat', 'done'], description: 'furthest stage reached' },
      chat_reply: { type: 'string', description: 'the assistant chat reply text on success, else ""' },
      error_summary: { type: 'string', description: 'one-line root cause if failed, else ""' },
      traceback_tail: { type: 'string', description: 'last ~40 lines of the real traceback/error from the vLLM log if failed, else ""' },
      culprit_source: { type: 'string', description: 'the relevant snippet from the INSTALLED package source that defines the contract being violated (with file:line), or "" — read it from nxdi_pkg_dir to ground the repair' },
      log_path: { type: 'string', description: 'absolute path to the captured vLLM log' },
    },
  }

  const verifyPrompt = (round) => `Verify that the GENERATED NxD Inference modeling file ACTUALLY RUNS on this Neuron host via vLLM, loading the generated file itself (NOT any builtin implementation). Load tools: ToolSearch "select:Bash,Read".

This is round ${round + 1}. Be rigorous and report the REAL outcome — never claim success you did not observe.

INPUTS (absolute):
- Generated model file: ${MODEL_FILE}
- Generated task-head class: ${GEN_CLASS}
- HF model id: ${MODEL}
- vLLM venv python: ${paths.vllm_python}
- vLLM venv bin dir (has libneuronpjrt-path): ${paths.neuron_bin_dir}
- Installed NxDI package dir (read its real source to ground contracts): ${paths.nxdi_pkg_dir}
- TP degree to use: ${tpDegree}
- Work dir: ${OUT_DIR}/_verify

STEPS:
1. Weights: ensure a local copy of ${MODEL} exists at ${OUT_DIR}/_verify/model (a dir with model.safetensors + config.json). If missing, download with: \`${paths.neuron_bin_dir}/huggingface-cli download ${MODEL} --local-dir ${OUT_DIR}/_verify/model\` (the venv bin dir holds huggingface-cli). Skip if already present.
2. Routing key: read config.json "architectures"[0] (e.g. "Qwen2ForCausalLM"). vLLM-neuron routes by splitting on "For": model=lower(part before "For"), task="causal-lm". Apply the same special-cases the loader uses (gptoss->gpt_oss, qwen3moe->qwen3_moe, qwen2vl->qwen2_vl, qwen3vl->qwen3_vl). Call this ROUTEKEY. (You can confirm by reading ${paths.nxdi_pkg_dir}/utils/constants.py MODEL_TYPES keys.)
3. Copy the generated file to an importable module: \`cp ${MODEL_FILE} ${OUT_DIR}/_verify/modeling_gen_mod.py\`.
4. Write ${OUT_DIR}/_verify/sitecustomize.py as a LAZY meta-path import hook that swaps MODEL_TYPES[ROUTEKEY]["causal-lm"] to the generated class. CRITICAL: do NOT \`import neuronx_distributed_inference\` at sitecustomize top level — sitecustomize runs in EVERY python process including the venv's libneuronpjrt-path helper, and eagerly importing NxDI there triggers torch_xla.init -> spawns libneuronpjrt-path -> re-runs sitecustomize -> FORK BOMB. Use a meta-path finder that wraps the loader of "neuronx_distributed_inference.utils.constants" and only then imports the generated class (\`sys.path.insert(0, "${OUT_DIR}/_verify"); from modeling_gen_mod import ${GEN_CLASS} as G; module.MODEL_TYPES[ROUTEKEY]["causal-lm"]=G\`) and prints "[sitecustomize] PATCHED <ROUTEKEY> -> generated" to stderr.
5. Launch vLLM in the background, capturing all output to ${OUT_DIR}/_verify/verify.log:
   \`cd ${OUT_DIR}/_verify && PATH="${paths.neuron_bin_dir}:/opt/aws/neuron/bin:$PATH" PYTHONPATH="${OUT_DIR}/_verify:$PYTHONPATH" PYTHONUNBUFFERED=1 VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_COMPILED_ARTIFACTS=${OUT_DIR}/_verify/artifacts nohup ${paths.vllm_python} -m vllm.entrypoints.openai.api_server --model=${OUT_DIR}/_verify/model --served-model-name verifytarget --max-num-seqs=4 --max-model-len=4096 --tensor-parallel-size=${tpDegree} --no-enable-prefix-caching > ${OUT_DIR}/_verify/verify.log 2>&1 &\`
   GOTCHAS (already known — do not rediscover): NO --device flag (removed in vllm 0.16; the neuron platform plugin auto-activates). --no-enable-prefix-caching is REQUIRED (Neuron has no prefix caching; omitting it crashes pydantic config validation). The PATH MUST include the venv bin dir or you get FileNotFoundError: 'libneuronpjrt-path'.
6. Confirm the patch took: grep the log for "PATCHED" inside an EngineCore/worker line — that proves the GENERATED class is the one being built. If you never see PATCHED, the verification is INVALID (you'd be testing a builtin) — report failure stage "patch".
7. Poll the log (sleep in ~30-60s steps, up to ~12 min total — compilation of a small model takes a few minutes): success signal = "Application startup complete" AND \`curl -s http://localhost:8000/v1/models\` lists "verifytarget". Watch for failure signals: "values to unpack", "missing key"/"unexpected key", "KeyError", "RuntimeError", Python "Traceback".
8. On startup success, send a chat: \`curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"verifytarget","messages":[{"role":"user","content":"What is 17 * 24? Answer with just the number."}],"max_tokens":40,"temperature":0}'\` and read choices[0].message.content. status=success only if you get a coherent reply.
9. ALWAYS kill the server before returning: \`for p in $(pgrep -f vllm.entrypoints); do kill -9 $p; done\` and also \`pkill -9 -f libneuronpjrt-path\` (in case of stray forks); confirm none remain. Leaving it running holds the Neuron device and blocks re-verify.
10. On FAILURE: capture the last ~40 lines of the real error from the log into traceback_tail, write a one-line error_summary, and — crucially — READ the INSTALLED package source under ${paths.nxdi_pkg_dir} that defines the violated contract (e.g. how models/model_base.py consumes each decoder layer's return tuple; what modules/attention/attention_base.py's forward returns; what the builtin models/${profile.architecture_family || 'llama'}/modeling_*.py converter does) and put the exact relevant snippet + file:line into culprit_source. This is what the Repair step needs.
Return the schema with the TRUE observed result.`

  for (let round = 0; round <= MAX_REPAIR_ROUNDS; round++) {
    phase('Verify')
    repairRounds = round
    verifyResult = await agent(verifyPrompt(round), { label: `verify:round${round + 1}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'opus' })

    if (!verifyResult) { log(`Verify round ${round + 1}: agent returned null — stopping.`); break }
    log(`Verify round ${round + 1}: ${verifyResult.status} @ stage=${verifyResult.stage}${verifyResult.status === 'success' ? ` — reply: ${String(verifyResult.chat_reply).slice(0, 60)}` : ` — ${verifyResult.error_summary}`}`)

    if (verifyResult.status === 'success') break
    if (round === MAX_REPAIR_ROUNDS) { log(`Reached max repair rounds (${MAX_REPAIR_ROUNDS}); leaving last error for manual review.`); break }

    // ---- Repair: patch the generated file on disk, grounded in real source ----
    phase('Repair')
    const repair = await agent(
      `The generated NxD Inference modeling file FAILED to run on hardware. Fix the file ON DISK so it loads/compiles/runs, grounding EVERY change in the INSTALLED package source (not guesses). Load tools: ToolSearch "select:Read,Edit,Bash,WebFetch".

FILE TO FIX (edit in place): ${MODEL_FILE}
INSTALLED PACKAGE SOURCE (read freely to confirm contracts): ${paths.nxdi_pkg_dir}
Reference builtin of the same family to copy patterns from: ${paths.nxdi_pkg_dir}/models/${profile.architecture_family || 'llama'}/
${grounded && grounded.contrib_match && grounded.contrib_match.raw_url ? `Community contrib impl that ALREADY RAN on Neuron for this/a-similar arch (WebFetch to compare — its working code is strong evidence for the right contract): ${grounded.contrib_match.raw_url} [validation: ${grounded.contrib_match.validation}]` : ''}

REAL FAILURE (round ${round + 1}):
- stage: ${verifyResult.stage}
- root cause: ${verifyResult.error_summary}
- traceback tail:
${verifyResult.traceback_tail}
- relevant installed-source contract:
${verifyResult.culprit_source}

COMMON CONTRACT BUGS in generated NxDI ports (verify each against the installed source before trusting):
- Decoder layer forward must return the FULL tuple the base consumes. Read ${paths.nxdi_pkg_dir}/models/model_base.py where it does \`layer_outputs = decoder_layer(...)\` then indexes layer_outputs[0..N] — match that arity exactly (often a 5-tuple: hidden, present_kv, cos_cache, sin_cache, residual).
- Attention forward return arity: read modules/attention/attention_base.py (its __iter__ / return) — the call site usually unpacks 4 values (hidden, present_kv, cos_cache, sin_cache). Generated code often wrongly unpacks 2.
- MLP modules frequently return a TUPLE — index [0].
- Converter/module layout MUST agree: if the MLP defines separate gate_proj/up_proj, the converter must NOT fuse them into gate_up_proj (and vice-versa). Missing/unexpected key errors point here.
- Many converters must seed a top-level \`state_dict["rank_util.rank"] = torch.arange(0, tp_degree)\` AND per-layer \`self_attn.rank_util.rank\` — compare to the builtin family converter.
- Imports: confirm each symbol's real module by importing it with the venv python (PATH must include ${paths.neuron_bin_dir}); drop any symbol the package does not export.

Make the MINIMAL set of edits that fixes the observed failure (and any identical sibling bug you can confirm from source). After editing, run \`${paths.vllm_python} -m py_compile ${MODEL_FILE}\` (with PATH including ${paths.neuron_bin_dir}) to confirm it still compiles. Report exactly what you changed and the file:line of the installed-source contract that justifies each change.`,
      { label: `repair:round${round + 1}`, phase: 'Repair', schema: { type: 'object', additionalProperties: false, required: ['changes', 'justification'], properties: { changes: { type: 'array', items: { type: 'string' } }, justification: { type: 'string' } } }, model: 'opus' }
    )
    if (!repair) { log(`Repair round ${round + 1}: agent returned null — stopping.`); break }
    log(`Repair round ${round + 1}: applied ${repair.changes.length} change(s).`)
  }
}

return {
  model: profile.model_name,
  mode: GENERIC ? 'generic-reference' : 'targeted',
  is_moe: profile.is_moe,
  recommended_tp_degrees: profile.recommended_tp_degrees,
  output_dir: OUT_DIR,
  files: [`modeling_${slug}.py`, 'ONBOARDING_PLAN.md', 'RUNBOOK.md'],
  review: { total: allFindings.length, blockers, majors },
  changelog: final.changelog,
  emit_report: emit,
  open_questions: scaffold.open_questions,
  verify: canVerify ? {
    attempted: true,
    status: verifyResult ? verifyResult.status : 'unknown',
    repair_rounds_used: repairRounds,
    chat_reply: verifyResult ? verifyResult.chat_reply : '',
    final_error: verifyResult && verifyResult.status !== 'success' ? verifyResult.error_summary : '',
    tp_degree: tpDegree,
  } : { attempted: false, reason: WANT_VERIFY ? (paths.notes || 'no neuron device / vLLM venv') : 'verify disabled or generic mode' },
}

// ----- helpers -----
function cap(s) {
  return String(s).split(/[_\-]/).filter(Boolean).map(w => w[0].toUpperCase() + w.slice(1)).join('')
}
