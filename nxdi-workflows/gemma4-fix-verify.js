export const meta = {
  name: 'gemma4-fix-verify',
  description: 'Fix the generated Gemma4-unified text-decoder NxDI modeling file against grounded findings, then verify it compiles + generates on Neuron via inference_demo (TP=8), with a source-grounded repair loop.',
  phases: [
    { title: 'Fix', detail: 'apply 8 grounded scaffold fixes against the real Gemma4 reference + checkpoint + installed attention base', model: 'opus' },
    { title: 'Verify', detail: 'compile + generate on Neuron via inference_demo (native NxDI engine; vLLM is blocked by transformers 4.57 vs needed 5.10)', model: 'opus' },
    { title: 'Repair', detail: 'read installed package source + real traceback, patch file, re-verify (loop)', model: 'opus' },
  ],
}

const VENV = '/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16'
const PY = `${VENV}/bin/python`
const BIN = `${VENV}/bin`
const PKG = `${VENV}/lib/python3.12/site-packages/neuronx_distributed_inference`
const OUT = '/home/ubuntu/nxdi-onboarding/model_google_gemma_4_12b_it_instance_trn2_48xlar'
const MODEL_FILE = `${OUT}/modeling_model_google_gemma_4_12b_it_instance_trn2_48xlar.py`
const CKPT = `${OUT}/_verify/model`
const ROUTEKEY = 'model_google_gemma_4_12b_it_instance_trn2_48xlar'
const GEN_CLASS = 'NeuronModelGoogleGemma412bItInstanceTrn248xlarForCausalLM'
const TP = 8
const MAX_REPAIR = Number.isInteger(args && args.max_repair) ? args.max_repair : 3

// Grounding gathered BEFORE this workflow (real Gemma4 reference @ transformers main,
// the actual model.safetensors header, and the installed NeuronAttentionBase source).
// Agents must TRUST this and not waste turns rediscovering it.
const GROUNDING = `
GROUNDED FACTS (verified against the real Gemma4 reference, the actual checkpoint header, and the INSTALLED package — trust these):

A. CHECKPOINT (single model.safetensors, 677 tensors). After the base strips the leading "model." prefix, the converter sees keys like:
   - language_model.embed_tokens.weight [262144,3840]
   - language_model.norm.weight [3840]
   - language_model.layers.N.input_layernorm.weight / post_attention_layernorm.weight / pre_feedforward_layernorm.weight / post_feedforward_layernorm.weight  [3840]  (FOUR norms per layer — all present)
   - language_model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
   - language_model.layers.N.self_attn.q_norm.weight / k_norm.weight   (per-head; dim = head_dim of THAT layer)
   - language_model.layers.N.mlp.{gate_proj,up_proj,down_proj}.weight   (SEPARATE — never fuse)
   - language_model.layers.N.layer_scalar  [1]   <-- 48 of these, ONE PER LAYER. The current scaffold has NO such param and would crash on this UNEXPECTED key.
   - NON-TEXT towers to DROP (the scaffold's drop-list MISSES these real names): model.vision_embedder.*, model.embed_vision.*, model.embed_audio.*  (note: these still carry the leading "model." at converter time only if not stripped — actually after the base strips ONE "model." they look like vision_embedder.* / embed_vision.* / embed_audio.*; drop BOTH "model.X." and "X." forms to be safe). There is NO v_norm in the checkpoint.
   Per-layer geometry CONFIRMED: sliding layers (indices where (i+1)%6 != 0): q_proj [4096,3840]=16x256, k/v_proj [2048,3840]=8x256, q/k_norm [256]. global/full layers (i in {5,11,17,23,29,35,41,47}): q_proj [8192,3840]=16x512, k/v_proj [512,3840]=1x512 (MQA), q/k_norm [512].

B. REAL GEMMA4 TEXT-DECODER FORWARD (from transformers main modeling_gemma4_unified.py):
   1. layer_scalar: registered buffer torch.ones(1); applied at the VERY END of the decoder layer forward: \`hidden_states *= self.layer_scalar\` (scales full post-residual layer output). MUST be added as a parameter/buffer on the decoder layer AND applied last, or the checkpoint key is unexpected and the math is wrong.
   2. Embedding scaling: embeddings are multiplied by embed_scale = hidden_size**0.5 (sqrt(3840)). MUST apply after embed_tokens.
   3. final_logit_softcapping=30.0 applied as: logits = tanh(logits/cap)*cap. MUST wire into the logits path.
   4. Attention scaling: self.scaling = 1.0 HARDCODED (NOT 1/sqrt(head_dim)). In NeuronAttentionBase the score is QK / softmax_scale, and softmax_scale defaults to sqrt(head_dim). To get scaling=1.0 you must pass softmax_scale=1.0 to NeuronAttentionBase (so QK/1.0). DO NOT leave it default.
   5. q_norm/k_norm: per-head RMSNorm over head_dim, applied PRE-ROPE (q_norm(Q) then rope; same for K). v has NO norm. The base supports this NATIVELY: pass use_qk_norm appropriately OR pass q_layernorm/k_layernorm callables — see C. The current scaffold defines q_layernorm/k_layernorm but does NOT wire them; the base DOES apply self.q_layernorm/self.k_layernorm inside move_heads_front, so passing them via __init__ is the correct wiring (NOT use_qk_norm, which builds its OWN qk_norm using config-driven dims).
   6. proportional RoPE (global layers): rope_type='proportional', theta=1e6, partial_rotary_factor=0.25 over head_dim=512. The HF reference routes through ROPE_INIT_FUNCTIONS['proportional'] with head_dim_key='global_head_dim'. The scaffold approximates with rotary_dim=int(512*0.25)=128. KEEP this approximation for v1 (it is the best available; flag it), but make sure RotaryEmbedding actually accepts/uses a reduced rotary_dim < head_dim — verify against the installed modules/attention/rotary or modules/attention/utils RotaryEmbedding signature; if it does NOT support partial rotary, the safest v1 fallback is full rotary on global layers (document the deviation).
   7. RMSNorm convention: Gemma4 multiplies by raw weight (NOT 1+weight). Confirm CustomRMSNorm matches (multiply by weight, ones-init). If CustomRMSNorm uses (1+w), the checkpoint norm weights (which are ~1.0-centered for the multiply-by-w convention) would be wrong — verify the installed CustomRMSNorm.

C. INSTALLED NeuronAttentionBase.__init__ ACCEPTS (verified): head_dim, rotary_emb, softmax_scale, use_qk_norm, qk_norm_placement(=QKNormPlacement.PRE_ROPE default), q_layernorm, k_layernorm, sliding_window, qkv_bias, o_bias, rms_norm_eps, num_attention_heads, num_key_value_heads, tensor_model_parallel_group, config. It applies self.q_layernorm/self.k_layernorm inside move_heads_front(Q,...layernorm=self.q_layernorm). Scores = QK/self.softmax_scale (default sqrt(head_dim)).

D. ENGINE: vLLM is BLOCKED (installed transformers 4.57.6 has no 'gemma4_unified'; vLLM ModelConfig routes through transformers AutoConfig). Use the NATIVE NxDI engine: inference_demo. Its config load is plain json (InferenceConfig.load -> json.loads), and weights load via load_state_dict(dir)+our convert_hf_to_neuron_state_dict — BOTH transformers-independent. So load_hf_model is NOT on the compile/generate path (only accuracy checks use it) — do not let load_hf_model block bring-up.

E. config.json at ${CKPT} is ALREADY FLATTENED (text_config hoisted to top level, model_type='${ROUTEKEY}', architectures=['${GEN_CLASS}']). The original multimodal config is saved as config.multimodal.json. Do NOT re-flatten.
`

phase('Fix')

const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['changes', 'py_compile_ok', 'notes'],
  properties: {
    changes: { type: 'array', items: { type: 'string' } },
    py_compile_ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

const fix = await agent(
  `You are fixing a GENERATED NxD Inference modeling file for the Gemma-4-12B-it TEXT DECODER so it will load + compile + generate on Neuron. Edit the file IN PLACE. Load tools: ToolSearch "select:Read,Edit,Bash".

FILE TO FIX: ${MODEL_FILE}
INSTALLED PACKAGE (read freely to confirm contracts): ${PKG}
  - Attention base: ${PKG}/modules/attention/attention_base.py  (read __init__ + move_heads_front + forward)
  - RMSNorm: ${PKG}/modules/custom_calls.py (CustomRMSNorm — confirm multiply-by-weight vs 1+weight)
  - RotaryEmbedding: ${PKG}/modules/attention/utils.py (confirm whether rotary_dim < head_dim partial rotary is supported)
  - Base model loop + logits: ${PKG}/models/model_base.py (how decoder layer return tuple is consumed; where/whether logit softcap or lm_head is applied; whether embed scaling hook exists)
Reference family already shipped: ${PKG}/models/gemma3/modeling_gemma3.py (Gemma3 is the closest ANCESTOR — copy proven patterns for four-norm layout, embed scaling, logit softcap, qk handling).

${GROUNDING}

APPLY THESE FIXES (each grounded above). For every change, first READ the relevant installed source to confirm the exact contract, then edit:
1. layer_scalar: add a per-layer buffer \`self.register_buffer("layer_scalar", torch.ones(1))\` on GemmaDecoderLayer (so the checkpoint key layers.N.layer_scalar loads) and apply \`hidden_states = hidden_states * self.layer_scalar\` as the LAST op before return. Confirm the converter does NOT drop/rename it.
2. Embedding scale by sqrt(hidden_size): apply after embed_tokens. Check how gemma3 does it in the installed source and match (e.g. an embed_scale buffer or a forward hook). If NeuronBaseModel has no hook, the cleanest is to scale in the model forward / wrap embed_tokens — match gemma3.
3. final_logit_softcapping=30.0: wire tanh cap into the logits path. Check gemma3 + NeuronBaseForCausalLM for the supported override hook; match it.
4. Attention softmax_scale: pass softmax_scale=1.0 to NeuronAttentionBase for BOTH layer types (Gemma4 scaling is hardcoded 1.0). Verify the kwarg name against the base __init__.
5. qk_norm wiring: the scaffold passes q_layernorm/k_layernorm-as-locals but the CORRECT path is to pass q_layernorm=<RMSNorm(head_dim)> and k_layernorm=<RMSNorm(head_dim)> INTO NeuronAttentionBase.__init__ (the base applies them in move_heads_front PRE-ROPE). Remove the post-hoc self.q_layernorm/self.k_layernorm assignments that are never used, and instead pass them as __init__ kwargs. Make sure NOT to also enable use_qk_norm (that builds a DIFFERENT qk_norm). v has no norm.
6. Converter: (a) fix the vision/audio drop-list to the REAL names (vision_embedder, embed_vision, embed_audio — both with and without a leading "model."). (b) Ensure q_norm/k_norm rename targets match the module attribute names you actually pass to the base (if you pass q_layernorm=..., the loaded key must be self_attn.q_layernorm.weight — so rename self_attn.q_norm.weight -> self_attn.q_layernorm.weight, which the scaffold already does; CONFIRM the base names the submodule q_layernorm). (c) Keep gate/up SEPARATE. (d) Keep the global-KV-head replication (1 -> tp_degree) for k_proj AND v_proj AND the matching q? no — only k/v are MQA; q stays 16 heads. (e) Keep rank_util.rank buffer. (f) layer_scalar passes through untouched.
7. proportional RoPE: confirm RotaryEmbedding(rotary_dim=128, ...) for global layers is accepted by the installed RotaryEmbedding (rotary_dim < head_dim). If NOT supported, fall back to full rotary on global layers and add a clear comment documenting the v1 deviation.
8. RMSNorm: confirm CustomRMSNorm matches Gemma's multiply-by-weight (ones-init) convention by reading custom_calls.py. If it uses (1+w), subclass/adjust so the checkpoint norm weights apply correctly.

After editing, run: \`PATH="${BIN}:/opt/aws/neuron/bin:$PATH" ${PY} -m py_compile ${MODEL_FILE}\` and report py_compile_ok. Keep edits MINIMAL and grounded; cite the installed file:line that justifies each non-trivial change in notes. Do NOT attempt to run the model (that is the next phase).`,
  { label: 'fix', phase: 'Fix', schema: FIX_SCHEMA, model: 'opus' }
)

log(`Fix: ${fix ? fix.changes.length + ' changes, py_compile=' + fix.py_compile_ok : 'agent returned null'}`)
if (!fix || !fix.py_compile_ok) {
  log('Fix did not produce a compiling file; continuing to verify anyway to capture the real error.')
}

// ---- Verify + Repair loop via inference_demo ----
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'stage', 'generated_text', 'error_summary', 'traceback_tail', 'culprit_source', 'log_path'],
  properties: {
    status: { type: 'string', enum: ['success', 'failure'] },
    stage: { type: 'string', enum: ['register', 'config', 'weights', 'trace', 'compile', 'load', 'generate', 'done'] },
    generated_text: { type: 'string' },
    error_summary: { type: 'string' },
    traceback_tail: { type: 'string' },
    culprit_source: { type: 'string' },
    log_path: { type: 'string' },
  },
}

const verifyPrompt = (round) => `Verify the GENERATED Gemma-4 text-decoder modeling file ACTUALLY COMPILES + GENERATES on this Neuron host using the NATIVE NxDI engine (NOT vLLM — vLLM is blocked by transformers 4.57 vs the 5.10 this model needs). Round ${round + 1}. Report the REAL outcome; never claim success you did not observe. Load tools: ToolSearch "select:Bash,Read".

INPUTS (absolute):
- Generated model file: ${MODEL_FILE}
- Generated class: ${GEN_CLASS}   routing key (model_type): ${ROUTEKEY}
- Flattened checkpoint dir (config.json already text-only): ${CKPT}
- vLLM/NxDI venv python: ${PY}    venv bin (MUST be on PATH for libneuronpjrt-path): ${BIN}
- Installed NxDI package: ${PKG}
- TP degree: ${TP}
- Work dir: ${OUT}/_verify

${GROUNDING}

STEPS:
1. Free Neuron devices first: \`pkill -9 -f inference_demo; pkill -9 -f neuronx_distributed; pkill -9 -f libneuronpjrt-path; pkill -9 -f vllm\` (ignore errors), then confirm with \`(neuron-ls 2>/dev/null||${BIN}/neuron-ls)|grep -iE "python|EngineCore" || echo FREE\`.
2. Make the generated class importable + registered. Copy the file: \`cp ${MODEL_FILE} ${OUT}/_verify/modeling_gen_mod.py\`. Determine HOW inference_demo resolves model_cls from --model-type: read ${PKG}/inference_demo.py and ${PKG}/utils/constants.py MODEL_TYPES. The robust approach: write ${OUT}/_verify/sitecustomize.py that, AFTER the constants module loads, injects MODEL_TYPES['${ROUTEKEY}']={'causal-lm': <generated class>}. CRITICAL anti-fork-bomb rule: do NOT \`import neuronx_distributed_inference\` at sitecustomize top level (sitecustomize runs in EVERY python proc incl. the libneuronpjrt-path helper -> torch_xla.init -> re-spawn -> fork bomb). Use a meta-path finder that wraps the loader for "neuronx_distributed_inference.utils.constants" and only in its exec_module post-step does \`sys.path.insert(0,"${OUT}/_verify"); from modeling_gen_mod import ${GEN_CLASS} as G; mod.MODEL_TYPES["${ROUTEKEY}"]={"causal-lm":G}\` then prints "[sitecustomize] REGISTERED ${ROUTEKEY}" to stderr. (If reading inference_demo shows an easier official registration hook, use that instead — but the sitecustomize approach is known-good.)
3. Run inference_demo on-device in the background, logging to ${OUT}/_verify/verify_r${round + 1}.log:
   \`cd ${OUT}/_verify && PATH="${BIN}:/opt/aws/neuron/bin:$PATH" PYTHONPATH="${OUT}/_verify:$PYTHONPATH" PYTHONUNBUFFERED=1 NEURON_RT_NUM_CORES=${TP} nohup ${BIN}/inference_demo --model-type ${ROUTEKEY} --task-type causal-lm run --model-path ${CKPT} --tp-degree ${TP} --prompt "The capital of France is" --top-k 1 --do-sample false --pad-token-id 0 --max-context-length 256 --seq-len 512 --check-accuracy-mode skip-accuracy-check > ${OUT}/_verify/verify_r${round + 1}.log 2>&1 &\`
   (First read \`${BIN}/inference_demo --help\` and the run-subcommand args in inference_demo.py to confirm EXACT flag names — adjust the above to the real flags; the must-haves are: model-type, task-type causal-lm, model-path, tp-degree ${TP}, a prompt, skip accuracy check, and small context/seq len to keep compile fast.)
4. Confirm registration took: grep the log for "REGISTERED ${ROUTEKEY}". If absent, the run is testing the WRONG class -> report stage "register" failure.
5. Poll the log (sleep 30-60s, up to ~15 min — first compile of a 12B model takes several minutes). SUCCESS = the log shows generated output text for the prompt (a coherent continuation) and no traceback. Capture the generated continuation into generated_text.
   FAILURE signals: Python "Traceback", "KeyError", "Unexpected key(s)"/"Missing key(s)", "size mismatch", "not divisible", "AssertionError", compiler errors.
6. ALWAYS kill all procs before returning: \`pkill -9 -f inference_demo; pkill -9 -f libneuronpjrt-path; pkill -9 -f neuronx_distributed\`; confirm devices FREE.
7. On FAILURE: put the last ~40 lines of the REAL error in traceback_tail, a one-line error_summary, and READ the INSTALLED source under ${PKG} that defines the violated contract (model_base.py decoder-loop unpacking, attention_base.py move_heads_front/forward, custom_calls.py RMSNorm, the converter expectations) and put the exact snippet + file:line into culprit_source.
Return the schema with the TRUE observed result.`

let verifyResult = null
let rounds = 0
for (let round = 0; round <= MAX_REPAIR; round++) {
  phase('Verify')
  rounds = round
  verifyResult = await agent(verifyPrompt(round), { label: `verify:r${round + 1}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'opus' })
  if (!verifyResult) { log(`Verify r${round + 1}: null — stop.`); break }
  log(`Verify r${round + 1}: ${verifyResult.status} @ ${verifyResult.stage}${verifyResult.status === 'success' ? ` — "${String(verifyResult.generated_text).slice(0, 80)}"` : ` — ${verifyResult.error_summary}`}`)
  if (verifyResult.status === 'success') break
  if (round === MAX_REPAIR) { log(`Max repair rounds reached.`); break }

  phase('Repair')
  const repair = await agent(
    `The generated Gemma-4 text-decoder file FAILED on Neuron. Fix it ON DISK, grounding EVERY change in INSTALLED package source (not guesses). Load tools: ToolSearch "select:Read,Edit,Bash,WebFetch".

FILE TO FIX (edit in place): ${MODEL_FILE}
INSTALLED PACKAGE SOURCE: ${PKG}
Reference ancestor (copy proven patterns): ${PKG}/models/gemma3/modeling_gemma3.py
Real Gemma4 reference (WebFetch if you need exact forward math): https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py

${GROUNDING}

REAL FAILURE (round ${round + 1}):
- stage: ${verifyResult.stage}
- root cause: ${verifyResult.error_summary}
- traceback tail:
${verifyResult.traceback_tail}
- installed-source contract:
${verifyResult.culprit_source}

Make the MINIMAL grounded edits that fix the observed failure (and any identical sibling bug you can confirm from source). Common NxDI contract bugs to check against installed source: decoder-layer return tuple arity (model_base.py), attention forward return (attention_base.py — it returns a NeuronAttentionBaseOutput dataclass; read .hidden_states/.present_key_value), MLP returning a tuple, converter key mismatches (missing/unexpected keys), softmax_scale kwarg name, RMSNorm convention. After editing run \`PATH="${BIN}:/opt/aws/neuron/bin:$PATH" ${PY} -m py_compile ${MODEL_FILE}\`. Report exactly what changed and the file:line justifying each change.`,
    { label: `repair:r${round + 1}`, phase: 'Repair', schema: { type: 'object', additionalProperties: false, required: ['changes', 'justification'], properties: { changes: { type: 'array', items: { type: 'string' } }, justification: { type: 'string' } } }, model: 'opus' }
  )
  if (!repair) { log(`Repair r${round + 1}: null — stop.`); break }
  log(`Repair r${round + 1}: ${repair.changes.length} change(s).`)
}

return {
  model: 'google/gemma-4-12B-it (text decoder)',
  engine: 'inference_demo (native NxDI; vLLM blocked by transformers 4.57<5.10)',
  tp_degree: TP,
  verify_status: verifyResult ? verifyResult.status : 'unknown',
  stage: verifyResult ? verifyResult.stage : '',
  generated_text: verifyResult ? verifyResult.generated_text : '',
  final_error: verifyResult && verifyResult.status !== 'success' ? verifyResult.error_summary : '',
  repair_rounds_used: rounds,
  model_file: MODEL_FILE,
}
