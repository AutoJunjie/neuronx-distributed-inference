#!/usr/bin/env python3
"""
Integration tests for the Gemma-4-12B-it (text decoder) NeuronX implementation.

Tests model compilation, loading, greedy next-token parity against a known
reference token, and a chat-template smoke test. Validated on trn2.48xlarge
(TP=8, bf16).

Set MODEL_PATH / COMPILED_MODEL_PATH below to your checkpoint and compile dir.
The checkpoint dir must contain the (text-decoder) config.json, the safetensors
weights, and the tokenizer files.
"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.accuracy import (
    get_generate_outputs_from_token_ids,
)

# Import the model classes from src/ (this also AutoConfig.register-s the
# custom model_type so the config loads on older transformers builds).
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_gemma4 import NeuronGemma4ForCausalLM, Gemma4InferenceConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/gemma-4-12B-it/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/gemma-4-12B-it/"

TP_DEGREE = 8
SEQ_LEN = 512
MAX_CONTEXT_LENGTH = 256

# Golden reference (transformers-main fp32 CPU) for the RAW prompt
# "The capital of France is" (input_ids [818, 5279, 529, 7001, 563]):
# greedy next-token argmax is id 236770 ('1'). gemma-4-12B-it is instruction
# tuned, so the raw-prompt continuation is intentionally degenerate ("111...");
# we assert the FIRST greedy token id, which is a strict numerical-parity check.
RAW_PROMPT_INPUT_IDS = [818, 5279, 529, 7001, 563]
GOLDEN_FIRST_TOKEN_ID = 236770


def _build_config():
    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=SEQ_LEN,
        max_context_length=MAX_CONTEXT_LENGTH,
        torch_dtype=torch.bfloat16,
    )
    return Gemma4InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (if needed) and load the model."""
    if not (Path(COMPILED_MODEL_PATH) / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH} ...")
        model = NeuronGemma4ForCausalLM(MODEL_PATH, _build_config())
        model.compile(COMPILED_MODEL_PATH)
    model = NeuronGemma4ForCausalLM(MODEL_PATH, _build_config())
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_model_loads(compiled_model):
    """Smoke test: the model loads and exposes its config."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: smoke test (model loaded)")


def test_greedy_token_parity(compiled_model, tokenizer):
    """Greedy next-token for the raw prompt must match the golden reference id."""
    ids = torch.tensor(RAW_PROMPT_INPUT_IDS, dtype=torch.long)
    gen_cfg = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=1, pad_token_id=0,
    )
    outputs, _ = get_generate_outputs_from_token_ids(
        compiled_model, [ids], tokenizer, is_hf=False, generation_config=gen_cfg,
    )
    seq = outputs[0] if not hasattr(outputs, "sequences") else outputs.sequences[0]
    first_new = seq.tolist()[len(RAW_PROMPT_INPUT_IDS)]
    assert first_new == GOLDEN_FIRST_TOKEN_ID, (
        f"first greedy token {first_new} != golden {GOLDEN_FIRST_TOKEN_ID}"
    )
    print(f"PASS: greedy token parity (id {first_new})")


def test_chat_template_paris(compiled_model, tokenizer):
    """Chat-template end-to-end smoke test: should answer with Paris."""
    msgs = [{"role": "user",
             "content": "What is the capital of France? Answer in one short sentence."}]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
    )[0].to(torch.long)
    gen_cfg = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=40,
        pad_token_id=0, eos_token_id=[1, 106],
    )
    outputs, _ = get_generate_outputs_from_token_ids(
        compiled_model, [ids], tokenizer, is_hf=False, generation_config=gen_cfg,
    )
    seq = outputs[0] if not hasattr(outputs, "sequences") else outputs.sequences[0]
    text = tokenizer.decode(seq.tolist()[ids.shape[0]:], skip_special_tokens=True)
    print(f"  chat output: {text!r}")
    assert "Paris" in text, f"expected 'Paris' in output, got: {text!r}"
    print("PASS: chat-template smoke test")


if __name__ == "__main__":
    print("=" * 70)
    print("Gemma-4-12B-it integration tests")
    print("=" * 70)
    if not (Path(COMPILED_MODEL_PATH) / "model.pt").exists():
        print(f"Compiling to {COMPILED_MODEL_PATH} ...")
        m = NeuronGemma4ForCausalLM(MODEL_PATH, _build_config())
        m.compile(COMPILED_MODEL_PATH)
    m = NeuronGemma4ForCausalLM(MODEL_PATH, _build_config())
    m.load(COMPILED_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test_model_loads(m)
    test_greedy_token_parity(m, tok)
    test_chat_template_paris(m, tok)
    print("=" * 70)
    print("All tests passed.")
    print("=" * 70)
