# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the TPU Reflection Beam model.

Covers registration, the config checks, the decoder-stack wiring and the pure
PyTorch ops, on CPU and on TPU.
"""

import functools
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import (
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.models.beam.nvidia.model import (
    _depth_scaled_coeffs as upstream_depth_scaled_coeffs,
)
from vllm.transformers_utils.configs.beam import BeamConfig

import vllm_torchtpu.models.beam.model as beam_model
from vllm_torchtpu.models.beam import ops
from vllm_torchtpu.models.beam.config import TPUBeamConfig
from vllm_torchtpu.models.beam.model import BeamForCausalLM, _depth_scaled_coeffs
from vllm_torchtpu.models.vllm import register_models

BF16, FP32 = torch.bfloat16, torch.float32

# The fields of the released Beam-1.0-501B-A23B config.json.
RELEASED_CONFIG = dict(
    vocab_size=151936,
    hidden_size=5120,
    intermediate_size=20480,
    num_hidden_layers=52,
    num_attention_heads=48,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=1048576,
    rms_norm_eps=1e-06,
    tie_word_embeddings=False,
    rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    },
    sliding_window=512,
    # Five sliding layers per full one, then three sliding and one full: 52.
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 8
    + ["sliding_attention"] * 3
    + ["full_attention"],
    mlp_layer_types=["dense"] * 3 + ["sparse"] * 49,
    moe_intermediate_size=2560,
    num_experts_per_tok=8,
    num_experts=256,
    norm_topk_prob=True,
    n_group=8,
    topk_group=4,
    routed_scaling_factor=2.5,
    final_logit_softcapping=30.0,
)

# Op-test heads: 6 Q heads and 1 K / V head of 128 channels.
HEAD_DIM, Q_HEADS = 128, 6
EPS, SOFTCAP = 1e-6, 30.0
# A two-layer Beam small enough to build and load on the host.
TINY_CONFIG = dict(
    vocab_size=256, hidden_size=128, intermediate_size=256, moe_intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=8, head_dim=16,
    num_experts=16, num_experts_per_tok=4, rms_norm_eps=EPS,
    layer_types=["sliding_attention", "full_attention"],
    mlp_layer_types=["dense", "sparse"],
)  # fmt: skip
# Positions up to the last of the released max_position_embeddings.
SSMAX_POSITIONS = [0, 1, 15, 2047, 2048, 16384, 100000, 1048575]


def test_beam_resolves_to_this_package():
    """BeamForCausalLM resolves to this package (over upstream's registry entry)."""
    register_models()
    model_cls = ModelRegistry.models["BeamForCausalLM"].load_model_cls()
    assert model_cls is BeamForCausalLM
    assert MODELS_CONFIG_MAP["BeamForCausalLM"] is TPUBeamConfig


UNSUPPORTED_FLAGS = (
    "TPU_MOE_SEQUENCE_PARALLEL",
    "VLLM_TPU_BLOCK_MAJOR_KV",
    "TPU_GREEDY_TOP_TOKENS",
)


@pytest.fixture
def clean_check_flags(monkeypatch):
    """Unsets the TPU flags the config checks reject."""
    for name in UNSUPPORTED_FLAGS:
        monkeypatch.delenv(name, raising=False)


_PARALLEL_DEFAULTS = dict(
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    enable_eplb=False,
)


def _config_for_checks(
    dtype=BF16, speculative_config=None, lora_config=None, **overrides
):
    """A VllmConfig stand-in: the released Beam config, no extra features.

    ``overrides`` set parallel_config fields, or else BeamConfig fields.
    """
    parallel = {
        key: overrides.pop(key, value) for key, value in _PARALLEL_DEFAULTS.items()
    }
    hf_config = BeamConfig(**{**RELEASED_CONFIG, **overrides})
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=dtype),
        parallel_config=SimpleNamespace(**parallel),
        speculative_config=speculative_config,
        lora_config=lora_config,
        # As arg_utils fills it from the model's sliding_window.
        cache_config=SimpleNamespace(sliding_window=hf_config.sliding_window),
    )


def test_config_accepts_released_model(clean_check_flags):
    config = _config_for_checks()
    assert config.cache_config.sliding_window is not None
    TPUBeamConfig.verify_and_update_config(config)
    # The full-attention layers must not inherit the window.
    assert config.cache_config.sliding_window is None


_LINEAR_ROPE = {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0}
REJECTED_CONFIGS = {  # id: (overrides, error match)
    "dtype": ({"dtype": torch.float16}, "--dtype bfloat16"),
    "spec": ({"speculative_config": object()}, "--speculative-config"),
    "lora": ({"lora_config": object()}, "--enable-lora"),
    "pp": ({"pipeline_parallel_size": 2}, "--pipeline-parallel-size 1"),
    "pcp": ({"prefill_context_parallel_size": 2}, "context parallelism"),
    "dcp": ({"decode_context_parallel_size": 2}, "context parallelism"),
    "eplb": ({"enable_eplb": True}, "--enable-eplb"),
    # Divides the 48 Q heads but not the 8 KV heads.
    "tp": ({"tensor_parallel_size": 3}, r"use one of \[1, 2, 4, 8\], got 3"),
    "hidden_act": ({"hidden_act": "gelu"}, "hidden_act='silu'"),
    "rope_type": ({"rope_parameters": _LINEAR_ROPE}, "rope_type='default'"),
    "tied_embeddings": ({"tie_word_embeddings": True}, "tie_word_embeddings=False"),
}


@pytest.mark.parametrize(
    "overrides,match", REJECTED_CONFIGS.values(), ids=REJECTED_CONFIGS.keys()
)
def test_config_rejects_unsupported(clean_check_flags, overrides, match):
    """Rejected when the config is built, before any worker starts."""
    with pytest.raises(NotImplementedError, match=match):
        TPUBeamConfig.verify_and_update_config(_config_for_checks(**overrides))


@pytest.mark.parametrize("flag", UNSUPPORTED_FLAGS)
def test_config_rejects_unsupported_flag(clean_check_flags, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(NotImplementedError, match=f"unset {flag}"):
        TPUBeamConfig.verify_and_update_config(_config_for_checks())


@pytest.mark.parametrize("is_attention", [True, False], ids=["attention", "mlp"])
def test_depth_scaled_coeffs_match_upstream(is_attention):
    """The copied residual coefficients equal upstream's on all 52 released layers."""
    for layer_id in range(52):
        assert _depth_scaled_coeffs(layer_id, is_attention) == (
            upstream_depth_scaled_coeffs(layer_id, is_attention)
        )


def _randn(generator, *shape, dtype=BF16, scale=1.0):
    return (torch.randn(*shape, generator=generator) * scale).to(dtype)


def _strided(generator, rows, cols, *, scale=1.0):
    """An unaligned ``[rows, cols]`` slice inside wider padded rows, like packed QKV."""
    row_padding = 64
    wide = _randn(generator, rows, cols + row_padding, scale=scale)
    return wide[:, 1 : cols + 1]


def _ref_rms(x, weight):
    normed = F.rms_norm(x.float(), (x.shape[-1],), eps=EPS)
    return (normed * weight.float()).to(x.dtype)


def _rms_norm_case(generator):
    x, weight = _strided(generator, 33, 513), _randn(generator, 513)
    return ops.beam_rms_norm, (x, weight, EPS), (_ref_rms(x, weight),)


def _norm_residual_norm_case(generator):
    x = _strided(generator, 33, 256)
    residual = _randn(generator, 33, 256)
    post_weight, next_weight = _randn(generator, 256), _randn(generator, 256)
    # Not powers of two, so rounding after each scale changes bits (1/sqrt(n)).
    scales = ops.ResidualScales(branch=3**-0.5, residual=(2 / 3) ** 0.5)

    def op(x, residual, post_weight, next_weight):
        return ops.beam_norm_residual_norm(
            x,
            residual,
            post_weight=post_weight,
            next_weight=next_weight,
            eps=EPS,
            scales=scales,
        )

    # Rounds after the post-norm affine, after each scale and after the add.
    scaled_branch = (_ref_rms(x, post_weight).float() * scales.branch).to(BF16)
    scaled_residual = (residual.float() * scales.residual).to(BF16)
    hidden = (scaled_branch.float() + scaled_residual.float()).to(BF16)
    args = (x, residual, post_weight, next_weight)
    return op, args, (hidden, _ref_rms(hidden, next_weight))


def _silu_mul_case(generator):
    x = _randn(generator, 33, 1024, scale=4.0)
    gate, up = x.float().chunk(2, -1)
    return ops.beam_silu_mul, (x,), ((F.silu(gate) * up).to(BF16),)


def _softcap_case(generator, cap=SOFTCAP):
    logits = _strided(generator, 8, 257, scale=90.0)
    # Upstream's _softcap_logits_kernel: one FP32 expression, rounded once.
    expected = (torch.tanh(logits.float() * (1.0 / cap)) * cap).to(BF16)
    return ops.beam_softcap_logits, (logits, cap), (expected,)


def _qk_inputs(generator, rows):
    """Q and K sliced from a packed QKV projection, and their norm weights."""
    qkv = _strided(generator, rows, (Q_HEADS + 2) * HEAD_DIM)
    q, k, _ = qkv.split((Q_HEADS * HEAD_DIM, HEAD_DIM, HEAD_DIM), -1)
    q_weight = _randn(generator, HEAD_DIM)
    k_weight = _randn(generator, HEAD_DIM)
    return q, k, q_weight, k_weight


def _head_rms(x, weight):
    return _ref_rms(x.unflatten(-1, (-1, HEAD_DIM)), weight).flatten(-2)


def _qk_norm(q, k, q_weight, k_weight, positions, rotary_emb):
    return ops.beam_qk_norm_rope(
        q,
        k,
        q_weight=q_weight,
        k_weight=k_weight,
        eps=EPS,
        positions=positions,
        rotary_emb=rotary_emb,
    )


def _qk_norm_rope_case(generator):
    q, k, q_weight, k_weight = _qk_inputs(generator, 17)
    positions = torch.arange(17) * 3
    max_position = 51
    # Random angles rather than a RoPE schedule, so every channel pair differs.
    angle = torch.randn(max_position, HEAD_DIM // 2, generator=generator)
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        # Base is unused; cos_sin_cache is replaced below.
        rope = RotaryEmbedding(
            HEAD_DIM, HEAD_DIM, max_position, 10000.0, is_neox_style=True, dtype=FP32
        )
    rope.cos_sin_cache = cache
    q_rotated, k_rotated = RotaryEmbedding.forward_static(
        positions, _head_rms(q, q_weight).float(), _head_rms(k, k_weight).float(),
        HEAD_DIM, HEAD_DIM, cache, True,
    )  # fmt: skip
    args = (q, k, q_weight, k_weight, positions, rope)
    return _qk_norm, args, (q_rotated.to(BF16), k_rotated.to(BF16))


def _qk_norm_ssmax_case(generator):
    q, k, q_weight, k_weight = _qk_inputs(generator, len(SSMAX_POSITIONS))
    positions = torch.tensor(SSMAX_POSITIONS, dtype=torch.int32)
    # Beam's SSMax: Q * (1 + (position + 1) / 2048) ** 0.1, here in float64.
    scale = torch.pow(1.0 + (positions.double() + 1.0) / 2048.0, 0.1).float()
    q_heads = _head_rms(q, q_weight).float().unflatten(-1, (Q_HEADS, HEAD_DIM))
    q_ref = (q_heads * scale[:, None, None]).to(BF16).flatten(-2)
    # K is normalized but never position-scaled.
    args = (q, k, q_weight, k_weight, positions, None)
    return _qk_norm, args, (q_ref, _head_rms(k, k_weight))


def _output_gate_case(generator):
    attn_output = _randn(generator, 17, Q_HEADS * HEAD_DIM)
    bias = _randn(generator, Q_HEADS)
    gate = _strided(generator, 17, Q_HEADS * HEAD_DIM, scale=8.0)
    heads = (17, Q_HEADS, HEAD_DIM)
    gate_logits = (gate.float().view(heads) + bias.float()[:, None]).to(BF16)
    gate_sigmoid = torch.sigmoid(gate_logits.float()).to(BF16)
    gated = attn_output.float().view(heads) * gate_sigmoid.float()
    args = (attn_output, gate, bias)
    return ops.beam_output_gate, args, (gated.flatten(-2).to(BF16),)


# (op, args, expected) in BF16, where the kernels' rounding points show.
OP_CASES = {
    "rms_norm": _rms_norm_case,
    "norm_residual_norm": _norm_residual_norm_case,
    "silu_mul": _silu_mul_case,
    "softcap": _softcap_case,
    "qk_norm_rope": _qk_norm_rope_case,
    "qk_norm_ssmax": _qk_norm_ssmax_case,
    "output_gate": _output_gate_case,
}


def _as_tuple(output):
    return output if isinstance(output, tuple) else (output,)


@pytest.mark.parametrize("case", OP_CASES.values(), ids=OP_CASES.keys())
@torch.no_grad()
def test_beam_op_matches_reference(case):
    """Each op matches its fp32 reference and its rounding points, bit-exact."""
    op, args, expected = case(torch.Generator().manual_seed(0))
    for value, reference in zip(_as_tuple(op(*args)), expected, strict=True):
        assert value.dtype == reference.dtype == BF16
        # Bit-exact: a dropped or extra rounding step changes some elements.
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@torch.no_grad()
def test_softcap_logits_writes_in_place():
    """The softcap writes into its (strided) input and leaves the rest alone."""
    vocab, padded_vocab = 257, 264
    storage = _randn(torch.Generator().manual_seed(0), 8, padded_vocab, scale=90.0)
    padding = storage[:, vocab:].clone()
    logits = storage[:, :vocab]
    capped = (torch.tanh(logits.float() * (1.0 / SOFTCAP)) * SOFTCAP).to(BF16)
    assert ops.beam_softcap_logits(logits, SOFTCAP) is logits
    assert torch.equal(storage[:, :vocab], capped)
    assert torch.equal(storage[:, vocab:], padding)


class _StubSublayer(nn.Module):
    """Stands in for attention or an MLP: ``tanh`` of a fixed random linear."""

    def __init__(self, vllm_config, prefix):
        super().__init__()
        hidden = vllm_config.model_config.hf_config.hidden_size
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.positions = []

    def forward(self, *args):
        # Attention is called as (positions, x), the MLP as (x).
        *positions, x = args
        self.positions.extend(positions)
        return torch.tanh(self.proj(x))


class _StubEmbedding(nn.Embedding):
    """VocabParallelEmbedding without a TP group; ignores quant_config and prefix."""

    def __init__(self, vocab_size, hidden_size, **kwargs):
        super().__init__(vocab_size, hidden_size)


@torch.no_grad()
def test_model_matches_reference_decoder_stack(monkeypatch):
    """Two decoder layers with stubbed sublayers match an unfused reference.

    Covers what the op tests cannot: the depth-scaled coefficients of each add,
    the order of the four norms, the next layer's input norm fused into the
    previous layer (``next_norm_weight``) and the final norm. FP32, so the ops'
    rounding points (tested per op) do not matter here.
    """
    for name in ("BeamAttention", "BeamMLP", "BeamSparseMoeBlock"):
        monkeypatch.setattr(beam_model, name, _StubSublayer)
    monkeypatch.setattr(beam_model, "VocabParallelEmbedding", _StubEmbedding)
    config = BeamConfig(**TINY_CONFIG)
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = SimpleNamespace(hf_config=config)
    vllm_config.compilation_config.mode = CompilationMode.NONE
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.tensor([1, 5, 7, 30, 2])
    positions = torch.arange(input_ids.numel())
    with set_current_vllm_config(vllm_config):
        model = beam_model.BeamModel(vllm_config=vllm_config, prefix="model")
        # Distinct norm weights, so a swapped or skipped norm changes the output.
        for name, param in model.named_parameters():
            mean, std = (1.0, 0.5) if name.endswith("norm.weight") else (0.0, 0.1)
            param.normal_(mean, std, generator=generator)
        output = model(input_ids, positions)
    # Each attention stub saw the model's own positions.
    for layer in model.layers:
        assert [p is positions for p in layer.self_attn.positions] == [True]

    hidden = model.embed_tokens(input_ids)
    for one_based, layer in enumerate(model.layers, start=1):
        for num_adds, block, pre_norm, post_norm in (
            (2 * one_based, functools.partial(layer.self_attn, positions),
             layer.input_layernorm, layer.post_attention_layernorm),
            (2 * one_based + 1, layer.mlp, layer.pre_feedforward_layernorm,
             layer.post_feedforward_layernorm),
        ):  # fmt: skip
            branch = block(_ref_rms(hidden, pre_norm.weight))
            hidden = (
                _ref_rms(branch, post_norm.weight) / num_adds**0.5
                + hidden * ((num_adds - 1) / num_adds) ** 0.5
            )
    expected = _ref_rms(hidden, model.norm.weight)

    # Two FP32 RMSNorm spellings differ by a few ulp; 1e-5 (~100 FP32 ulp at
    # O(1) outputs) is far below the O(0.1) change of a wrong coefficient or norm.
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def _to_device(arg, device):
    """Moves an op argument, keeping a tensor view's strides (padded-row slices)."""
    if isinstance(arg, nn.Module):
        return arg.to(device)
    if not isinstance(arg, torch.Tensor):
        return arg
    base = arg if arg._base is None else arg._base
    return base.to(device).as_strided(arg.size(), arg.stride(), arg.storage_offset())


@pytest.mark.tpu_test
@pytest.mark.parametrize("case", OP_CASES.values(), ids=OP_CASES.keys())
@torch.no_grad()
def test_beam_op_on_tpu_matches_cpu(case):
    """Compiled on TPU, each op gives the CPU result (rel. error <= 5e-4)."""
    op, args, _ = case(torch.Generator().manual_seed(0))
    expected = _as_tuple(op(*args))
    # Fresh inputs from the same seed: the softcap writes into its own.
    op, args, _ = case(torch.Generator().manual_seed(0))
    args = [_to_device(arg, "tpu") for arg in args]
    compiled = torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)
    for value, reference in zip(_as_tuple(compiled(*args)), expected, strict=True):
        value = value.cpu()
        assert value.shape == reference.shape and value.dtype == reference.dtype
        error = (value.float() - reference.float()).norm() / reference.float().norm()
        # Bit-exact on v7x (libtpu 0.0.50) except qk_norm_rope: TPU's RoPE leaves
        # one-ulp BF16 differences (~4e-6). The headroom is for libtpu changes.
        assert error.item() <= 5e-4
