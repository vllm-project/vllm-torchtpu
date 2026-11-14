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

Covers registration, the config checks, checkpoint loading, the pure PyTorch
ops and the fp32-accurate router, on CPU and on TPU.
"""

import hashlib
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
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.utils.torch_utils import set_default_torch_dtype

import vllm_torchtpu.models.vllm.beam.model as beam_model
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.beam import ops
from vllm_torchtpu.models.vllm.beam.config import TPUBeamConfig
from vllm_torchtpu.models.vllm.beam.model import BeamForCausalLM
from vllm_torchtpu.models.vllm.beam.router import BeamRouter

BF16, FP32, FP8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn

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

# The real router: 256 experts over hidden 5120, top-8.
ROUTER_EXPERTS, ROUTER_HIDDEN, ROUTER_TOPK = 256, 5120, 8
ROUTER_INIT_STD = 0.02
FP32_ROUTER_MAX_REL_ERROR = 1e-6

# Op-test heads: 6 Q heads and 1 K / V head of 128 channels.
HEAD_DIM, Q_HEADS = 128, 6
EPS, SOFTCAP = 1e-6, 30.0
# Positions up to the last of the released max_position_embeddings.
SSMAX_POSITIONS = [0, 1, 15, 2047, 2048, 16384, 100000, 1048575]


def test_beam_resolves_to_this_package():
    """BeamForCausalLM resolves to this package (over upstream's registry entry)."""
    register_models()
    model_cls = ModelRegistry.models["BeamForCausalLM"].load_model_cls()
    assert model_cls is BeamForCausalLM
    assert MODELS_CONFIG_MAP["BeamForCausalLM"] is TPUBeamConfig


@pytest.fixture
def clean_check_flags(monkeypatch):
    """Unsets the TPU flags the config checks reject."""
    for name in ("TPU_MOE_SEQUENCE_PARALLEL", "VLLM_TPU_BLOCK_MAJOR_KV"):
        monkeypatch.delenv(name, raising=False)


def _config_for_checks(
    *,
    dtype=BF16,
    speculative_config=None,
    lora_config=None,
    hf_overrides=None,
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    enable_eplb=False,
):
    """A VllmConfig stand-in: the released Beam config, no extra features."""
    hf_config = BeamConfig(**{**RELEASED_CONFIG, **(hf_overrides or {})})
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=dtype),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
            enable_eplb=enable_eplb,
        ),
        speculative_config=speculative_config,
        lora_config=lora_config,
    )


def test_config_accepts_released_model(clean_check_flags):
    TPUBeamConfig.verify_and_update_config(_config_for_checks())


@pytest.mark.parametrize(
    "overrides,match",
    [
        pytest.param({"dtype": torch.float16}, "--dtype bfloat16", id="dtype"),
        pytest.param(
            {"speculative_config": object()}, "--speculative-config", id="spec"
        ),
        pytest.param({"lora_config": object()}, "--enable-lora", id="lora"),
        pytest.param(
            {"pipeline_parallel_size": 2}, "--pipeline-parallel-size 1", id="pp"
        ),
        pytest.param(
            {"prefill_context_parallel_size": 2}, "context parallelism", id="pcp"
        ),
        pytest.param(
            {"decode_context_parallel_size": 2}, "context parallelism", id="dcp"
        ),
        pytest.param({"enable_eplb": True}, "--enable-eplb", id="eplb"),
        # Divides the 48 Q heads but not the 8 KV heads.
        pytest.param(
            {"tensor_parallel_size": 3}, r"use one of \[1, 2, 4, 8\], got 3", id="tp"
        ),
        pytest.param(
            {"hf_overrides": {"hidden_act": "gelu"}},
            "hidden_act='silu'",
            id="hidden_act",
        ),
        pytest.param(
            {
                "hf_overrides": {
                    "rope_parameters": {
                        "rope_type": "linear",
                        "rope_theta": 10000.0,
                        "factor": 2.0,
                    }
                }
            },
            "rope_type='default'",
            id="rope_type",
        ),
        pytest.param(
            {"hf_overrides": {"tie_word_embeddings": True}},
            "tie_word_embeddings=False",
            id="tied_embeddings",
        ),
    ],
)
def test_config_rejects_unsupported(clean_check_flags, overrides, match):
    """Rejected when the config is built, before any worker starts."""
    with pytest.raises(NotImplementedError, match=match):
        TPUBeamConfig.verify_and_update_config(_config_for_checks(**overrides))


@pytest.mark.parametrize(
    "flag", ["TPU_MOE_SEQUENCE_PARALLEL", "VLLM_TPU_BLOCK_MAJOR_KV"]
)
def test_config_rejects_unsupported_flag(clean_check_flags, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(NotImplementedError, match=f"unset {flag}"):
        TPUBeamConfig.verify_and_update_config(_config_for_checks())


# The quantization_config of Beam-1.0-501B-A23B-FP8 (revision 077872da), without
# its null and bookkeeping fields.
_FP8_TENSORS = dict(num_bits=8, symmetric=True, type="float")
FP8_QUANTIZATION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {
        "FP8_DYNAMIC": {
            "targets": ["Linear"],
            "weights": dict(_FP8_TENSORS, dynamic=False, strategy="channel"),
            "input_activations": dict(_FP8_TENSORS, dynamic=True, strategy="token"),
        }
    },
    "ignore": ["lm_head", "model.embed_tokens", "re:.*mlp\\.gate$"],
}


def _checkpoint_tensors(config: BeamConfig, *, fp8: bool):
    """Yields (name, shape, dtype) of every tensor of a Beam checkpoint.

    FP8 checkpoints hold e4m3 linear weights with a bf16 scale per output row.
    """
    hidden, head_dim, experts = config.hidden_size, config.head_dim, config.num_experts
    q_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    def linears(prefix, *projs):
        for proj, out_size, in_size in projs:
            yield (
                f"{prefix}.{proj}_proj.weight",
                (out_size, in_size),
                FP8 if fp8 else BF16,
            )
            if fp8:
                yield f"{prefix}.{proj}_proj.weight_scale", (out_size, 1), BF16

    yield "model.embed_tokens.weight", (config.vocab_size, hidden), BF16
    yield "model.norm.weight", (hidden,), BF16
    yield "lm_head.weight", (config.vocab_size, hidden), BF16
    for layer, mlp_type in enumerate(config.mlp_layer_types):
        prefix = f"model.layers.{layer}"
        for norm in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            yield f"{prefix}.{norm}_layernorm.weight", (hidden,), BF16
        attn = f"{prefix}.self_attn"
        yield from linears(
            attn,
            ("q", q_size, hidden),
            ("k", kv_size, hidden),
            ("v", kv_size, hidden),
            ("gate", q_size, hidden),
            ("o", hidden, q_size),
        )
        yield f"{attn}.gate_bias", (config.num_attention_heads,), BF16
        yield f"{attn}.q_norm.weight", (head_dim,), BF16
        yield f"{attn}.k_norm.weight", (head_dim,), BF16
        if mlp_type == "dense":
            mlps, size = [f"{prefix}.mlp"], config.intermediate_size
        else:
            yield f"{prefix}.mlp.gate.weight", (experts, hidden), FP32
            yield f"{prefix}.mlp.gate.e_score_correction_bias", (experts,), FP32
            mlps = [f"{prefix}.mlp.experts.{e}" for e in range(experts)]
            size = config.moe_intermediate_size
        for mlp in mlps:
            yield from linears(
                mlp,
                ("gate", size, hidden),
                ("up", size, hidden),
                ("down", hidden, size),
            )


@pytest.mark.parametrize(
    "fp8,count,digest",
    [
        (False, 38366, "9161379b68290704feb12d19059067bb3517578c734bce86faa83674ed06e75d"),  # noqa: E501
        (True,  76267, "ee4b6ffe4fcb9af2b76104d924dc60fe1fcfd88604af241bbed78f80399fc4a1"),  # noqa: E501
    ],
    ids=["bf16", "fp8"],
)  # fmt: skip
def test_checkpoint_tensors_match_released_index(fp8, count, digest):
    """``_checkpoint_tensors`` names exactly the released checkpoint's tensors.

    The digests are sha256("\\n".join(sorted(index["weight_map"]))) of the
    model.safetensors.index.json of reflection/Beam-1.0-501B-A23B and
    Beam-1.0-501B-A23B-FP8 (rev 077872da). Without this, the load test's names
    would only echo the model's own.
    """
    config = BeamConfig(**RELEASED_CONFIG)
    names = sorted(name for name, _, _ in _checkpoint_tensors(config, fp8=fp8))
    assert len(names) == count
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == digest


def _meta_model_config(config: BeamConfig, monkeypatch, *, fp8: bool) -> VllmConfig:
    """A CPU VllmConfig for building the full-size model on the meta device."""
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = SimpleNamespace(
        hf_config=config, max_model_len=4096, dtype=BF16, head_dtype=None
    )
    vllm_config.compilation_config.mode = CompilationMode.NONE
    if fp8:
        vllm_config.quant_config = VllmCompressedTensorsConfig.from_config(
            FP8_QUANTIZATION_CONFIG
        )
        # Mirrors VllmConfig._set_quant_config / set_configs on TPU quant classes.
        for cls in (VllmCompressedTensorsConfig, VllmFp8Config):
            monkeypatch.setattr(cls, "vllm_config", vllm_config, raising=False)
    return vllm_config


def _zeros_for_load(name, shape, dtype):
    # BeamRouter splits its fp32 weight on the host, so it needs data.
    device = "cpu" if name.endswith(".mlp.gate.weight") else "meta"
    return torch.zeros(shape, dtype=dtype, device=device)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_released_checkpoint_loads_into_every_parameter(monkeypatch, request, fp8):
    """Every released tensor loads into the full-size meta model, filling it."""
    config = BeamConfig(**RELEASED_CONFIG)
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")  # A CPU (gloo) group, after the patch.
    monkeypatch.setattr(beam_model, "Attention", lambda *args, **kwargs: nn.Identity())
    vllm_config = _meta_model_config(config, monkeypatch, fp8=fp8)
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(BF16),
        torch.device("meta"),
    ):
        model = BeamForCausalLM(vllm_config=vllm_config)

    tensors = _checkpoint_tensors(config, fp8=fp8)
    with set_current_vllm_config(vllm_config):
        loaded = model.load_weights(
            (name, _zeros_for_load(name, shape, dtype))
            for name, shape, dtype in tensors
        )
    assert loaded == {name for name, _ in model.named_parameters()}


def _randn(generator, *shape, dtype=BF16, scale=1.0):
    return (torch.randn(*shape, generator=generator) * scale).to(dtype)


def _strided(generator, rows, cols, *, scale=1.0, device="cpu"):
    """An unaligned ``[rows, cols]`` slice inside wider padded rows, like packed QKV."""
    row_padding = 64
    wide = _randn(generator, rows, cols + row_padding, scale=scale).to(device)
    return wide[:, 1 : cols + 1]


def _ref_rms(x, weight):
    normed = F.rms_norm(x.float(), (x.shape[-1],), eps=EPS)
    return (normed * weight.float()).to(x.dtype)


def _rms_norm_case(generator):
    x, weight = _strided(generator, 33, 513), _randn(generator, 513)
    return (ops.beam_rms_norm(x, weight, EPS),), (_ref_rms(x, weight),)


def _norm_residual_norm_case(generator):
    x = _strided(generator, 33, 256)
    residual = _randn(generator, 33, 256)
    post_weight, next_weight = _randn(generator, 256), _randn(generator, 256)
    # Not powers of two, so rounding after each scale changes bits (1/sqrt(n)).
    scales = ops.ResidualScales(branch=3**-0.5, residual=(2 / 3) ** 0.5)
    actual = ops.beam_norm_residual_norm(
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
    return actual, (hidden, _ref_rms(hidden, next_weight))


def _silu_mul_case(generator):
    x = _randn(generator, 33, 1024, scale=4.0)
    gate, up = x.float().chunk(2, -1)
    return (ops.beam_silu_mul(x),), ((F.silu(gate) * up).to(BF16),)


def _softcap_case(generator, cap=SOFTCAP):
    logits = _strided(generator, 8, 257, scale=90.0)
    # Upstream's _softcap_logits_kernel: one FP32 expression, rounded once.
    expected = (torch.tanh(logits.float() * (1.0 / cap)) * cap).to(BF16)
    return (ops.beam_softcap_logits(logits, cap),), (expected,)


def _qk_inputs(generator, rows, device="cpu"):
    """Q and K sliced from a packed QKV projection, and their norm weights."""
    qkv = _strided(generator, rows, (Q_HEADS + 2) * HEAD_DIM, device=device)
    q, k, _ = qkv.split((Q_HEADS * HEAD_DIM, HEAD_DIM, HEAD_DIM), -1)
    q_weight = _randn(generator, HEAD_DIM).to(device)
    k_weight = _randn(generator, HEAD_DIM).to(device)
    return q, k, q_weight, k_weight


def _head_rms(x, weight):
    return _ref_rms(x.unflatten(-1, (-1, HEAD_DIM)), weight).flatten(-2)


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
    actual = ops.beam_qk_norm_rope(
        q,
        k,
        q_weight=q_weight,
        k_weight=k_weight,
        eps=EPS,
        positions=positions,
        rotary_emb=rope,
    )
    q_rotated, k_rotated = RotaryEmbedding.forward_static(
        positions, _head_rms(q, q_weight).float(), _head_rms(k, k_weight).float(),
        HEAD_DIM, HEAD_DIM, cache, True,
    )  # fmt: skip
    return actual, (q_rotated.to(BF16), k_rotated.to(BF16))


def _qk_norm_ssmax_case(generator):
    q, k, q_weight, k_weight = _qk_inputs(generator, len(SSMAX_POSITIONS))
    positions = torch.tensor(SSMAX_POSITIONS, dtype=torch.int32)
    actual = ops.beam_qk_norm_rope(
        q,
        k,
        q_weight=q_weight,
        k_weight=k_weight,
        eps=EPS,
        positions=positions,
        rotary_emb=None,
    )
    # Beam's SSMax: Q * (1 + (position + 1) / 2048) ** 0.1, here in float64.
    scale = torch.pow(1.0 + (positions.double() + 1.0) / 2048.0, 0.1).float()
    q_heads = _head_rms(q, q_weight).float().unflatten(-1, (Q_HEADS, HEAD_DIM))
    q_ref = (q_heads * scale[:, None, None]).to(BF16).flatten(-2)
    # K is normalized but never position-scaled.
    return actual, (q_ref, _head_rms(k, k_weight))


def _output_gate_case(generator):
    attn_output = _randn(generator, 17, Q_HEADS * HEAD_DIM)
    bias = _randn(generator, Q_HEADS)
    gate = _strided(generator, 17, Q_HEADS * HEAD_DIM, scale=8.0)
    actual = ops.beam_output_gate(attn_output, gate, bias)
    heads = (17, Q_HEADS, HEAD_DIM)
    gate_logits = (gate.float().view(heads) + bias.float()[:, None]).to(BF16)
    gate_sigmoid = torch.sigmoid(gate_logits.float()).to(BF16)
    gated = attn_output.float().view(heads) * gate_sigmoid.float()
    return (actual,), (gated.flatten(-2).to(BF16),)


# In BF16, where the kernels' rounding points show.
OP_CASES = {
    "rms_norm": _rms_norm_case,
    "norm_residual_norm": _norm_residual_norm_case,
    "silu_mul": _silu_mul_case,
    "softcap": _softcap_case,
    "qk_norm_rope": _qk_norm_rope_case,
    "qk_norm_ssmax": _qk_norm_ssmax_case,
    "output_gate": _output_gate_case,
}


@pytest.mark.parametrize("case", OP_CASES.values(), ids=OP_CASES.keys())
@torch.no_grad()
def test_beam_op_matches_reference(case):
    """Each op matches its fp32 reference and its rounding points, bit-exact."""
    actual, expected = case(torch.Generator().manual_seed(0))
    for value, reference in zip(actual, expected, strict=True):
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
    assert ops.beam_softcap_logits(logits, SOFTCAP) is logits
    assert torch.equal(storage[:, vocab:], padding)


def _router_weight(generator=None):
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(ROUTER_EXPERTS, ROUTER_HIDDEN, generator=generator)
        * ROUTER_INIT_STD
    )


def _router_inputs(num_tokens):
    """A seeded fp32 router weight and bf16 activations at the released shape."""
    generator = torch.Generator().manual_seed(0)
    weight = _router_weight(generator)
    x = torch.randn(num_tokens, ROUTER_HIDDEN, generator=generator).to(BF16)
    return weight, x


def _loaded_router(weight):
    router = BeamRouter(ROUTER_HIDDEN, ROUTER_EXPERTS)
    router.weight.weight_loader(router.weight, weight)
    return router


def _router_error(logits, reference):
    return ((logits.double() - reference).abs().max() / reference.abs().max()).item()


def _topk_flip_rate(logits, reference):
    """Fraction of tokens whose top-k expert set differs from the reference."""
    top = logits.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    ref_top = reference.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    return (top != ref_top).any(-1).float().mean().item()


def test_router_loads_fp32_weight_as_bf16_parts():
    """The fp32 [E, H] weight loads as 3 bf16 parts that sum back to it exactly."""
    weight = _router_weight()
    parts = _loaded_router(weight).weight
    assert parts.shape == (3, ROUTER_EXPERTS, ROUTER_HIDDEN)
    assert parts.dtype == BF16
    torch.testing.assert_close(parts.double().sum(0), weight.double(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(torch.zeros(ROUTER_EXPERTS, ROUTER_HIDDEN, dtype=BF16), id="bf16"),
        pytest.param(torch.zeros(ROUTER_EXPERTS // 2, ROUTER_HIDDEN), id="shape"),
    ],
)
def test_router_rejects_other_weights(weight):
    with pytest.raises(ValueError, match=r"fp32 mlp\.gate\.weight"):
        _loaded_router(weight)


@torch.no_grad()
def test_router_is_fp32_accurate_on_cpu():
    weight, x = _router_inputs(64)
    logits, bias = _loaded_router(weight)(x)
    assert bias is None and logits.dtype == FP32
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) <= FP32_ROUTER_MAX_REL_ERROR


def _softcap_as_tuple(logits, cap):
    return (ops.beam_softcap_logits(logits, cap),)


def _tpu_softcap_inputs(device):
    generator = torch.Generator().manual_seed(42)
    logits = _strided(generator, 8, 257, scale=90.0, device=device)
    return _softcap_as_tuple, (logits, SOFTCAP)


def _qk_norm_ssmax(q, k, q_weight, k_weight, positions):
    return ops.beam_qk_norm_rope(
        q,
        k,
        q_weight=q_weight,
        k_weight=k_weight,
        eps=EPS,
        positions=positions,
        rotary_emb=None,
    )


def _tpu_qk_norm_ssmax_inputs(device):
    generator = torch.Generator().manual_seed(42)
    q, k, q_weight, k_weight = _qk_inputs(generator, len(SSMAX_POSITIONS), device)
    positions = torch.tensor(SSMAX_POSITIONS, dtype=torch.int32).to(device)
    return _qk_norm_ssmax, (q, k, q_weight, k_weight, positions)


# The ops whose log / tanh lowering on TPU could differ from CPU.
TPU_OP_CASES = {
    "qk_norm_ssmax": _tpu_qk_norm_ssmax_inputs,
    "softcap_logits": _tpu_softcap_inputs,
}


@pytest.mark.tpu_test
@pytest.mark.parametrize("make_inputs", TPU_OP_CASES.values(), ids=TPU_OP_CASES.keys())
@torch.no_grad()
def test_beam_op_on_tpu_matches_cpu(make_inputs):
    """Compiled on TPU, the log/tanh ops give the CPU result (rel. error <= 5e-4)."""
    op, cpu_args = make_inputs("cpu")
    expected = op(*cpu_args)
    op, tpu_args = make_inputs("tpu")
    actual = torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)(*tpu_args)
    for value, reference in zip(actual, expected, strict=True):
        value = value.cpu()
        assert value.shape == reference.shape and value.dtype == reference.dtype
        error = (value.float() - reference.float()).norm() / reference.float().norm()
        # Bit-exact on v7x (libtpu 0.0.50); the headroom is for libtpu changes to
        # the transcendentals.
        assert error.item() <= 5e-4


def _tpu_router_logits(router, x, compiled):
    router = router.to("tpu")
    run = torch.compile(router, backend="tpu", fullgraph=True) if compiled else router
    logits, bias = run(x.to("tpu"))
    assert bias is None and logits.dtype == FP32
    return logits.cpu().double()


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.no_grad()
def test_router_is_fp32_accurate_on_tpu(dist_ctx, compiled):
    """BeamRouter's TPU logits are fp32-accurate and pick the fp32 top-k."""
    weight, x = _router_inputs(1024)
    logits = _tpu_router_logits(_loaded_router(weight), x, compiled)
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) <= FP32_ROUTER_MAX_REL_ERROR
    assert _topk_flip_rate(logits, reference) == 0


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.no_grad()
def test_gate_linear_control_is_not_fp32_accurate_on_tpu(dist_ctx, compiled):
    """Control: upstream's fp32 GateLinear runs one bf16 pass on TPU.

    That is the precision loss BeamRouter fixes; without it, the BeamRouter
    test would not show anything.
    """
    weight, x = _router_inputs(1024)
    # Upstream's fp32 router, as its BeamSparseMoeBlock builds it.
    with set_current_vllm_config(VllmConfig()):
        router = GateLinear(
            ROUTER_HIDDEN, ROUTER_EXPERTS, bias=False, params_dtype=FP32,
            out_dtype=FP32, prefix="gate",
        )  # fmt: skip
    router.weight.weight_loader(router.weight, weight)
    logits = _tpu_router_logits(router, x, compiled)
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) > FP32_ROUTER_MAX_REL_ERROR
    assert _topk_flip_rate(logits, reference) > 0
