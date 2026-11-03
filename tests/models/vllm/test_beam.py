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
"""Unit tests for Reflection Beam: registration, config checks, the pure PyTorch
ops and the fp32-accurate router, on CPU and on TPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.transformers_utils.configs.beam import BeamConfig

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.beam import ops
from vllm_torchtpu.models.vllm.beam.config import TPUBeamConfig
from vllm_torchtpu.models.vllm.beam.model import BeamForCausalLM
from vllm_torchtpu.models.vllm.beam.router import BeamRouter

BF16, FP32 = torch.bfloat16, torch.float32
# The real router: 256 experts over hidden 5120, top-8.
ROUTER_EXPERTS, ROUTER_HIDDEN, ROUTER_TOPK = 256, 5120, 8
SSMAX_POSITIONS = [0, 1, 15, 2047, 2048, 16384, 100000, 1048575]  # up to 2**20 - 1


def test_beam_resolves_to_this_package():
    """BeamForCausalLM resolves to this package (over upstream's registry entry)."""
    register_models()
    model_cls = ModelRegistry.models["BeamForCausalLM"].load_model_cls()
    assert model_cls is BeamForCausalLM
    assert MODELS_CONFIG_MAP["BeamForCausalLM"] is TPUBeamConfig


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


def _config_for_checks(*, dtype=BF16, tensor_parallel_size=8):
    """A VllmConfig stand-in with the released Beam config and no extra features."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=BeamConfig(**RELEASED_CONFIG), dtype=dtype
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
            enable_eplb=False,
        ),
        speculative_config=None,
        lora_config=None,
    )


def test_config_accepts_released_model(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_BLOCK_MAJOR_KV", raising=False)
    TPUBeamConfig.verify_and_update_config(_config_for_checks())


@pytest.mark.parametrize(
    "overrides,match",
    [
        # BeamRouter's split weight is exact only for bf16 activations.
        ({"dtype": torch.float16}, "--dtype bfloat16"),
        ({"tensor_parallel_size": 3}, r"use one of \[1, 2, 4, 8\], got 3"),
    ],
    ids=["fp16", "tp3"],
)
def test_config_rejects_unsupported(monkeypatch, overrides, match):
    """Rejected when the config is built, before any worker starts."""
    monkeypatch.delenv("VLLM_TPU_BLOCK_MAJOR_KV", raising=False)
    with pytest.raises(NotImplementedError, match=match):
        TPUBeamConfig.verify_and_update_config(_config_for_checks(**overrides))


def _tuple(x):
    return x if isinstance(x, tuple) else (x,)


def _randn(gen, *shape, dtype=BF16, scale=1.0):
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def _ref_rms(x, w, eps=1e-6):
    return (F.rms_norm(x.float(), (x.shape[-1],), eps=eps) * w.float()).to(x.dtype)


def _rms_norm_case(gen, dtype):
    x, w = _randn(gen, 33, 577, dtype=dtype)[:, 1:514], _randn(gen, 513, dtype=dtype)
    return ops.beam_rms_norm(x, w, 1e-6), _ref_rms(x, w)


def _norm_residual_norm_case(gen, dtype):
    x = _randn(gen, 33, 288, dtype=dtype)[:, 1:257]
    res = _randn(gen, 33, 256, dtype=FP32)  # The op casts it to the activation dtype.
    post_w, next_w = _randn(gen, 256), _randn(gen, 256)
    # Not powers of two, so rounding after each scale changes bits (1/sqrt(n)).
    scales = (3**-0.5, (2 / 3) ** 0.5)
    actual = ops.beam_norm_residual_norm(x, res, post_w, 1e-6, next_w, 1e-6, scales)
    # Rounds after the post-norm affine, after each scale and after the add.
    h = (_ref_rms(x, post_w).float() * scales[0]).to(dtype)
    r = (res.to(dtype).float() * scales[1]).to(dtype)
    hidden = (h.float() + r.float()).to(dtype)
    return actual, (hidden, _ref_rms(hidden, next_w))


def _silu_mul_case(gen, dtype):
    x = _randn(gen, 33, 1024, dtype=dtype, scale=4.0)
    gate, up = x.float().chunk(2, -1)
    return ops.beam_silu_mul(x), (F.silu(gate) * up).to(dtype)


def _softcap_case(gen, dtype, cap=30.0):
    storage = _randn(gen, 8, 264, dtype=dtype, scale=90.0)
    padding = storage[:, 257:].clone()
    logits = storage[:, :257]
    # Upstream's _softcap_logits_kernel: one FP32 expression, rounded once.
    expected = (torch.tanh(logits.float() * (1.0 / cap)) * cap).to(dtype)
    actual = ops.beam_softcap_logits(logits, cap)
    # In place, and the padding past the slice is untouched.
    assert actual is logits and torch.equal(storage[:, 257:], padding)
    return actual, expected


def _qk_inputs(gen, rows, device="cpu"):
    qkv = _randn(gen, rows, 1025).to(device)[:, 1:]  # 6 Q, 1 K, 1 V head of 128
    q, k, _ = qkv.split((768, 128, 128), -1)
    return q, k, _randn(gen, 128).to(device), _randn(gen, 128).to(device)


def _head_rms(x, w):
    return _ref_rms(x.unflatten(-1, (-1, 128)), w).flatten(-2)


def _qk_norm_rope_case(gen, dtype):
    q, k, qw, kw = _qk_inputs(gen, 17)
    positions = torch.arange(17) * 3
    angle = torch.randn(51, 64, generator=gen)
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    actual = ops.beam_qk_norm_rope(q, k, qw, kw, 1e-6, positions, cache, 128, 128)
    qr, kr = RotaryEmbedding.forward_static(
        positions, _head_rms(q, qw).float(), _head_rms(k, kw).float(), 128, 128,
        cache, True,
    )  # fmt: skip
    return actual, (qr.to(dtype), kr.to(dtype))


def _qk_norm_ssmax_case(gen, dtype):
    q, k, qw, kw = _qk_inputs(gen, 8)
    positions = torch.tensor(SSMAX_POSITIONS, dtype=torch.int32)
    actual = ops.beam_qk_norm_rope(q, k, qw, kw, 1e-6, positions, None, 128, ssmax=True)
    scale = torch.pow(1.0 + (positions.double() + 1.0) / 2048.0, 0.1).float()
    q_ref = (_head_rms(q, qw).float().view(8, 6, 128) * scale[:, None, None]).to(dtype)
    # K is normalized but never position-scaled.
    return actual, (q_ref.flatten(-2), _head_rms(k, kw))


def _output_gate_case(gen, dtype):
    x, bias = _randn(gen, 17, 768), _randn(gen, 6)
    gate = _randn(gen, 17, 1792, scale=8.0)[:, -768:]
    actual = ops.beam_output_gate(x, gate, bias, num_heads=6, head_dim=128)
    g = (gate.float().view(17, 6, 128) + bias.float()[:, None]).to(dtype)
    s = torch.sigmoid(g.float()).to(dtype)
    return actual, (x.float().view(17, 6, 128) * s.float()).flatten(-2).to(dtype)


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
    """Each op matches its fp32 reference and rounding points; softcap is in place."""
    actual, expected = case(torch.Generator().manual_seed(0), BF16)
    for value, reference in zip(_tuple(actual), _tuple(expected), strict=True):
        assert value.dtype == reference.dtype == BF16
        # Bit-exact: a dropped or extra rounding step changes some elements.
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def _router_error(logits, reference):
    return ((logits.double() - reference).abs().max() / reference.abs().max()).item()


@pytest.mark.parametrize("bad", [None, "bf16", "shape"])
def test_router_weight_loader(bad):
    """The fp32 [E, H] weight loads as 3 bf16 parts; anything else is rejected."""
    router = BeamRouter(ROUTER_HIDDEN, ROUTER_EXPERTS)
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(ROUTER_EXPERTS, ROUTER_HIDDEN, generator=gen) * 0.02
    if bad:
        weight = weight.to(BF16) if bad == "bf16" else weight[:128]
        with pytest.raises(ValueError, match=r"fp32 mlp\.gate\.weight"):
            router.weight.weight_loader(router.weight, weight)
        return
    router.weight.weight_loader(router.weight, weight)
    parts = router.weight.view(3, ROUTER_EXPERTS, ROUTER_HIDDEN)
    assert parts.dtype == BF16
    torch.testing.assert_close(parts.double().sum(0), weight.double(), rtol=0, atol=0)
    x = torch.randn(64, ROUTER_HIDDEN, generator=gen).to(BF16)
    with torch.no_grad():
        logits, bias = router(x)
    assert bias is None and logits.dtype == FP32
    assert _router_error(logits, x.double() @ weight.double().T) <= 1e-6


def _tpu_op_inputs(case, device):
    gen = torch.Generator().manual_seed(42)
    if case == "softcap_logits":
        logits = _randn(gen, 8, 264, scale=90.0).to(device)[:, :257]
        return ops.beam_softcap_logits, (logits, 30.0), {}
    q, k, qw, kw = _qk_inputs(gen, 8, device)
    positions = torch.tensor(SSMAX_POSITIONS, dtype=torch.int32)
    args = (q, k, qw, kw, 1e-6, positions.to(device), None, 128)
    return ops.beam_qk_norm_rope, args, {"ssmax": True}


@pytest.mark.tpu_test
@pytest.mark.parametrize("case", ["qk_norm_ssmax", "softcap_logits"])
def test_beam_op_on_tpu_matches_cpu(case):
    """Compiled on TPU, the log/tanh ops give the CPU result (rel. error <= 5e-4)."""
    with torch.no_grad():
        fn, args, kwargs = _tpu_op_inputs(case, "cpu")
        expected = fn(*args, **kwargs)
        fn, args, kwargs = _tpu_op_inputs(case, "tpu")
        compiled = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
        actual = compiled(*args, **kwargs)
    for value, reference in zip(_tuple(actual), _tuple(expected), strict=True):
        value = value.cpu()
        assert value.shape == reference.shape and value.dtype == reference.dtype
        error = (value.float() - reference.float()).norm() / reference.float().norm()
        # Bit-exact on v7x (libtpu 0.0.50); the headroom is for libtpu changes to
        # the transcendentals.
        assert error.item() <= 5e-4, f"{case}: {error.item():.2e}"


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("router_cls", [BeamRouter, GateLinear])
def test_router_is_fp32_accurate_on_tpu(dist_ctx, router_cls, compiled):
    """BeamRouter is fp32-accurate on TPU; the GateLinear control is not."""
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(ROUTER_EXPERTS, ROUTER_HIDDEN, generator=gen) * 0.02
    x = torch.randn(1024, ROUTER_HIDDEN, generator=gen).to(BF16)
    if router_cls is BeamRouter:
        router = BeamRouter(ROUTER_HIDDEN, ROUTER_EXPERTS)
    else:  # Upstream's fp32 router, as its BeamSparseMoeBlock builds it.
        with set_current_vllm_config(VllmConfig()):
            router = GateLinear(
                ROUTER_HIDDEN, ROUTER_EXPERTS, bias=False, params_dtype=FP32,
                out_dtype=FP32, prefix="gate",
            )  # fmt: skip
    router.weight.weight_loader(router.weight, weight)
    router = router.to("tpu")
    fn = torch.compile(router, backend="tpu", fullgraph=True) if compiled else router
    with torch.no_grad():
        logits, bias = fn(x.to("tpu"))
    assert bias is None and logits.dtype == FP32
    logits, reference = logits.cpu().double(), x.double() @ weight.double().T
    error = _router_error(logits, reference)
    top = logits.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    ref_top = reference.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    flips = (top != ref_top).any(-1).float().mean().item()
    if router_cls is BeamRouter:
        assert error <= 1e-6 and flips == 0
    else:  # Control: one bf16 pass over the fp32 weight, the bug being fixed.
        assert error > 1e-6 and flips > 0
