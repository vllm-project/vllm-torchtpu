# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Token-cap dispatch of the fused-MoE-decode op's per-rank body: up to
FUSED_MOE_DECODE_KERNEL_MAX_TOKENS the kernel serves it; above, the GMM MoE
fallback. Admission and fallback of GLM-5.2's convention (ungated shared
expert, routed scale). No TPU needed: the kernels are stubs."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

H = 8


def test_arm_runner_models_without_drafter(monkeypatch):
    class Runner:
        model = torch.nn.Linear(1, 1)
        drafter = None

        @property
        def _async_drafter(self):
            raise AssertionError("no async drafter")

    armed = []
    monkeypatch.setattr(
        bridge, "arm_fused_moe_decode", lambda m, label: armed.append(label) or 1
    )
    assert bridge.arm_runner_models(Runner()) == 1
    assert armed == ["target model"]


def test_max_tokens_env(monkeypatch):
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL", raising=False)
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", raising=False)
    assert envs.FUSED_MOE_DECODE_KERNEL is False
    assert envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS == 256
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", "64")
    assert envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS == 64


@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("tokens", [1, 256, 257, 1024])
def test_token_cap_dispatch(monkeypatch, tokens, expert_parallel):
    calls = []

    def kernel(rank, x, *args, expert_start, **kwargs):
        calls.append(("kernel", expert_start))
        return x

    def gmm(*args, topk, **kwargs):
        calls.append(("gmm", args[12]))
        return args[0]

    monkeypatch.setattr(bridge, "_kernel_body", kernel)
    monkeypatch.setattr(bridge, "_gmm_layer", gmm)
    # The trace-time ring plan reads the TPU's VMEM size; CPU has none.
    monkeypatch.setattr(bridge, "plan_weight_ring", lambda *a, **k: (None, 3))
    rank = np.array([[3, 192]], np.int32)  # [[mesh index, first expert]]
    w = np.zeros((1, 1), np.float32)
    bridge._layer(
        rank,
        np.zeros((tokens, H), np.float32),
        *([w] * 9),
        np.zeros((tokens, 2), np.float32),
        np.zeros((tokens, 2), np.int32),
        n_ranks=8,
        topk=2,
        expert_parallel=expert_parallel,
        max_tokens=256,
    )
    path, start = calls[0]
    assert len(calls) == 1
    assert path == ("kernel" if tokens <= 256 else "gmm")
    assert start == (192 if expert_parallel else None)


class DeepseekV2MLP(torch.nn.Module):
    """Same class name and layout as vLLM's (the admission keys on both)."""

    def __init__(self, h, i_s):
        super().__init__()
        f8 = torch.float8_e4m3fn
        self.gate_up_proj = SimpleNamespace(
            weight=torch.zeros((h, 2 * i_s), dtype=f8), weight_scale=torch.ones(2 * i_s)
        )
        self.down_proj = SimpleNamespace(
            weight=torch.zeros((i_s, h), dtype=f8), weight_scale=torch.ones(h)
        )
        self.act_fn = SiluAndMul.__new__(SiluAndMul)


def _glm_runner(mlp, *, n=2, h=512, i=128, e=4, dtype=torch.bfloat16):
    """A GLM-5.2-like EP layer: ungated shared expert, routed_scaling_factor 2.5."""
    qm = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
    qm._tpu_activation_str = "silu"
    par = SimpleNamespace(use_ep=True, ep_size=n, dp_size=1, tp_size=n)
    return SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=par, in_dtype=dtype),
        _quant_method=qm,
        routed_experts=SimpleNamespace(
            w13_weight=torch.zeros((e, h, 2 * i), dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros((e, i, h), dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones((e, 1, 1, 2 * i)),
            w2_weight_scale_inv=torch.ones((e, 1, 1, h)),
        ),
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=2.5,
        shared_expert_gate=None,
        _shared_experts=SimpleNamespace(_layer=mlp),
    )


def test_glm_admission(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    shared, expert_parallel, routed_scale = bridge._admit(
        _glm_runner(DeepseekV2MLP(512, 128)), 2
    )
    assert shared[4] is None and expert_parallel and routed_scale == 2.5
    # FP16 vLLM divides the shared output instead; an ungated MLP of another
    # class is not known to be gate_up -> SiluAndMul -> down.
    fp16 = _glm_runner(DeepseekV2MLP(512, 128), dtype=torch.float16)
    assert bridge._admit(fp16, 2) is None
    other = DeepseekV2MLP(512, 128)
    other.act_fn = torch.nn.GELU()
    assert bridge._admit(_glm_runner(other), 2) is None


def test_admission_declines_attention_narrower_than_mesh(monkeypatch):
    """PCP2 (or DP2) at TP1: the EP group spans the mesh, attention does not, so
    MoERunner still runs a collective after the op."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    assert bridge._admit(_glm_runner(DeepseekV2MLP(512, 128)), 2) is None


def test_glm_fallback_scales_routed_only(monkeypatch):
    """Above the cap: routed x routed_scale + ungated shared expert, as vLLM's
    MoERunner computes it; scaling the sum instead must not match."""
    monkeypatch.setattr(bridge, "fused_moe_func", lambda x, *a, **k: x)
    monkeypatch.setattr(bridge, "_quantized_matmul_jax", lambda a, w, s: a @ (w * s))
    monkeypatch.setattr(bridge, "resolve_onehot_permute_threshold", lambda: 0)
    rng = np.random.default_rng(0)
    x, w1s, w2s = (
        rng.normal(size=s).astype(np.float32) for s in ((4, H), (H, 4), (2, H))
    )
    w = np.zeros((1, 1), np.float32)

    def body(x):
        return bridge._layer(
            np.array([[0, 0]], np.int32),
            x,
            *([w] * 4),
            w1s,
            np.ones(4, np.float32),
            w2s,
            np.ones(H, np.float32),
            w,
            np.zeros((4, 2), np.float32),
            np.zeros((4, 2), np.int32),
            n_ranks=1,
            topk=2,
            expert_parallel=False,
            max_tokens=1,
            shared_gated=False,
            routed_scale=2.5,
        )

    with jax.default_matmul_precision("float32"):
        got = np.asarray(jax.vmap(body, axis_name=EP_AXIS_NAME)(x[None])[0])
    gate_up = x @ w1s
    shared = (np.asarray(jax.nn.silu(gate_up[:, :2])) * gate_up[:, 2:]) @ w2s
    np.testing.assert_allclose(got, 2.5 * x + shared, rtol=1e-4, atol=1e-4)
    assert not np.allclose(got, 2.5 * (x + shared), rtol=1e-2, atol=1e-2)


def _bf16_linear(weight):
    """An unquantized linear as VllmUnquantizedLinearMethod leaves it: BF16,
    K-major ([in, out]), flipped, no scale or bias."""
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    return SimpleNamespace(
        **{"weight": weight, "bias": None, WEIGHT_FLIPPED_ATTR: True}
    )


class Qwen2MoeMLP(torch.nn.Module):
    """Qwen3.8-Flash's shared expert: BF16 in the checkpoint, its TP slice of
    i_s = 640 / 8 = 80, gated by the replicated shared_expert_gate."""

    def __init__(self, h, i_s, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = _bf16_linear(
            torch.randn((h, 2 * i_s), generator=g).to(torch.bfloat16)
        )
        self.down_proj = _bf16_linear(
            torch.randn((i_s, h), generator=g).to(torch.bfloat16)
        )
        self.expert_gate = _bf16_linear(
            torch.randn((h, 1), generator=g).to(torch.bfloat16)
        )
        self.act_fn = SiluAndMul.__new__(SiluAndMul)


def _qwen38_runner(mlp, *, n=8, h=2560, i=640, e=4):
    runner = _glm_runner(mlp, n=n, h=h, i=i, e=e)
    runner.routed_scaling_factor = 1.0
    return runner


def test_qwen38_flash_admission_pads_bf16_shared_expert(monkeypatch):
    """h = 2560 (an 8-way chunk of 320 columns, padded by the kernel), i = 640
    and a BF16 80-wide shared slice: admitted, with the slice zero-padded to
    128 and unit scales; the gate keeps its flipped [h, 1] layout."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 8
    )
    mlp = Qwen2MoeMLP(2560, 80)
    why = []
    admitted = bridge._admit(_qwen38_runner(mlp), 8, why)
    assert admitted is not None, why
    (w1, s1, w2, s2, gate), expert_parallel, routed_scale = admitted
    assert expert_parallel and routed_scale == 1.0
    assert w1.shape == (2560, 256) and w1.dtype == torch.bfloat16
    assert w2.shape == (128, 2560) and w2.dtype == torch.bfloat16
    assert torch.equal(s1, torch.ones(256)) and torch.equal(s2, torch.ones(2560))
    assert torch.equal(gate, mlp.expert_gate.weight)
    # A shared expert in neither layout is declined with a reason.
    mlp.gate_up_proj = SimpleNamespace(weight=mlp.gate_up_proj.weight, bias=None)
    why = []
    assert bridge._admit(_qwen38_runner(mlp), 8, why) is None
    assert "shared expert neither FP8" in why[0]


def test_pad_shared_intermediate_is_exact():
    """Zero padding of the shared intermediate leaves Qwen2MoeMLP's output
    unchanged: padded gate/up columns give SiLU(0) * 0 = 0."""
    g = torch.Generator().manual_seed(1)
    h, i_s = 64, 80
    w1 = torch.randn((h, 2 * i_s), generator=g)
    w2 = torch.randn((i_s, h), generator=g)
    x = torch.randn((5, h), generator=g)
    p1, p2 = bridge.pad_shared_intermediate(w1, w2, 128)
    assert p1.shape == (h, 256) and p2.shape == (128, h)
    assert torch.equal(p1[:, :i_s], w1[:, :i_s])
    assert torch.equal(p1[:, 128 : 128 + i_s], w1[:, i_s:])

    def mlp(a, b, width):
        gu = x @ a
        return (torch.nn.functional.silu(gu[:, :width]) * gu[:, width:]) @ b

    torch.testing.assert_close(mlp(p1, p2, 128), mlp(w1, w2, i_s))
    q1, q2 = bridge.pad_shared_intermediate(w1, w2, i_s)
    assert q1 is w1 and q2 is w2


def test_qwen38_fallback_bf16_shared_expert_gated(monkeypatch):
    """Above the cap with a BF16 shared expert: routed + sigmoid(x . w_g) *
    shared, as Qwen2MoeMLP computes it; dropping the gate must not match."""
    import jax.numpy as jnp

    monkeypatch.setattr(bridge, "fused_moe_func", lambda x, *a, **k: x)
    monkeypatch.setattr(bridge, "resolve_onehot_permute_threshold", lambda: 0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, H)).astype(np.float32)
    w1s, w2s, wg = (rng.normal(size=s) for s in ((H, 4), (2, H), (H, 1)))
    w = np.zeros((1, 1), np.float32)

    def body(x):
        return bridge._layer(
            np.array([[0, 0]], np.int32),
            x,
            *([w] * 4),
            jnp.asarray(w1s, jnp.bfloat16),
            np.ones(4, np.float32),
            jnp.asarray(w2s, jnp.bfloat16),
            np.ones(H, np.float32),
            jnp.asarray(wg, jnp.bfloat16),
            np.zeros((4, 2), np.float32),
            np.zeros((4, 2), np.int32),
            n_ranks=1,
            topk=2,
            expert_parallel=False,
            max_tokens=1,
        )

    xb = jnp.asarray(x, jnp.bfloat16)
    got = np.asarray(
        jax.vmap(body, axis_name=EP_AXIS_NAME)(xb[None])[0].astype(jnp.float32)
    )
    xf = np.asarray(xb.astype(jnp.float32))
    bf = [
        np.asarray(jnp.asarray(a, jnp.bfloat16).astype(jnp.float32))
        for a in (w1s, w2s, wg)
    ]
    gate_up = xf @ bf[0]
    shared = (np.asarray(jax.nn.silu(gate_up[:, :2])) * gate_up[:, 2:]) @ bf[1]
    gate = 1 / (1 + np.exp(-(xf @ bf[2])))
    np.testing.assert_allclose(got, xf + gate * shared, rtol=5e-2, atol=5e-2)
    assert not np.allclose(got, xf + shared, rtol=5e-2, atol=5e-2)
