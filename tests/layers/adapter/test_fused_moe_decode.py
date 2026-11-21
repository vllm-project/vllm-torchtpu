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
"""Token-cap dispatch of the fused-MoE-decode op's per-rank body: up to
FUSED_MOE_DECODE_KERNEL_MAX_TOKENS the kernel serves it; above, the GMM MoE
fallback. Admission and fallback of GLM-5.2's convention (ungated shared
expert, routed scale). Which modules arm_runner_models arms, with and without
a draft model. No TPU needed: the kernels are stubs."""

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


def _glm_runner(mlp, *, n=2, h=256, i=128, e=4, dtype=torch.bfloat16):
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
        _glm_runner(DeepseekV2MLP(256, 128)), 2
    )
    assert shared[4] is None and expert_parallel and routed_scale == 2.5
    # FP16 vLLM divides the shared output instead; an ungated MLP of another
    # class is not known to be gate_up -> SiluAndMul -> down.
    fp16 = _glm_runner(DeepseekV2MLP(256, 128), dtype=torch.float16)
    assert bridge._admit(fp16, 2) is None
    other = DeepseekV2MLP(256, 128)
    other.act_fn = torch.nn.GELU()
    assert bridge._admit(_glm_runner(other), 2) is None


def test_admission_declines_silu_bf16(monkeypatch):
    """The kernel fuses float32 SiLU; a "silu_bf16" layer must stay on gmm_v2
    for every token count, or one server would mix two roundings."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    runner = _glm_runner(DeepseekV2MLP(256, 128))
    assert bridge._admit(runner, 2) is not None
    runner._quant_method._tpu_activation_str = "silu_bf16"
    assert bridge._admit(runner, 2) is None


def test_admission_declines_topk_weights_float32(monkeypatch):
    """The kernel rounds the top-k weights to bfloat16; under
    tpu_moe_topk_weights_float32 the layer keeps gmm_v2's float32 combine."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    runner = _glm_runner(DeepseekV2MLP(256, 128))
    runner._quant_method._tpu_topk_weights_float32 = True
    assert bridge._admit(runner, 2) is None


def test_admission_declines_attention_narrower_than_mesh(monkeypatch):
    """PCP2 (or DP2) at TP1: the EP group spans the mesh, attention does not, so
    MoERunner still runs a collective after the op."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    assert bridge._admit(_glm_runner(DeepseekV2MLP(256, 128)), 2) is None


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


class _Runner:
    """The parts of the TPU runner arm_runner_models reads."""

    def __init__(self, drafter):
        self.model = torch.nn.Linear(1, 1)
        self.drafter = drafter

    @property
    def _async_drafter(self):
        # TPUModelRunner._async_drafter asserts the drafter is a DFlash/Eagle3
        # proposer, so it raises for no drafter and for an n-gram one alike;
        # arm_runner_models must never touch it.
        raise AssertionError("_async_drafter read")


def _record_arms(monkeypatch):
    armed = []
    monkeypatch.setattr(
        bridge,
        "arm_fused_moe_decode",
        lambda model, what="model": armed.append((model, what)) or 1,
    )
    return armed


@pytest.mark.parametrize(
    "drafter",
    [None, SimpleNamespace()],
    ids=["no-spec-decoding", "ngram-without-draft-model"],
)
def test_arm_runner_models_without_a_draft_model(monkeypatch, drafter):
    """Only the target is armed, and the asserting _async_drafter property is
    never touched."""
    armed = _record_arms(monkeypatch)
    runner = _Runner(drafter)
    assert bridge.arm_runner_models(runner) == 1
    assert armed == [(runner.model, "target model")]


def test_arm_runner_models_arms_the_draft_model(monkeypatch):
    armed = _record_arms(monkeypatch)
    drafter = SimpleNamespace(draft_model=torch.nn.Linear(1, 1))
    runner = _Runner(drafter)
    assert bridge.arm_runner_models(runner) == 2
    assert armed == [
        (runner.model, "target model"),
        (drafter.draft_model, "draft model"),
    ]


def test_arm_runner_models_skips_a_draft_that_is_the_target(monkeypatch):
    """A drafter sharing the target module is not armed twice."""
    armed = _record_arms(monkeypatch)
    runner = _Runner(None)
    runner.drafter = SimpleNamespace(draft_model=runner.model)
    assert bridge.arm_runner_models(runner) == 1
    assert armed == [(runner.model, "target model")]


def test_shared_slice_declines_bias_and_reduce_results(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    mlp_bias = DeepseekV2MLP(256, 128)
    mlp_bias.down_proj.bias = torch.zeros(256)
    assert bridge._admit(_glm_runner(mlp_bias), 2) is None

    mlp_reduced = DeepseekV2MLP(256, 128)
    mlp_reduced.down_proj.reduce_results = True
    assert bridge._admit(_glm_runner(mlp_reduced), 2) is None


# ---------------------------------------------------------------------------
# Trace-time static bucket dispatch and shape_variants placeholder parity
# ---------------------------------------------------------------------------

_fmd_test_lib = torch.library.Library("tpu_fmd_static_dispatch_test", "FRAGMENT")
_fmd_test_lib.define(
    "decode_gated("
    "Tensor x, Tensor w1, Tensor w2, Tensor w1s, Tensor w2s, "
    "Tensor sw1, Tensor sw1s, Tensor sw2, Tensor sw2s, Tensor sg, "
    "Tensor tw, Tensor ti, Tensor rk"
    ") -> Tensor"
)
_fmd_test_lib.impl(
    "decode_gated",
    lambda x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, sg, tw, ti, rk: (
        x + sw1.float().sum() + sw2.float().sum() + sg.float().sum()
    ),
    "CPU",
)
torch.library.register_fake(
    "tpu_fmd_static_dispatch_test::decode_gated",
    lambda x, *_a: torch.empty_like(x),
)
_fmd_test_lib.define(
    "decode_ungated("
    "Tensor x, Tensor w1, Tensor w2, Tensor w1s, Tensor w2s, "
    "Tensor sw1, Tensor sw1s, Tensor sw2, Tensor sw2s, "
    "Tensor tw, Tensor ti, Tensor rk"
    ") -> Tensor"
)
_fmd_test_lib.impl(
    "decode_ungated",
    lambda x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, tw, ti, rk: (
        x + sw1.float().sum() + sw2.float().sum()
    ),
    "CPU",
)
torch.library.register_fake(
    "tpu_fmd_static_dispatch_test::decode_ungated",
    lambda x, *_a: torch.empty_like(x),
)
_fmd_test_lib.define(
    "gmm("
    "Tensor x, Tensor w1, Tensor w2, Tensor w1s, Tensor w2s, "
    "Tensor tw, Tensor ti, Tensor es"
    ") -> Tensor"
)
_fmd_test_lib.impl("gmm", lambda x, *_a: x * 2.0, "CPU")
torch.library.register_fake(
    "tpu_fmd_static_dispatch_test::gmm",
    lambda x, *_a: torch.empty_like(x),
)
_DECODE_GATED_OP = torch.ops.tpu_fmd_static_dispatch_test.decode_gated
_DECODE_UNGATED_OP = torch.ops.tpu_fmd_static_dispatch_test.decode_ungated
_GMM_OP = torch.ops.tpu_fmd_static_dispatch_test.gmm


class _StubLinear(torch.nn.Module):
    def __init__(self, k: int, n: int, *, scaled: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros((k, n), dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = None
        self.reduce_results = False
        if scaled:
            self.weight_scale = torch.nn.Parameter(
                torch.ones(n, dtype=torch.float32), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = x @ self.weight.to(x.dtype)
        if hasattr(self, "weight_scale"):
            out = out * self.weight_scale.to(x.dtype)
        return out, None


class _StubSharedMLP(torch.nn.Module):
    def __init__(self, h: int, i_s: int, *, gated: bool):
        super().__init__()
        self.gate_up_proj = _StubLinear(h, 2 * i_s, scaled=True)
        self.down_proj = _StubLinear(i_s, h, scaled=True)
        self.expert_gate = _StubLinear(h, 1, scaled=False) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu, _ = self.gate_up_proj(x)
        i_s = gu.shape[-1] // 2
        out, _ = self.down_proj(torch.nn.functional.silu(gu[:, :i_s]) * gu[:, i_s:])
        if self.expert_gate is not None:
            out = torch.sigmoid(self.expert_gate(x)[0]) * out
        return out


class _StubSharedExperts(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self._layer = layer
        self.output: torch.Tensor | None = None

    def maybe_forward_async(self, shared_experts_input: torch.Tensor | None) -> bool:
        return False

    def __call__(self, shared_experts_input: torch.Tensor, order: object) -> None:
        self.output = self._layer(shared_experts_input)


class _StubRoutedExperts(torch.nn.Module):
    def __init__(self, qm: VllmFp8MoEMethodTPU, e: int, h: int, i: int, *, ep: bool):
        super().__init__()
        self.quant_method = qm
        self.w13_weight = torch.nn.Parameter(
            torch.zeros((e, h, 2 * i), dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros((e, i, h), dtype=torch.bfloat16), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones((e, 1, 1, 2 * i), dtype=torch.float32), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.ones((e, 1, 1, h), dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "_experts_start",
            torch.tensor(0, dtype=torch.int32) if ep else None,
            persistent=False,
        )
        self.moe_config = SimpleNamespace(experts_per_token=2)

    def _ensure_moe_quant_config_init(self) -> None:
        return None

    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply_monolithic(self, x, router_logits)


def _build_static_dispatch_runner(
    *,
    gated: bool,
    ep: bool,
    max_tokens: int = 16,
    routed_scale: float = 1.0,
    layer_name: str = "model.layers.0.mlp.experts",
):
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

    h, i, i_s, e = 8, 4, 4, 2
    qm = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
    qm._tpu_activation_str = "silu"
    qm._tpu_topk_weights_float32 = False
    qm._use_adaptive_fused_moe = False
    qm.moe_kernel = None
    setattr(
        qm,
        bridge.FUSED_MOE_DECODE_OP_ATTR,
        _DECODE_GATED_OP if gated else _DECODE_UNGATED_OP,
    )
    setattr(qm, bridge._STATIC_DISPATCH_ATTR, True)

    par = SimpleNamespace(
        use_ep=ep, ep_size=2 if ep else 1, dp_size=1, tp_size=2, pcp_size=1
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=par,
        hidden_dim=h,
        hidden_dim_unpadded=0,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=2 if ep else 1,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        use_ep=ep,
        in_dtype=torch.bfloat16,
    )

    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.layer_name = layer_name
    runner.moe_config = moe_config
    runner.router = SimpleNamespace()
    runner.routed_experts = _StubRoutedExperts(qm, e, h, i, ep=ep)
    runner._shared_experts = _StubSharedExperts(_StubSharedMLP(h, i_s, gated=gated))
    runner.gate = None
    runner.shared_expert_gate = None
    runner._fse_fuse_gate = False
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = routed_scale
    runner._forward_entry = lambda h_, l_, s_, i_, _n, _u: runner._forward_impl(
        h_, l_, s_, i_
    )
    object.__setattr__(runner, bridge._RUNNER_ARMED_ATTR, True)
    object.__setattr__(runner, bridge._RUNNER_STATIC_DISPATCH_ATTR, True)
    object.__setattr__(runner, bridge._RUNNER_MAX_TOKENS_ATTR, max_tokens)
    return runner


def _patch_cpu_moe_runtime(monkeypatch, *, all_reduce_calls: list | None = None):
    from contextlib import nullcontext

    from vllm.model_executor.layers.fused_moe.runner import moe_runner as vllm_mr

    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

    monkeypatch.setattr(
        bridge, "_RANK_BUFFER", torch.tensor([[0, 0]], dtype=torch.int32)
    )
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(envs, "TPU_MOE_FP8_ALL_GATHER", False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(
        fp8_mod, "enable_pipelined_collective_and_compute", lambda: False
    )
    monkeypatch.setattr(
        moe_routing,
        "route",
        lambda layer, x, logits, topk_weights_float32=False: (
            logits[:, :2],
            torch.zeros((x.shape[0], 2), dtype=torch.int32, device=x.device),
        ),
    )

    def _stub_gmm(**kwargs):
        es = kwargs["experts_start"]
        if es is None:
            es = torch.zeros(
                (), dtype=torch.int32, device=kwargs["hidden_states"].device
            )
        return _GMM_OP(
            kwargs["hidden_states"],
            kwargs["w1"],
            kwargs["w2"],
            kwargs["w1_scale"],
            kwargs["w2_scale"],
            kwargs["topk_weights"],
            kwargs["topk_ids"],
            es,
        )

    monkeypatch.setattr(fp8_mod, "fused_moe_gmm", _stub_gmm)

    def _stub_all_reduce(x):
        if all_reduce_calls is not None:
            all_reduce_calls.append(x.shape[0])
        return x * 3.0

    monkeypatch.setattr(vllm_mr, "tensor_model_parallel_all_reduce", _stub_all_reduce)
    monkeypatch.setattr(
        vllm_mr.MoERunner, "_sequence_parallel_context", lambda self: nullcontext()
    )


def test_arm_static_dispatch_preserves_shared_experts_and_scaling(monkeypatch):
    """Arming a VllmMoERunner with dp=1, pcp=1 keeps shared experts and
    routed_scaling_factor intact so m > max_tokens runs unmodified MoERunner."""
    import vllm.distributed

    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

    monkeypatch.setattr(envs, "FUSED_MOE_DECODE_KERNEL", True)
    monkeypatch.setattr(envs, "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", 64)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        bridge, "build_ep_mesh", lambda: SimpleNamespace(shape={EP_AXIS_NAME: 2})
    )
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "_check_vmem", lambda *a, **k: None)
    monkeypatch.setattr(
        bridge, "_check_same_arming_on_every_rank", lambda *a, **k: None
    )
    monkeypatch.setattr(bridge, "_RANK_BUFFER", None)
    built_calls = []
    monkeypatch.setattr(
        bridge,
        "_build_op",
        lambda *a, **k: built_calls.append((a, k)) or _DECODE_UNGATED_OP,
    )

    mlp = DeepseekV2MLP(256, 128)
    qm = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
    qm._tpu_activation_str = "silu"
    qm._tpu_topk_weights_float32 = False
    qm.moe_kernel = None
    par = SimpleNamespace(
        use_ep=True, ep_size=2, ep_rank=0, dp_size=1, tp_size=2, pcp_size=1
    )

    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.layer_name = "model.layers.0.mlp.experts"
    runner.moe_config = SimpleNamespace(
        moe_parallel_config=par,
        in_dtype=torch.bfloat16,
        skip_final_all_reduce=False,
        experts_per_token=8,
        num_experts=8,
        ep_size=2,
        ep_rank=0,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=qm,
        moe_config=runner.moe_config,
        global_num_experts=8,
        w13_weight=torch.zeros((4, 256, 256), dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros((4, 128, 256), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones((4, 1, 1, 256)),
        w2_weight_scale_inv=torch.ones((4, 1, 1, 256)),
        _experts_start=torch.tensor(0, dtype=torch.int32),
    )
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 2.5
    runner.shared_expert_gate = None
    runner._shared_experts = SimpleNamespace(_layer=mlp)

    model = torch.nn.Module()
    model.add_module("runner", runner)

    assert bridge.arm_fused_moe_decode(model) == 1
    assert runner._shared_experts is not None
    assert runner.routed_scaling_factor == 2.5
    assert getattr(runner, bridge._RUNNER_STATIC_DISPATCH_ATTR) is True
    assert getattr(runner, bridge._RUNNER_MAX_TOKENS_ATTR) == 64
    assert bridge.fused_moe_decode_supported(qm) is False
    assert runner._fused_output_is_reduced is False
    assert built_calls[0][0][5] is False  # shared_gated=False


def test_static_dispatch_eager_routing_and_live_weight_update(monkeypatch):
    """m <= max_tokens calls the fused op directly against live shared weights;
    m > max_tokens falls through to stock MoERunner + GMM + all-reduce."""
    ar_calls: list[int] = []
    _patch_cpu_moe_runtime(monkeypatch, all_reduce_calls=ar_calls)
    runner = _build_static_dispatch_runner(
        gated=True, ep=False, max_tokens=16, routed_scale=2.5
    )

    x_small = torch.ones((16, 8), dtype=torch.float32)
    logits_small = torch.ones((16, 4), dtype=torch.float32)
    out_before = runner(x_small, logits_small)
    assert ar_calls == []  # decode op reduces internally
    torch.testing.assert_close(out_before, x_small)

    # In-place parameter mutation (e.g. RL weight sync) takes effect without re-arming.
    runner._shared_experts._layer.gate_up_proj.weight.data.fill_(1.0)
    out_after = runner(x_small, logits_small)
    assert ar_calls == []
    expected_delta = runner._shared_experts._layer.gate_up_proj.weight.numel()
    torch.testing.assert_close(out_after, x_small + expected_delta)

    # Above the token cap: unmodified stock path (shared + 2.5 * gmm, then all-reduce).
    runner._shared_experts._layer.gate_up_proj.weight.data.zero_()
    x_large = torch.ones((24, 8), dtype=torch.float32)
    logits_large = torch.ones((24, 4), dtype=torch.float32)
    out_large = runner(x_large, logits_large)
    assert ar_calls == [24]
    # shared=0, gmm=2*x, routed_scale=2.5, all_reduce=3* -> 15 * x.
    torch.testing.assert_close(out_large, x_large * 15.0)


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("ep", [False, True])
def test_static_dispatch_shape_variants_placeholder_parity(monkeypatch, gated, ep):
    """Both sides of the m <= max_tokens trace-time branch produce identical
    FX placeholder signatures and 1-to-1 placeholder order across multiple
    layers, with complementary shape_variants guards."""
    from vllm_torchtpu.compilation import shape_variants

    _patch_cpu_moe_runtime(monkeypatch)
    r0 = _build_static_dispatch_runner(
        gated=gated, ep=ep, max_tokens=16, layer_name="model.layers.0.mlp.experts"
    )
    r1 = _build_static_dispatch_runner(
        gated=gated, ep=ep, max_tokens=16, layer_name="model.layers.1.mlp.experts"
    )

    class TwoLayerMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = r0
            self.l1 = r1

        def forward(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            x = self.l0(x, logits)
            return self.l1(x, logits)

    model = TwoLayerMoE()
    captured: list[tuple[tuple, list[str], object]] = []

    def capture_backend(gm, _example_inputs):
        captured.append(
            (
                shape_variants.graph_signature(gm),
                [n.name for n in gm.graph.nodes if n.op == "placeholder"],
                shape_variants.trace_shape_env(gm),
            )
        )
        return gm

    with shape_variants.warmup():
        for tokens in (16, 32):
            torch._dynamo.reset()
            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            x = torch.ones((tokens, 8), dtype=torch.float32)
            logits = torch.ones((tokens, 4), dtype=torch.float32)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(logits, 0)
            compiled(x, logits)

    (sig_dec, names_dec, env_dec), (sig_pre, names_pre, env_pre) = captured
    assert sig_dec == sig_pre
    assert names_dec == names_pre
    assert shape_variants.unsupported_reason(env_dec, 16) is None
    assert shape_variants.unsupported_reason(env_dec, 32) is not None
    assert shape_variants.unsupported_reason(env_pre, 16) is not None
    assert shape_variants.unsupported_reason(env_pre, 32) is None


def test_static_dispatch_outside_warmup_emits_no_shape_guard(monkeypatch):
    """Modules compiled outside shape_variants.warmup() (e.g. an MTP drafter)
    must not branch on m <= max_tokens so one graph serves every bucket."""
    from vllm_torchtpu.compilation import shape_variants

    _patch_cpu_moe_runtime(monkeypatch)
    runner = _build_static_dispatch_runner(gated=True, ep=False, max_tokens=16)
    captured = []

    def capture_backend(gm, _example_inputs):
        captured.append(gm)
        return gm

    torch._dynamo.reset()
    compiled = torch.compile(runner, backend=capture_backend, fullgraph=True)
    x = torch.ones((32, 8), dtype=torch.float32)
    logits = torch.ones((32, 4), dtype=torch.float32)
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(logits, 0)
    compiled(x, logits)

    env = shape_variants.trace_shape_env(captured[0])
    assert shape_variants.unsupported_reason(env, 16) is None
    assert shape_variants.unsupported_reason(env, 32) is None
    assert any("decode_gated" in str(n.target) for n in captured[0].graph.nodes)


def test_anchor_rank_buffer_for_trace_bitwise_identity(monkeypatch):
    x = torch.tensor([0.0, -0.0, 1.5, float("nan")], dtype=torch.float32)
    monkeypatch.setattr(bridge, "_RANK_BUFFER", None)
    assert bridge.anchor_rank_buffer_for_trace(x) is x

    monkeypatch.setattr(
        bridge, "_RANK_BUFFER", torch.tensor([[2, 0]], dtype=torch.int32)
    )
    assert bridge.anchor_rank_buffer_for_trace(x) is x

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    anchored = bridge.anchor_rank_buffer_for_trace(x)
    torch.testing.assert_close(anchored, x, equal_nan=True)
    assert torch.equal(anchored.view(torch.int32), x.view(torch.int32))
