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
        self.calls = 0
        self.gate_up_proj = SimpleNamespace(
            weight=torch.zeros((h, 2 * i_s), dtype=f8), weight_scale=torch.ones(2 * i_s)
        )
        self.down_proj = SimpleNamespace(
            weight=torch.zeros((i_s, h), dtype=f8),
            weight_scale=torch.ones(h),
            reduce_results=False,
        )
        self.act_fn = SiluAndMul.__new__(SiluAndMul)
        torch.nn.Module.__init__(self.act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full_like(x, 10.0)


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


@pytest.mark.parametrize("ep_impl", ["default", "adaptive"])
def test_stands_down_when_fused_ep_serves_layer(monkeypatch, ep_impl):
    """An FP8 pure-EP layer armed with a fused EP kernel must not also be armed
    by fused_moe_decode: apply_monolithic returns from the fused EP branch
    before fused_moe_decode, so arming decode would drop the runner's shared
    expert and routed_scaling_factor."""
    import vllm.config as vllm_config
    import vllm.distributed
    import vllm.distributed.parallel_state as parallel_state
    from jax.experimental.pallas import tpu as pltpu
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner import (
        moe_runner as moe_runner_mod,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as adaptive_moe,
    )
    from vllm_torchtpu.layers.adapter import fused_moe_ep, moe_routing
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import (
        VllmMoERunner,  # noqa: F401
    )

    n_ranks, h, e, inter, i_s, topk = 8, 1024, 4, 128, 128, 2
    mesh = SimpleNamespace(size=n_ranks, shape={EP_AXIS_NAME: n_ranks})
    ep_calls = 0
    decode_calls = 0

    def fake_ep_op(x, *_args):
        nonlocal ep_calls
        ep_calls += 1
        return torch.full_like(x, 4.0)

    def fake_decode_op(x, *_args):
        nonlocal decode_calls
        decode_calls += 1
        return torch.full_like(x, 99.0)

    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", False)
    monkeypatch.setattr(envs, "FUSED_MOE_DECODE_KERNEL", True)
    monkeypatch.setattr(envs, "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", 256)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(envs, "TPU_MOE_FP8_ALL_GATHER", False)
    monkeypatch.setattr(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: n_ranks
    )
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    cfg_stub = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048)
    )
    monkeypatch.setattr(vllm_config, "get_current_vllm_config", lambda: cfg_stub)
    monkeypatch.setattr(adaptive_moe, "get_current_vllm_config", lambda: cfg_stub)

    for mod in (fused_moe_ep, adaptive_moe, bridge):
        monkeypatch.setattr(mod, "build_ep_mesh", lambda: mesh)
        monkeypatch.setattr(mod, "ep_mesh_index", lambda: 0)
    for mod in (fused_moe_ep, adaptive_moe):
        monkeypatch.setattr(mod, "ep_rank_order", lambda: tuple(range(n_ranks)))
        monkeypatch.setattr(mod, "ep_token_replica_groups", lambda **_k: ())
        monkeypatch.setattr(mod, "_build_op", lambda *_a, **_k: fake_ep_op)
        monkeypatch.setattr(mod, "_RANK_BUFFER", None)
    monkeypatch.setattr(bridge, "_RANK_BUFFER", None)
    monkeypatch.setattr(bridge, "_RANK_BUFFER_START", 0)
    monkeypatch.setattr(bridge, "_RANK_BUFFER_INDEX", None)
    monkeypatch.setattr(bridge, "_check_same_arming_on_every_rank", lambda *_a: None)
    monkeypatch.setattr(bridge, "_build_op", lambda *_a, **_k: fake_decode_op)

    layers: dict[str, MoERunner] = {}
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda _c, _l: None
    )
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name", layers.__getitem__)
    monkeypatch.setattr(
        moe_runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(moe_runner_mod, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce", lambda t: t)

    par = SimpleNamespace(
        use_ep=True,
        ep_size=n_ranks,
        ep_rank=0,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        enable_eplb=False,
        is_sequence_parallel=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
        use_all2all_kernels=False,
    )
    moe_cfg = SimpleNamespace(
        moe_parallel_config=par,
        experts_per_token=topk,
        in_dtype=torch.bfloat16,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=1,
        ep_size=n_ranks,
        ep_rank=0,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        hidden_dim=h,
        hidden_dim_unpadded=h,
    )
    qm = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
    FusedMoEMethodBase.__init__(qm, moe_cfg)
    qm._tpu_activation_str = "silu"

    class _EpRoutedExperts(torch.nn.Module):
        forward_monolithic = RoutedExperts.forward_monolithic

        def __init__(self):
            super().__init__()
            self.moe_config = moe_cfg
            self.quant_method = qm
            self.global_num_experts = e * n_ranks
            self.scoring_func = "softmax"
            self.renormalize = True
            self.use_grouped_topk = False
            self.num_expert_group = 1
            self.topk_group = 1
            self.custom_routing_function = None
            self.routed_scaling_factor = 1.0
            self.e_score_correction_bias = None
            self.expert_placement_strategy = "linear"
            f8 = torch.float8_e4m3fn
            self.w13_weight = torch.nn.Parameter(
                torch.zeros((e, h, 2 * inter), dtype=f8), requires_grad=False
            )
            self.w2_weight = torch.nn.Parameter(
                torch.zeros((e, inter, h), dtype=f8), requires_grad=False
            )
            self.w13_weight_scale_inv = torch.nn.Parameter(
                torch.ones((e, 1, 1, 2 * inter)), requires_grad=False
            )
            self.w2_weight_scale_inv = torch.nn.Parameter(
                torch.ones((e, 1, 1, h)), requires_grad=False
            )
            moe_routing.register_experts_start_buffer(self, device=torch.device("cpu"))

        def _ensure_moe_quant_config_init(self):
            pass

    routed = _EpRoutedExperts()
    if ep_impl == "adaptive":
        qm._use_adaptive_fused_moe = True
        qm._adaptive_fused_moe_op = adaptive_moe.prebuild_adaptive_fused_moe(
            routed, topk=topk, renormalize=True, activation="silu"
        )
    else:
        ep_op = fused_moe_ep.prebuild_fused_moe_ep(
            routed, topk=topk, renormalize=True, activation="silu"
        )
        assert ep_op is not None
        setattr(qm, fused_moe_ep.FUSED_MOE_EP_OP_ATTR, ep_op)

    shared_mlp = DeepseekV2MLP(h, i_s)
    runner = MoERunner(
        layer_name="model.layers.0.mlp.experts",
        moe_config=moe_cfg,
        router=SimpleNamespace(prepare_monolithic_logits=lambda logits: logits),
        routed_experts=routed,
        shared_experts=shared_mlp,
        routed_scaling_factor=2.5,
    )
    layers[runner.layer_name] = runner
    model = torch.nn.Module()
    model.add_module("moe", runner)

    armed = bridge.arm_fused_moe_decode(model)
    x = torch.ones((4, h), dtype=torch.bfloat16)
    logits = torch.zeros((4, e * n_ranks), dtype=torch.float32)
    out = runner(hidden_states=x, router_logits=logits)

    torch.testing.assert_close(out, torch.full_like(x, 20.0))
    assert (ep_calls, shared_mlp.calls, decode_calls) == (1, 1, 0)
    assert armed == 0 and not bridge.fused_moe_decode_supported(qm)
    assert runner.shared_experts is not None and runner.routed_scaling_factor == 2.5
