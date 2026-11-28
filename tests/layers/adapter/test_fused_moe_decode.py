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


def test_admit_tp1_single_rank(monkeypatch):
    """A TP=1 non-EP FP8 MoE layer is admitted with n_ranks=1."""
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    runner = _glm_runner(DeepseekV2MLP(256, 128), n=1)
    runner.moe_config.moe_parallel_config.use_ep = False
    admitted = bridge._admit(runner, 1)
    assert admitted is not None
    shared, expert_parallel, routed_scale = admitted
    assert shared[4] is None and not expert_parallel and routed_scale == 2.5


@pytest.mark.parametrize("global_dev_id", [0, 1])
def test_build_single_rank_mesh_aligns_device_id(monkeypatch, global_dev_id):
    """Odd TPU v7x TensorCores report PJRT device id 1 while get_pallas_mesh()
    hardcodes id 0 for world_size=1; _build_single_rank_mesh re-wraps when they
    differ, keeps the mesh as-is when they match, and caches the result."""
    import sys
    import types

    import vllm.distributed
    from jax.sharding import Mesh

    class _FakePallasDevice:
        def __init__(self, *, id, process_index, device_kind):
            self.id = id
            self.process_index = process_index
            self.device_kind = device_kind

    ranked = Mesh(
        np.array([_FakePallasDevice(id=0, process_index=0, device_kind="TPU7x")]),
        axis_names=(EP_AXIS_NAME,),
    )
    tpu_mod = getattr(torch, "tpu", None)
    pallas_mod = getattr(tpu_mod, "pallas", None) if tpu_mod is not None else None
    if pallas_mod is not None:
        monkeypatch.setattr(pallas_mod, "get_pallas_mesh", lambda axis_names: ranked)
    else:
        fake_tpu = types.ModuleType("torch.tpu")
        fake_pallas = types.ModuleType("torch.tpu.pallas")
        fake_pallas.get_pallas_mesh = lambda axis_names: ranked
        fake_tpu.pallas = fake_pallas
        monkeypatch.setattr(torch, "tpu", fake_tpu, raising=False)
        monkeypatch.setitem(sys.modules, "torch.tpu", fake_tpu)
        monkeypatch.setitem(sys.modules, "torch.tpu.pallas", fake_pallas)
    pallas_sub = types.ModuleType("torch_tpu._internal.pallas.pallas")
    pallas_sub._PallasDevice = _FakePallasDevice
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas.pallas", pallas_sub)
    monkeypatch.setattr(bridge, "_SINGLE_RANK_MESH_CACHE", {})
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.mesh_utils._get_tpu_global_device_id",
        lambda: global_dev_id,
    )

    mesh = bridge._build_single_rank_mesh()
    assert mesh is not None
    assert mesh.shape == {EP_AXIS_NAME: 1}
    dev = mesh.devices.flat[0]
    assert int(dev.id) == global_dev_id
    assert dev.process_index == 0
    assert dev.device_kind == "TPU7x"
    if global_dev_id == 0:
        assert mesh is ranked
    # Subsequent calls reuse the cached Mesh object so _OPS hits its id(mesh) key.
    assert bridge._build_single_rank_mesh() is mesh


@pytest.mark.parametrize(
    "tp_size, dist_init, world_size",
    [(2, True, 1), (1, False, 1), (1, True, 2)],
)
def test_build_single_rank_mesh_declines_non_single_rank(
    monkeypatch, tp_size, dist_init, world_size
):
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: dist_init)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    assert bridge._build_single_rank_mesh() is None


def test_arm_fused_moe_decode_single_rank_fallback(monkeypatch):
    """When build_ep_mesh() returns None at TP=1, arm_fused_moe_decode falls
    back to _build_single_rank_mesh() and uses mesh index 0."""
    import vllm.distributed
    from jax.sharding import Mesh
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class _FakeMoERunner(MoERunner):
        @property
        def _quant_method(self):
            return self._qm

        def __init__(self):
            torch.nn.Module.__init__(self)
            base = _glm_runner(DeepseekV2MLP(256, 128), n=1)
            base.moe_config.moe_parallel_config.use_ep = False
            base.moe_config.experts_per_token = 2
            self.moe_config = base.moe_config
            self._qm = base._quant_method
            self.routed_experts = torch.nn.Module()
            self.routed_experts.moe_config = base.moe_config
            for attr in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale_inv",
                "w2_weight_scale_inv",
            ):
                setattr(self.routed_experts, attr, getattr(base.routed_experts, attr))
            self.routed_input_transform = None
            self.routed_output_transform = None
            self.routed_scaling_factor = 1.0
            self.shared_expert_gate = None
            self._shared_experts = base._shared_experts

        def _select_forward(self):
            return lambda *a, **k: None

    class _Dev:
        id = 1
        process_index = 0
        device_kind = "TPU7x"

    fake_mesh = Mesh(np.array([_Dev()]), axis_names=(EP_AXIS_NAME,))
    model = torch.nn.ModuleList([_FakeMoERunner()])
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "1")
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(bridge, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(bridge, "_build_single_rank_mesh", lambda: fake_mesh)
    monkeypatch.setattr(bridge, "_check_vmem", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_build_op", lambda *a, **k: "fake_op")
    monkeypatch.setattr(bridge, "_RANK_BUFFER", None)
    monkeypatch.setattr(bridge, "_RANK_BUFFER_START", 0)
    monkeypatch.setattr(bridge, "_RANK_BUFFER_INDEX", None)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.get_experts_start", lambda _: 0
    )

    armed = bridge.arm_fused_moe_decode(model, "target model")
    assert armed == 1
    assert bridge._RANK_BUFFER_INDEX == 0
    assert bridge._RANK_BUFFER.tolist() == [[0, 0]]
    assert getattr(model[0]._quant_method, bridge.FUSED_MOE_DECODE_OP_ATTR) == "fake_op"
