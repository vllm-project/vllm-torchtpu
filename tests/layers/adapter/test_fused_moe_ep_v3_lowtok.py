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
"""CPU unit tests for the v3 fused EP + few-token MoE + small reduce-scatter bridge."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.moe_lowtok import expert_parallel as lowtok_ep
from vllm_torchtpu.kernels.moe_lowtok.kernel import LANES
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as bridge_v3
from vllm_torchtpu.layers.adapter import moe_lowtok_scores as scores_entry
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

EP = 8
LOCAL_EXPERTS = LANES // EP  # 64
HIDDEN = 512
INTER = 256


@pytest.fixture(autouse=True)
def _reset_cached_env_attrs(monkeypatch):
    """Clear stale module-dict attrs left by monkeypatch.setattr(envs, ...)."""
    for key in tuple(envs.__dict__):
        if key in envs.environment_variables:
            monkeypatch.delattr(envs, key, raising=False)


@pytest.fixture
def clean_v3_cache(monkeypatch):
    monkeypatch.setattr(bridge_v3, "_OPS", {})


def test_lowtok_keeps_small_buckets_inside_v3_op(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", "33")
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "0")
    assert bridge_v3.fused_kernel_takes(8) is False

    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "256")
    assert bridge_v3.fused_kernel_takes(8) is True
    assert bridge_v3.fused_kernel_takes(64) is True

    out = torch.ones(8, 16)
    buf = torch.tensor(5, dtype=torch.int32)
    kept = bridge_v3.retain_fork_graph_inputs(out, buf)
    assert kept is out


def test_v3_build_op_routes_small_steps_to_lowtok_and_large_to_v3(
    clean_v3_cache, monkeypatch
):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", "16")
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")

    captured = {}

    def fake_jax_op(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        captured["name"] = name
        captured["fn"] = fn
        captured["in_specs"] = input_partition_specs
        captured["out_specs"] = output_partition_specs
        return SimpleNamespace(register_fake=lambda f: None, fn=fn)

    calls = []
    monkeypatch.setattr(bridge_v3, "jax_op", fake_jax_op)
    monkeypatch.setattr(
        lowtok_ep,
        "moe_lowtok_ep",
        lambda *a, **kw: calls.append(("lowtok", a, kw)) or a[0],
    )

    import vllm_torchtpu.kernels.fused_moe.v3 as v3_pkg

    monkeypatch.setattr(
        v3_pkg,
        "fused_ep_moe",
        lambda *a, **kw: calls.append(("v3", a, kw)) or a[0],
    )

    mesh = SimpleNamespace(
        shape={bridge.EP_AXIS_NAME: EP},
        axis_names=(bridge.EP_AXIS_NAME,),
    )
    bridge_v3.build_op(
        mesh,
        topk=8,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        hidden=HIDDEN,
        inter=INTER,
        local_experts=LOCAL_EXPERTS,
    )
    assert len(captured["in_specs"]) == 7

    fn = captured["fn"]
    w1 = jnp.zeros((LANES, HIDDEN, 2 * INTER), dtype=jnp.float8_e4m3fn)
    w2 = jnp.zeros((LANES, INTER, HIDDEN), dtype=jnp.float8_e4m3fn)
    w1s = jnp.ones((LANES, 1, 1, 2 * INTER), dtype=jnp.float32)
    w2s = jnp.ones((LANES, 1, 1, HIDDEN), dtype=jnp.float32)
    rank = jnp.arange(EP, dtype=jnp.int32).reshape(EP, 1)

    # Small step (64 gathered rows <= max_tokens=64) -> moe_lowtok_ep
    x_small = jnp.zeros((64, HIDDEN), dtype=jnp.bfloat16)
    gate_small = jnp.zeros((64, LANES), dtype=jnp.bfloat16)
    fn(x_small, w1, w2, w1s, w2s, gate_small, rank)
    assert calls[-1][0] == "lowtok"
    assert calls[-1][2]["small_reduce_scatter_max_rows"] == 16

    # Large step (128 gathered rows > max_tokens=64) -> fused_ep_moe
    x_large = jnp.zeros((128, HIDDEN), dtype=jnp.bfloat16)
    gate_large = jnp.zeros((128, LANES), dtype=jnp.bfloat16)
    fn(x_large, w1, w2, w1s, w2s, gate_large, rank)
    assert calls[-1][0] == "v3"


def test_v3_build_op_scores_entry_wires_16_operands_and_copies(
    clean_v3_cache, monkeypatch
):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", "32")
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")

    captured = {}

    def fake_jax_op(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        captured["fn"] = fn
        captured["in_specs"] = input_partition_specs
        return SimpleNamespace(register_fake=lambda f: None)

    calls = []
    monkeypatch.setattr(bridge_v3, "jax_op", fake_jax_op)
    monkeypatch.setattr(
        lowtok_ep,
        "moe_lowtok_scores_ep",
        lambda *a, **kw: calls.append(("scores", a, kw)) or a[0],
    )

    import vllm_torchtpu.kernels.fused_moe.v3 as v3_pkg

    monkeypatch.setattr(
        v3_pkg,
        "fused_ep_moe",
        lambda *a, **kw: calls.append(("v3", a, kw)) or a[0],
    )

    mesh = SimpleNamespace(
        shape={bridge.EP_AXIS_NAME: EP},
        axis_names=(bridge.EP_AXIS_NAME,),
    )
    bridge_v3.build_op(
        mesh,
        topk=8,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        hidden=HIDDEN,
        inter=INTER,
        local_experts=LOCAL_EXPERTS,
        lowtok_scores=True,
        lowtok_scores_copies=True,
    )
    assert len(captured["in_specs"]) == 16

    fn = captured["fn"]
    w1 = jnp.zeros((LANES, HIDDEN, 2 * INTER), dtype=jnp.float8_e4m3fn)
    w2 = jnp.zeros((LANES, INTER, HIDDEN), dtype=jnp.float8_e4m3fn)
    w1s = jnp.ones((LANES, 1, 1, 2 * INTER), dtype=jnp.float32)
    w2s = jnp.ones((LANES, 1, 1, HIDDEN), dtype=jnp.float32)
    cw1 = jnp.zeros((EP * 4, HIDDEN, 2 * INTER), dtype=jnp.float8_e4m3fn)
    cw2 = jnp.zeros((EP * 4, INTER, HIDDEN), dtype=jnp.float8_e4m3fn)
    cw1s = jnp.ones((EP * 4, 1, 1, 2 * INTER), dtype=jnp.float32)
    cw2s = jnp.ones((EP * 4, 1, 1, HIDDEN), dtype=jnp.float32)
    rank = jnp.arange(EP, dtype=jnp.int32).reshape(EP, 1)
    pad = jnp.zeros((64,), dtype=jnp.int32)
    tables = jnp.zeros((EP, 16), dtype=jnp.int32)
    prim = jnp.zeros((EP, LANES), dtype=jnp.int32)
    copy = jnp.zeros((EP, LANES), dtype=jnp.int32)
    hot = jnp.zeros((EP * LANES, 128), dtype=jnp.float32)

    x_small = jnp.zeros((64, HIDDEN), dtype=jnp.bfloat16)
    gate_small = jnp.zeros((64, LANES), dtype=jnp.bfloat16)
    fn(
        x_small,
        w1,
        w2,
        w1s,
        w2s,
        gate_small,
        rank,
        pad,
        tables,
        prim,
        copy,
        hot,
        cw1,
        cw2,
        cw1s,
        cw2s,
    )
    assert calls[-1][0] == "scores"
    _, pos_args, kw_args = calls[-1]
    assert pos_args[11] == (cw1, cw2, cw1s, cw2s)
    assert kw_args["rank"] is rank
    assert kw_args["small_reduce_scatter_max_rows"] == 32

    # Larger than max_tokens falls through to v3 fused_ep_moe
    x_large = jnp.zeros((128, HIDDEN), dtype=jnp.bfloat16)
    gate_large = jnp.zeros((128, LANES), dtype=jnp.bfloat16)
    fn(
        x_large,
        w1,
        w2,
        w1s,
        w2s,
        gate_large,
        rank,
        jnp.zeros((128,), dtype=jnp.int32),
        tables,
        prim,
        copy,
        hot,
        cw1,
        cw2,
        cw1s,
        cw2s,
    )
    assert calls[-1][0] == "v3"


def test_fp8_apply_monolithic_passes_scores_operands_on_v3(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    monkeypatch.setenv("TPU_MOE_LOWTOK_FROM_SCORES", "1")
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "256")
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)

    layer = torch.nn.Module()
    for name, shape, dtype in (
        ("w13_weight", (LOCAL_EXPERTS, HIDDEN, 2 * INTER), torch.float8_e4m3fn),
        ("w2_weight", (LOCAL_EXPERTS, INTER, HIDDEN), torch.float8_e4m3fn),
        ("w13_weight_scale_inv", (LOCAL_EXPERTS, 1, 1, 2 * INTER), torch.float32),
        ("w2_weight_scale_inv", (LOCAL_EXPERTS, 1, 1, HIDDEN), torch.float32),
    ):
        setattr(
            layer,
            name,
            torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False),
        )
    layer.register_buffer("_experts_start", torch.tensor(0, dtype=torch.int32))
    layer.global_num_experts = LANES
    layer.renormalize = True
    layer.e_score_correction_bias = None
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.moe_config = SimpleNamespace(
        ep_rank=0,
        ep_size=EP,
        experts_per_token=8,
        moe_parallel_config=SimpleNamespace(use_ep=True, pcp_size=1),
    )

    tables = scores_entry.arm(layer, "silu", adaptive=True)
    assert tables is not None

    received = {}

    def fake_op(*args):
        received["arity"] = len(args)
        return args[0]

    method = object.__new__(VllmFp8MoEMethodTPU)
    method._tpu_activation_str = "silu"
    method._use_adaptive_fused_moe = False
    setattr(method, bridge.FUSED_MOE_EP_OP_ATTR, fake_op)
    monkeypatch.setattr(bridge, "_RANK_BUFFER", torch.zeros(1, 1, dtype=torch.int32))

    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    logits = torch.zeros(4, LANES, dtype=torch.bfloat16)
    out = method.apply_monolithic(layer, x, logits)
    assert out.shape == x.shape
    assert received["arity"] == 16


def test_combine_partials_uses_small_reduce_scatter_below_cap(monkeypatch):
    calls = []

    def fake_srs(partial, *, num_cores, axis, rank):
        calls.append((partial.shape, num_cores, axis, rank.shape))
        local_rows = partial.shape[0] // num_cores
        return partial[:local_rows]

    assert lowtok_ep.sgs is not None
    monkeypatch.setattr(lowtok_ep.sgs, "small_reduce_scatter", fake_srs)

    partial = jnp.ones((EP * 8, 256), dtype=jnp.bfloat16)
    rank_l = jnp.zeros((1, 1), dtype=jnp.int32)

    out = lowtok_ep._combine_partials(
        partial,
        axis="ep",
        ep=EP,
        rank_l=rank_l,
        out_dtype=jnp.bfloat16,
        small_reduce_scatter_max_rows=16,
        interpret=False,
    )
    assert out.shape == (8, 256)
    assert calls == [((EP * 8, 256), EP, "ep", (1,))]

    # Above cap or interpret=True bypasses small_reduce_scatter
    calls.clear()
    monkeypatch.setattr(
        jax.lax,
        "psum_scatter",
        lambda x, axis, scatter_dimension=0, tiled=True: x[: x.shape[0] // EP],
    )
    out_above = lowtok_ep._combine_partials(
        partial,
        axis="ep",
        ep=EP,
        rank_l=rank_l,
        out_dtype=jnp.bfloat16,
        small_reduce_scatter_max_rows=4,
        interpret=False,
    )
    assert out_above.shape == (8, 256)
    assert calls == []
