# SPDX-License-Identifier: Apache-2.0
"""Opt-in forward post-processing dispatch checks; run this file explicitly."""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.core import block_table_delta as btd
from vllm_torchtpu.layers.core import table_lookup

_STUB_EP = 2


def _layer(experts=4, hidden=8, inter=16, **overrides):
    """A layer whose weights and routing the kernel would accept."""
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
        global_num_experts=experts * _STUB_EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _prebuild(
    layer,
    *,
    has_mesh=True,
    node_tokens=4096,
    smem_bytes=1024 * 1024,
    ep_order=None,
    replica_groups=((0,), (1,)),
    build_op=None,
    **kwargs,
):
    """Call prebuild with the distributed lookups stubbed out.

    `build_ep_mesh` does an `all_gather_object` over the EP group, so it is
    replaced by a two-way mesh stub. Everything under test decides before the
    op would be built, and `_build_op` is stubbed so the one accepting case
    does not need a device.
    """
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP}) if has_mesh else None
    build_op = build_op or (lambda *_args, **_kwargs: "op")
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=ep_order),
        patch.object(bridge, "ep_token_replica_groups", return_value=replica_groups),
        patch.object(bridge, "_max_node_tokens", return_value=node_tokens),
        patch.object(bridge, "_smem_capacity_bytes", return_value=smem_bytes),
        patch.object(bridge, "_build_op", side_effect=build_op),
    ):
        defaults = dict(topk=2, renormalize=True, activation="silu")
        defaults.update(kwargs)
        return bridge.prebuild_fused_moe_ep(layer, **defaults)


@pytest.fixture
def armed_moe():
    """Enable the kernel for the marked MoE dispatch checks."""
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        patch.object(envs, "MOE_FUSED_EP_ENABLE_W4A8", True),
        patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024),
        patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", False),
    ):
        yield


@pytest.mark.usefixtures("armed_moe")
@pytest.mark.parametrize("tuning", [True, False], ids=["tuned", "flag_off"])
def test_dspark_draft_op_gets_its_own_push_knobs(monkeypatch, tuning):
    """A DSpark draft layer's op runs sc_prepush=1 and two rungs (measured
    161.8 -> 135.8 us per draft layer on EP16), while a target layer built in
    the same process keeps the process-wide knobs: the draft tuning was never
    measured on the target's shapes, so it must not leak onto them. The two
    ops are distinct cache entries, and TPU_DSPARK_DRAFT_MOE_TUNING=0 gives
    the draft layer the target's knobs back."""
    from unittest.mock import Mock

    from vllm_torchtpu.kernels.experimental import push_fused_moe

    monkeypatch.setattr(bridge, "_OPS", {})
    closures = []

    def build(name, fn, **kwargs):
        closures.append(fn)
        return Mock()

    monkeypatch.setattr(bridge, "jax_op", build)
    push_kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
    monkeypatch.setattr(push_fused_moe, "fused_ep_moe_push", push_kernel)
    monkeypatch.setitem(vars(envs), "MOE_FUSED_EP_KERNEL_IMPL", "push")
    monkeypatch.setitem(vars(envs), "MOE_FUSED_EP_PUSH_SC_PREPUSH", 2)
    monkeypatch.setitem(vars(envs), "MOE_FUSED_EP_PUSH_TC_EXCHANGE", True)
    monkeypatch.setitem(vars(envs), "MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", 0)
    monkeypatch.setitem(vars(envs), "TPU_DSPARK_DRAFT_MOE_TUNING", tuning)
    target = bridge._build_op(None, 2, True, "silu", None, False)
    draft = bridge._build_op(None, 2, True, "silu", None, False, dspark_draft=True)
    assert (draft is target) != tuning
    operands = (
        jnp.ones((8, 8), dtype=jnp.bfloat16),
        None,
        None,
        None,
        None,
        jnp.zeros((8, 4)),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    seen = []
    for closure in closures:
        closure(*operands)
        kw = push_kernel.call_args.kwargs
        seen.append((kw["sc_prepush"], kw["height_rungs_override"], kw["tc_exchange"]))
    assert seen[0] == (2, 0, True)
    if tuning:
        assert seen[1] == (1, 2, True)
    else:
        assert len(seen) == 1


@pytest.mark.usefixtures("armed_moe")
@pytest.mark.parametrize("marked", [False, True])
def test_prebuild_reads_the_dspark_draft_mark_off_the_layer(marked):
    """prebuild learns a layer is a draft layer only from the mark the DSpark
    model puts on its experts; an unmarked (target) layer builds as before."""
    layer = _layer()
    if marked:
        bridge.mark_dspark_draft_moe(SimpleNamespace(routed_experts=layer))
    seen = []
    assert _prebuild(layer, build_op=lambda *a, **kw: seen.append(kw) or "op") == "op"
    assert seen[0]["dspark_draft"] is marked


@pytest.mark.usefixtures("armed_moe")
@pytest.mark.parametrize("impl", ["push", "v2"])
def test_scale_lookup_flag_specializes_op_and_stays_bound(monkeypatch, impl):
    """An existing op keeps its option when another env selection is built."""
    from unittest.mock import Mock

    from vllm_torchtpu.kernels.experimental import push_fused_moe
    from vllm_torchtpu.kernels.fused_moe import v2

    monkeypatch.setattr(bridge, "_OPS", {})
    built = []

    def build(name, fn, **kwargs):
        op = Mock()
        built.append((name, fn, op))
        return op

    kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
    monkeypatch.setattr(v2, "fused_ep_moe_v2", kernel)
    monkeypatch.setattr(push_fused_moe, "fused_ep_moe_push", kernel)
    monkeypatch.setattr(bridge, "jax_op", build)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", impl)
    selected = []
    for enabled in (False, True, False):
        monkeypatch.setattr(envs, "TPU_MOE_ONEHOT_SCALE_LOOKUP", enabled)
        selected.append(bridge._build_op(None, 2, True, "silu", None, False))
    assert selected[0] is selected[2]
    assert selected[0] is not selected[1]
    assert len(built) == 2 and built[0][0] != built[1][0]
    x = jnp.ones((8, 8), dtype=jnp.bfloat16)
    for enabled, (_, fn, _) in zip((False, True), built, strict=True):
        fn(x, None, None, None, None, jnp.zeros((8, 4)), jnp.zeros((1, 1), jnp.int32))
        assert kernel.call_args.kwargs["onehot_scale_lookup"] is enabled


def test_donated_delta_matches_functional():
    # The JAX body of the donated op must produce the functional table,
    # duplicate (padding) pairs included.
    import jax.numpy as jnp
    import numpy as np

    staged = torch.arange(6, dtype=torch.int32)
    update = torch.tensor([[1, 4, 1], [10, 40, 10]], dtype=torch.int32)
    want = btd._apply_block_table_delta(staged, update)
    got, _ = btd._jax_apply_delta(
        jnp.asarray(staged.numpy()), jnp.asarray(update.numpy())
    )
    assert np.array_equal(np.asarray(got), want.numpy())


def _starts(rng, requests, tokens, first=0):
    counts = rng.integers(0, 2 * tokens // requests + 2, requests)
    counts[rng.random(requests) < 0.2] = 0
    starts = np.concatenate(([first], first + np.cumsum(counts)))
    return np.minimum(starts, first + tokens)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("requests,tokens", [(48, 288), (48, 240), (1, 7), (5, 64)])
def test_int32_table_lookups_match(monkeypatch, seed, requests, tokens):
    rng = np.random.default_rng(seed)
    starts = torch.tensor(_starts(rng, requests, tokens), dtype=torch.int32)
    rows = torch.arange(tokens)
    index = torch.tensor(rng.integers(-requests, requests, tokens))
    table = torch.tensor(rng.integers(-1000, 1000, requests + 1), dtype=torch.int32)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("TPU_ONEHOT_TABLE_LOOKUP", flag)
        out[flag] = (
            table_lookup.bucketize_right(rows, starts[1:]),
            table_lookup.bucketize_right(rows.int(), starts[1:], out_int32=True),
            table_lookup.take(table, index),
            table_lookup.take(table, index.clamp(min=0)[:, None]),
        )
    for ref, new in zip(out["0"], out["1"], strict=True):
        assert ref.dtype == new.dtype and ref.shape == new.shape
        assert torch.equal(ref, new)
