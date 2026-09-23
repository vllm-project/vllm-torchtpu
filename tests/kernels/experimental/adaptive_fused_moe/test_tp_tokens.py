# SPDX-License-Identifier: Apache-2.0
"""PR #1084 replica layouts ported to the adaptive kernel, against original v2.

Single-host tests use eight local TPU devices, with deliberately permuted mesh
order. Run with TPU_SKIP_MDS_QUERY=true and select ``-k single_host``. Device
queries happen during fixture setup, never during pytest collection.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    AXIS,
    WeightFormat,
    adaptive_fused_moe,
)
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.host import U32_SUBLANE_TILE
from vllm_torchtpu.kernels.fused_moe.v2 import fused_ep_moe_v2

pytestmark = pytest.mark.multichip

_TP2_PERMUTED = ((6, 1), (4, 3), (0, 7), (2, 5))
_TP4_PERMUTED = ((7, 1, 4, 2), (6, 0, 5, 3))


@pytest.fixture(scope="module")
def single_host_mesh():
    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    # Group members are mesh positions, not physical device ids.
    order = (3, 0, 6, 1, 7, 4, 2, 5)
    return Mesh(np.asarray([devices[i] for i in order]), (AXIS,))


def _make_inputs(mesh, groups, rows, weight_format):
    ep = mesh.shape[AXIS]
    owner = [0] * ep
    for i, group in enumerate(groups):
        for device in group:
            owner[device] = i
    rank = jax.device_put(
        np.arange(ep, dtype=np.int32).reshape(ep, 1), NamedSharding(mesh, P(AXIS))
    )

    def init(rank_l):
        me = rank_l[0, 0]
        group_id = jnp.asarray(owner, dtype=jnp.int32)[me]
        tk = jax.random.split(jax.random.fold_in(jax.random.key(123), group_id), 2)
        wk = jax.random.split(jax.random.fold_in(jax.random.key(456), me), 2)
        x = (jax.random.normal(tk[0], (rows, 512)) * 0.2).astype(jnp.bfloat16)
        gate = jax.random.normal(tk[1], (rows, 2 * ep))

        def weights(key, width):
            raw = jax.random.normal(key, (2, 512, width)) / jnp.sqrt(512.0)
            fp4 = weight_format == WeightFormat.FP4
            dtype = jnp.float4_e2m1fn if fp4 else jnp.float8_e4m3fn
            maximum = 6.0 if fp4 else float(jnp.finfo(dtype).max)
            scale = jnp.maximum(
                jnp.max(jnp.abs(raw), axis=1, keepdims=True) / maximum, 1e-12
            )
            weight = (raw / scale).astype(dtype)
            # FP4 uses one contraction block, FP8 uses per-channel scales.
            return weight, scale if fp4 else scale[:, 0, :]

        w1, s1 = weights(wk[0], 1024)
        w2, s2 = weights(wk[1], 512)
        return x, w1, w2, s1, s2, gate

    inputs = jax.jit(
        jax.shard_map(
            init, mesh=mesh, in_specs=P(AXIS), out_specs=(P(AXIS),) * 6, check_vma=False
        )
    )(rank)
    return rank, inputs


def _run_layer(mesh, rank, inputs, groups, weight_format):
    input_rows = inputs[0].shape[0] // mesh.shape[AXIS]
    # The original router accepts aligned blocks or a whole power-of-two
    # array. Pad unsupported reference inputs here, as the serving runner
    # does, without widening the kernel's no-replica contract.
    reference_padding = 0
    if groups is None and input_rows & (input_rows - 1):
        reference_padding = -input_rows % U32_SUBLANE_TILE
    if reference_padding:

        def pad(x, gate):
            return (
                jnp.pad(x, ((0, reference_padding), (0, 0))),
                jnp.pad(
                    gate, ((0, reference_padding), (0, 0)), constant_values=-jnp.inf
                ),
            )

        x, gate = jax.shard_map(
            pad,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=(P(AXIS), P(AXIS)),
            check_vma=False,
        )(inputs[0], inputs[-1])
        inputs = (x, *inputs[1:-1], gate)

    def run(rank_arg, *args):
        fn = fused_ep_moe_v2 if groups is None else adaptive_fused_moe
        options = (
            {"sharded_plan": True}
            if groups is None
            else {"token_replica_groups": groups}
        )
        return fn(
            *args,
            topk=4,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=weight_format,
            rhs_qb=512 if weight_format == WeightFormat.FP4 else None,
            rank=rank_arg,
            **options,
        )

    result = jax.jit(run)(rank, *inputs)
    if reference_padding:
        result = jax.shard_map(
            lambda rows: rows[:input_rows],
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(AXIS),
            check_vma=False,
        )(result)
    return jax.block_until_ready(result)


def _assert_outputs_match(actual, reference, rows, groups):
    assert actual.shape == reference.shape
    refs = {
        s.device.id: np.asarray(s.data).astype(np.float32)
        for s in reference.addressable_shards
    }
    stats = {}
    by_rank = {}
    for shard in actual.addressable_shards:
        got = np.asarray(shard.data).astype(np.float32)
        want = refs[shard.device.id]
        assert np.isfinite(got).all() and np.isfinite(want).all()
        assert got.shape == want.shape == (rows, 512)
        assert np.linalg.norm(want) > 1e-8, "zero output is not a useful oracle"
        delta = got - want
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(want), 1e-20))
        worst = float(
            np.max(
                np.linalg.norm(delta, axis=1)
                / np.maximum(np.linalg.norm(want, axis=1), 1e-20)
            )
        )
        assert relative < 0.002, (shard.device.id, relative)
        assert worst < 0.01, (shard.device.id, worst)
        stats[str(shard.device.id)] = {
            "relative_l2": relative,
            "worst_token_relative_l2": worst,
            "bitwise_equal": bool(np.array_equal(got, want)),
        }
        by_rank[shard.index[0].start // rows] = got
    # Every TP member must regain the complete output in the same row order.
    for group in groups:
        local = [by_rank[r] for r in group if r in by_rank]
        for output in local[1:]:
            np.testing.assert_array_equal(output, local[0])
    return stats


@pytest.mark.parametrize(
    "weight_format", [WeightFormat.FP8, WeightFormat.FP4], ids=["fp8", "fp4"]
)
@pytest.mark.parametrize(
    "groups,rows",
    [
        pytest.param(tuple((i,) for i in range(8)), 32, id="tp1"),
        pytest.param(((0, 1), (2, 3), (4, 5), (6, 7)), 256, id="tp2-aligned"),
        pytest.param(_TP2_PERMUTED, 24, id="tp2-permuted-padding"),
        pytest.param(_TP4_PERMUTED, 7, id="tp4-permuted-tail"),
        pytest.param(_TP4_PERMUTED, 1, id="tp4-empty-members"),
    ],
)
def test_single_host_replicas_match_original_layer(
    single_host_mesh, groups, rows, weight_format
):
    mesh = single_host_mesh
    rank, inputs = _make_inputs(mesh, groups, rows, weight_format)
    reference = _run_layer(mesh, rank, inputs, None, weight_format)
    actual = _run_layer(mesh, rank, inputs, groups, weight_format)
    _assert_outputs_match(actual, reference, rows, groups)
