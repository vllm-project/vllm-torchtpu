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
"""Direct HBM -> peer VMEM weight streaming (`ep_tp_size > 1`) in fused_ep_moe_v2."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
)
from vllm_torchtpu.kernels.fused_moe.v2 import (
    host as host_mod,
)
from vllm_torchtpu.kernels.fused_moe.v2 import (
    kernel as kernel_mod,
)
from vllm_torchtpu.kernels.fused_moe.v2 import (
    layer as layer_mod,
)


def _dummy_ep_tp_operands(
    *, ep=4, ep_tp_size=2, g_local=2, rows=32, hidden=512, inter=256
):
    t_total = ep * rows
    e_total = ep * g_local
    w1_experts = e_total * ep_tp_size
    inter_slice = inter // ep_tp_size
    x = jnp.zeros((t_total, hidden), dtype=jnp.bfloat16)
    w1 = jnp.zeros((w1_experts, hidden, 2 * inter_slice), dtype=jnp.float8_e4m3fn)
    w2 = jnp.zeros((w1_experts, inter_slice, hidden), dtype=jnp.float8_e4m3fn)
    w1s = jnp.ones((w1_experts, 2 * inter_slice), dtype=jnp.float32)
    w2s = jnp.ones((w1_experts, hidden), dtype=jnp.float32)
    gate = jnp.zeros((t_total, e_total), dtype=jnp.float32)
    mesh = SimpleNamespace(axis_names=(AXIS,), shape={AXIS: ep})
    return x, w1, w2, w1s, w2s, gate, mesh


def test_ep_tp_derives_global_experts_and_full_inter_for_kernel_builder():
    x, w1, w2, w1s, w2s, gate, mesh = _dummy_ep_tp_operands(
        ep=4, ep_tp_size=2, g_local=2, rows=32, hidden=512, inter=256
    )
    captured = {}

    class _StopAfterKernelBuild(RuntimeError):
        pass

    def fake_builder(**kw):
        captured.update(kw)
        raise _StopAfterKernelBuild

    with (
        patch.object(layer_mod, "build_fused_ep_moe_kernel", side_effect=fake_builder),
        pytest.raises(_StopAfterKernelBuild),
    ):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            gate,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            ep_tp_size=2,
        )

    assert captured["g_local"] == 2
    assert captured["inter"] == 256
    assert captured["ep"] == 4
    assert captured["ep_tp_size"] == 2


@pytest.mark.parametrize(
    "mutate,exc,match",
    [
        (
            lambda a: (*a[:6], a[6], dict(ep_tp_size=0)),
            ValueError,
            "ep_tp_size must be >= 1",
        ),
        (
            lambda a: (
                a[0],
                a[1].astype(jnp.bfloat16),
                a[2].astype(jnp.bfloat16),
                None,
                None,
                a[5],
                a[6],
                dict(ep_tp_size=2, weight_format=WeightFormat.BF16),
            ),
            NotImplementedError,
            "FP8 weights only",
        ),
        (
            lambda a: (
                a[0],
                a[1][:7],
                a[2][:7],
                a[3][:7],
                a[4][:7],
                a[5],
                a[6],
                dict(ep_tp_size=2),
            ),
            ValueError,
            "not divisible by ep_tp_size",
        ),
        (
            lambda a: (
                a[0],
                a[1],
                a[2][:, :64, :],
                a[3],
                a[4],
                a[5],
                a[6],
                dict(ep_tp_size=2),
            ),
            ValueError,
            "w2 layout",
        ),
        (
            lambda a: (
                a[0],
                a[1],
                a[2],
                jnp.ones((a[1].shape[0], 512), dtype=jnp.float32),
                a[4],
                a[5],
                a[6],
                dict(ep_tp_size=2),
            ),
            ValueError,
            "w1_scale layout",
        ),
        (
            lambda _: (
                *_dummy_ep_tp_operands(ep=4, ep_tp_size=3, inter=384)[:7],
                dict(ep_tp_size=3),
            ),
            ValueError,
            "mesh size 4 is not divisible",
        ),
        (
            lambda a: (*a[:6], a[6], dict(ep_tp_size=2, mesh_ep_ranks=(0, 2, 1, 3))),
            ValueError,
            "rank-ordered EP mesh",
        ),
    ],
)
def test_ep_tp_validation_refuses_invalid_contracts(mutate, exc, match):
    base = _dummy_ep_tp_operands()
    x, w1, w2, w1s, w2s, gate, mesh, extra = mutate(base)
    with pytest.raises(exc, match=match):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            gate,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            **extra,
        )


@pytest.mark.multichip
@pytest.mark.parametrize(
    "wire_dtype", [None, jnp.bfloat16], ids=["fp8_wire", "bf16_wire"]
)
@pytest.mark.parametrize(
    "sparse_empty_experts", [False, True], ids=["dense_route", "empty_experts"]
)
def test_single_host_ep_tp2_matches_ep_tp1_reference(wire_dtype, sparse_empty_experts):
    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    ep = 8
    ep_tp = 2
    g_local = 2
    e_total = ep * g_local
    rows = 128
    hidden = 512
    inter = 256
    inter_s = inter // ep_tp
    mesh = Mesh(np.asarray(devices[:ep]), (AXIS,))

    rng = np.random.default_rng(17)
    x_np = (rng.standard_normal((ep * rows, hidden)) * 0.2).astype(np.float32)
    w1_raw = (
        rng.standard_normal((e_total, hidden, 2 * inter)) / np.sqrt(hidden)
    ).astype(np.float32)
    w2_raw = (rng.standard_normal((e_total, inter, hidden)) / np.sqrt(inter)).astype(
        np.float32
    )

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    w1s_np = np.maximum(np.max(np.abs(w1_raw), axis=1) / fp8_max, 1e-8).astype(
        np.float32
    )
    w1_q = np.asarray(
        jnp.asarray(w1_raw / w1s_np[:, None, :]).astype(jnp.float8_e4m3fn)
    )

    # Quantize each TP slice of w2 independently per output channel, matching
    # vLLM's TP-sharded FP8 MoE weight loader (`w2_scale` differs across m=0,1).
    w2_tp_q = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, inter_s, hidden), dtype=w1_q.dtype
    )
    w2s_tp = np.empty((ep // ep_tp, ep_tp, ep_tp * g_local, hidden), dtype=np.float32)
    w2_ref_deq = np.empty_like(w2_raw)
    for grp in range(ep // ep_tp):
        e0 = grp * ep_tp * g_local
        e1 = e0 + ep_tp * g_local
        for m in range(ep_tp):
            raw_slice = w2_raw[e0:e1, m * inter_s : (m + 1) * inter_s, :]
            s_slice = np.maximum(
                np.max(np.abs(raw_slice), axis=1) / fp8_max, 1e-8
            ).astype(np.float32)
            q_slice = np.asarray(
                jnp.asarray(raw_slice / s_slice[:, None, :]).astype(jnp.float8_e4m3fn)
            )
            w2_tp_q[grp, m] = q_slice
            w2s_tp[grp, m] = s_slice
            w2_ref_deq[e0:e1, m * inter_s : (m + 1) * inter_s, :] = (
                jnp.asarray(q_slice).astype(jnp.float32) * s_slice[:, None, :]
            )

    w1_tp_q = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, hidden, 2 * inter_s), dtype=w1_q.dtype
    )
    w1s_tp = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, 2 * inter_s), dtype=np.float32
    )
    for grp in range(ep // ep_tp):
        e0 = grp * ep_tp * g_local
        e1 = e0 + ep_tp * g_local
        g_half = w1_q[e0:e1, :, :inter]
        u_half = w1_q[e0:e1, :, inter:]
        gs_half = w1s_np[e0:e1, :inter]
        us_half = w1s_np[e0:e1, inter:]
        for m in range(ep_tp):
            c = slice(m * inter_s, (m + 1) * inter_s)
            w1_tp_q[grp, m] = np.concatenate([g_half[:, :, c], u_half[:, :, c]], axis=2)
            w1s_tp[grp, m] = np.concatenate([gs_half[:, c], us_half[:, c]], axis=1)

    gate_np = rng.standard_normal((ep * rows, e_total)).astype(np.float32)
    if sparse_empty_experts:
        gate_np[:, ::2] = -1e9

    sh = NamedSharding(mesh, P(AXIS))
    rank = jax.device_put(np.arange(ep, dtype=np.int32).reshape(ep, 1), sh)
    x_dev = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16), sh)
    gate_dev = jax.device_put(jnp.asarray(gate_np), sh)

    out_tp2 = jax.jit(
        lambda r, x_a, w1_a, w2_a, s1_a, s2_a, g_a: fused_ep_moe_v2(
            x_a,
            w1_a,
            w2_a,
            s1_a,
            s2_a,
            g_a,
            topk=4,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            rank=r,
            sharded_plan=True,
            scoring_fn="sigmoid",
            wire_dtype=wire_dtype,
            ep_tp_size=ep_tp,
        )
    )(
        rank,
        x_dev,
        jax.device_put(jnp.asarray(w1_tp_q.reshape(-1, hidden, 2 * inter_s)), sh),
        jax.device_put(jnp.asarray(w2_tp_q.reshape(-1, inter_s, hidden)), sh),
        jax.device_put(jnp.asarray(w1s_tp.reshape(-1, 2 * inter_s)), sh),
        jax.device_put(jnp.asarray(w2s_tp.reshape(-1, hidden)), sh),
        gate_dev,
    )
    got = np.asarray(jax.block_until_ready(out_tp2)).astype(np.float32)
    assert np.isfinite(got).all()

    # Compare against exact float32 reference using the identical quantized
    # per-TP-slice weights and scales.
    scores = 1.0 / (1.0 + np.exp(-gate_np))
    topk_idx = np.argpartition(-scores, kth=3, axis=-1)[:, :4]
    topk_w = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_w = topk_w / np.maximum(topk_w.sum(axis=-1, keepdims=True), 1e-20)
    x_bf16 = np.asarray(jnp.asarray(x_np, dtype=jnp.bfloat16).astype(jnp.float32))
    w1_deq = np.asarray(jnp.asarray(w1_q).astype(jnp.float32)) * w1s_np[:, None, :]
    ref = np.zeros_like(got)
    for t in range(ep * rows):
        for k_i in range(4):
            e_id = int(topk_idx[t, k_i])
            gu = x_bf16[t] @ w1_deq[e_id]
            gate_v, up_v = gu[:inter], gu[inter:]
            silu_g = gate_v / (1.0 + np.exp(-gate_v))
            act = np.asarray(
                jnp.asarray(silu_g * up_v, dtype=jnp.bfloat16).astype(jnp.float32)
            )
            down = act @ w2_ref_deq[e_id]
            ref[t] += topk_w[t, k_i] * down

    cos = float(
        np.dot(got.ravel(), ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref))
    )
    assert cos > 0.9985, cos


def test_should_select_peer_w2_slots_within_vmem_budget():
    fake_info = SimpleNamespace(
        vmem_capacity_bytes=64 * 1024 * 1024,
        num_lanes=128,
        generation=host_mod.MIN_GENERATION,
        get_sublane_tiling=lambda dtype: 32 // jnp.dtype(dtype).itemsize,
    )
    limit_small = host_mod.fit_weight_slots(
        16,
        128,
        5120,
        1536,
        max_distance=0,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    ).limit
    vmem_small_w2p_2 = host_mod.tp_pair_vmem_bytes(
        16,
        128,
        5120,
        1536,
        tp_w1_slots=2,
        tp_w1p_slots=2,
        tp_w2p_slots=2,
        tp_scl_slots=2,
        ep_tp_size=2,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    )
    assert vmem_small_w2p_2 <= limit_small

    limit_large = host_mod.fit_weight_slots(
        16,
        128,
        6144,
        2048,
        max_distance=0,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    ).limit
    vmem_large_w2p_1 = host_mod.tp_pair_vmem_bytes(
        16,
        128,
        6144,
        2048,
        tp_w1_slots=1,
        tp_w1p_slots=2,
        tp_w2p_slots=1,
        tp_scl_slots=2,
        ep_tp_size=2,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    )
    vmem_large_w2p_2 = host_mod.tp_pair_vmem_bytes(
        16,
        128,
        6144,
        2048,
        tp_w1_slots=1,
        tp_w1p_slots=2,
        tp_w2p_slots=2,
        tp_scl_slots=2,
        ep_tp_size=2,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    )
    assert vmem_large_w2p_1 <= limit_large < vmem_large_w2p_2


def test_should_rebalance_skewed_ep_tp_pairs_and_preserve_sentinels():
    ep, ep_tp_size, g_local = 4, 2, 4
    e_total = ep * g_local
    # Put all heavy load on Core 0 (local experts 0..3) of chip 0 (0..7).
    counts = np.array(
        [
            [50, 40, 30, 20, 1, 2, 3, 4, 10, 10, 10, 10, 10, 10, 10, 10],
            [50, 40, 30, 20, 1, 2, 3, 4, 10, 10, 10, 10, 10, 10, 10, 10],
        ],
        dtype=np.int32,
    )
    rows_by_dest = jnp.asarray(counts)
    block_hist = rows_by_dest
    pairs_g = jnp.asarray(
        list(range(e_total)) + [e_total, -1],
        dtype=jnp.int32,
    )
    local_blocks = pairs_g.reshape(1, -1)

    new_pairs, new_local, new_bh, new_rbd, slot_to_phys = (
        layer_mod.rebalance_ep_tp_pair_experts(
            pairs_g,
            local_blocks,
            block_hist,
            rows_by_dest,
            ep=ep,
            ep_tp_size=ep_tp_size,
            g_local=g_local,
            e_total=e_total,
        )
    )
    # Sentinels outside [0, e_total) must stay untouched.
    assert int(new_pairs[-2]) == e_total
    assert int(new_pairs[-1]) == -1
    assert np.array_equal(np.asarray(new_local).ravel(), np.asarray(new_pairs))
    # Every valid expert ID maps back to its exact physical expert on the same chip.
    slot_phys_np = np.asarray(slot_to_phys)
    for orig_id in range(e_total):
        remapped = int(new_pairs[orig_id])
        chip = orig_id // (2 * g_local)
        assert remapped // (2 * g_local) == chip
        assert int(slot_phys_np[chip, remapped % (2 * g_local)]) == orig_id % (
            2 * g_local
        )
    # Positive control: chip 0's heavy Core-0 experts split alternating across
    # Core 0 ([100, 4, 60, 8] = 172) and Core 1 ([2, 80, 6, 40] = 128), cutting
    # imbalance from 260 down to 44.
    chip0_counts = np.asarray(new_rbd).sum(axis=0)[: 2 * g_local].reshape(2, g_local)
    assert abs(int(chip0_counts[0].sum()) - int(chip0_counts[1].sum())) == 44
    assert np.array_equal(np.asarray(new_bh), np.asarray(new_rbd))


def test_should_retire_local_w1_and_w2_halves_in_expert_ffn_fp8():
    rng = np.random.default_rng(7)
    m, k, inter = 8, 32, 16
    i_slice = inter // 2
    act_q = jnp.asarray(rng.standard_normal((m, k)), dtype=jnp.float8_e4m3fn)
    act_s = jnp.ones((m, 1), dtype=jnp.float32) * 0.25
    w1_own = jnp.asarray(rng.standard_normal((k, 2 * i_slice)), dtype=jnp.float8_e4m3fn)
    w1_peer = jnp.asarray(
        rng.standard_normal((k, 2 * i_slice)), dtype=jnp.float8_e4m3fn
    )
    w2_own = jnp.asarray(rng.standard_normal((i_slice, k)), dtype=jnp.float8_e4m3fn)
    w2_peer = jnp.asarray(rng.standard_normal((i_slice, k)), dtype=jnp.float8_e4m3fn)
    w1s = jnp.ones((1, 2 * inter), dtype=jnp.float32) * 0.5
    w2s0 = jnp.ones((1, k), dtype=jnp.float32) * 0.3
    w2s1 = jnp.ones((1, k), dtype=jnp.float32) * 0.7

    events: list[str] = []
    acc_split, mid_s_split = kernel_mod.expert_ffn_fp8(
        act_q,
        act_s,
        (lambda: w1_own, lambda: w1_peer),
        (lambda: w2_own, lambda: w2_peer),
        w1s,
        retire_w1=lambda: events.append("retire_w1"),
        retire_w2=lambda: events.append("retire_w2"),
        tp_w2s=lambda idx: w2s0 if idx == 0 else w2s1,
        ep_tp_size=2,
    )
    acc_joined, mid_s_joined = kernel_mod.expert_ffn_fp8(
        act_q,
        act_s,
        (lambda: w1_own, lambda: w1_peer),
        jnp.concatenate([w2_own, w2_peer], axis=0),
        w1s,
        tp_w2s=lambda idx: w2s0 if idx == 0 else w2s1,
        ep_tp_size=2,
    )
    assert events == ["retire_w1", "retire_w2"]
    np.testing.assert_allclose(
        np.asarray(acc_split), np.asarray(acc_joined), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(mid_s_split), np.asarray(mid_s_joined), rtol=1e-5, atol=1e-5
    )


@pytest.mark.multichip
@pytest.mark.parametrize(
    "wire_dtype", [None, jnp.bfloat16], ids=["fp8_wire", "bf16_wire"]
)
def test_should_match_reference_under_skewed_routing_and_two_hop_commits(wire_dtype):
    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    ep = 8
    ep_tp = 2
    g_local = 4
    e_total = ep * g_local
    rows = 512
    hidden = 512
    inter = 256
    inter_s = inter // ep_tp
    mesh = Mesh(np.asarray(devices[:ep]), (AXIS,))

    rng = np.random.default_rng(91)
    x_np = (rng.standard_normal((ep * rows, hidden)) * 0.2).astype(np.float32)
    # Give every expert a distinct magnitude multiplier so any physical-weight
    # mismatch under intra-chip pair flipping breaks cosine parity immediately.
    expert_gain = np.linspace(0.5, 2.0, e_total, dtype=np.float32)[:, None, None]
    w1_raw = (
        rng.standard_normal((e_total, hidden, 2 * inter)) / np.sqrt(hidden)
    ).astype(np.float32) * expert_gain
    w2_raw = (rng.standard_normal((e_total, inter, hidden)) / np.sqrt(inter)).astype(
        np.float32
    )

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    w1s_np = np.maximum(np.max(np.abs(w1_raw), axis=1) / fp8_max, 1e-8).astype(
        np.float32
    )
    w1_q = np.asarray(
        jnp.asarray(w1_raw / w1s_np[:, None, :]).astype(jnp.float8_e4m3fn)
    )

    w2_tp_q = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, inter_s, hidden), dtype=w1_q.dtype
    )
    w2s_tp = np.empty((ep // ep_tp, ep_tp, ep_tp * g_local, hidden), dtype=np.float32)
    w2_ref_deq = np.empty_like(w2_raw)
    for grp in range(ep // ep_tp):
        e0 = grp * ep_tp * g_local
        e1 = e0 + ep_tp * g_local
        for m in range(ep_tp):
            raw_slice = w2_raw[e0:e1, m * inter_s : (m + 1) * inter_s, :]
            s_slice = np.maximum(
                np.max(np.abs(raw_slice), axis=1) / fp8_max, 1e-8
            ).astype(np.float32)
            q_slice = np.asarray(
                jnp.asarray(raw_slice / s_slice[:, None, :]).astype(jnp.float8_e4m3fn)
            )
            w2_tp_q[grp, m] = q_slice
            w2s_tp[grp, m] = s_slice
            w2_ref_deq[e0:e1, m * inter_s : (m + 1) * inter_s, :] = (
                jnp.asarray(q_slice).astype(jnp.float32) * s_slice[:, None, :]
            )

    w1_tp_q = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, hidden, 2 * inter_s), dtype=w1_q.dtype
    )
    w1s_tp = np.empty(
        (ep // ep_tp, ep_tp, ep_tp * g_local, 2 * inter_s), dtype=np.float32
    )
    for grp in range(ep // ep_tp):
        e0 = grp * ep_tp * g_local
        e1 = e0 + ep_tp * g_local
        g_half = w1_q[e0:e1, :, :inter]
        u_half = w1_q[e0:e1, :, inter:]
        gs_half = w1s_np[e0:e1, :inter]
        us_half = w1s_np[e0:e1, inter:]
        for m in range(ep_tp):
            c = slice(m * inter_s, (m + 1) * inter_s)
            w1_tp_q[grp, m] = np.concatenate([g_half[:, :, c], u_half[:, :, c]], axis=2)
            w1s_tp[grp, m] = np.concatenate([gs_half[:, c], us_half[:, c]], axis=1)

    # Skew all hot experts onto Core 0 of every chip so rebalancing flips half.
    gate_np = rng.standard_normal((ep * rows, e_total)).astype(np.float32) * 0.25
    for grp in range(ep // ep_tp):
        e0 = grp * ep_tp * g_local
        gate_np[:, e0 : e0 + g_local] += 2.5

    sh = NamedSharding(mesh, P(AXIS))
    rank = jax.device_put(np.arange(ep, dtype=np.int32).reshape(ep, 1), sh)
    x_dev = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16), sh)
    gate_dev = jax.device_put(jnp.asarray(gate_np), sh)

    out_tp2 = jax.jit(
        lambda r, x_a, w1_a, w2_a, s1_a, s2_a, g_a: fused_ep_moe_v2(
            x_a,
            w1_a,
            w2_a,
            s1_a,
            s2_a,
            g_a,
            topk=4,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            rank=r,
            sharded_plan=True,
            scoring_fn="sigmoid",
            wire_dtype=wire_dtype,
            ep_tp_size=ep_tp,
        )
    )(
        rank,
        x_dev,
        jax.device_put(jnp.asarray(w1_tp_q.reshape(-1, hidden, 2 * inter_s)), sh),
        jax.device_put(jnp.asarray(w2_tp_q.reshape(-1, inter_s, hidden)), sh),
        jax.device_put(jnp.asarray(w1s_tp.reshape(-1, 2 * inter_s)), sh),
        jax.device_put(jnp.asarray(w2s_tp.reshape(-1, hidden)), sh),
        gate_dev,
    )
    got = np.asarray(jax.block_until_ready(out_tp2)).astype(np.float32)
    assert np.isfinite(got).all()

    scores = 1.0 / (1.0 + np.exp(-gate_np))
    topk_idx = np.argpartition(-scores, kth=3, axis=-1)[:, :4]
    topk_w = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_w = topk_w / np.maximum(topk_w.sum(axis=-1, keepdims=True), 1e-20)
    x_bf16 = np.asarray(jnp.asarray(x_np, dtype=jnp.bfloat16).astype(jnp.float32))
    w1_deq = np.asarray(jnp.asarray(w1_q).astype(jnp.float32)) * w1s_np[:, None, :]
    ref = np.zeros_like(got)
    for t in range(ep * rows):
        for k_i in range(4):
            e_id = int(topk_idx[t, k_i])
            gu = x_bf16[t] @ w1_deq[e_id]
            gate_v, up_v = gu[:inter], gu[inter:]
            silu_g = gate_v / (1.0 + np.exp(-gate_v))
            act = np.asarray(
                jnp.asarray(silu_g * up_v, dtype=jnp.bfloat16).astype(jnp.float32)
            )
            down = act @ w2_ref_deq[e_id]
            ref[t] += topk_w[t, k_i] * down

    cos = float(
        np.dot(got.ravel(), ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref))
    )
    assert cos > 0.9985, cos


def test_should_split_skewed_ep_tp_experts_and_keep_balanced_noop():
    # n_chips * g_local = 64 exercises the multi-word (n_words=2) VPU bitmask.
    ep, ep_tp_size, g_local, k_split, topk, t_local, capacity = 8, 2, 16, 2, 4, 128, 128
    n_tokens = ep * t_local
    e_total = ep * g_local
    g_plan = g_local + k_split
    e_plan = ep * g_plan
    block = host_mod.routing_block(t_local, topk)
    stride = host_mod.ragged_stride_bound(n_tokens, topk, e_plan, capacity)
    blocks_per_dest = (t_local * topk) // block

    rng = np.random.default_rng(19)
    pairs = rng.integers(0, e_total, size=(n_tokens, topk), dtype=np.int32)
    # Chip 0: single dominant expert (~970 rows = 8 tiles) -> splitting halves it.
    pairs[: int(n_tokens * 0.95), 0] = 1
    # Chip 1: two equal heavy experts on opposite cores -> skip splitting.
    pairs[: int(n_tokens * 0.45), 1] = 2 * g_local
    pairs[int(n_tokens * 0.45) : int(n_tokens * 0.90), 1] = 2 * g_local + g_local
    pairs[-2:, :] = e_total
    pairs_2d = jnp.asarray(pairs.reshape(ep, t_local * topk))
    pairs_flat = pairs_2d.reshape(-1)

    all_bh = []
    rbd_rows = []
    for r in range(ep):
        lb = pairs_2d[r].reshape(blocks_per_dest, block)
        bh = host_mod.pair_block_hist(lb, e_total)
        all_bh.append(bh)
        rbd_rows.append(bh.sum(axis=0))
    rows_by_dest = jnp.stack(rbd_rows, axis=0)

    ref_pairs = ref_rbd = ref_phys = None
    for r in (0, 1, ep - 1):
        lb = pairs_2d[r].reshape(blocks_per_dest, block)
        new_p, new_l, new_bh, new_rbd, slot_phys = (
            layer_mod.rebalance_ep_tp_pair_experts(
                pairs_2d,
                lb,
                all_bh[r],
                rows_by_dest,
                ep=ep,
                ep_tp_size=ep_tp_size,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=capacity,
                me=jnp.int32(r),
                t_local=t_local,
                topk=topk,
            )
        )
        assert new_p.shape == pairs_2d.shape
        if ref_pairs is None:
            ref_pairs, ref_rbd, ref_phys = new_p, new_rbd, slot_phys
        else:
            assert np.array_equal(np.asarray(new_p), np.asarray(ref_pairs))
            assert np.array_equal(np.asarray(new_rbd), np.asarray(ref_rbd))
            assert np.array_equal(np.asarray(slot_phys), np.asarray(ref_phys))
        expected_bh = host_mod.pair_block_hist(new_l, e_plan)
        assert np.array_equal(np.asarray(new_bh), np.asarray(expected_bh))
        assert np.array_equal(np.asarray(new_bh.sum(axis=0)), np.asarray(new_rbd[r]))

        gathered = host_mod.GatheredPairs(
            expert_blocks=new_p.reshape(n_tokens * topk // block, block),
            local_blocks=new_l,
            block_hist=new_bh,
            rows_by_dest=new_rbd,
            n_tokens=n_tokens,
            topk=topk,
        )
        rt_sh = host_mod.build_routing_tables_sharded(
            None,
            jnp.int32(r),
            gathered=gathered,
            e_total=e_plan,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=stride,
        )
        rt_full = host_mod.build_routing_tables(
            new_p.reshape(n_tokens, topk),
            e_total=e_plan,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=stride,
        )
        assert np.array_equal(
            np.asarray(rt_sh.pos),
            np.asarray(rt_full.arrival_row[r * t_local : (r + 1) * t_local]),
        )
        assert np.array_equal(
            np.asarray(rt_sh.mirror_pos),
            np.asarray(rt_full.mirror_row[r * t_local : (r + 1) * t_local]),
        )

    assert ref_pairs is not None and ref_rbd is not None and ref_phys is not None
    orig_np = np.asarray(pairs_flat)
    new_np = np.asarray(ref_pairs).reshape(-1)
    phys_np = np.asarray(ref_phys)
    for orig_id, remapped in zip(orig_np, new_np):
        o_i, n_i = int(orig_id), int(remapped)
        if o_i == e_total:
            assert n_i == e_plan
        else:
            chip = o_i // (2 * g_local)
            assert n_i // (2 * g_plan) == chip
            assert int(phys_np[chip, n_i % (2 * g_plan)]) == o_i % (2 * g_local)

    chip0_counts = np.asarray(ref_rbd).sum(axis=0)[: 2 * g_plan].reshape(2, g_plan)
    assert int(chip0_counts[0, g_local]) > 400
    assert int(chip0_counts[1, g_local]) > 400
    chip1_counts = (
        np.asarray(ref_rbd).sum(axis=0)[2 * g_plan : 4 * g_plan].reshape(2, g_plan)
    )
    assert int(chip1_counts[:, g_local:].sum()) == 0

    bal_pairs = jnp.asarray(np.arange(n_tokens * topk, dtype=np.int32) % e_total)
    lb0 = bal_pairs[: t_local * topk].reshape(blocks_per_dest, block)
    bh0 = host_mod.pair_block_hist(lb0, e_total)
    rbd_bal = jnp.stack(
        [
            host_mod.pair_block_hist(
                bal_pairs[r * t_local * topk : (r + 1) * t_local * topk].reshape(
                    blocks_per_dest, block
                ),
                e_total,
            ).sum(axis=0)
            for r in range(ep)
        ],
        axis=0,
    )
    _, _, _, rbd_bal_new, _ = layer_mod.rebalance_ep_tp_pair_experts(
        bal_pairs,
        lb0,
        bh0,
        rbd_bal,
        ep=ep,
        ep_tp_size=ep_tp_size,
        g_local=g_local,
        e_total=e_total,
        k_split=k_split,
        capacity=capacity,
        me=jnp.int32(0),
        t_local=t_local,
        topk=topk,
    )
    virtual_counts = (
        np.asarray(rbd_bal_new).sum(axis=0).reshape(ep, g_plan)[:, g_local:]
    )
    assert int(virtual_counts.sum()) == 0


def test_should_expand_virtual_slots_for_large_ep_tp_and_honor_explicit_stride():
    x, w1, w2, w1s, w2s, gate, mesh = _dummy_ep_tp_operands(
        ep=4, ep_tp_size=2, g_local=2, rows=64, hidden=512, inter=256
    )

    class _StopAfterBuild(RuntimeError):
        pass

    captured = {}

    def _capture(**kw):
        captured.clear()
        captured.update(kw)
        raise _StopAfterBuild

    with (
        patch.object(layer_mod, "build_fused_ep_moe_kernel", side_effect=_capture),
        pytest.raises(_StopAfterBuild),
    ):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            gate,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            ep_tp_size=2,
        )
    assert captured["g_local"] == 4

    # When an explicit ragged_stride is smaller than the split-slot bound,
    # virtual row-split slots must stay off instead of tripping stride validation.
    x_lg, w1_lg, w2_lg, w1s_lg, w2s_lg, gate_lg, mesh_lg = _dummy_ep_tp_operands(
        ep=8, ep_tp_size=2, g_local=16, rows=64, hidden=512, inter=256
    )
    base_stride = host_mod.ragged_stride_bound(8 * 64, 4, 8 * 16, 128)
    split_stride = host_mod.ragged_stride_bound(8 * 64, 4, 8 * 20, 128)
    assert base_stride < split_stride
    with (
        patch.object(layer_mod, "build_fused_ep_moe_kernel", side_effect=_capture),
        pytest.raises(_StopAfterBuild),
    ):
        fused_ep_moe_v2(
            x_lg,
            w1_lg,
            w2_lg,
            w1s_lg,
            w2s_lg,
            gate_lg,
            topk=4,
            renormalize=True,
            mesh=mesh_lg,
            capacity=128,
            ragged_stride=base_stride,
            ep_tp_size=2,
        )
    assert captured["g_local"] == 16

    with (
        patch.object(layer_mod, "build_fused_ep_moe_kernel", side_effect=_capture),
        pytest.raises(_StopAfterBuild),
    ):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            gate,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            ep_tp_size=2,
            ep_tp_row_splits=0,
        )
    assert captured["g_local"] == 2


def test_should_trace_sharded_ep_tp_row_split_end_to_end():
    ep, ep_tp_size, g_local, rows, hidden, inter, topk, cap = (
        4,
        2,
        4,
        128,
        512,
        256,
        4,
        128,
    )
    mesh = jax.sharding.AbstractMesh((ep,), ("ep",))
    sh = jax.sharding.NamedSharding(mesh, P("ep"))
    x = jax.ShapeDtypeStruct((ep * rows, hidden), jnp.bfloat16, sharding=sh)
    w1 = jax.ShapeDtypeStruct(
        (ep * ep_tp_size * g_local, hidden, inter), jnp.float8_e4m3fn, sharding=sh
    )
    w2 = jax.ShapeDtypeStruct(
        (ep * ep_tp_size * g_local, inter // ep_tp_size, hidden),
        jnp.float8_e4m3fn,
        sharding=sh,
    )
    w1s = jax.ShapeDtypeStruct(
        (ep * ep_tp_size * g_local, inter), jnp.float32, sharding=sh
    )
    w2s = jax.ShapeDtypeStruct(
        (ep * ep_tp_size * g_local, hidden), jnp.float32, sharding=sh
    )
    gate = jax.ShapeDtypeStruct((ep * rows, ep * g_local), jnp.float32, sharding=sh)
    rank = jax.ShapeDtypeStruct((ep, 1), jnp.int32, sharding=sh)
    captured_visit_shape: list[tuple[int, ...]] = []

    def _stub_kernel(**kw):
        # Hardware-only Pallas boundary stub so shard_map(local_fn) traces
        # _unpack_routing_exchange -> rebalance -> routing -> combine.
        g_plan = kw["g_local"]

        def _call(*args, recv_rows, visit, counts, rank):
            del args, counts, rank
            captured_visit_shape.append(tuple(visit.shape))
            n_recv_scl = recv_rows // cap + 2 * ep * g_plan
            return (
                jnp.zeros((recv_rows, hidden // 128, 128), dtype=jnp.float8_e4m3fn),
                jnp.zeros((n_recv_scl, 128), dtype=jnp.float32),
            )

        return _call

    def _stub_combine(**kw):
        del kw
        return lambda *args: jnp.zeros((rows, hidden), dtype=jnp.bfloat16)

    with (
        patch.object(layer_mod, "build_fused_ep_moe_kernel", side_effect=_stub_kernel),
        patch.object(layer_mod, "build_combine_kernel", side_effect=_stub_combine),
    ):
        out = jax.eval_shape(
            lambda x_a, w1_a, w2_a, w1s_a, w2s_a, g_a, r_a: fused_ep_moe_v2(
                x_a,
                w1_a,
                w2_a,
                w1s_a,
                w2s_a,
                g_a,
                topk=topk,
                renormalize=True,
                mesh=mesh,
                capacity=cap,
                rank=r_a,
                sharded_plan=True,
                scoring_fn="sigmoid",
                ep_tp_size=ep_tp_size,
                ep_tp_row_splits=2,
            ),
            x,
            w1,
            w2,
            w1s,
            w2s,
            gate,
            rank,
        )
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # visit has ep_tp_size + 1 + ep_tp_size = 5 rows and g_local + 2 = 6 cols.
    assert captured_visit_shape == [(5, 6)]


def test_should_resolve_tp_down_chunk_and_fit_ep_tp2_peer_w2_slots_in_vmem():
    assert host_mod.resolve_tp_down_chunk(6144) == 2048
    assert host_mod.resolve_tp_down_chunk(3072) == 1024
    assert host_mod.resolve_tp_down_chunk(2048) == 1024
    assert host_mod.resolve_tp_down_chunk(512) == 256
    assert host_mod.resolve_tp_down_chunk(384) is None
    assert host_mod.resolve_tp_down_chunk(128) is None
    assert host_mod.resolve_tp_down_chunk(6144, ep_tp_size=1) is None
    assert host_mod.resolve_tp_down_chunk(6144, weight_format=WeightFormat.BF16) is None
    assert host_mod.resolve_tp_down_chunk(6144, wire_dtype=jnp.bfloat16) is None
    assert host_mod.resolve_tp_down_chunk(6144, has_w1_bias=True) is None
    assert host_mod.resolve_tp_down_chunk(6144, has_w2_bias=True) is None

    fake_info = SimpleNamespace(
        vmem_capacity_bytes=64 * 1024 * 1024,
        num_lanes=128,
        generation=host_mod.MIN_GENERATION,
        get_sublane_tiling=lambda dtype: 32 // jnp.dtype(dtype).itemsize,
    )
    limit_large = host_mod.fit_weight_slots(
        16,
        128,
        6144,
        2048,
        max_distance=0,
        weight_format=WeightFormat.FP8,
        info=fake_info,
    ).limit
    vmem_chunked_w1_1_w2p_2 = host_mod.tp_pair_vmem_bytes(
        16,
        128,
        6144,
        2048,
        tp_w1_slots=1,
        tp_w1p_slots=2,
        tp_w2p_slots=2,
        tp_scl_slots=2,
        ep_tp_size=2,
        weight_format=WeightFormat.FP8,
        info=fake_info,
        down_chunk=2048,
    )
    vmem_chunked_w1_2_w2p_2 = host_mod.tp_pair_vmem_bytes(
        16,
        128,
        6144,
        2048,
        tp_w1_slots=2,
        tp_w1p_slots=2,
        tp_w2p_slots=2,
        tp_scl_slots=2,
        ep_tp_size=2,
        weight_format=WeightFormat.FP8,
        info=fake_info,
        down_chunk=2048,
    )
    assert vmem_chunked_w1_1_w2p_2 <= limit_large < vmem_chunked_w1_2_w2p_2

    vmem_bf16_wire = host_mod.tp_pair_vmem_bytes(
        4,
        128,
        1024,
        512,
        weight_format=WeightFormat.FP8,
        wire_dtype=jnp.bfloat16,
        info=fake_info,
    )
    vmem_fp4 = host_mod.tp_pair_vmem_bytes(
        4,
        128,
        1024,
        512,
        weight_format=WeightFormat.FP4,
        info=fake_info,
    )
    vmem_int8 = host_mod.tp_pair_vmem_bytes(
        4,
        128,
        1024,
        512,
        weight_format=WeightFormat.INT8,
        info=fake_info,
    )
    assert vmem_bf16_wire > 0
    assert vmem_fp4 > vmem_int8 > 0


def test_should_match_unchunked_fp8_wire_and_retire_w2_only_after_final_chunk():
    rng = np.random.default_rng(29)
    m, hidden, inter, down_chunk = 16, 512, 256, 128
    i_slice = inter // 2
    act_q = jnp.asarray(rng.standard_normal((m, hidden)), dtype=jnp.float8_e4m3fn)
    act_s = jnp.asarray(
        0.15 + 0.4 * rng.random((m, 1), dtype=np.float32), dtype=jnp.float32
    )
    w1_own = jnp.asarray(
        rng.standard_normal((hidden, 2 * i_slice)), dtype=jnp.float8_e4m3fn
    )
    w1_peer = jnp.asarray(
        rng.standard_normal((hidden, 2 * i_slice)), dtype=jnp.float8_e4m3fn
    )
    w2_own = jnp.asarray(
        rng.standard_normal((i_slice, hidden)), dtype=jnp.float8_e4m3fn
    )
    w2_peer = jnp.asarray(
        rng.standard_normal((i_slice, hidden)), dtype=jnp.float8_e4m3fn
    )
    w1s = jnp.asarray(
        0.2 + 0.5 * rng.random((1, 2 * inter), dtype=np.float32), dtype=jnp.float32
    )
    w2s0 = jnp.asarray(
        0.1 + 0.6 * rng.random((1, hidden), dtype=np.float32), dtype=jnp.float32
    )
    w2s1 = jnp.asarray(
        0.3 + 0.8 * rng.random((1, hidden), dtype=np.float32), dtype=jnp.float32
    )

    events: list[str] = []

    def own_slice(c0: int, cw: int):
        events.append(f"own:{c0}")
        return w2_own[:, c0 : c0 + cw]

    def peer_slice(c0: int, cw: int):
        events.append(f"peer:{c0}")
        return w2_peer[:, c0 : c0 + cw]

    def w2s_slice(member: int, c0: int, cw: int):
        table = w2s0 if member == 0 else w2s1
        return table[:, c0 : c0 + cw]

    got_rows, got_scales = kernel_mod.expert_ffn_fp8_tp_chunked_wire(
        act_q,
        act_s,
        (lambda: w1_own, lambda: w1_peer),
        own_slice,
        peer_slice,
        w1s,
        w2s_slice,
        hidden=hidden,
        down_chunk=down_chunk,
        wait_w1=lambda: events.append("wait_w1"),
        retire_w1=lambda: events.append("retire_w1"),
        wait_w2=lambda: events.append("wait_w2"),
        retire_w2=lambda: events.append("retire_w2"),
        ep_tp_size=2,
    )

    ref_acc2, ref_mid_s = kernel_mod.expert_ffn_fp8(
        act_q,
        act_s,
        (lambda: w1_own, lambda: w1_peer),
        (lambda: w2_own, lambda: w2_peer),
        w1s,
        tp_w2s=lambda idx: w2s0 if idx == 0 else w2s1,
        ep_tp_size=2,
    )
    ref_rows, ref_scales = kernel_mod.wire_row_and_scale(
        ref_acc2,
        ref_mid_s,
        w2s=None,
        w2b=None,
        wire_dtype=jnp.float8_e4m3fn,
        tile_m=m,
    )

    assert events == [
        "wait_w1",
        "retire_w1",
        "wait_w2",
        "own:0",
        "peer:0",
        "own:128",
        "peer:128",
        "own:256",
        "peer:256",
        "own:384",
        "retire_w2",
        "peer:384",
    ]
    np.testing.assert_array_equal(
        np.asarray(got_rows).view(np.uint8),
        np.asarray(ref_rows).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(got_scales).view(np.uint32),
        np.asarray(ref_scales).view(np.uint32),
    )


@pytest.mark.parametrize(
    ("z_clone", "wire_dtype", "expect_w2p2"),
    [
        (False, None, True),
        (True, None, True),
        (False, jnp.bfloat16, False),
    ],
    ids=["fp8_wire_no_zclone", "fp8_wire_zclone", "bf16_wire_fallback"],
)
def test_should_export_ep_tp2_w2p_schedule_and_bf16_wire_fallback_to_tpu7x_mlir(
    z_clone: bool,
    wire_dtype,
    expect_w2p2: bool,
):
    fake_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    mesh = AbstractMesh((16,), ("ep",))
    sh = NamedSharding(mesh, P("ep"))

    def trace_fn(x, w1, w2, w1s, w2s, g, r):
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            g,
            topk=8,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP8,
            wire_dtype=wire_dtype,
            rank=r,
            sharded_plan=True,
            scoring_fn="sigmoid",
            ep_tp_size=2,
            ep_tp_row_splits=4,
            ep_tp_z_clone=z_clone,
            ep_tp_z_clone_delay=2,
        )

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    with (
        patch.object(pltpu, "get_tpu_info", return_value=fake_info),
        patch.object(tpu_info, "get_tpu_info", return_value=fake_info),
    ):
        exported = jax.export.export(jax.jit(trace_fn), platforms=("tpu",))(
            sds((16384, 6144), jnp.bfloat16),
            sds((512, 6144, 2048), jnp.float8_e4m3fn),
            sds((512, 1024, 6144), jnp.float8_e4m3fn),
            sds((512, 2048), jnp.float32),
            sds((512, 6144), jnp.float32),
            sds((16384, 256), jnp.float32),
            sds((16, 1), jnp.int32),
        )
    mlir = exported.mlir_module()
    assert "partition_id" not in mlir
    assert ("_w2p2" in mlir) is expect_w2p2
    assert ("_zclone" in mlir) is z_clone
    assert ("_bf16wire" in mlir) is (wire_dtype is not None)
