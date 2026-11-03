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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
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
