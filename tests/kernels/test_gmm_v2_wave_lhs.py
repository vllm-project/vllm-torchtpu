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
"""The scattering expert kernel on rows that land while it runs
(`gmm_v2_mlp_scatter(wave_lhs=...)`): bitwise the kernel on `gather_rows`'
rows.

A SparseCore kernel copies each route's fp8 row words to its route-order row
of the receive buffer, one 8 KB row per DMA, expert after expert, each
signalling its expert's (wave's) DMA semaphore in the TensorCore's memory,
and exits with its copies in flight (a second SC kernel waits for them), as
the SC dispatch does; see tests/kernels/test_sc_wave_dispatch.py on branch
gxd/q38-sc-wave-dispatch-test for the pattern and the flags.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2_mlp_scatter
from vllm_torchtpu.kernels.megablox.row_copy import gather_rows
from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor
from vllm_torchtpu.layers.core.ragged_a2a_moe import DP_AXIS, TP_AXIS

N_DEV = 8
HIDDEN = 4096
INTER = 512
TOKENS = 48
# Rows per local expert (rolled per member): 0 rows, more than one 64-row
# tile, and tiles in each of the 16 / 32 / 64-row buckets.
SIZES = (10, 0, 30, 70, 50, 3, 16, 17)
NUM_ROWS = 224  # past the routed rows (196) and their 16-row block
BUCKETS = (16, 32)
FLAGS = dict(
    xla_mosaic_unsafe_allow_multicore_remote_dma=True,
    xla_msa_enable=False,
    xla_tpu_vmem_scavenging_mode="NONE",
)


def require_tpu_sc():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < N_DEV:
        pytest.skip(f"needs {N_DEV} TPU devices")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs SparseCore")
    return devices[:N_DEV]


def _chunk(limit):
    b = 1
    while b * 2 <= min(64, limit):
        b *= 2
    return b


def _wait_count(n, wait_rows, big):
    """Waits n (dynamic) rows with static power-of-two waits wait_rows(b)."""

    @pl.loop(0, n // big)
    def _(i):
        wait_rows(big)

    b = big // 2
    while b:

        @pl.when((n & b) != 0)
        def _(b=b):
            wait_rows(b)

        b //= 2


def sc_land_rows(rows, sizes, tc_mesh, axes):
    """SC kernel: copies `rows` [P, 8, H / 32] row by row to the same rows of
    a new buffer on this device, lhs group after lhs group (``sizes``), row
    `j` of group `e` signalling TC DMA semaphore `e`; exits with the copies
    in flight. Returns (buffer, semaphores, send semaphore, rows sent)."""
    sc_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
    groups = sizes.shape[0]

    def body(rows_hbm, sizes_hbm, buf, sems, send_sems, sizes_s, dma_sem):
        pltpu.async_copy(sizes_hbm, sizes_s, dma_sem).wait()
        me = {a: lax.axis_index(a) for a in axes}

        def wave(e, start):
            @pl.loop(0, sizes_s[e])
            def _(r):
                pltpu.async_remote_copy(
                    rows_hbm.at[start + r],
                    buf.at[start + r],
                    send_sems.at[0],
                    sems.at[e],
                    device_id={**me, "tc": 0},
                )

            return start + sizes_s[e]

        lax.fori_loop(0, groups, wave, jnp.int32(0))

    buf, sems, send_sem = pl.kernel(
        body,
        out_type=(
            pltpu.HBM(rows.shape, rows.dtype),
            pltpu.SemaphoreType.DMA((groups,)) @ tc_mesh,
            pltpu.SemaphoreType.DMA((1,)) @ sc_mesh,
        ),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM(sizes.shape, jnp.int32), pltpu.SemaphoreType.DMA(())],
        name="sc_land_rows",
    )(rows, sizes)
    return buf, sems, send_sem


def sc_drain(send_sem, rows, sizes):
    """SC kernel: waits `send_sem` for the rows `sc_land_rows` sent."""
    sc_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
    sem_ref = jax.new_ref(send_sem, memory_space=pltpu.SEMAPHORE @ sc_mesh)
    total = jnp.sum(sizes, dtype=jnp.int32)[None]

    def body(sem_r, rows_hbm, total_hbm, total_s, dma_sem):
        pltpu.async_copy(total_hbm, total_s, dma_sem).wait()

        def wait_sent(b):
            part = rows_hbm.at[pl.ds(0, b)]
            pltpu.make_async_remote_copy(
                part, part, sem_r.at[0], sem_r.at[0], device_id={DP_AXIS: 0, TP_AXIS: 0}
            ).wait_send()

        _wait_count(total_s[0], wait_sent, _chunk(rows.shape[0]))

    pl.kernel(
        body,
        out_type=(),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM((1,), jnp.int32), pltpu.SemaphoreType.DMA(())],
        name="sc_land_drain",
    )(sem_ref, rows, total)


def fp8_words(rng, num, hidden):
    """Random fp8 row words (every bit pattern: zeros, subnormals, NaNs)."""
    return rng.integers(0, 2**32, size=(num, 8, hidden // 32), dtype=np.uint32)


def expert_weights(key, groups, hidden, inter):
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (groups, hidden, 2 * inter)) / 8
    w2 = jax.random.normal(k2, (groups, inter, hidden)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    return w1, w2, s1, s2


def _plan(layout, rng):
    """Per member: group sizes, route tokens, destinations, receive rows and
    (with the owner tail) its [K, blk] tail arrays."""
    n = N_DEV
    sizes = np.stack([np.roll(SIZES, m) for m in range(n)]).astype(np.int32)
    routed = int(sizes[0].sum())
    token = np.zeros((n, NUM_ROWS), np.int32)
    token[:, :routed] = rng.integers(0, TOKENS, size=(n, routed))
    members = 1 if layout == "single" else n
    dest = np.zeros((n, NUM_ROWS), np.int32)
    if members > 1:
        dest[:, :routed] = rng.integers(0, n, size=(n, routed))
    topk, blk = 10, 32
    num_recv = topk * blk if layout == "tail" else n * NUM_ROWS
    slot = np.zeros((n, NUM_ROWS), np.int32)
    expected = np.zeros((n, 1), np.int32)
    tail = None
    if layout == "tail":
        tp_rank = np.full((n, topk, blk), 4, np.int32)
        dp_pos = np.zeros((n, topk, blk), np.int32)
        sent = np.zeros((n, topk, blk), np.int32)
        weight = np.zeros((n, topk, blk), np.float32)
    for d in range(n):
        if members == 1:
            senders = [(d, j) for j in range(routed)]
        else:
            senders = [
                (m, j) for m in range(n) for j in range(routed) if dest[m, j] == d
            ]
        perm = rng.permutation(num_recv)[: len(senders)]
        for (m, j), r in zip(senders, perm):
            slot[m, j] = r
            if layout == "tail":
                tp_rank[d].flat[r] = m % 4
                dp_pos[d].flat[r] = m // 4
                sent[d].flat[r] = 1
                weight[d].flat[r] = rng.random()
        expected[d] = sum(1 for m, _ in senders if m != d)
    if layout == "tail":
        tail = (tp_rank, dp_pos, sent, weight, np.ones((n, 1), np.int32))
    return sizes, token, dest, slot, expected, num_recv, tail, routed


@pytest.mark.multichip
@pytest.mark.parametrize("layout", ["single", "members", "tail"])
def test_wave_lhs_matches_gathered_rows(layout):
    """Every output row bitwise the gathered-rows kernel's, over three back
    to back calls (a wave semaphore left unbalanced corrupts or hangs the
    next one): one member, eight members, and (2 DP x 4 TP) members with the
    owner tail."""
    devices = require_tpu_sc()
    n = N_DEV
    if layout == "tail":
        mesh = Mesh(np.asarray(devices).reshape(2, 4), (DP_AXIS, TP_AXIS))
        axis, num_members = (DP_AXIS, TP_AXIS), (2, 4)
    else:
        mesh = Mesh(np.asarray(devices).reshape(n, 1), (DP_AXIS, TP_AXIS))
        axis, num_members = (TP_AXIS, 1) if layout == "single" else (DP_AXIS, n)
    rng = np.random.default_rng(["single", "members", "tail"].index(layout))
    sizes, token, dest, slot, expected, num_recv, tail, routed = _plan(layout, rng)
    groups = len(SIZES)
    w1, w2, s1, s2 = expert_weights(jax.random.key(5), groups, HIDDEN, INTER)
    words = np.stack([fp8_words(rng, TOKENS, HIDDEN) for _ in range(n)])
    scale = rng.random((n, TOKENS)).astype(np.float32) * 2.0 ** rng.integers(
        -20, 4, size=(n, TOKENS)
    )
    scale[:, 0] = 0.0
    tc_mesh = pltpu.create_tensorcore_mesh("tc", num_cores=1)
    calls = 3

    def body(words, scale, sizes, token, dest, slot, me, expected, *tail):
        words, scale, sizes, token, dest, slot, me, expected = (
            a.reshape(a.shape[2:])
            for a in (words, scale, sizes, token, dest, slot, me, expected)
        )
        tail = tuple(a.reshape(a.shape[2:]) for a in tail) or None
        kwargs = dict(
            num_recv_rows=num_recv,
            axis=axis,
            num_members=num_members,
            fuse_act="silu",
            buckets=BUCKETS,
            maybe_quantize_lhs=False,
        )

        def call(i, lhs, wave_lhs):
            return gmm_v2_mlp_scatter(
                lhs,
                w1,
                w2,
                sizes,
                s1,
                s2,
                jnp.zeros((1,), jnp.int32),
                dest,
                slot,
                me,
                expected,
                None,
                tail,
                wave_lhs,
                collective_id=i % 2,
                **kwargs,
            )

        def step(_, i):
            # Each call scales the rows differently.
            scale_i = scale * (1.0 + i.astype(jnp.float32) / 4)
            x = gather_rows(
                words, token, sizes, jnp.full((1,), TOKENS, jnp.int32), scale_i
            )
            base = call(0, x, None)
            rows = words[token]
            recv, sems, send_sem = sc_land_rows(
                rows, sizes, tc_mesh, (DP_AXIS, TP_AXIS)
            )
            sc_drain(send_sem, rows, sizes)
            got = call(1, None, (recv, scale_i[token], sems))
            return None, (base, got)

        _, (base, got) = lax.scan(step, None, jnp.arange(calls))
        return base[None, None], got[None, None]

    spec = P(DP_AXIS, TP_AXIS)
    tp = mesh.shape[TP_AXIS]

    def per_member(a):
        a = np.asarray(a)
        return jnp.asarray(a.reshape(n // tp, tp, *a.shape[1:]))

    args = [
        words,
        scale,
        sizes,
        token,
        dest,
        slot,
        np.arange(n, dtype=np.int32)[:, None] * (layout != "single"),
        expected,
    ]
    if tail is not None:
        args += list(tail)
    base, got = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * len(args),
            out_specs=(spec, spec),
            check_vma=False,
        ),
        compiler_options=FLAGS,
    )(*map(per_member, args))
    base = np.asarray(base).reshape(n, calls, -1, HIDDEN).view(np.uint16)
    got = np.asarray(got).reshape(n, calls, -1, HIDDEN).view(np.uint16)
    if tail is not None:
        assert np.abs(base.view(jnp.bfloat16).astype(np.float32)).max() > 0
        np.testing.assert_array_equal(got, base)
        return
    for m in range(n):
        for j in range(routed):
            d = m if layout == "single" else dest[m, j]
            np.testing.assert_array_equal(
                got[d, :, slot[m, j]], base[d, :, slot[m, j]], err_msg=f"{m}, {j}"
            )
