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
"""The run-copy kernel builds both slab tables as the scatters do.

Where the step is large enough, the layer builds the token-gather table and
the activation-scale slab in one pass: every shard sends its routed pairs
sorted by expert, each carrying its token's number and scale bits, and
`kernel.build_slab_run_copy` copies each (local expert, source) run of the
gathered payloads to its rows of the slab. The host half -- the sender sort
and the run offsets -- is checked against the scatter tables on the CPU in
`test_fused_ep_moe_v2_tables.py`; this runs the kernel on the chip, on the
same inputs, and requires the scatter's tables bit for bit.

The routing plan and the payloads are built on the host CPU; only the copy
runs on the chip.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel  # noqa: E402

pytestmark = pytest.mark.tpu_test

# (ep, t_local, g_local, topk): Qwen3.8's EP32 layout at a prefill and a
# decode step, and the 397B layout over eight shards.
GEOMETRIES = [
    pytest.param(32, 512, 16, 10, id="qwen38_prefill_t16384"),
    pytest.param(32, 48, 16, 10, id="qwen38_decode_t1536"),
    pytest.param(8, 256, 64, 10, id="ep8_g64_t2048"),
]


def _inputs(ep, t_local, g_local, topk, routing, seed):
    """The routing, its sharded plans' inputs and every source's payload."""
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if routing == "skewed":
        hot = rng.choice(e_total, size=max(4, e_total // 12), replace=False)
        idx = rng.choice(hot, size=(T, topk))
    else:
        idx = rng.integers(0, e_total, (T, topk))
    if routing == "padded":
        for d in range(ep):
            start = d * t_local if d == 1 else d * t_local + t_local // 2
            idx[start : (d + 1) * t_local, :] = e_total
    idx = idx.astype(np.int32)
    capacity = 128
    block = host.routing_block(t_local, topk)
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, capacity),
    )
    # One positive f32 per token, so a live row's bits are never zero.
    scale_bits = (rng.random(T, dtype=np.float32) + 0.5).view(np.int32)
    pair_w = host.align_up(t_local * topk, host.HIDDEN_LANE_BLOCK)
    tokens, scales = [], []
    with jax.default_device(jax.devices("cpu")[0]):
        for s in range(ep):
            rows = slice(s * t_local, (s + 1) * t_local)
            t_s, c_s = host.sender_sorted_pairs(
                jnp.asarray(idx[rows]),
                jnp.int32(s),
                t_local=t_local,
                topk=topk,
                scale_bits=jnp.asarray(scale_bits[rows]),
            )
            tokens.append(np.pad(np.asarray(t_s), (0, pair_w - t_s.size)))
            scales.append(np.pad(np.asarray(c_s), (0, pair_w - c_s.size)))
        counts = np.stack(
            [
                np.bincount(
                    idx[s * t_local : (s + 1) * t_local][
                        idx[s * t_local : (s + 1) * t_local] < e_total
                    ],
                    minlength=e_total,
                )
                for s in range(ep)
            ]
        ).astype(np.int32)
    return idx, kwargs, scale_bits, np.stack(tokens), np.stack(scales), pair_w, counts


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "padded"])
def test_run_copy_builds_the_scattered_tables(ep, t_local, g_local, topk, routing):
    idx, kwargs, scale_bits, tokens, scales, pair_w, counts = _inputs(
        ep, t_local, g_local, topk, routing, seed=9
    )
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    T = ep * t_local
    scale_rows = host.act_scale_slab_rows(stride)
    view_rows = scale_rows * host.HIDDEN_LANE_BLOCK
    out_rows = host.align_up(scale_rows, host.U32_SUBLANE_TILE)
    streams = np.stack([tokens.reshape(-1), scales.reshape(-1)])
    src_rows = host.align_up(
        streams.shape[1], host.HIDDEN_LANE_BLOCK * host.U32_SUBLANE_TILE
    )
    streams = np.pad(streams, ((0, 0), (0, src_rows - streams.shape[1])))
    src_rows //= host.HIDDEN_LANE_BLOCK
    copy = kernel.build_slab_run_copy(
        n_streams=2, src_rows=src_rows, out_rows=out_rows, n_runs=g_local * ep
    )
    lo, hi = host.gather_clamp_bounds(T)

    for me in sorted({0, 1, ep // 2, ep - 1}):
        with jax.default_device(jax.devices("cpu")[0]):
            plan = host.build_routing_tables_sharded(
                jnp.asarray(idx),
                jnp.int32(me),
                all_gather_rows=lambda _row: jnp.asarray(counts),
                **kwargs,
            )
            rows = host.local_slab_rows(plan, me, shard_stride=stride)
            want_tokens = np.asarray(
                host.shard_token_gather(plan, me, shard_stride=stride, rows=rows)
            )
            want_scales = np.asarray(
                host.slab_scatter(
                    view_rows,
                    rows,
                    jnp.asarray(np.repeat(scale_bits, topk)),
                    by_set=False,
                )
            )
            runs = [
                np.asarray(a)
                for a in host.slab_run_copy_runs(
                    plan, jnp.int32(me), e_total=e_total, ep=ep, src_stride=pair_w
                )
            ]
        got = np.asarray(
            copy(
                *(jnp.asarray(a) for a in runs),
                jnp.asarray(streams.reshape(2, src_rows, host.HIDDEN_LANE_BLOCK)),
            )
        ).reshape(2, -1)
        np.testing.assert_array_equal(
            np.clip(got[0, :stride], lo, hi),
            want_tokens,
            err_msg=f"token-gather table, shard {me}",
        )
        np.testing.assert_array_equal(
            got[1, :stride], want_scales[:stride], err_msg=f"scale slab, shard {me}"
        )
        # Nothing lands past the slab: the kernel's zeros stay.
        assert not got[:, stride:].any(), f"shard {me} wrote past the slab"
