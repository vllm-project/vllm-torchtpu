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
"""GDN states in vLLM's unified pools, as the decode megakernel holds them.

A GDN layer's state lives in a state block of the pool it shares with an
attention layer, addressed in 512-byte granules (``gdn_pcp_fused.memory_ref``,
which the prefill writes it through): the FP32 recurrent state ``[V heads, K,
V]`` from granule 0, then at granule ``V heads * K * V / 128`` the FP32 conv
window ``[taps - 1, conv]`` (columns ``[q | k | v]``, this rank's heads). The
decoder keeps each request's states in private BF16 slots: conv ``[taps - 1,
conv]`` and recurrent ``[V heads, V, K]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

LANES = 128  # u32 lanes of a granule


def state_granules(v_heads, k_dim, v_dim, taps, conv):
    """(conv offset, granules) of a state block's states."""
    offset = v_heads * k_dim * v_dim // LANES
    return offset, offset + pl.cdiv((taps - 1) * conv, LANES)


def logical_blocks(pool, pages_per_block):
    """An HND pool ``[pages, 2, D / 4, 4, page]`` with dim 0 its state blocks
    (``pages_per_block`` kernel pages each), as the GDN kernels address it."""
    return pool.reshape(pool.shape[0] // pages_per_block, pool.shape[1] * pages_per_block, *pool.shape[2:])


def read_states(pools, gdn_pools, blocks, *, v_heads, k_dim, v_dim, taps, conv, interpret=False):
    """Every request's states of every GDN layer, from its state blocks (inside
    ``shard_map``; rank-local operands).

    ``pools`` are the logical-block views (:func:`logical_blocks`), GDN layer
    ``l`` in ``pools[gdn_pools[l]]``; ``blocks`` int32 ``[requests, layers]``.
    Returns BF16 conv ``[requests, layers, taps - 1, conv]`` and recurrent
    ``[requests, layers, V heads, V, K]``."""
    requests, layers = blocks.shape
    offset, granules = state_granules(v_heads, k_dim, v_dim, taps, conv)

    def kernel(blocks_ref, *refs):
        sources, out, sem = refs[: len(pools)], refs[len(pools)], refs[len(pools) + 1]

        def copy(i, layer):
            source = sources[gdn_pools[layer]]
            view = source.bitcast(jnp.uint32).reshape(source.shape[0], -1, 1, LANES)
            return tpu.make_async_copy(
                view.at[blocks_ref[i, layer], pl.ds(0, granules)], out.at[i, layer], sem
            )

        for i in range(requests):
            for layer in range(layers):
                copy(i, layer).start()
        for i in range(requests):
            for layer in range(layers):
                copy(i, layer).wait()

    words = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((requests, layers, granules, 1, LANES), jnp.uint32),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * len(pools),
            out_specs=pl.BlockSpec(memory_space=tpu.HBM),
            scratch_shapes=(tpu.SemaphoreType.DMA,),
        ),
        interpret=interpret,
        name="qwen_decode_read_gdn_states",
    )(blocks.astype(jnp.int32), *pools)
    values = jax.lax.bitcast_convert_type(words[:, :, :, 0], jnp.float32)
    recurrent = values[:, :, :offset].reshape(requests, layers, v_heads, k_dim, v_dim)
    window = values[:, :, offset:].reshape(requests, layers, -1)[:, :, : (taps - 1) * conv]
    return (
        window.reshape(requests, layers, taps - 1, conv).astype(jnp.bfloat16),
        jnp.swapaxes(recurrent, -1, -2).astype(jnp.bfloat16),
    )
