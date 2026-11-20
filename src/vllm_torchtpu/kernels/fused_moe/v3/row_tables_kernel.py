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
"""The row tables, scattered on the sparse cores.

Two tables are indexed by this shard's routed row: the token each row
computes and that token's activation scale. Every vector subcore streams
its own run of the routed pairs. It reads the pair's routed row from HBM,
computes its token in place, gathers its scale word directly from the
gathered routing messages, and scatters the token and the scale word by
index-driven DMA. A pair of another shard carries an index at or past the
routed rows' end. Clamped to that length, the DMA filters it out (XLA's
mode="drop"). The tables arrive zeroed from the shard-tables kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    SCATTER_CHUNK_PAIRS_DEFAULT,
    SCATTER_RING_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES

# Added to an integer before a float multiply by an inverse, so the
# conversion back to int32 floors the exact quotient (no integer divide
# on the sparse core).
FLOOR_NUDGE = 0.5


def _scatter_body(
    routed_row_hbm,
    message_hbm,
    token_table_hbm,
    scale_table_hbm,
    rows_vmem,
    tokens_vmem,
    scale_index_vmem,
    scales_vmem,
    scale_row_vmem,
    fetch_sems,
    gather_sems,
    token_scatter_sems,
    scale_scatter_sems,
    *,
    run,
    chunk,
    ring,
    top_k,
    routed_rows,
    tokens_per_shard,
    message_words,
    scale_offset,
    messages_words,
    n_scales,
    plane_words,
    scale_words,
    pairs_per_shard,
    message_tokens,
):
    subcore = lax.axis_index((gather.CORE_AXIS, gather.SUBCORE_AXIS))
    base = subcore * run
    n_chunks = run // chunk
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    lane = lax.broadcasted_iota(jnp.int32, (lanes,), 0)
    # Integer division as (x + FLOOR_NUDGE) * (1 / d) truncated: exact while x stays
    # far below 2**23. A padding pair (past tokens * top_k in its shard)
    # decodes to a slot past top_k; its routed row is the drop row, so
    # nothing it decodes to is written.
    inv_pairs = jnp.float32(1.0 / pairs_per_shard)
    inv_tokens = jnp.float32(1.0 / message_tokens)

    def part(buffer, slot):
        # The ring is one flat buffer: an index-driven DMA needs its offsets
        # and its data contiguous, and 2-D VMEM rows are tile-interleaved.
        return buffer.at[pl.ds(slot * chunk, chunk)]

    def plane_part(buffer, slot, plane):
        # The scale buffers hold every plane of a slot, plane-major.
        return buffer.at[pl.ds((slot * n_scales + plane) * chunk, chunk)]

    def fetch(c, slot):
        return pltpu.make_async_copy(
            routed_row_hbm.at[pl.ds(base + c * chunk, chunk)],
            part(rows_vmem, slot),
            fetch_sems.at[slot],
        )

    def gather_scales(slot, plane):
        # A foreign pair's scale is never written, so it is never fetched:
        # prepare points it at the ignored index.
        return pltpu.make_async_copy(
            message_hbm.at[
                plsc.Indices(
                    plane_part(scale_index_vmem, slot, plane),
                    ignored_value=messages_words,
                )
            ],
            plane_part(scales_vmem, slot, plane),
            gather_sems.at[slot],
        )

    def scatter_tokens(slot):
        return pltpu.make_async_copy(
            part(tokens_vmem, slot),
            token_table_hbm.at[
                plsc.Indices(part(rows_vmem, slot), ignored_value=routed_rows)
            ],
            token_scatter_sems.at[slot],
        )

    def scatter_scales(slot, plane):
        # Plane p of the table starts at word p * scale_words; a foreign
        # pair's row is clamped to the drop row of that plane.
        return pltpu.make_async_copy(
            plane_part(scales_vmem, slot, plane),
            scale_table_hbm.at[
                plsc.Indices(
                    plane_part(scale_row_vmem, slot, plane),
                    ignored_value=routed_rows + plane * scale_words,
                )
            ],
            scale_scatter_sems.at[slot],
        )

    def prepare(c, slot):
        """A pair's number is its shard's word in the gathered routing
        messages: shard *
        pairs + top-k slot * tokens + token, so the token is the shard's
        first plus the word's place in its top-k slot. Its scale word is
        its shard's routing message at scale_offset words plus that place.
        Its routed
        row is clamped so every foreign row is the single value the scatter
        drops."""
        for vector in range(chunk // lanes):
            span = pl.ds(slot * chunk + vector * lanes, lanes)
            pair = base + c * chunk + vector * lanes + lane
            shard = ((pair.astype(jnp.float32) + FLOOR_NUDGE) * inv_pairs).astype(
                jnp.int32
            )
            local = pair - shard * jnp.int32(pairs_per_shard)
            top_k_slot = (
                (local.astype(jnp.float32) + FLOOR_NUDGE) * inv_tokens
            ).astype(jnp.int32)
            token = local - top_k_slot * jnp.int32(message_tokens)
            tokens_vmem[span] = shard * jnp.int32(tokens_per_shard) + token
            row = jnp.minimum(rows_vmem[span], jnp.int32(routed_rows))
            rows_vmem[span] = row
            scale_word = (
                shard * jnp.int32(message_words) + jnp.int32(scale_offset) + token
            )
            for plane in range(n_scales):
                plane_span = pl.ds(
                    (slot * n_scales + plane) * chunk + vector * lanes, lanes
                )
                scale_index_vmem[plane_span] = jnp.where(
                    row < jnp.int32(routed_rows),
                    scale_word + jnp.int32(plane * plane_words),
                    jnp.int32(messages_words),
                )
                scale_row_vmem[plane_span] = row + jnp.int32(plane * scale_words)

    for c in range(min(ring, n_chunks)):
        fetch(c, c).start()

    def retire(c):
        slot = c % ring
        scatter_tokens(slot).wait()
        for plane in range(n_scales):
            scatter_scales(slot, plane).wait()

        @pl.when(c + ring < n_chunks)
        def _():
            fetch(c + ring, slot).start()

    def step(c, carry):
        slot = c % ring
        fetch(c, slot).wait()
        prepare(c, slot)
        for plane in range(n_scales):
            gather_scales(slot, plane).start()
        scatter_tokens(slot).start()
        for plane in range(n_scales):
            gather_scales(slot, plane).wait()
            scatter_scales(slot, plane).start()

        @pl.when(c > 0)
        def _():
            retire(c - 1)

        return carry

    lax.fori_loop(0, n_chunks, step, 0)
    retire(n_chunks - 1)


def scatter_row_tables(
    routed_row,
    routing_messages,
    token_table,
    scale_table,
    *,
    top_k,
    tokens_per_shard,
    message_words,
    scale_offset,
    n_scales=1,
    plane_words=0,
    pairs_per_shard=None,
    message_tokens=None,
    chunk=None,
    chunk_max=SCATTER_CHUNK_PAIRS_DEFAULT,
    ring=SCATTER_RING_DEFAULT,
):
    """Fills the two row tables on the sparse cores, in place.

    Args:
      routed_row: [P] int32, routed pair p's row in this shard's routed
        rows, or any value at or past routed_rows for a pair of another
        shard. P = num_shards *
        tokens_per_shard * top_k pairs numbered as the words of the
        gathered routing messages.
      routing_messages: [num_shards * message_words] int32, every shard's
        routing message, flat.
        Token t's scale is the f32 bits of word (t // tokens_per_shard) *
        message_words + scale_offset + t % tokens_per_shard.
      token_table: [routed_rows] int32, zeroed.
      scale_table: [n_scales * scale words] int32, zeroed; plane p of a
        row's scales at word p * scale words + row, scale words >=
        routed_rows.
      n_scales: scale words per token (one, or one per activation block).
      plane_words: words between one token's scale planes in its shard's
        routing message.
      pairs_per_shard: words of pair indices per shard in the messages
        (message_tokens * top_k when None; more when the pair list is
        padded with pairs that route nowhere).
      message_tokens: tokens per slot in a shard's message (tokens_per_shard
        when None; the count the select kernel padded the tokens to, whose
        extra tokens route nowhere). tokens_per_shard is the gathered rows'
        count per shard, which the token table indexes.
      chunk: pairs per chunk (whole vectors of whole tokens); None takes
        the largest count up to `chunk_max` that divides a subcore's run.
      ring: chunks in flight per subcore.

    Returns:
      (token_table, scale_table) filled: the token of each routed row and
      that token's scale words, zero where no pair lands.
    """
    for name, a in (
        ("routed_row", routed_row),
        ("routing messages", routing_messages),
        ("token_table", token_table),
        ("scale_table", scale_table),
    ):
        if a.ndim != 1 or a.dtype != jnp.int32:
            raise ValueError(
                f"{name} is {a.shape} {a.dtype}. Flat int32 arrays are expected"
            )
    n_pairs = routed_row.shape[0]
    if message_tokens is None:
        message_tokens = tokens_per_shard
    if message_tokens < tokens_per_shard:
        raise ValueError(
            f"a message of {message_tokens} tokens per slot "
            f"holds fewer than the shard's {tokens_per_shard}"
        )
    if pairs_per_shard is None:
        pairs_per_shard = message_tokens * top_k
    if pairs_per_shard < message_tokens * top_k:
        raise ValueError(
            f"{pairs_per_shard} pair words per shard hold fewer "
            f"than {message_tokens} tokens x {top_k}"
        )
    if n_pairs % pairs_per_shard:
        raise ValueError(
            f"{n_pairs} routed pairs is not a whole number of "
            f"shards of {tokens_per_shard} tokens x {top_k}"
        )
    if routing_messages.shape[0] != n_pairs // pairs_per_shard * message_words:
        raise ValueError(
            f"the gathered routing messages hold "
            f"{routing_messages.shape[0]} words. "
            f"{n_pairs // pairs_per_shard} shards of {message_words} "
            "were expected"
        )
    if n_scales > 1 and plane_words < tokens_per_shard:
        raise ValueError(
            f"scale planes {plane_words} words apart cannot "
            f"hold {tokens_per_shard} tokens' words each"
        )
    if message_words < (
        scale_offset + (n_scales - 1) * plane_words + tokens_per_shard
    ):  # scales exist for real rows
        raise ValueError(
            f"a shard's routing message of {message_words} words has no room "
            f"for {n_scales} planes of {tokens_per_shard} scales at word "
            f"{scale_offset}"
        )
    routed_rows = token_table.shape[0]
    if scale_table.shape[0] % n_scales:
        raise ValueError(
            f"the scale table ({scale_table.shape[0]} words) is "
            f"not {n_scales} equal planes"
        )
    scale_words = scale_table.shape[0] // n_scales
    if scale_words < routed_rows:
        raise ValueError(
            f"a scale plane ({scale_words} words) is shorter "
            f"than the token table ({routed_rows}). The FFN "
            "kernel's window reads run past the last routed row"
        )
    if ring < 2:
        raise ValueError(
            f"ring {ring}: a chunk's fetch is issued when an "
            "earlier chunk retires, so at least two slots"
        )
    mesh, n_subcores = gather.subcore_mesh()
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    if n_pairs % n_subcores:
        raise ValueError(
            f"{n_pairs} routed pairs do not split evenly over "
            f"the chip's {n_subcores} vector subcores"
        )
    run = n_pairs // n_subcores
    if chunk is None:
        # The largest whole number of 128-word ring parts, up to
        # chunk_max, that divides a subcore's run; a pair's decode
        # does not depend on where the chunk boundaries fall. The chunk is
        # the same at every plane count: the plane buffers grow with the
        # planes (ring x planes x chunk words each) instead.
        chunk = next((c for c in range(chunk_max, 0, -LANES) if run % c == 0), run)
    if chunk % lanes or chunk % gather.SLICE_ALIGN_WORDS:
        raise ValueError(
            f"chunk {chunk} is not a whole number of {lanes}-lane "
            f"vectors and of {gather.SLICE_ALIGN_WORDS}-word slices"
        )
    run = gather.rows_per_subcore(n_pairs, n_subcores, chunk)
    if run != chunk and chunk % LANES:
        raise ValueError(
            f"chunk {chunk}: a ring part starts at a multiple of "
            f"{LANES} words, so a chunk is a multiple of {LANES} "
            f"unless it is a subcore's whole run of {run}"
        )
    program = gather.sparse_core_program(
        functools.partial(
            _scatter_body,
            run=run,
            chunk=chunk,
            ring=ring,
            top_k=top_k,
            routed_rows=routed_rows,
            tokens_per_shard=tokens_per_shard,
            message_words=message_words,
            scale_offset=scale_offset,
            messages_words=routing_messages.shape[0],
            n_scales=n_scales,
            plane_words=plane_words,
            scale_words=scale_words,
            pairs_per_shard=pairs_per_shard,
            message_tokens=message_tokens,
        ),
        out_types=[
            jax.ShapeDtypeStruct((routed_rows,), jnp.int32),
            jax.ShapeDtypeStruct((n_scales * scale_words,), jnp.int32),
        ],
        mesh=mesh,
        scratch_types=[
            pltpu.VMEM((ring * chunk,), jnp.int32),  # routed rows
            pltpu.VMEM((ring * chunk,), jnp.int32),  # tokens
            pltpu.VMEM((ring * n_scales * chunk,), jnp.int32),  # scale indices
            pltpu.VMEM((ring * n_scales * chunk,), jnp.int32),  # scale words
            pltpu.VMEM((ring * n_scales * chunk,), jnp.int32),  # table rows
            pltpu.SemaphoreType.DMA((ring,)),  # fetches
            pltpu.SemaphoreType.DMA((ring,)),  # scale gathers
            pltpu.SemaphoreType.DMA((ring,)),  # token scatters
            pltpu.SemaphoreType.DMA((ring,)),  # scale scatters
        ],
        compiler_params=pltpu.CompilerParams(**gather.COMPILER_PARAMS),
        name=f"fused_ep_row_tables_c{chunk}x{ring}"
        f"{'_p' + str(n_scales) if n_scales > 1 else ''}",
        in_place_outputs=2,
    )
    return program(routed_row, routing_messages, token_table, scale_table)
