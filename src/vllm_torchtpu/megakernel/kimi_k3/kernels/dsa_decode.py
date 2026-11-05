# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1's DSA indexer inside the K3 decode megakernel (TP32, DCP32).

Each rank holds one index head per MLA layer: its ``wq_b`` rows in ``m_qb``
columns 768:896 and its ``weights_proj`` row in ``m_knorm`` lanes 768:.
:class:`IndexScorer` is one layer's indexer: a step whose rows all have
context of at most ``INDEX_TOPK`` (:func:`scores_needed`) runs none of it,
since every token is then selected.

Scores are FP32 ``[planes, V, 128]`` (:func:`score_shape`), the select's
virtual-row layout: below eight rows a row splits into ``g = 8 / R`` parts of
``P`` tokens on virtual rows ``r g .. r g + g - 1``, from eight rows ``g = 1``
(:func:`virtual_rows`); local token ``t`` of row ``r`` is at plane
``(t % P) // 128``, virtual row ``r g + t // P``, lane ``t % 128``.

Wire fields: per batch row this rank's FP8 query values, its head weight's
BF16 projection ``v`` and the query scale's biased exponent ``e``
(:func:`query_fields`) ride the DCP query packets' fifth lane tile
(``mla_dcp.DcpAttention``), whose lanes the latent query leaves free, so every
rank receives all 32 heads with no exchange of its own:

* BF16 packets: slot 0 packet lanes 576:640 hold values 0:64, slot 1 values
  64:128, slot 2 lane 576 ``v`` and 577 ``e`` (all exact in BF16); the
  receiver zeroes lanes 576:640 before the latent attention
  (:meth:`IndexScorer.clean`).
* uint8 packets (plain FP8 B32 and B64): lanes 576:584 keep the latent
  query's scale bytes; slot 0 lanes 584:640 hold value bytes 0:56, slot 1
  bytes 56:112, slot 2 lanes 584:600 bytes 112:128, lanes 600 and 601
  ``v``'s two BF16 bytes and 602 ``e``; ``mla_dcp._unpack_fp8_query``
  clears lanes 576:640.

Every receiver forms the head weight ``w = v 2^(e - 127) 128^-0.5``
(:func:`head_weight`, ``dsa_reference.index_query``'s order), every
member's in the scorer's ``queries`` and ``weights``.

Chunked batches (B16 and up) score every row before the first chunk attends,
while the query packets of later chunks are not sent yet: there the fields
go on an all-gather of their own instead (:meth:`IndexScorer.send_fields`),
each row's :func:`field_packet` (its values as BF16 pairs, exact) into slot
``member`` of every member's persistent inbox (:func:`field_persistent_scratch`),
a row to a 32-bit lane row: Mosaic reads a traced row of a 32-bit array one lane
tile wide, not of a wider 8-bit one.

The scorer (:meth:`IndexScorer.score`) scores each row over its own pages:
per 128 keys one ``[32, 128] x [128, 128]`` MXU dot of the FP8 query values
held in BF16 with the page's FP8 keys, ReLU, the head weights, a pairwise sum
of the four 8-head tiles, sublane rolls by 1, 2 and 4, then the keys'
power-of-two scales. Pages stream through a ring of ``RING_DEPTH`` DMAs in
flight (:meth:`IndexScorer.prime` at the layer start, rows in order); the new
token, not in the cache until the op commits it, is scored from the layer's own
key on every rank (``new_scores``) and stored on the rank owning its page. Once
every row is scored, the layer's ``dsa_select.TopkSelect`` selects each row's
top-k across the group (:meth:`IndexScorer.score_and_select`). Per row, rank
and the select's virtual-row layout, let ``L`` be :func:`local_length`, this
rank's local tokens among the row's first ``position + 1`` tokens (the new
token included on its owner): entries ``[0, L)`` hold scores, ``[L, ceil to
128 of L)`` -inf, and every other entry is arbitrary (unwritten; stale VMEM on TPU), so the select bounds
each row by ``L``, not by the context length. With a row split into parts of
``P`` local tokens (``P`` from :func:`score_shape`), part ``p``'s bound is
``clip(L - p P, 0, P)``.

The indexer's VMEM views overlay the projection/expert pool where an MLA
layer leaves it free (``decode_megakernel._dsa_layout``); each is a whole
number of native tiles.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.dsa import (
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    QUERY_NORM_LANES,
)
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, mla_dcp

RING_DEPTH = 8  # index pages in flight while one is scored
RING_SLOTS = RING_DEPTH + 1
BLOCK_UNITS = 8  # block-major: the (sequence, page) items a loop step scores from 16 rows (half of it at 8)
FIELD_SLOTS = 8  # the fields buffer's rows a batch row: one FP32 tile (packet head slots are at most 4)
BF16_FIELD_LANE = 64  # of the packet's fifth lane tile: BF16 packet lanes 576:640
FP8_FIELD_LANE = 72  # uint8 packet lanes 584:640; 576:584 keep the latent query's scale bytes
DEBUG_OUTPUTS = ("x", "q_c", "queries", "weights", "scores", "mask", "new_selected", "absorbed", "merged",
                 "new_latent")
DEBUG_VIEWS = ("query_latent", "absorbed", "merged")  # the check outputs staged in VMEM views
SCOPED_NAMES = ("dsa row pages", "dsa row starts", "dsa next rows", "dsa fetch cursor", "dsa semaphores")
FIELD_NAMES = ("dsa field inbox", "dsa field semaphores")  # field_persistent_scratch's entries
BATCH_CLASSES = (1, 2, 4, 8, 16, 32, 64)  # TP32 decode batches the scorer is built and checked for
# Diagnostic ablations of the scorer (KernelConfig.dcp_skip; results invalid, timing only): never run; no page DMAs
# (each fetch's table read and slot are kept, sunk into SMEM); no cursor walk or table reads; no dots (a constant
# store: every load that feeds them, keys, exponents, queries and weights, goes too); no store; the select
# compiled, never run (dsa_select.TopkSelect).
INDEX_SKIPS = ("index_off", "index_dma", "index_fetch", "index_dots", "index_store", "index_select")


def scores_needed(position_ref, batch: int):
    """Whether a step scores: some row's context (its position plus one)
    exceeds ``INDEX_TOPK``. Below it every token is selected and the step is
    Kimi-K3's dense attention."""
    last = position_ref[0]
    for b in range(1, batch):
        last = jnp.maximum(last, position_ref[b])
    return last + 1 > INDEX_TOPK


def local_pages(tokens, owner, page_tokens: int, group: int):
    """This rank's pages among a row's first ``tokens`` tokens: the global
    pages ``g < ceil(tokens / page_tokens)`` with ``g % group == owner``
    (``tokens >= 0``, ``0 <= owner < group``; Python or traced integers)."""
    return ((tokens + page_tokens - 1) // page_tokens + group - 1 - owner) // group


def local_length(position, owner, page_tokens: int, group: int):
    """``L``, the scores contract's bound: this rank's local tokens among a
    row's first ``position + 1`` tokens (the new token included on its owner),
    the full global pages ``g < (position + 1) // page_tokens`` with ``g %
    group == owner`` plus the partial page's tokens on its owner (Python or
    traced integers)."""
    tokens = position + 1
    full = tokens // page_tokens
    whole = (full + group - 1 - owner) // group
    return whole * page_tokens + jnp.where(full % group == owner, tokens - full * page_tokens, 0)


def select_rows(batch: int, local_tokens: int) -> int:
    """The rows one score-and-select pass holds: ``batch``, or half of it when
    ``batch`` rows of ``local_tokens`` scores exceed
    ``KernelConfig.dsa_select_row_tokens`` (the halves are whole 8-row tiles
    and whole verify sequences; decode_megakernel._dsa_layout)."""
    if batch >= 32 and batch * local_tokens > _config().dsa_select_row_tokens:
        return batch // 2
    return batch


def block_units(rows: int, sequence_rows: int) -> int:
    """``KernelConfig.dsa_score_block_major``: the (sequence, page) items a
    block-major loop step scores in calls of ``rows`` rows (:func:`select_rows`),
    ``BLOCK_UNITS`` from 16 rows and half of it at 8; 0 (the per-sequence page
    loop) when off, below 8 rows, for sequences that do not divide 8, or
    without grouped new tokens."""
    config = _config()
    if not (config.dsa_score_block_major and config.dsa_group_new_tokens) or rows < 8 or 8 % sequence_rows:
        return 0
    return BLOCK_UNITS if rows >= 16 else BLOCK_UNITS // 2


def block_steps_ahead(units: int) -> int:
    """Block-major steps whose pages are in flight while one is scored."""
    return 2


def ring_slots(batch: int, local_tokens: int, sequence_rows: int = 1) -> int:
    """Index pages the scorer's ring holds: ``RING_SLOTS``, or block-major
    (:func:`block_units`) a slot per item of each step in flight and the one
    scored."""
    units = block_units(select_rows(batch, local_tokens), sequence_rows)
    return units * (block_steps_ahead(units) + 1) if units else RING_SLOTS


def ring_in_attention(batch: int, local_tokens: int, sequence_rows: int = 1) -> bool:
    """Whether the ring overlays the select's views in the MLA layer's
    attention allocation (``decode_megakernel._dsa_layout``) instead of region
    B: block-major in one score-and-select pass, where the select first writes
    its views after the scorer's last read of the ring; primed once the down
    dots are done."""
    rows = select_rows(batch, local_tokens)
    return rows == batch and block_units(rows, sequence_rows) > 0


def semaphore_ring(batch: int, sequence_rows: int = 1) -> int:
    """The most ring slots :func:`ring_slots` gives at ``batch`` over every
    context (one pass or halves): the semaphores :func:`scoped_scratch` holds."""
    best = RING_SLOTS
    for rows in {batch, batch // 2}:
        units = block_units(rows, sequence_rows)
        if units:
            best = max(best, units * (block_steps_ahead(units) + 1))
    return best


def virtual_rows(rows: int) -> tuple[int, int]:
    """``(V, g)``: the score buffer's virtual rows and each row's parts. Below
    eight rows, rows pad to 1, 2, 4 or 8 and split into ``8 / R`` parts on
    eight virtual rows; from eight, rows pad to a multiple of 8, one part each."""
    if rows < 8:
        return 8, 8 // (1 << (rows - 1).bit_length())
    return -(-rows // 8) * 8, 1


def score_shape(local_tokens: int, rows: int) -> tuple[int, int, int]:
    """``(planes, V, 128)``: ``local_tokens`` a row padded to whole 128-token
    lane rows in each of its parts."""
    vrows, parts = virtual_rows(rows)
    return (-(-local_tokens // (parts * 128)), vrows, 128)


def scores_by_row(scores, rows: int):
    """numpy ``[planes, V, 128]`` -> ``[rows, T]``: the inverse of the placement."""
    planes, _, lanes = scores.shape
    parts = virtual_rows(rows)[1]
    out = np.asarray(scores)[:, : rows * parts].reshape(planes, rows, parts, lanes)
    return out.transpose(1, 2, 0, 3).reshape(rows, parts * planes * lanes)


def head_weight(v, exponent):
    """FP32 ``v 2^(e - 127) 128^-0.5``: a head weight with its query's
    power-of-two scale and ``128^-0.5`` folded in (dsa_reference.index_query)."""
    return v * dsa_cache.power_of_two(exponent - 127) * INDEX_DIM**-0.5


def query_fields(q_projection, v_projection):
    """``(values f32 [B, 128], v f32 [B, 1], exponent int32 [B, 1])`` of one
    index head: ``q_projection`` (``q_c @ wq_b``, BF16) as FP8 values with a
    power-of-two scale per row (``dsa_cache.encode_index_keys`` with the query
    floor; ``exponent`` its biased exponent), and ``v_projection`` (FP32 ``x .
    weights_proj``) rounded to BF16, times ``32^-0.5`` rounded to BF16 again."""
    values, exponent = dsa_cache.encode_index_keys(q_projection, floor=dsa_cache.QUERY_AMAX_FLOOR)
    v = (v_projection.astype(jnp.bfloat16).astype(jnp.float32) * INDEX_HEADS**-0.5).astype(jnp.bfloat16)
    return values, v.astype(jnp.float32), exponent


def row_dot(x, row):
    """FP32 ``x [B, H] . row [1, H]`` per row: exact products of 128-lane
    tiles accumulated in order, then one lane sum (no ``[B, H]`` FP32
    temporary, no one-column MXU dot)."""
    total = None
    for start in range(0, x.shape[1], 128):
        term = x[:, start:start + 128].astype(jnp.float32) * row[:, start:start + 128].astype(jnp.float32)
        total = term if total is None else total + term
    return jnp.sum(total, axis=1, keepdims=True)


def bf16_fields(values, v, exponent, *, roll=pltpu.roll):
    """The BF16 packets' fifth lane tile of head slots 0, 1 and 2: FP32
    ``[B, 128]`` each, zero below ``BF16_FIELD_LANE`` (module docstring)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    upper = lane >= BF16_FIELD_LANE
    third = jnp.where(lane == BF16_FIELD_LANE, v,
                      jnp.where(lane == BF16_FIELD_LANE + 1, exponent.astype(jnp.float32), 0.0))
    return jnp.where(upper, roll(values, BF16_FIELD_LANE, 1), 0.0), jnp.where(upper, values, 0.0), third


def fp8_fields(values, v, exponent, *, roll=pltpu.roll):
    """The uint8 packets' fifth lane tile of head slots 0, 1 and 2 as int32
    byte values, zero below ``FP8_FIELD_LANE`` (module docstring)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    codes = jax.lax.bitcast_convert_type(values.astype(jnp.float8_e4m3fn), jnp.uint8).astype(jnp.int32)
    upper = lane >= FP8_FIELD_LANE
    bits = jax.lax.bitcast_convert_type(v, jnp.int32)
    third = (jnp.where(upper & (lane < 88), roll(codes, 88, 1), 0)
             + jnp.where(lane == 88, (bits >> 16) & 255, 0) + jnp.where(lane == 89, (bits >> 24) & 255, 0)
             + jnp.where(lane == 90, exponent, 0))
    return jnp.where(upper, roll(codes, 72, 1), 0), jnp.where(upper, roll(codes, 16, 1), 0), third


def e4m3_values(codes):
    """FP32 values of finite E4M3 byte codes (int32), by bit arithmetic; the
    sign is set through the bits (Mosaic lowers a negation as ``0 - x``,
    which turns -0 into +0)."""
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    normal = jax.lax.bitcast_convert_type(((exponent + 120) << 23) | (mantissa << 20), jnp.float32)
    magnitude = jnp.where(exponent > 0, normal, mantissa.astype(jnp.float32) * 2.0**-9)
    signed = jax.lax.bitcast_convert_type(magnitude, jnp.int32) | ((codes & 128) << 24)
    return jax.lax.bitcast_convert_type(signed, jnp.float32)


def read_bf16_fields(first, second, third, *, roll=pltpu.roll):
    """``(values f32 [G, 128], w f32 [G, 1])`` of every member from its BF16
    packet's head slots 0, 1 and 2 (fifth lane tile, FP32 ``[G, 128]`` each)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, first.shape, 1)
    values = jnp.where(lane < BF16_FIELD_LANE, roll(first, BF16_FIELD_LANE, 1), second)
    v = jnp.sum(jnp.where(lane == BF16_FIELD_LANE, third, 0.0), axis=1, keepdims=True)
    exponent = jnp.sum(jnp.where(lane == BF16_FIELD_LANE + 1, third, 0.0), axis=1, keepdims=True)
    return values, head_weight(v, exponent.astype(jnp.int32))


def read_fp8_fields(first, second, third, *, roll=pltpu.roll):
    """As :func:`read_bf16_fields` from uint8 packets (int32 byte values)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, first.shape, 1)
    codes = jnp.where(lane < 56, roll(first, 56, 1), jnp.where(lane < 112, roll(second, 112, 1), roll(third, 40, 1)))

    def byte(at):
        return jnp.sum(jnp.where(lane == at, third, 0), axis=1, keepdims=True)

    v = jax.lax.bitcast_convert_type((byte(89) << 24) | (byte(88) << 16), jnp.float32)
    return e4m3_values(codes), head_weight(v, byte(90))


def field_packet(values, v, exponent, *, roll=pltpu.roll):
    """The fields' all-gather packet of every row, int32 ``[B, 128]``: lane
    ``w < 64`` the BF16 bits of ``values`` lanes ``w`` (low half) and ``w +
    64`` (high half), exact for FP8 values; lane 64 ``v``'s bits, 65
    ``exponent``, the rest zero."""
    bits = jax.lax.shift_right_logical(jax.lax.bitcast_convert_type(
        values.astype(jnp.bfloat16).astype(jnp.float32), jnp.int32), 16)
    lane = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    words = bits | jnp.left_shift(roll(bits, 64, 1), 16)
    return (jnp.where(lane < 64, words, 0) + jnp.where(lane == 64, jax.lax.bitcast_convert_type(v, jnp.int32), 0)
            + jnp.where(lane == 65, exponent, 0))


def read_field_packet(packet, *, roll=pltpu.roll):
    """``(values f32 [G, 128], w f32 [G, 1])`` of every member from its
    :func:`field_packet` ``[G, 128]`` (lanes past 65 ignored)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1)
    low = jax.lax.bitcast_convert_type(jnp.left_shift(packet, 16), jnp.float32)
    high = jax.lax.bitcast_convert_type(packet & jnp.int32(-65536), jnp.float32)
    values = jnp.where(lane < 64, low, roll(high, 64, 1))

    def lane_value(at):
        return jnp.sum(jnp.where(lane == at, packet, 0), axis=1, keepdims=True)

    v = jax.lax.bitcast_convert_type(lane_value(64), jnp.float32)
    return values, head_weight(v, lane_value(65))


def field_persistent_scratch(batch: int, group: int, heads: int = 1) -> tuple:
    """The fields' all-gather buffers (``FIELD_NAMES``), which peers write: the
    inbox ``[group, heads * batch, 128]`` int32 (slot ``member`` each member's
    :func:`field_packet` rows, head ``k``'s at rows ``k * batch ..``; attention
    DP2 has two index heads a rank) and its received / sent DMA semaphores."""
    if heads not in (1, 2):
        raise ValueError(f"the index fields' all-gather carries one or two heads a rank, not {heads}")
    if batch % 8:
        raise ValueError(f"the index fields' all-gather takes whole 8-row tiles, not {batch} rows")
    return pltpu.VMEM((group, heads * batch, 128), jnp.int32), pltpu.SemaphoreType.DMA((2,))


def scratch_shapes(batch: int, local_tokens: int, page_tokens: int, debug: bool, sequence_rows: int = 1, *,
                   query_head: bool = True,
                   attention_heads: int = mla_dcp.HEADS_PER_RANK) -> dict[str, tuple[tuple[int, ...], Any]]:
    """The indexer's VMEM views by ``_dsa_layout`` name, ``(shape, dtype)``:
    with ``query_head`` (attention DP1; attention DP's index queries come with
    its query-up shards) ``query_head`` (this rank's ``wq_b`` head), ``ring`` (index pages),
    ``queries`` and ``weights`` (every member's head, per row), ``fields``
    (this rank's wire fields; read as int32 for uint8 packets; one unused
    tile at chunked batches, whose fields have their own all-gather), ``scores``,
    ``new_scores`` (each row's new token's score, lane-broadcast, on every
    rank), and with ``debug`` the q-down latent ``query_latent`` and this
    rank's ``attention_heads`` heads' absorbed queries ``absorbed`` and merged
    latent outputs ``merged``, heads-major (``DEBUG_VIEWS``)."""
    shapes: dict[str, tuple[tuple[int, ...], Any]] = {
        **({"query_head": ((QUERY_NORM_LANES, INDEX_DIM), jnp.bfloat16)} if query_head else {}),
        "ring": ((ring_slots(batch, local_tokens, sequence_rows), *dsa_cache.index_block_shape(page_tokens)),
                 jnp.uint8),
        "queries": ((batch, INDEX_HEADS, INDEX_DIM), jnp.bfloat16),
        "weights": ((batch, INDEX_HEADS, INDEX_DIM), jnp.float32),
        # Chunked batches send the fields on their own all-gather: one tile, unused.
        "fields": ((8 if mla_dcp.chunked(batch) else batch, FIELD_SLOTS, 128), jnp.float32),
        "scores": (score_shape(local_tokens, select_rows(batch, local_tokens)), jnp.float32),
        "new_scores": ((-(-batch // 8) * 8, 128), jnp.float32),
    }
    if debug:
        shapes["query_latent"] = ((-(-batch // 16) * 16, QUERY_NORM_LANES), jnp.bfloat16)
        # Heads-major: a row's three heads would pad to a whole tile.
        shapes["absorbed"] = ((attention_heads, -(-batch // 8) * 8, mla_dcp.LANES), jnp.float32)
        shapes["merged"] = ((attention_heads, -(-batch // 8) * 8, mla_dcp.LATENT), jnp.float32)
    return shapes


def scoped_scratch(batch: int, debug: bool, ring: int = RING_SLOTS) -> tuple:
    """The indexer's scoped SMEM and DMA semaphores (``SCOPED_NAMES``): each
    row's pages, first item and next row with pages (``batch + 1`` entries),
    the fetch cursor, the block-major walk and each row's local length, and the semaphores of the ``ring`` slots
    (:func:`ring_slots`), the query head and with ``debug`` one per check
    output."""
    return (
        pltpu.SMEM((batch + 1,), jnp.int32),
        pltpu.SMEM((batch + 1,), jnp.int32),
        pltpu.SMEM((batch + 1,), jnp.int32),
        pltpu.SMEM((4 + batch,), jnp.int32),
        pltpu.SemaphoreType.DMA((ring + 1 + (len(DEBUG_OUTPUTS) if debug else 0),)),
    )


def debug_shapes(batch: int, hidden: int, local_tokens: int, layers: int, key_heads: int,
                 attention_heads: int = mla_dcp.HEADS_PER_RANK) -> tuple:
    """The stack's check outputs (``index_debug``), in ``DEBUG_OUTPUTS``
    order, each with one entry per MLA layer: the scores before the select
    turns them into keys, the select's mask words, each row's new-token
    flag (:meth:`IndexScorer.row_selected_new`, row ``b`` lane-broadcast),
    this rank's ``attention_heads`` heads' absorbed queries (score scale folded in) and
    merged latent outputs (``mla_dcp.DcpAttention.attend``'s), head ``h`` of
    row ``b`` at ``[h, b]``, and the MLA layer's new-row handoff (``[b,
    key_heads, 256]``, the first three slots lanes 0:768 of row ``b``'s new
    latent as the attention merges it)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    mask = dsa_select.scratch_shapes(batch, local_tokens, dsa_select.SELECT_FORMS[batch])["mask"][0]
    return (
        jax.ShapeDtypeStruct((layers, batch, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((layers, -(-batch // 16) * 16, QUERY_NORM_LANES), jnp.bfloat16),
        jax.ShapeDtypeStruct((layers, batch, INDEX_HEADS, INDEX_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((layers, batch, INDEX_HEADS, INDEX_DIM), jnp.float32),
        jax.ShapeDtypeStruct((layers, *score_shape(local_tokens, batch)), jnp.float32),
        jax.ShapeDtypeStruct((layers, *mask), jnp.int32),
        jax.ShapeDtypeStruct((layers, -(-batch // 8) * 8, 128), jnp.int32),
        jax.ShapeDtypeStruct((layers, attention_heads, -(-batch // 8) * 8, mla_dcp.LANES), jnp.float32),
        jax.ShapeDtypeStruct((layers, attention_heads, -(-batch // 8) * 8, mla_dcp.LATENT), jnp.float32),
        jax.ShapeDtypeStruct((layers, batch, key_heads, 256), jnp.bfloat16),
    )


def _head_sum(queries, weights, keys):
    """``sum_h w_h ReLU(q_h . k)`` for 128 keys: ``[8, 128]`` FP32, every
    sublane the row's total (the four 8-head tiles summed pairwise, then the
    sublanes by rolls of 4, 2 and 1: each stage adds sublanes ``s`` and ``s ^
    shift``, so every sublane holds the same bits and a score does not depend
    on its virtual row)."""
    logits = jax.lax.dot_general(queries, keys, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    terms = [jnp.maximum(logits[8 * t:8 * t + 8], 0.0) * weights[8 * t:8 * t + 8] for t in range(INDEX_HEADS // 8)]
    total = (terms[0] + terms[1]) + (terms[2] + terms[3])
    for shift in (4, 2, 1):
        total = total + pltpu.roll(total, shift, 0)
    return total


def _head_sums(queries, weights, keys):
    """:func:`_head_sum` of several rows over the same 128 keys: one dot of the
    rows' stacked queries ``[32 rows, 128]``, then each row's head sum, the
    same arithmetic per element as one row's dot."""
    logits = jax.lax.dot_general(jnp.concatenate(queries, axis=0), keys, (((1,), (1,)), ((), ())),
                                 preferred_element_type=jnp.float32)
    out = []
    for r, weight in enumerate(weights):
        own = logits[INDEX_HEADS * r:INDEX_HEADS * (r + 1)]
        terms = [jnp.maximum(own[8 * t:8 * t + 8], 0.0) * weight[8 * t:8 * t + 8] for t in range(INDEX_HEADS // 8)]
        total = (terms[0] + terms[1]) + (terms[2] + terms[3])
        for shift in (4, 2, 1):
            total = total + pltpu.roll(total, shift, 0)
        out.append(total)
    return out


def _head_partials(logits, weights):
    """Each row's sum of its four 8-head tiles of ``w_h ReLU(q_h . k)``
    (``[32 rows, 128]`` ``logits``), as :func:`_head_sum` forms it before its
    sublane rolls."""
    out = []
    for r, weight in enumerate(weights):
        own = logits[INDEX_HEADS * r:INDEX_HEADS * (r + 1)]
        terms = [jnp.maximum(own[8 * t:8 * t + 8], 0.0) * weight[8 * t:8 * t + 8] for t in range(INDEX_HEADS // 8)]
        out.append((terms[0] + terms[1]) + (terms[2] + terms[3]))
    return out


def _butterfly(partials):
    """``[8, 128]``: on sublane ``r`` the sum over the 8 sublanes of
    ``partials[r]`` (8 ``[8, 128]`` values, repeats the same object), in
    :func:`_head_sum`'s order (rolls of 4, 2 and 1, the same bits on every
    sublane): each distinct partial reduced once, then one select per further
    distinct partial."""
    sublane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    totals: dict[int, Any] = {}
    for x in partials:
        if id(x) not in totals:
            total = x
            for shift in (4, 2, 1):
                total = total + pltpu.roll(total, shift, 0)
            totals[id(x)] = total
    out = totals[id(partials[-1])]
    for key, total in totals.items():
        if key != id(partials[-1]):
            picks = functools.reduce(jnp.logical_or, [sublane == r for r, x in enumerate(partials) if id(x) == key])
            out = jnp.where(picks, total, out)
    return out


def _on_sublanes(values, sublanes):
    """``[8, 128]`` with ``values[j]`` (a tile or a scalar) on sublane
    ``sublanes[j]`` (traced, distinct); the other sublanes are ``values[-1]``'s."""
    sub = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    out = jnp.broadcast_to(values[-1], (8, 128))
    for value, at in zip(values[:-1], sublanes[:-1], strict=True):
        out = jnp.where(sub == at, value, out)
    return out


def _scale(exponents):
    """``2^(e - 127)`` FP32 of UE8M0 exponent bytes (int32)."""
    return pltpu.bitcast(exponents << 23, jnp.float32)


@dataclasses.dataclass(frozen=True)
class IndexScratch:
    """One MLA layer's indexer buffers (see :func:`scratch_shapes`): ``cache``
    the index-key cache in HBM ``[blocks, 24, tokens + 32, 128]`` uint8, read
    only; the VMEM views; the scoped SMEM and semaphores
    (:func:`scoped_scratch`); ``debug`` with ``index_debug`` the check outputs
    by ``DEBUG_OUTPUTS`` name (HBM) and the views by ``DEBUG_VIEWS`` name;
    ``new_scores`` the view of the new tokens' scores; ``select`` the
    select's buffers (``dsa_select.SelectScratch``); at chunked batches
    ``field_inbox`` and ``field_semaphores`` (:func:`field_persistent_scratch`)."""

    cache: Any
    query_head: Any
    ring: Any
    queries: Any
    weights: Any
    fields: Any
    scores: Any
    counts: Any
    starts: Any
    following: Any
    cursor: Any
    semaphores: Any
    debug: tuple[dict[str, Any], dict[str, Any]] | None = None
    new_scores: Any = None
    select: Any = None
    field_inbox: Any = None
    field_semaphores: Any = None


class IndexScorer:
    """One MLA layer's indexer on this rank, for ``batch`` rows at
    ``position_ref`` (SMEM).

    ``active`` (traced, :func:`scores_needed`) gates every method. The page
    table is the DCP cache's (``table_ref`` flat, ``pages`` entries a row,
    ``global_pages`` its convention); this rank holds global pages ``g`` with
    ``g % group == owner``. ``slots`` and ``uint8_packets`` describe the query
    packets (``mla_dcp.packet_layout``, ``persistent_scratch``); at chunked
    batches (``mla_dcp.chunked``) the fields go on their own all-gather
    instead (``packet_fields`` False), among the group's mesh ranks ``base +
    m``, this rank ``base + member``. ``heads``: the rank's index heads (two
    under attention DP2, its held shards'); ``table_row(r)``: the table row of
    row ``r`` (attention DP: the group's rows of the batch-wide table, as
    ``mla_dcp.DcpCache.table_row``; default ``r``). ``new_value_ref`` is the layer's
    handoff, whose rows 0 and 1 hold the new token's encoded key. ``counts``:
    each row's local pages below its position, ``DcpAttention.counts``
    (required by :meth:`prime`). ``skip``: the ``KernelConfig.dcp_skip``
    names, of which the scorer takes its ``INDEX_SKIPS`` (timing only,
    resolved at trace time); ``scope(name)``: the context its
    ``index_fields`` and ``index_score`` work runs in
    (``decode_megakernel._phase``; default none). ``selection``: the layer's
    ``dsa_select.TopkSelect`` (set by the MLA layer), which
    :meth:`score_and_select` runs.

    ``sequence_rows`` > 1 (speculative verify): consecutive groups of that
    many rows are one sequence's tokens at consecutive positions, each row
    with its own query and top-k. A row's candidates are the cache below its
    sequence's first position, its earlier rows' tokens and its own; the
    earlier rows' tokens are not in the cache until the op commits them, so
    the row scores them from their keys in ``new_value_ref`` and stores them
    on their pages' owners (:meth:`_score_earlier`).

    ``group_new_tokens`` (default ``KernelConfig.dsa_group_new_tokens``): a
    call of a multiple of 8 rows whose sequences divide 8 scores the new
    tokens per aligned group of 8 rows (:meth:`_score_new_tokens`), the same
    bits as each row's own dots."""

    def __init__(self, scratch: IndexScratch, *, position_ref, batch: int, active, table_ref, pages: int,
                 layer, page_tokens: int, group: int, global_pages: bool, owner, slots: int, uint8_packets: bool,
                 new_value_ref, counts: Any = None, member: Any = None, base: Any = None,
                 skip: tuple[str, ...] = (), scope: Any = None, sequence_rows: int = 1, order: Any = None,
                 group_new_tokens: bool | None = None, heads: int = 1, table_row: Any = None):
        if heads not in (1, 2):
            raise ValueError(f"a rank holds one or two index heads (attention DP1 or DP2), not {heads}")
        self.heads, self.table_row = heads, table_row
        if sequence_rows < 1 or batch % sequence_rows:
            raise ValueError(f"{sequence_rows}-row sequences must divide the {batch} rows")
        self.sequence_rows = sequence_rows
        self.group_new_tokens = _config().dsa_group_new_tokens if group_new_tokens is None else group_new_tokens
        self.s, self.position_ref, self.batch, self.active = scratch, position_ref, batch, active
        self.table_ref, self.pages, self.layer, self.counts = table_ref, pages, layer, counts
        self.page_tokens, self.group, self.global_pages, self.owner = page_tokens, group, global_pages, owner
        self.slots, self.uint8_packets, self.new_value_ref = slots, uint8_packets, new_value_ref
        self.member, self.base = member, base
        # dsa_select.sort_rows's order (SMEM [3, batch]): rows on virtual rows by slot; None: by row.
        self.order = order
        self.packet_fields = not mla_dcp.chunked(batch)
        if scratch.field_inbox is not None and scratch.field_inbox.shape[:2] != (group, batch * heads):
            raise ValueError(f"the field inbox {scratch.field_inbox.shape[:2]} is not {group} members' "
                             f"{batch * heads} rows")
        self.selection: Any = None  # one TopkSelect per pass of select_rows rows (a list when there are two)
        # The scores view holds one pass's rows (select_rows): all of them, or a half.
        self.select_rows = batch if scratch.scores is None else min(batch, scratch.scores.shape[1])
        self.parts = virtual_rows(batch)[1]
        self.part_tokens = None if scratch.scores is None else scratch.scores.shape[0] * 128
        self.skip = frozenset(name for name in skip if name.startswith("index_"))
        if self.skip - set(INDEX_SKIPS):
            raise ValueError(f"unknown scorer ablations {sorted(self.skip - set(INDEX_SKIPS))}")
        if "index_store" in self.skip and "index_dots" not in self.skip:
            raise ValueError("index_store needs index_dots: unstored dots are dead code")
        if "index_fetch" in self.skip and "index_dma" not in self.skip:
            raise ValueError("index_fetch needs index_dma: a wait needs its start")
        self.scope = scope or (lambda name: contextlib.nullcontext())
        # prime and score run when the step scores (index_off: compiled, never run)
        self.runs = active & (position_ref[0] < 0) if "index_off" in self.skip else active
        # Block-major (block_units): (sequence, page) items a step and steps in flight; the ring must hold them
        # (ring_slots).
        self.units = block_units(self.select_rows, sequence_rows) if self.parts == 1 else 0
        self.ring_count = RING_SLOTS if scratch.ring is None else scratch.ring.shape[0]
        if self.units:
            self.ahead = block_steps_ahead(self.units)
            if self.ring_count != self.units * (self.ahead + 1):
                raise ValueError(f"block-major scoring needs a ring of {self.units * (self.ahead + 1)} slots, "
                                 f"not {self.ring_count} (scratch_shapes with sequence_rows={sequence_rows})")

    def _debug_copy(self, outputs, name, source):
        return pltpu.make_async_copy(source, outputs[name].at[self.layer],
                                     self.s.semaphores.at[self.ring_count + 1 + DEBUG_OUTPUTS.index(name)])

    def write_debug(self, hidden_ref, new_key_ref):
        """With ``index_debug``: copy the layer's input ``hidden_ref``, q-down
        latent, queries, weights, the select's mask words, each row's
        new-token flag, the attention's absorbed queries and merged outputs
        (their views, stored by the MLA layer) and the new latent rows
        ``new_key_ref`` to the check outputs (the
        scores go before the select, :meth:`score_and_select`). The flags go
        out through the select's ``bounds`` view, which it no longer reads
        once it has run."""
        if self.s.debug is None:
            return
        outputs, views = self.s.debug
        select = self.selection.s
        sources = {"x": hidden_ref, "q_c": views["query_latent"], "queries": self.s.queries,
                   "weights": self.s.weights, "mask": select.mask, "new_selected": select.bounds,
                   "absorbed": views["absorbed"], "merged": views["merged"], "new_latent": new_key_ref}

        @pl.when(self.active)
        def _():
            sub = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)

            @pl.loop(0, self.batch)
            def flag(row):
                tile = pl.ds(pl.multiple_of(row // 8 * 8, 8), 8)
                chosen = self.row_selected_new(row).astype(jnp.int32)
                select.bounds[tile, :] = jnp.where(sub == row % 8, chosen, select.bounds[tile, :])

            copies = [self._debug_copy(outputs, name, source) for name, source in sources.items()]
            for copy in copies:
                copy.start()
            for copy in copies:
                copy.wait()

    def set_fields(self, values, v, exponent):
        """Store this rank's wire fields of every row (:func:`query_fields`;
        with ``heads`` > 1 a sequence of each per head) for
        :meth:`send_packet`, head ``k``'s in head slots ``3 k .. 3 k + 2``
        and the slots past them (B2's pad slot) zero; at chunked batches its
        :func:`field_packet` into slot ``member`` of its own field inbox for
        :meth:`send_fields` (:func:`field_persistent_scratch`'s layout)."""
        heads = list(zip(values, v, exponent)) if self.heads > 1 else [(values, v, exponent)]
        if len(heads) != self.heads:
            raise ValueError(f"{len(heads)} heads' fields for a rank of {self.heads} index heads")
        if not self.packet_fields:
            packets = [field_packet(*head) for head in heads]
            self.s.field_inbox[self.member] = packets[0] if self.heads == 1 else jnp.concatenate(packets, axis=0)
            return
        encode = fp8_fields if self.uint8_packets else bf16_fields
        tiles = [tile for head in heads for tile in encode(*head)]
        for slot in range(self.slots):
            self.s.fields[:, slot, :] = tiles[slot] if slot < len(tiles) else jnp.zeros_like(tiles[0])

    def send_fields(self):
        """Chunked batches: start this rank's field packet (slot ``member``
        of its inbox, :meth:`set_fields`) into slot ``member`` of each peer's
        inbox (when active; a rolled loop over the ``group - 1`` peers)."""
        inbox, semaphores = self.s.field_inbox, self.s.field_semaphores

        @pl.when(self.active)
        def _():
            @pl.loop(1, self.group)
            def send(offset):
                peer = (self.member + offset) % self.group
                pltpu.make_async_remote_copy(
                    inbox.at[self.member], inbox.at[self.member], semaphores.at[1], semaphores.at[0],
                    device_id=(self.base + peer,), device_id_type=pl.DeviceIdType.MESH,
                ).start()

    def receive_fields(self):
        """Chunked batches: wait for every peer's field packet and for this
        rank's to have left, then read every member's fields of each row (a
        rolled loop over rows) into the scorer's queries and weights (when
        active). A peer sends a layer's packet only after this rank's previous
        layer combined with it, so slots and semaphore counts never mix."""
        inbox, semaphores = self.s.field_inbox, self.s.field_semaphores
        members = self.group

        with self.scope("index_fields"):
            @pl.when(self.active)
            def _():
                peers = inbox.at[pl.ds(0, members - 1)]
                pltpu.make_async_copy(peers, peers, semaphores.at[0]).wait()
                pltpu.make_async_copy(peers, peers, semaphores.at[1]).wait()

                @pl.loop(0, self.batch)
                def row(r):
                    # head k of member m is query k members + m (TP32 rank order)
                    heads = [read_field_packet(inbox[:, pl.ds(k * self.batch + r if k else r, 1), :].reshape(members, 128))
                             for k in range(self.heads)]
                    if self.heads == 1:
                        values, weight = heads[0]
                    else:
                        values = jnp.concatenate([head[0] for head in heads], axis=0)
                        weight = jnp.concatenate([head[1] for head in heads], axis=0)
                    self.s.queries[r] = values.astype(jnp.bfloat16)
                    self.s.weights[r] = jnp.broadcast_to(weight, values.shape)

    def send_packet(self, dest, own, first, rows):
        """Store this rank's query packet ``own`` (``[slots rows, 640]`` FP32,
        batch rows ``first ..``; ``first`` may be traced) into ``dest`` as its
        wire encodes it, with the rows' index fields when active, else exactly
        as Kimi-K3 stores it (always at chunked batches, whose fields go on
        their own all-gather). Fields merge in 32-bit lanes before the cast."""
        uint8 = dest.dtype == jnp.uint8
        if not self.packet_fields:
            dest[...] = mla_dcp._pack_fp8_query(own) if uint8 else own.astype(jnp.bfloat16)
            return

        @pl.when(self.active)
        def with_fields():
            fields = self.s.fields[pl.ds(first, rows), pl.ds(0, self.slots), :].reshape(self.slots * rows, 128)
            fields = jnp.concatenate((jnp.zeros((self.slots * rows, 512), fields.dtype), fields), axis=1)
            if uint8:
                dest[...] = mla_dcp._pack_fp8_query(own, fields)
            else:
                lane = jax.lax.broadcasted_iota(jnp.int32, own.shape, 1)
                dest[...] = jnp.where(lane >= 512 + BF16_FIELD_LANE, fields, own).astype(jnp.bfloat16)

        @pl.when(jnp.logical_not(self.active))
        def plain():
            dest[...] = mla_dcp._pack_fp8_query(own) if uint8 else own.astype(jnp.bfloat16)

    def clean(self, query):
        """``query`` ``[heads, 640]`` FP32 from BF16 packets with the field
        lanes 576:640 zeroed when active (otherwise, and at chunked batches,
        unchanged)."""
        if not self.packet_fields:
            return query
        lane = jax.lax.broadcasted_iota(jnp.int32, query.shape, 1)
        first = jnp.where(self.active, 512 + BF16_FIELD_LANE, query.shape[1])
        return jnp.where(lane >= first, 0.0, query)

    def take_fields(self, gathered, first, rows, packet_row=None):
        """Every member's fields of batch rows ``first ..`` (``first`` may be
        traced) from a landed gather ``gathered`` ``[G, packet rows, 640]``
        into the scorer's queries and weights (when active); row ``first + j``'s
        head slots start at packet row ``slots (packet_row + j)``
        (``packet_row``, static, default ``first``)."""
        packet_row = first if packet_row is None else packet_row
        read = read_fp8_fields if self.uint8_packets else read_bf16_fields
        kind = jnp.int32 if self.uint8_packets else jnp.float32
        members = gathered.shape[0]

        with self.scope("index_fields"):
            @pl.when(self.active)
            def _():
                for j in range(rows):
                    heads = []
                    for k in range(self.heads):  # head k of member m is query k members + m
                        tiles = [gathered[:, pl.ds(self.slots * (packet_row + j) + 3 * k + h, 1), pl.ds(512, 128)]
                                 .astype(kind).reshape(members, 128) for h in range(3)]
                        heads.append(read(*tiles))
                    values = jnp.concatenate([head[0] for head in heads], axis=0)
                    weight = jnp.concatenate([head[1] for head in heads], axis=0)
                    self.s.queries[first + j] = values.astype(jnp.bfloat16)
                    self.s.weights[first + j] = jnp.broadcast_to(weight, values.shape)

    def prime(self):
        """Store each row's page count (``counts``) and start the first
        ``RING_DEPTH`` page DMAs (when active; call as early as the positions
        allow). Items are numbered in the order :meth:`score` takes the rows
        (:meth:`_row_at`), which the fetches follow: each ring slot's waits
        then meet its DMAs in order."""
        @pl.when(self.runs)
        def _():
            s, total = self.s, jnp.int32(0)
            for b in range(self.batch):
                s.counts[b] = self.counts[b]
            counts = [s.counts[self._row_at(i)] if self.order is not None else self.counts[i]
                      for i in range(self.batch)]
            for i in range(self.batch):
                s.starts[self._row_at(i)] = total
                total = total + counts[i]
            s.counts[self.batch], s.starts[self.batch], s.following[self.batch] = 0, total, self.batch
            following = jnp.int32(self.batch)
            for i in reversed(range(self.batch)):
                following = jnp.where(counts[i] > 0, i, following)
                s.following[i] = following
            s.cursor[0], s.cursor[1] = following, 0
            if "index_fetch" not in self.skip:
                if self.units:  # block-major: the first steps' items (one fetch body)
                    pl.loop(0, jnp.minimum(total, self.ahead * self.units))(lambda _: self._fetch())
                else:
                    for k in range(RING_DEPTH):
                        pl.when(k < total)(self._fetch)

    def _fetch(self):
        """Start the DMA of the page under the cursor into its item's ring
        slot and advance the cursor (the place of its row in :meth:`score`'s
        order, and the page): a row's pages in order, then the next row with
        pages (O(1) scalar work, no search)."""
        s = self.s
        at, page = s.cursor[0], s.cursor[1]
        row = self._row_at(at)
        item = s.starts[row] + page
        entry = page * self.group + self.owner if self.global_pages else page
        table_row = row if self.table_row is None else self.table_row(row)
        block = self.table_ref[table_row * self.pages + entry]
        slot = jax.lax.rem(item, self.ring_count)
        if "index_dma" in self.skip:  # no copy consumes them: a store keeps them (counts[batch] is never read)
            s.counts[self.batch] = block + slot
        else:
            pltpu.make_async_copy(s.cache.at[block, self.layer], s.ring.at[slot], s.semaphores.at[slot]).start()
        wrap = page + 1 >= s.counts[row]
        s.cursor[0] = jnp.where(wrap, s.following[at + 1], at)
        s.cursor[1] = jnp.where(wrap, 0, page + 1)

    def _cursor_item(self):
        """The item under the fetch cursor: the next :meth:`_fetch` starts
        it (the total once every item has started)."""
        s = self.s
        at = s.cursor[0]
        return jnp.where(at >= self.batch, s.starts[self.batch], s.starts[self._row_at(at)] + s.cursor[1])

    def _region(self, name):
        """``KernelConfig.phase_scopes``: an XProf scope ``name`` (else none)."""
        return jax.named_scope(name) if _config().phase_scopes else contextlib.nullcontext()

    def _row_at(self, i):
        """The row :meth:`score` takes ``i``-th: row ``i``, or with the order
        the row at slot ``i``."""
        return i if self.order is None else self.order[1, jnp.minimum(i, self.batch - 1)]

    def _selections(self):
        return self.selection if isinstance(self.selection, (list, tuple)) else [self.selection]

    def score_and_select(self):
        """Score every row, then (with ``index_debug``, after the scores' check
        copy: the select turns them into keys in place) select each row's
        top-k across the group (when the step scores). With two passes
        (:func:`select_rows`) each half of the rows is scored into the one
        scores view and selected by its own ``TopkSelect`` before the next."""
        selections = self._selections()
        if len(selections) > 1 and self.s.debug is not None:
            raise ValueError("index_debug takes one select pass: no halves")
        outputs = None if self.s.debug is None else self.s.debug[0]
        for g, selection in enumerate(selections):
            self.score(g * self.select_rows, self.select_rows if len(selections) > 1 else self.batch)
            if outputs is not None:

                @pl.when(self.active)
                def _():
                    copy = self._debug_copy(outputs, "scores", self.s.scores)
                    copy.start()
                    copy.wait()
            pl.when(self.runs)(lambda selection=selection: selection.run(self.s.scores))

    def _by_pass(self, row, pick):
        """``pick(selection, pass row)`` for ``row`` (traced): of its pass's
        select, computed for both passes and chosen when there are two."""
        selections = self._selections()
        slot = self._slot(row)
        if len(selections) == 1:
            return pick(selections[0], slot)
        rows = self.select_rows
        first = pick(selections[0], jnp.minimum(slot, rows - 1))
        second = pick(selections[1], jnp.maximum(slot - rows, 0))
        return jnp.where(slot < rows, first, second)

    def _slot(self, row):
        """Batch row ``row``'s slot: its virtual row (one part a row), or its
        place in the select's position order."""
        return row if self.order is None else self.order[0, row]

    def lanes(self, row, first, tokens: int):
        """Row ``row``'s selection of local tokens ``first ..`` (``dsa_select.TopkSelect.lanes``)."""
        return self._by_pass(row, lambda selection, r: selection.lanes(r, first, tokens))

    def new_score(self, row):
        """Row ``row``'s new token's score (traced row), lane-broadcast ``[1,
        128]`` FP32, as every rank computed it in :meth:`score`."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        tile = self.s.new_scores[pl.ds(pl.multiple_of(row // 8 * 8, 8), 8), :]
        return jnp.sum(jnp.where(sub == row % 8, tile, 0.0), axis=0, keepdims=True)

    def row_selected_new(self, row):
        """Whether row ``row``'s new token is among its selected tokens (a bool
        scalar): always for a row within the top-k, decided from its live
        position, since in a step that does not score the select's state is
        another layer's; else the select's decision in this step."""
        short = jnp.logical_not(self.position_ref[row] + 1 > INDEX_TOPK)
        score = self.new_score(row)
        return short | (self.runs & self._by_pass(row, lambda selection, r: selection.new_selected(r, score)))

    def rows_selected_new(self, first, rows: int):
        """Int32 ``[rows, 1]`` 0/1: :meth:`row_selected_new` of batch rows
        ``first .. first + rows`` (``first`` may be traced) with no scalar per
        row: whole 8-row tiles of the select's state when the rows are its
        virtual rows in batch order (one part a row, no order), else row by
        row, unrolled."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
        short = jnp.zeros((rows, 1), jnp.int32)
        for j in range(rows):
            short = jnp.where(sub == j, (self.position_ref[first + j] + 1 <= INDEX_TOPK).astype(jnp.int32), short)
        if self.order is None and self.parts == 1 and rows % 8 == 0:
            selections = self._selections()
            tiles = []
            for t in range(rows // 8):
                base = first + 8 * t
                tile = pl.ds(base if isinstance(base, int) else pl.multiple_of(base, 8), 8)
                score = self.s.new_scores[tile, :]
                if len(selections) == 1:
                    tiles.append(selections[0].new_selected_tile(base, score))
                    continue
                half = self.select_rows
                tiles.append(jnp.where(
                    base < half,
                    selections[0].new_selected_tile(jnp.minimum(base, half - 8), score),
                    selections[1].new_selected_tile(jnp.maximum(base - half, 0), score)))
            chosen = tiles[0] if len(tiles) == 1 else jnp.concatenate(tiles, axis=0)
        else:
            chosen = jnp.zeros((rows, 1), jnp.int32)
            for j in range(rows):
                score = self.new_score(first + j)
                chosen = jnp.where(
                    sub == j,
                    self._by_pass(first + j, lambda selection, r, score=score: selection.new_selected_lanes(r, score)),
                    chosen)
        return short | jnp.where(self.runs, chosen, 0)

    def score(self, first, rows: int):
        """Score the rows the scoring order (:meth:`_row_at`) takes ``first ..
        first + rows`` (``first`` may be traced, a multiple of 8 when ``rows``
        is) over their pages in the fetch order, each row's new token last
        (when active). Grouped (:meth:`_groups_new_tokens`): every sequence's
        pages, then each group of 8 rows' new tokens."""
        grouped = self._groups_new_tokens(rows)
        step = self.sequence_rows if grouped else 1  # a sequence's other rows have no pages
        with self.scope("index_score"):
            @pl.when(self.runs)
            def _():
                if self.units and grouped:
                    self._score_items(first, rows)

                    @pl.loop(0, rows // 8)
                    def new_tokens(g):
                        self._score_new_tokens(pl.multiple_of(first + 8 * g, 8))
                    return

                @pl.loop(0, rows // step)  # one row body: unrolled rows grow the program into instruction-memory overlays
                def row(j):
                    self._score_row(self._row_at(first + j * step), new_tokens=not grouped)

                if grouped:
                    @pl.loop(0, rows // 8)
                    def group(g):
                        with self._region("score_new"):
                            self._score_new_tokens(pl.multiple_of(first + 8 * g, 8))

    def _score_items(self, first, rows: int):
        """Block-major: the cached pages of the rows the scoring order takes
        ``first .. first + rows``, as (sequence, page) items in the fetch order,
        ``units`` items a loop step whatever sequences they belong to. A step
        tops the ring up to ``ahead`` steps beyond itself (each fetch into the
        slot of an item already scored), walks its items' rows and pages from
        the counts, issues every item's wait and reads, then scores each item
        as :meth:`_score_row`'s page loop does (the same arithmetic, so the same
        bits) and stores them last (masked to the item's rows). Items past the
        call's last score stale bytes and store nothing; the new tokens follow
        (:meth:`_score_new_tokens`)."""
        s, tokens, W, U = self.s, self.page_tokens, self.sequence_rows, self.units
        chunks = tokens // 128
        dots, store = "index_dots" not in self.skip, "index_store" not in self.skip
        sublane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
        last = first + rows
        total = s.starts[self.batch]
        item_lo = s.starts[self._row_at(first)]
        item_hi = jnp.where(last >= self.batch, total, s.starts[self._row_at(jnp.minimum(last, self.batch - 1))])
        s.cursor[2], s.cursor[3] = s.following[first], 0  # the walk: the scoring position and page of the next item

        @pl.loop(first, last)  # each row's local length (the scores' bound), once a call
        def length(at):
            row = self._row_at(at)
            s.cursor[4 + row] = local_length(self.position_ref[row], self.owner, tokens, self.group)

        @pl.loop(0, (item_hi - item_lo + U - 1) // U)
        def step(b):
            base = item_lo + U * b
            if "index_fetch" not in self.skip:
                limit = jnp.minimum(total, base + (self.ahead + 1) * U)
                pl.loop(0, jnp.clip(limit - self._cursor_item(), 0, 2 * U))(lambda _: self._fetch())
            items, at, page = [], s.cursor[2], s.cursor[3]
            for k in range(U):
                row = self._row_at(jnp.minimum(at, self.batch - 1))
                items.append((base + k < item_hi, row, page, jax.lax.rem(base + k, self.ring_count)))
                wrap = page + 1 >= s.counts[row]
                at = jnp.where(wrap, s.following[jnp.minimum(at + 1, self.batch)], at)
                page = jnp.where(wrap, 0, page + 1)
            s.cursor[2], s.cursor[3] = at, page
            if "index_dma" not in self.skip:
                for valid, _, _, slot in items:
                    copy = pltpu.make_async_copy(s.ring.at[slot], s.ring.at[slot], s.semaphores.at[slot])
                    pl.when(valid)(copy.wait)
            # The ring, queries, weights and scores may be views of one allocation (the decode stack's): every read
            # of the step goes before its stores, which would otherwise order the later reads behind them.
            reads = []
            for valid, row, page, slot in items:
                row = jnp.minimum(row, self.batch - W)  # a sequence's first row (clamped past the last item)
                held = s.ring.at[slot]
                exponents = held[pl.ds(tokens, dsa_cache.SCALE_ROWS), :].astype(jnp.int32) if dots else None
                keys = [pltpu.bitcast(held[pl.ds(128 * c, 128), :], jnp.float8_e4m3fn) for c in range(chunks)] if dots else []
                stacked = s.queries[pl.ds(row, W)].reshape(INDEX_HEADS * W, INDEX_DIM) if dots else None
                weights = s.weights[pl.ds(row, W)] if dots else None
                lengths = [s.cursor[4 + row + j] for j in range(W)]
                reads.append((row, exponents, keys, stacked, weights, lengths))
            out = []  # (target planes, chunk, value, store mask)
            for (valid, _, page, _), (row, exponents, keys, stacked, weights, lengths) in zip(items, reads, strict=True):
                vrow = self._score_vrow(row, 0)[1]
                sub_first, tile_base = vrow & 7, pl.multiple_of(vrow - (vrow & 7), 8)
                keep = (sublane >= sub_first) & (sublane < sub_first + W) & valid
                limits = _on_sublanes(lengths, [sub_first + j for j in range(W)])
                page = jnp.where(valid, page, 0)
                planes = s.scores.at[pl.ds(page * chunks, chunks), pl.ds(tile_base, 8), :]
                for c in range(chunks):
                    token = page * tokens + 128 * c
                    if dots:
                        assert stacked is not None and weights is not None and exponents is not None
                        logits = jax.lax.dot_general(stacked, keys[c], (((1,), (1,)), ((), ())),
                                                     preferred_element_type=jnp.float32)
                        partials = _head_partials(logits, [weights[j] for j in range(W)])
                        sums = _butterfly([partials[r % W] for r in range(8)])
                        value = jnp.where(token + lane < limits, sums * _scale(exponents[c:c + 1]), -jnp.inf)
                    else:
                        value = jnp.full((8, 128), -jnp.inf, jnp.float32)
                    out.append((planes, c, value, keep))
            if store:
                for planes, c, value, keep in out:
                    pltpu.store(planes.at[c], value, mask=keep)

    def _groups_new_tokens(self, rows: int) -> bool:
        """Whether a call of ``rows`` rows scores its new tokens per group of
        8 rows in the scoring order: whole groups, sequences within a group
        (the select's order keeps a sequence's rows together, in order), and a
        scores view of whole groups."""
        return (self.group_new_tokens and rows % 8 == 0 and 8 % self.sequence_rows == 0
                and self.select_rows % 8 == 0)

    def _score_new_tokens(self, at):
        """The new tokens of the 8 rows the scoring order takes ``at ..``, as
        :meth:`_score_earlier` and :meth:`_score_row` score and store them:
        one dot of the 8 rows' stacked queries against their 8 new keys (key
        row ``r`` is the ``r % 8``-th row's), each row's head sums over all of
        them with lane ``l`` scaled by token ``l % 8``'s exponent (every
        element the arithmetic of the row's dot against that token's broadcast
        key), then the new-token scores (one tile store when the rows are a
        tile's, else one masked store each) and each row's own and earlier
        rows' tokens stored on their pages' owners in position order."""
        s, tokens, W, rows = self.s, self.page_tokens, self.sequence_rows, range(8)
        dots, store = "index_dots" not in self.skip, "index_store" not in self.skip
        if self.order is None:
            batch_rows = [at + r for r in rows]
        else:  # whole sequences, each at a slot that is a multiple of W, its rows in order
            batch_rows = [self._row_at(at + W * (r // W)) + r % W for r in rows]
        sublane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
        if dots:
            keys = jnp.concatenate([self.new_value_ref[b, 0:1, :] for b in batch_rows], axis=0)
            logits = jax.lax.dot_general(jnp.concatenate([s.queries[b] for b in batch_rows], axis=0),
                                         jnp.concatenate([keys] * (128 // 8), axis=0),
                                         (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            partials = _head_partials(logits, [s.weights[b] for b in batch_rows])
            exponents = [self.new_value_ref[b, 1:2, :].astype(jnp.float32).astype(jnp.int32) for b in batch_rows]
            exponent = functools.reduce(lambda out, r: jnp.where(lane[0:1] % 8 == r, exponents[r], out), rows[:-1],
                                        exponents[-1])
            values = []
            for j in rows:
                total = partials[j]
                for shift in (4, 2, 1):
                    total = total + pltpu.roll(total, shift, 0)
                total = total * _scale(exponent)
                values.append([jnp.broadcast_to(total[0:1, t:t + 1], (8, 128)) for t in rows])
        else:
            values = [[jnp.full((8, 128), -jnp.inf, jnp.float32)] * 8 for _ in rows]
        if store and s.new_scores is not None:
            if self.order is None:
                s.new_scores[pl.ds(at, 8), :] = functools.reduce(
                    lambda out, j: jnp.where(sublane == j, values[j][j], out), rows[:-1], values[-1][-1])
            else:
                for j, b in enumerate(batch_rows):
                    pltpu.store(s.new_scores.at[pl.ds(pl.multiple_of(b // 8 * 8, 8), 8), :], values[j][j],
                                mask=sublane == b % 8)
        owner_scope = self._region("score_owner")
        owner_scope.__enter__()
        for j in rows:
            for t in range(j - j % W, j + 1):
                position = self.position_ref[batch_rows[t]]
                global_page = position // tokens

                @pl.when(global_page % self.group == self.owner)
                def owned(j=j, t=t, position=position, global_page=global_page):
                    local = global_page // self.group * tokens + position % tokens
                    if store:
                        self._store(batch_rows[j], local // 128 * 128,
                                    jnp.where(lane == local % 128, values[j][t], -jnp.inf), lane >= local % 128)
        owner_scope.__exit__(None, None, None)

    def _score_row(self, row, new_tokens: bool = True):
        """Row ``row``'s scores. A verify sequence's rows share its pages
        (``sequence_rows`` > 1): its first row streams them once and scores
        every row of the sequence over them, one MXU dot of the rows' stacked
        queries per 128 keys (:func:`_head_sums`); the other rows' page counts
        are 0. Each row then scores its earlier rows' tokens and its own
        (``new_tokens``; else :meth:`_score_new_tokens` does)."""
        setup = self._region("score_setup")
        setup.__enter__()
        s, tokens = self.s, self.page_tokens
        dots, store = "index_dots" not in self.skip, "index_store" not in self.skip
        queries, weights = s.queries[row], s.weights[row]
        position, start, total = self.position_ref[row], s.starts[row], s.starts[self.batch]
        rows = range(self.sequence_rows)
        # A non-first row's loop never runs; clamp its rows to the batch so no read leaves it.
        lengths = [local_length(self.position_ref[jnp.minimum(row + j, self.batch - 1)], self.owner, tokens, self.group)
                   for j in rows]
        lane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
        aligned = (self.sequence_rows == 1 or self._one_tile()) and self.parts == 1
        if aligned:
            # One part a row: the sequence's rows sit on sublanes first .. first + W - 1 of one tile row for every
            # token (_score_vrow), and token t is on plane t // 128 (constants of the sequence, outside the loop).
            vrow = self._score_vrow(row, 0)[1]
            first, base = vrow & 7, pl.multiple_of(vrow - (vrow & 7), 8)
            sublane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
            keep = (sublane >= first) & (sublane < first + self.sequence_rows)
            limits = _on_sublanes(lengths, [first + j for j in rows])
        setup.__exit__(None, None, None)

        @pl.loop(0, s.counts[row])
        def page(p):
            item = start + p
            slot = jax.lax.rem(item, RING_SLOTS)
            if "index_dma" not in self.skip:
                with self._region("score_wait"):
                    pltpu.make_async_copy(s.ring.at[slot], s.ring.at[slot], s.semaphores.at[slot]).wait()
            # The ring and the scores are views of one allocation: a page's reads go before its stores, which would
            # otherwise order each chunk's key loads behind the previous chunk's score store.
            exponents = s.ring[slot, pl.ds(tokens, dsa_cache.SCALE_ROWS), :].astype(jnp.int32)
            keys = [pltpu.bitcast(s.ring[slot, pl.ds(128 * c, 128), :], jnp.float8_e4m3fn)
                    for c in range(tokens // 128)] if dots else []
            if "index_fetch" not in self.skip:
                pl.when(item + RING_DEPTH < total)(self._fetch)  # into the slot item - 1 used
            if aligned:
                # Every row's partial head sum on the sublanes its sublane residue picks (W divides 8 and first is a
                # multiple of W), one butterfly for the sequence's sums, one mask and scale, one masked store.
                stacked = [s.queries[row + j] for j in rows] if dots else []
                sequence_weights = [s.weights[row + j] for j in rows] if dots else []
                for c in range(tokens // 128):
                    token = p * tokens + 128 * c
                    if dots:
                        logits = jax.lax.dot_general(jnp.concatenate(stacked, axis=0), keys[c],
                                                     (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                        partials = _head_partials(logits, sequence_weights)
                        sums = _butterfly([partials[r % self.sequence_rows] for r in range(8)])
                        value = jnp.where(token + lane < limits, sums * _scale(exponents[c:c + 1]), -jnp.inf)
                    else:
                        value = jnp.full((8, 128), -jnp.inf, jnp.float32)
                    if store:
                        target = self.s.scores.at[p * (tokens // 128) + c, pl.ds(base, 8), :]
                        if self.sequence_rows == 8:
                            target[...] = value
                        else:
                            pltpu.store(target, value, mask=keep)
                return
            if self._one_tile():
                # A sequence's rows share one tile of each 128 keys' scores: their sums go to their sublanes, then
                # one mask, scale and read-modify-write for all of them.
                for c in range(tokens // 128):
                    token = p * tokens + 128 * c
                    sublanes = [self._score_vrow(row + j, token)[1] % 8 for j in rows]
                    if dots:
                        sums = _head_sums([s.queries[row + j] for j in rows], [s.weights[row + j] for j in rows],
                                          keys[c])
                        value = jnp.where(token + lane < _on_sublanes(lengths, sublanes),
                                          _on_sublanes(sums, sublanes) * _scale(exponents[c:c + 1]), -jnp.inf)
                    else:
                        value = jnp.full((8, 128), -jnp.inf, jnp.float32)
                    if store:
                        self._store_tile(row, token, value, sublanes)
                return
            values = []  # [chunk][row of the sequence]
            for c in range(tokens // 128):
                if dots:
                    if self.sequence_rows == 1:
                        sums = [_head_sum(queries, weights, keys[c])]
                    else:
                        sums = _head_sums([s.queries[row + j] for j in rows], [s.weights[row + j] for j in rows],
                                          keys[c])
                    values.append([jnp.where(p * tokens + 128 * c + lane < lengths[j],
                                             sums[j] * _scale(exponents[c:c + 1]), -jnp.inf) for j in rows])
                else:
                    values.append([jnp.full((8, 128), -jnp.inf, jnp.float32) for _ in rows])
            if store:
                for c, by_row in enumerate(values):
                    for j, value in enumerate(by_row):
                        self._store(row + j, p * tokens + 128 * c, value)

        if not new_tokens:
            return
        if self.sequence_rows > 1:
            self._score_earlier(row, queries, weights, dots, store)
        global_page = position // tokens
        # The new token, not in the cache until the op commits it after the stack, is scored on every rank (the select
        # decides its flag everywhere) and stored on its page's owner.
        if dots:
            keys = jnp.broadcast_to(self.new_value_ref[row, 0:1, :], (128, INDEX_DIM))
            exponent = self.new_value_ref[row, 1:2, :].astype(jnp.float32).astype(jnp.int32)
            value = _head_sum(queries, weights, keys) * _scale(exponent)
        else:
            value = jnp.full((8, 128), -jnp.inf, jnp.float32)
        if store and s.new_scores is not None:
            tile = pl.ds(pl.multiple_of(row // 8 * 8, 8), 8)
            keep = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0) == row % 8
            s.new_scores[tile, :] = jnp.where(keep, value, s.new_scores[tile, :])

        @pl.when(global_page % self.group == self.owner)
        def new_token():
            local = global_page // self.group * tokens + position % tokens
            lane = jax.lax.broadcasted_iota(jnp.int32, value.shape, 1)
            if store:
                self._store(row, local // 128 * 128, jnp.where(lane == local % 128, value, -jnp.inf),
                            lane >= local % 128)

    def _score_earlier(self, row, queries, weights, dots: bool, store: bool):
        """Verify row ``row``'s earlier rows' tokens, in position order: each
        scored with the row's query from its key in ``new_value_ref`` and
        stored on its page's owner at its local token, over the stale entry
        the page loop left there, the rest of its 128 lanes cleared (the next
        token, if on the same lanes, rewrites its own)."""
        tokens = self.page_tokens

        @pl.loop(row - row % self.sequence_rows, row)
        def earlier(i):
            position = self.position_ref[i]
            global_page = position // tokens
            if dots:
                keys = jnp.broadcast_to(self.new_value_ref[i, 0:1, :], (128, INDEX_DIM))
                exponent = self.new_value_ref[i, 1:2, :].astype(jnp.float32).astype(jnp.int32)
                value = _head_sum(queries, weights, keys) * _scale(exponent)
            else:
                value = jnp.full((8, 128), -jnp.inf, jnp.float32)

            @pl.when(global_page % self.group == self.owner)
            def owned():
                local = global_page // self.group * tokens + position % tokens
                lane = jax.lax.broadcasted_iota(jnp.int32, value.shape, 1)
                if store:
                    self._store(row, local // 128 * 128, jnp.where(lane == local % 128, value, -jnp.inf),
                                lane >= local % 128)

    def _score_vrow(self, row, token):
        """``(plane, virtual row)`` of batch row ``row``'s local tokens
        ``token ..`` (``token`` a multiple of 128) in the scores: every score
        store goes through it. With one part a row (``parts`` 1) the plane is
        ``token // 128`` and a verify sequence's rows are consecutive virtual
        rows (the one-tile store relies on both; the select's row order keeps
        a sequence's rows together)."""
        slot = self._slot(row)
        return token % self.part_tokens // 128, slot % self.select_rows * self.parts + token // self.part_tokens

    def _one_tile(self) -> bool:
        """Whether a verify sequence's rows hold each 128 tokens' scores in one
        tile: rows in aligned groups of a divisor of eight, their parts on its
        sublanes (below eight rows a row's parts are ``8 / R`` apart)."""
        return (self.sequence_rows > 1 and 8 % self.sequence_rows == 0 and self.select_rows % self.sequence_rows == 0
                and self.sequence_rows * self.parts <= 8)

    def _store_tile(self, row, token, value, sublanes):
        """Write ``value``'s ``sublanes`` (traced) at the tile holding local
        tokens ``token ..`` of the sequence starting at batch row ``row``: one
        sublane-masked store, unmasked when they are the whole tile."""
        plane, vrow = self._score_vrow(row, token)
        base = pl.multiple_of(vrow // 8 * 8, 8)
        if len(sublanes) == 8:
            self.s.scores[plane, pl.ds(base, 8), :] = value
            return
        sub = jax.lax.broadcasted_iota(jnp.int32, value.shape, 0)
        keep = functools.reduce(jnp.logical_or, [sub == at for at in sublanes])
        pltpu.store(self.s.scores.at[plane, pl.ds(base, 8), :], value, mask=keep)

    def _store(self, row, token, value, lanes=None):
        """Write ``value`` (``[8, 128]``, every sublane equal) at batch row
        ``row``'s local tokens ``token .. token + 128`` (``token`` a multiple of
        128), only ``lanes`` if given: one masked store into its virtual row's
        tile."""
        plane, vrow = self._score_vrow(row, token)
        base = pl.multiple_of(vrow // 8 * 8, 8)
        keep = jax.lax.broadcasted_iota(jnp.int32, value.shape, 0) == vrow % 8
        if lanes is not None:
            keep = keep & lanes
        pltpu.store(self.s.scores.at[plane, pl.ds(base, 8), :], value, mask=keep)


def score_calls(scorer, rows: int, chunk: int):
    """Prime ``scorer`` and score its ``rows`` in calls of ``chunk`` rows: one
    call, or a rolled loop over several (one traced body), as the decode
    layer's chunked path calls it. Its attend path calls once per row group
    from a Python loop instead, so at B2 (two one-row groups) it traces two
    scorer bodies where this traces one."""
    scorer.prime()
    if rows == chunk:
        scorer.score(0, rows)
    else:
        @pl.loop(0, rows // chunk)
        def calls(c):
            scorer.score(c * chunk, chunk)


def score_rows(cache, table, positions, queries, weights, new_values, *, layer: int, page_tokens: int, group: int,
               owner: int, chunk: int, global_pages: bool = False, skip: tuple[str, ...] = (), scope: Any = None,
               order=None, interpret=None):
    """The scorer as a standalone call (tests): every row of the ``table``
    ``[rows, pages]`` (local, or global with ``global_pages``) at
    ``positions``, with the queries ``[rows, 32, 128]`` BF16, the
    lane-broadcast weights ``[rows, 32, 128]`` FP32 and the new tokens'
    encoded keys ``new_values`` ``[rows, 3, 128]`` BF16 in place, scored in
    calls of ``chunk`` rows as the decode layer makes them (a rolled loop over
    several): the scores (:func:`score_shape`). ``skip`` and ``scope`` as
    :class:`IndexScorer`'s; ``order``: ``dsa_select.sort_rows``'s ``[3, rows]``
    table (rows scored and stored in slot order); ``interpret``: Pallas TPU
    interpret mode."""
    rows, pages = table.shape
    if rows % chunk:
        raise ValueError("chunk must divide the rows")
    held = -(-pages // group) if global_pages else pages  # the most local pages any rank holds
    shapes = scratch_shapes(rows, held * page_tokens, page_tokens, False)

    def kernel(table_ref, position_ref, cache_ref, queries_ref, weights_ref, new_value_ref, order_ref, scores_ref,
               ring, *scoped):
        scratch = IndexScratch(cache_ref, None, ring, queries_ref, weights_ref, None, scores_ref, *scoped)
        scorer = IndexScorer(scratch, position_ref=position_ref, batch=rows, active=position_ref[0] >= 0,
                             table_ref=table_ref, pages=pages, layer=layer, page_tokens=page_tokens, group=group,
                             global_pages=global_pages, owner=owner, slots=3, uint8_packets=False,
                             new_value_ref=new_value_ref,
                             counts=[local_pages(position_ref[b], owner, page_tokens, group) for b in range(rows)],
                             skip=skip, scope=scope, order=None if order is None else order_ref)
        score_calls(scorer, rows, chunk)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    order_table = np.zeros((3, max(rows, 8)), np.int32) if order is None else np.asarray(order, np.int32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shapes["scores"][0], jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM), vmem, vmem, vmem,
                      pl.BlockSpec(memory_space=pltpu.SMEM)],
            out_specs=vmem,
            scratch_shapes=(pltpu.VMEM(*shapes["ring"]), *scoped_scratch(rows, False, shapes["ring"][0][0])),
        ),
        interpret=interpret,
        name="k31_index_scores",
    )(jnp.asarray(table, jnp.int32).reshape(-1), positions, cache, queries, weights, new_values,
      jnp.asarray(order_table))
