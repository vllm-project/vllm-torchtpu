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
"""Whole-step kernel QSA inputs from the runner's metadata, with the stock addressing.

Builds, inside the whole-step JAX callable and before the kernel, the stage-C extra inputs
(`wholestep.inputs.EXTRA_NAMES_C`) of every QSA layer from what the stock `_pooled_qsa`
receives: the unified pool, per-layer `(ring_ids, main_table, compressed_table)` flat block
tables, `(main_slot, compressed_slot)`, the rope cache, positions, `query_start_loc` and the
context lengths. Every address is the stock formula: main KV page = `main_table[req, col]`
(one pool block of 128 tokens per column), compressed-key carrier page =
`compressed_table[req, col] * physical_pages + slot` (`pooled_qsa.select_blocks`), ring and
tags read with `qsa_state.read_ring_and_tags`.

Rows: request `q` owns kernel rows `q*T .. q*T+T-1` (uniform verify windows, as the
eligibility check guarantees); per-layer tables are flattened `row = layer * nreq + q`.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags

RING_ROWS = 8  # newest positions kept per request (lane = position % 8), >= ratio
IDX_DIM = 128
GROUP = 4  # compress ratio


def _fit(x: Any, n: int) -> Any:
    """Rows (requests / positions) padded or cropped to the kernel's static count. Cropping only
    drops padding rows: the driver calls the kernel for steps whose requests fit its bucket."""
    x = x[:n]
    if x.shape[0] < n:
        x = jnp.pad(x, ((0, n - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))
    return x


def carrier_pages_per_block(pool: Any) -> int:
    """Compressed-key carrier pages (32 groups of 2 x 128 BF16 lanes) per pool block."""
    rows, width = 128 // GROUP, pool.shape[-1]
    return math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * width * 2)


def qsa_step_inputs(
    pool: Any,
    tables: list[tuple[Any, Any, Any]],
    slots: list[tuple[int, int]],
    trig: Any,
    positions: Any,
    starts: Any,
    context: Any,
    *,
    M: int,
    T: int,
    pages_per_seq: int,
    max_groups: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    layer_major_tables: bool = False,
) -> dict[str, Any]:
    """`{name: array}` of the stage-C QSA extras for `len(tables)` QSA layers.

    tables[o]: layer o's flat `(ring_ids, main_table, compressed_table)` as the runner hands
    them to the stock op; slots[o]: its `(main_slot, compressed_slot)` (-1 main slot = whole
    rows: head-pair offset 0). `layer_major_tables`: page tables `[layers, requests, cols]`
    (the indexer's `hbm_tables` form) instead of `[layers * requests, cols]`.
    """
    n_req = len(context)
    nreq = M // T
    ctx = _fit(context.astype(jnp.int32), nreq)
    q_len = _fit((starts[1:] - starts[:-1]).astype(jnp.int32), nreq)
    live = q_len > 0
    # Rows: lengths seen by verify row t of request q = context + t + 1 (0 for idle requests).
    t = jnp.arange(T, dtype=jnp.int32)
    lens = jnp.where(live[:, None], ctx[:, None] + t[None] + 1, 0).reshape(M)
    n_ck = max_groups // (128 // GROUP)
    # Carrier (kernel) pages per block-table column: a pool block of `pool.shape[1]` tokens
    # holds `pool.shape[1] // 128` pages of 32 groups for this layer (served 512-token blocks:
    # 4). Addressed by `pooled_qsa.compressed_kernel_pages`, the stock selection's own mapping.
    lp = pool.shape[1] // 128
    mtab, cktab, rings = [], [], []
    for (ring_ids, main, comp), (_, cslot) in zip(tables, slots, strict=True):
        main = _fit(main.reshape(n_req, -1).astype(jnp.int32), nreq)
        comp = _fit(comp.reshape(n_req, -1).astype(jnp.int32), nreq)
        ids = _fit(ring_ids.reshape(n_req, -1)[:, 0].astype(jnp.int32), nreq)
        ids = jnp.where(live, ids, 0)
        # Columns past `pages_per_seq` / the compressed width only address contexts past
        # `wholecap`, which the driver routes to stock (`_whole_eligible`); padding covers
        # tables narrower than the cap.
        mtab.append(jnp.pad(main, ((0, 0), (0, max(0, pages_per_seq - main.shape[1]))))[:, :pages_per_seq])
        cols = -(-n_ck // lp)
        comp = jnp.pad(comp, ((0, 0), (0, max(0, cols - comp.shape[1]))))[:, :cols]
        pages = compressed_kernel_pages(comp, pool, cslot, GROUP).reshape(nreq, cols * lp)
        cktab.append(pages[:, :n_ck].astype(jnp.int32))
        ring, _ = read_ring_and_tags(pool, ids, RING_ROWS * IDX_DIM, RING_ROWS * 2)
        ring = ring.reshape(nreq, RING_ROWS, IDX_DIM)
        rings.append(jnp.where((ctx > 0)[:, None, None], ring, 0).astype(jnp.bfloat16))
    # Gather the rows, then widen: converting the whole [max_model_len, 128] table every step cost
    # 64 us (xprof j-1e002dca, convert f32[262144,128]); same values.
    def trig32(rows):
        return trig[rows].astype(jnp.float32)

    pos = _fit(positions.astype(jnp.int32), M)
    # First position of the group a verify window of T = 4 completes (stock: the group's
    # first member's position, from the step or from the ring tags).
    j0 = ctx + (GROUP - 1 - ctx % GROUP) % GROUP - (GROUP - 1)
    return {
        "qsa_pool": pool,
        "qsa_ck": pool.reshape(-1, 128 // GROUP // 2, pool.shape[-2], pool.shape[-1]),
        "qsa_cktab": jnp.stack(cktab) if layer_major_tables else jnp.concatenate(cktab),
        "qsa_nvis": lens // GROUP,
        "qsa_tail": lens % GROUP,
        "qsa_mtab": jnp.stack(mtab) if layer_major_tables else jnp.concatenate(mtab),
        "qsa_off": jnp.asarray([max(m, 0) for m, _ in slots], jnp.int32),
        "qsa_scales": jnp.asarray([k_scale, v_scale], jnp.float32),
        "qsa_rope": trig32(pos),
        "qsa_ctx": ctx,
        "qsa_ring": jnp.concatenate(rings),
        "qsa_rope_ck": trig32(jnp.maximum(j0, 0))[:, None, :],
    }
