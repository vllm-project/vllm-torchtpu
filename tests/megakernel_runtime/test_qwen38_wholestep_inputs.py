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
"""The whole-step compressed-key page table addresses what the stock selection reads.

The stock `pooled_qsa.select_blocks` expands each compressed block-table column to
`pool.shape[1] // 128` kernel pages of 32 groups, `column * physical_pages + slot * that + j`.
Served pools have 512-token blocks (4 pages per column); the whole-step inputs once mapped one
page per column, which only selects correctly while a row selects all its groups (contexts up
to 4 * budget tokens): w07ar real AL 2.59 vs ~3.3 at AgentX contexts.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_inputs as wsi
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import (
    carrier_pages_per_block,
    qsa_step_inputs,
)


def _stock_pages(comp, pool, slot):
    """Stock expand_pages over a request's compressed columns."""
    physical = carrier_pages_per_block(pool)
    logical = pool.shape[1] // 128
    first = np.arange(logical) + slot * logical
    return (np.asarray(comp)[:, :, None] * physical + first).reshape(comp.shape[0], -1)


@pytest.mark.parametrize("block_tokens", [128, 512])
@pytest.mark.parametrize("slot", [0, 1])
def test_compressed_pages_match_stock_expand(block_tokens, slot, monkeypatch):
    # The ring/tag read is a TPU kernel; the page tables do not depend on it.
    monkeypatch.setattr(
        wsi, "read_ring_and_tags",
        lambda pool, ids, n, m: (jnp.zeros((len(ids), n), jnp.bfloat16), None),
    )
    pool = jnp.zeros((64, block_tokens, 1, 4, 256), jnp.float8_e4m3fn)
    nreq, T, max_groups = 2, 4, 1024
    n_ck = max_groups // 32
    cols = -(-n_ck // (block_tokens // 128))
    rng = np.random.default_rng(0)
    comp = rng.integers(1, 60, size=(nreq, cols)).astype(np.int32)
    main = rng.integers(1, 60, size=(nreq, 64)).astype(np.int32)
    ring = np.ones((nreq, 1), np.int32)
    tables = [(jnp.asarray(ring), jnp.asarray(main), jnp.asarray(comp))]
    out = qsa_step_inputs(
        pool, tables, [(-1, slot)], jnp.zeros((4096, 128), jnp.float32),
        jnp.arange(nreq * T, dtype=jnp.int32), jnp.arange(0, nreq * T + 1, T, dtype=jnp.int32),
        jnp.asarray([3000, 100], jnp.int32), M=nreq * T, T=T, pages_per_seq=64,
        max_groups=max_groups,
    )
    want = _stock_pages(comp, pool, slot)[:, :n_ck]
    np.testing.assert_array_equal(np.asarray(out["qsa_cktab"]), want)
    if block_tokens == 512:  # positive control: one page per column (the old mapping) differs
        old = comp[:, :n_ck] * carrier_pages_per_block(pool) + slot
        assert not np.array_equal(old, want[:, : old.shape[1]])


def test_long_request_tables_match_stock_selection_addressing(monkeypatch):
    """End to end for a served-shaped long request: a 100K-token context (25,000 groups over
    196 compressed blocks of 512 tokens, cap 131072 = 1024 kernel pages), packed slot 1, and a
    second short request. Every visible group's kernel page in the whole-step table is the one
    the stock selection reads (its expand_pages of the same block table)."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages

    monkeypatch.setattr(
        wsi, "read_ring_and_tags",
        lambda pool, ids, n, m: (jnp.zeros((len(ids), n), jnp.bfloat16), None),
    )
    pool = jnp.zeros((4096, 512, 1, 4, 256), jnp.float8_e4m3fn)
    nreq, T, max_groups, slot = 2, 4, 32768, 1
    ctx = np.array([100_000, 3_000], np.int32)
    blocks = -(-ctx // 512)
    rng = np.random.default_rng(1)
    width = 260  # the runner's table is wider than the cap needs (256 columns)
    comp = np.zeros((nreq, width), np.int32)
    for q in range(nreq):
        comp[q, : blocks[q]] = rng.choice(np.arange(1, 4096), size=blocks[q], replace=False)
    tables = [(jnp.ones((nreq, 1), jnp.int32), jnp.ones((nreq, 1024), jnp.int32), jnp.asarray(comp))]
    out = qsa_step_inputs(
        pool, tables, [(0, slot)], jnp.zeros((200_000, 128), jnp.float32),
        jnp.arange(nreq * T, dtype=jnp.int32), jnp.arange(0, nreq * T + 1, T, dtype=jnp.int32),
        jnp.asarray(ctx), M=nreq * T, T=T, pages_per_seq=1024, max_groups=max_groups,
        layer_major_tables=True,
    )
    cktab = np.asarray(out["qsa_cktab"])[0]  # layer 0: [requests, kernel pages]
    for q in range(nreq):
        stock = np.asarray(compressed_kernel_pages(jnp.asarray(comp[q]), pool, slot))
        n_pages = -(-(ctx[q] // 4) // 32)
        np.testing.assert_array_equal(cktab[q, :n_pages], stock[:n_pages])


def test_shared_addressing_is_the_former_inline_stock_formula():
    """compressed_kernel_pages is byte-for-byte select_blocks' former inline expand_pages."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages

    for shape in ((16, 512, 1, 4, 256), (16, 128, 1, 4, 256)):
        pool = jnp.zeros(shape, jnp.float8_e4m3fn)
        physical = carrier_pages_per_block(pool)
        logical = shape[1] // 128
        pages = jnp.asarray([[3, 7, 0], [1, 2, 5]], jnp.int32)
        for slot in (0, 1):
            first = jnp.arange(logical) + (slot * logical if slot else 0)
            want = (pages[..., None] * physical + first).reshape(-1)
            np.testing.assert_array_equal(compressed_kernel_pages(pages, pool, slot), want)
