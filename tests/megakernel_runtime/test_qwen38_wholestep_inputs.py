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
`pool.shape[1] // 128` kernel pages of 32 groups, `column * physical_pages + slot * that
+ j`.
Served pools have 512-token blocks (4 pages per column); the whole-step inputs once
mapped one
page per column, which only selects correctly while a row selects all its groups
(contexts up
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
        wsi,
        "read_ring_and_tags",
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
        pool,
        tables,
        [(-1, slot)],
        jnp.zeros((4096, 128), jnp.float32),
        jnp.arange(nreq * T, dtype=jnp.int32),
        jnp.arange(0, nreq * T + 1, T, dtype=jnp.int32),
        jnp.asarray([3000, 100], jnp.int32),
        M=nreq * T,
        T=T,
        pages_per_seq=64,
        max_groups=max_groups,
    )
    want = _stock_pages(comp, pool, slot)[:, :n_ck]
    np.testing.assert_array_equal(np.asarray(out["qsa_cktab"]), want)
    if (
        block_tokens == 512
    ):  # positive control: one page per column (the old mapping) differs
        old = comp[:, :n_ck] * carrier_pages_per_block(pool) + slot
        assert not np.array_equal(old, want[:, : old.shape[1]])


def test_long_request_tables_match_stock_selection_addressing(monkeypatch):
    """End to end for a served-shaped long request: a 100K-token context (25,000 groups
    over
    196 compressed blocks of 512 tokens, cap 131072 = 1024 kernel pages), packed slot 1,
    and a
    second short request. Every visible group's kernel page in the whole-step table is
    the one
    the stock selection reads (its expand_pages of the same block table)."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages

    monkeypatch.setattr(
        wsi,
        "read_ring_and_tags",
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
        comp[q, : blocks[q]] = rng.choice(
            np.arange(1, 4096), size=blocks[q], replace=False
        )
    tables = [
        (
            jnp.ones((nreq, 1), jnp.int32),
            jnp.ones((nreq, 1024), jnp.int32),
            jnp.asarray(comp),
        )
    ]
    out = qsa_step_inputs(
        pool,
        tables,
        [(0, slot)],
        jnp.zeros((200_000, 128), jnp.float32),
        jnp.arange(nreq * T, dtype=jnp.int32),
        jnp.arange(0, nreq * T + 1, T, dtype=jnp.int32),
        jnp.asarray(ctx),
        M=nreq * T,
        T=T,
        pages_per_seq=1024,
        max_groups=max_groups,
        layer_major_tables=True,
    )
    cktab = np.asarray(out["qsa_cktab"])[0]  # layer 0: [requests, kernel pages]
    for q in range(nreq):
        stock = np.asarray(compressed_kernel_pages(jnp.asarray(comp[q]), pool, slot))
        n_pages = -(-(ctx[q] // 4) // 32)
        np.testing.assert_array_equal(cktab[q, :n_pages], stock[:n_pages])


def test_shared_addressing_is_the_former_inline_stock_formula():
    """compressed_kernel_pages is byte-for-byte select_blocks' former inline
    expand_pages."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_kernel_pages

    for shape in ((16, 512, 1, 4, 256), (16, 128, 1, 4, 256)):
        pool = jnp.zeros(shape, jnp.float8_e4m3fn)
        physical = carrier_pages_per_block(pool)
        logical = shape[1] // 128
        pages = jnp.asarray([[3, 7, 0], [1, 2, 5]], jnp.int32)
        for slot in (0, 1):
            first = jnp.arange(logical) + (slot * logical if slot else 0)
            want = (pages[..., None] * physical + first).reshape(-1)
            np.testing.assert_array_equal(
                compressed_kernel_pages(pages, pool, slot), want
            )


@pytest.mark.parametrize("layer_major", [False, True])
def test_layer_batched_inputs_match_the_per_layer_loop(layer_major, monkeypatch):
    """VLLM_TPU_MK_WHOLE_BATCHED_INPUTS: every output equals the per-layer loop's, for
    several layers with their own tables and slots, an idle request and a fresh one."""
    from vllm_torchtpu import envs

    def fake_read(pool, ids, n, m):  # ring rows that name the page they were read from
        rows = ids.astype(jnp.float32)[:, None] + jnp.arange(n, dtype=jnp.float32)[None] / n
        return rows.astype(jnp.bfloat16), None

    monkeypatch.setattr(wsi, "read_ring_and_tags", fake_read)
    pool = jnp.zeros((256, 512, 1, 4, 256), jnp.float8_e4m3fn)
    n_req, nreq, T, max_groups, L = 3, 4, 4, 1024, 5
    rng = np.random.default_rng(2)
    tables = [
        (
            jnp.asarray(rng.integers(1, 250, size=(n_req, 1)), jnp.int32),
            jnp.asarray(rng.integers(1, 250, size=(n_req, 40)), jnp.int32),
            jnp.asarray(rng.integers(1, 250, size=(n_req, 6)), jnp.int32),
        )
        for _ in range(L)
    ]
    slots = [(o % 2 if o % 3 else -1, o % 2) for o in range(L)]
    starts = jnp.asarray([0, 4, 4, 8], jnp.int32)  # request 1 idle
    args = (
        pool,
        tables,
        slots,
        jnp.asarray(rng.standard_normal((8192, 128)), jnp.bfloat16),
        jnp.arange(nreq * T, dtype=jnp.int32),
        starts,
        jnp.asarray([3000, 100, 0], jnp.int32),  # request 2 fresh (no ring yet)
    )
    kw = dict(M=nreq * T, T=T, pages_per_seq=64, max_groups=max_groups,
              layer_major_tables=layer_major)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", False, raising=False)
    want = qsa_step_inputs(*args, **kw)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", True, raising=False)
    got = qsa_step_inputs(*args, **kw)
    assert want.keys() == got.keys()
    for k in want:
        a, b = np.asarray(want[k]), np.asarray(got[k])
        assert a.shape == b.shape and a.dtype == b.dtype, k
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=k)


@pytest.mark.parametrize("n_blocks", [None, 50])
def test_layer_batched_gdn_blocks_match_the_per_layer_loop(n_blocks, monkeypatch):
    """VLLM_TPU_MK_WHOLE_BATCHED_INPUTS: gdn_blocks' vmapped pass equals the per-layer
    loop, padding / idle / negative / out-of-range ids included."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks

    rng = np.random.default_rng(3)
    n, NS, slots = 3, 4, 9
    meta = []
    off = jnp.asarray(rng.integers(0, 5, size=slots), jnp.int32)  # the runner's one table
    for _ in range(6):
        meta += [
            jnp.asarray(rng.integers(-1, slots, size=n), jnp.int32),  # state indices
            off,  # read offsets (every mamba group's metadata holds the same tensor)
            jnp.asarray(rng.integers(-3, 64, size=(n, 5)), jnp.int32),  # ckpt ids
            jnp.asarray([0, 4, 4, 8], jnp.int32),  # query_start_loc (request 1 idle)
            jnp.asarray(rng.integers(0, 9, size=n), jnp.int32),  # seq lens
        ]
    for write in (True, False):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", False, raising=False)
        want = gdn_blocks(meta, NS, write, n_blocks)
        monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", True, raising=False)
        got = gdn_blocks(meta, NS, write, n_blocks)
        for a, b in zip(want, got, strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
