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
"""CPU-interpret harness for the SparseCore-layout sparse MLA kernel.

Lets kernel tests run `sparse_ragged_paged_attention` with the SparseCore
cache layout on CPU (Pallas TPU interpret mode) by replacing the SparseCore
gather with a CPU stand-in, and builds a compact SparseCore cache problem.
"""

from __future__ import annotations

import contextlib
import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import (  # noqa: E402
    native_sc_cache_layout,
    native_sc_gather,
)


def dsa_gather_cpu_interpret(
    nope_cache,
    rope_cache,
    indices,
    *,
    out_size=None,
    atoms_per_batch=16,
    num_live_indices=None,
    live_multiple=1,
    live_vec=None,
    live_vec_offset=None,
    nope_words=128,
    **_unused,
):
    """CPU stand-in for `native_sc_gather.dsa_gather_native_sc`.

    Gathers NoPE rows and banded RoPE quads for `indices`, and poisons rows
    past the gathered live bound with `0x7F7F7F7F`. It re-derives the live
    bound and occupancy math of the SparseCore gather. It only needs to be
    self-consistent: tests compare two kernel paths that both read through
    this stand-in, so drift in its live-bound math cannot hide a word-layout
    or merge bug.
    """
    if out_size is None:
        out_size = indices.size
    num_cores, num_subcores, atoms_per_batch, occ, num_batches = (
        native_sc_gather.live_vec_geometry(out_size, atoms_per_batch)
    )
    num_owners = num_cores * num_subcores
    guard_tail = live_multiple % (num_owners * occ) != 0
    if live_vec_offset is not None:
        lv = indices.reshape(-1)[
            live_vec_offset : live_vec_offset + 2 * native_sc_gather.NUM_LANES
        ]
        eff = lv[0] if guard_tail else lv[native_sc_gather.NUM_LANES] * num_owners
        gathered = jnp.minimum(eff * occ, out_size)
    elif live_vec is not None:
        eff = (
            live_vec[0]
            if guard_tail
            else live_vec[native_sc_gather.NUM_LANES] * num_owners
        )
        gathered = jnp.minimum(eff * occ, out_size)
    elif num_live_indices is not None:
        lb = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occ - 1) // occ, 0, num_batches
        )
        ob = (lb + num_owners - 1) // num_owners
        gathered = jnp.minimum((lb if guard_tail else ob * num_owners) * occ, out_size)
    else:
        gathered = jnp.int32(out_size)
    poison = np.uint32(0x7F7F7F7F)
    flat_idx = indices.reshape(-1)[:out_size]
    nope_flat = nope_cache.reshape(-1, nope_words)
    rope_flat = rope_cache.reshape(-1, native_sc_gather.ROPE_WORDS)
    safe = jnp.clip(flat_idx, 0, nope_flat.shape[0] - 1)
    nope_out = jnp.where(
        jnp.arange(out_size, dtype=jnp.int32)[:, None] < gathered,
        nope_flat[safe],
        poison,
    )
    quads = rope_flat[safe].reshape(
        out_size // native_sc_gather.ROPE_PACKING,
        native_sc_gather.ROPE_PACKING,
        native_sc_gather.ROPE_WORDS,
    )
    bands = native_sc_gather.transpose_quad(
        [quads[:, s, :] for s in range(native_sc_gather.ROPE_PACKING)]
    )
    rope_out = jnp.concatenate(bands, axis=-1)
    rope_out = jnp.where(
        jnp.arange(rope_out.shape[0], dtype=jnp.int32)[:, None]
        < (gathered // native_sc_gather.ROPE_PACKING),
        rope_out,
        poison,
    )
    return nope_out, rope_out


@contextlib.contextmanager
def cpu_interpret_context():
    """Runs the sparse MLA kernel in Pallas TPU interpret mode on CPU.

    Patches the chip info to TPU7x and the SparseCore gather with
    `dsa_gather_cpu_interpret`. On a TPU it does nothing.
    """
    if jax.devices()[0].platform == "tpu":
        yield
        return
    pltpu.reset_tpu_interpret_mode_state()
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=info),
        mock.patch.object(
            native_sc_gather, "dsa_gather_native_sc", dsa_gather_cpu_interpret
        ),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield


def compact_problem(query_tokens: int, seq_q: tuple[int, ...], heads: int) -> dict:
    """Compact 2048-token SparseCore cache problem with empty (-inf LSE),
    capped-overflow (>512), and mid-range live rows."""
    rng = np.random.default_rng(2026 + query_tokens * 7 + heads)
    page_size = 256
    cache_tokens = 2048
    pages = cache_tokens // page_size
    topk = 2048

    def quantized_bytes(dim: int) -> np.ndarray:
        vals = rng.random((pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(vals / prologue.K_SCALE).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope = quantized_bytes(prologue.NOPE_DIM).reshape(pages, page_size, 4, 128)
    rope = np.pad(
        quantized_bytes(prologue.ROPE_DIM),
        ((0, 0), (0, 0), (0, prologue.ROPE_DIM)),
    ).reshape(pages, page_size // 4, 4, 128)

    page_table = rng.permutation(pages).astype(np.int32)
    inv = np.argsort(page_table)
    physical = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    )
    logical = (inv[physical // page_size] * page_size + physical % page_size).astype(
        np.int32
    )
    counts = np.resize(np.array([0, 640, 129, 257, 512, 65, 384, 700]), query_tokens)
    counts[0] = 0  # empty row -> lse = -inf (0xFF800000)
    counts[min(1, query_tokens - 1)] = 640  # > topk_cap=512 -> overflow fix-up
    live = np.arange(topk)[None, :] < counts[:, None]
    topk_idx = np.where(live, logical, -1).astype(np.int32)

    num_seqs = len(seq_q)
    tables = [page_table] + [
        rng.permutation(pages).astype(np.int32) for _ in range(max(num_seqs - 1, 0))
    ]
    num_decode = sum(1 for n in seq_q if n <= 4)
    q_raw = jnp.asarray(
        rng.normal(size=(query_tokens, heads, prologue.RAW_HEAD)).astype(np.float32),
        jnp.bfloat16,
    )
    ql_nope = jnp.asarray(
        rng.normal(size=(query_tokens, heads, prologue.NOPE_DIM)).astype(np.float32),
        jnp.bfloat16,
    )
    return dict(
        nope=jnp.asarray(native_sc_cache_layout.pack_nope(nope)),
        rope=jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope)),
        topk=jnp.asarray(topk_idx),
        page_table=jnp.asarray(np.concatenate(tables).astype(np.int32)),
        cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(seq_q)]), jnp.int32),
        distribution=jnp.array([num_decode, num_decode, num_seqs], jnp.int32),
        q_raw=q_raw,
        ql_nope=ql_nope,
        positions=jnp.asarray(
            rng.integers(0, prologue.MAX_POS, size=query_tokens), jnp.int32
        ),
    )
