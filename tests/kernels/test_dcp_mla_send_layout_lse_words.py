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
"""CPU-interpret regression test for `build_send_lse_words` when `T < W`.

When `num_tokens < send_world`, `send_rows = num_tokens * (num_q_heads //
send_world) < num_q_heads` (for example `T=4, W=8, H=64 -> send_rows=32 < 64`
and `T=4, W=8, H=32 -> send_rows=16 < 32`). `build_send_lse_words` must keep
the block-reversed `hi` (`[0, num_q_heads)`) and `lo` (`[send_rows, send_rows +
num_q_heads)`) LSE word sources separate instead of packing them into a single
shared row where lanes `[send_rows, num_q_heads)` overwrite `lo` with `hi`.
"""

from __future__ import annotations

import contextlib
import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(__file__))

import test_dcp_mla_send_layout as send_layout_test  # noqa: E402
import test_mla_decode_prologue as prologue  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import (  # noqa: E402
    native_sc_cache_layout,
    native_sc_gather,
)

PAD = sparse_mla.SEND_PAD_ROWS


def _dsa_gather_cpu_interpret(
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
def _cpu_interpret_context():
    if jax.devices()[0].platform == "tpu":
        yield
        return
    pltpu.reset_tpu_interpret_mode_state()
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=info),
        mock.patch.object(
            native_sc_gather, "dsa_gather_native_sc", _dsa_gather_cpu_interpret
        ),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield


def _compact_problem(query_tokens: int, seq_q: tuple[int, ...], heads: int) -> dict:
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


@pytest.mark.parametrize(
    ("world", "heads", "cell"),
    [
        (8, 64, (4, (3,))),
        (8, 32, (4, (3,))),
        (16, 64, (8, (4, 3))),
        (8, 64, (8, (4, 3))),
        (8, 64, (16, (4, 4, 4, 1))),
    ],
    ids=["w8_h64_t4", "w8_h32_t4", "w16_h64_t8", "w8_h64_t8_ctrl", "w8_h64_t16_ctrl"],
)
def test_send_layout_lse_words_match_packed_reference(
    world: int, heads: int, cell: tuple[int, tuple[int, ...]]
) -> None:
    query_tokens, seq_q = cell
    real = int(sum(seq_q))
    assert sparse_mla.send_layout_supported(query_tokens, heads, world, 512, 4)
    p = _compact_problem(query_tokens, seq_q, heads)
    with _cpu_interpret_context():
        out, lse = prologue._kernel(p, candidate=True)
        send = send_layout_test._kernel_send(p, world)

    per = heads // world
    rows = query_tokens * per
    assert send.shape == (world, PAD + rows, out.shape[-1])

    nw, no = send_layout_test._diffs(
        send, prologue._bits16(out), np.asarray(lse), world, real
    )
    assert nw == 0 and no == 0, f"W={world} H={heads} T={query_tokens}: {nw} / {no}"

    # Verify unpacked LSE bit patterns on real rows contain no spurious NaNs and
    # preserve both -inf (empty row 0) and finite LSEs bit for bit.
    words = send.view(np.int16)[:, PAD - 1]
    hi = words[:, :rows].astype(np.int32)
    lo = words[:, rows : 2 * rows].astype(np.int32)
    lse_unpacked = ((hi << 16) | (lo & 0xFFFF)).view(np.float32)[:, : real * per]
    assert not np.isnan(lse_unpacked).any()
    assert np.isneginf(lse_unpacked).any()
    assert np.isfinite(lse_unpacked).any()
    assert not send.view(np.int16)[:, : PAD - 1].any()


def test_send_layout_lse_lo_skip_fault_caught_at_t4() -> None:
    p = _compact_problem(4, (3,), 64)
    with _cpu_interpret_context():
        out, lse = prologue._kernel(p, candidate=True)
        send = send_layout_test._kernel_send(p, 8, fault="send_lse_lo_skip")
    nw, no = send_layout_test._diffs(send, prologue._bits16(out), np.asarray(lse), 8, 3)
    assert nw > 0 and no == 0
