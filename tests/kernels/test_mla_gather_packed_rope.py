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
"""TPU_SPARSE_MLA_GATHER_PACKED_ROPE: packed gathered rope rows attend exactly
like the padded rows they replace.

The live-group attention kernel reads one chunk's gathered buffers twice, once
with the padded rope rows the gather writes today and once with the same keys
regrouped eight per row (`dsa_gather_native_sc(packed_out=True)`), and must
return the same output and lse bit for bit. Keys past each token's kv_len and
every row the gather never writes hold fp8 NaN bytes, so a key that picked up
its row neighbour's bytes would turn a live score into NaN. The kernel runs on
TPU, or on CPU in the Pallas TPU interpreter; the SparseCore gather test needs
a TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout, native_sc_gather

pytestmark = pytest.mark.tpu_test

HEADS, NOPE, RAW_PE, TOPK, CAP, BZ = 32, 512, 256, 2048, 512, 16
WINDOWS = TOPK // CAP
ATOM = native_sc_gather.OCCURRENCES_PER_ATOM
NAN_BYTE = 0x7F
SERVED_LEVERS = sparse_mla.MIXED_BODY_DEFAULT + sparse_mla.MIXED_KERNEL_V2_LEVERS

# kv_len per token of a 64-token chunk: a group within 256 rows (one token
# empty), one within 384, a whole group (one token at 512) and one whose
# tokens overflow window 0 into the fix-up windows.
KV_LENS = np.concatenate(
    [
        np.r_[0, 1, 63, 64, 65, 255, 256, np.arange(9) * 23 + 7],
        np.r_[384, 300, 257, 129, np.arange(12) * 29 + 11],
        np.r_[512, 511, 449, np.arange(13) * 37 + 3],
        np.r_[2048, 1500, 700, 513, 512, np.arange(11) * 41 + 5],
    ]
).astype(np.int32)


def regroup_rope(padded: np.ndarray) -> np.ndarray:
    """The padded gathered rope rows as `packed_out` writes them."""
    rows = native_sc_gather.PACKED_OUT_ROWS
    quads = padded.reshape(-1, 2, rows, 128)[..., : native_sc_gather.QUAD_ROPE_WORDS]
    return np.concatenate([quads[:, 0], quads[:, 1]], axis=-1).reshape(-1, 128)


def _finite_bytes(rng, shape):
    """Random fp8 e4m3fn bytes without the two NaN codes."""
    b = rng.integers(0, 0x7F, shape, dtype=np.uint8)
    return b | (rng.integers(0, 2, shape, dtype=np.uint8) << 7)


def _occurrence(t, p, tab, tokens):
    """Flat gathered slot of key `p` of token `t` (None: not gathered)."""
    w, j = divmod(p, CAP)
    if w:
        return (w * tokens + t) * CAP + j
    groups = tokens // BZ
    g, b = divmod(t, BZ)
    live, off = int(tab[g]), int(tab[groups + g])
    if live == CAP // ATOM:
        return (off * (BZ * ATOM // CAP) + b) * CAP + j
    a, r = divmod(j, ATOM)
    if a >= live:
        return None
    return (off + a) * BZ * ATOM + b * ATOM + r


def gathered_chunk(seed, kv_lens=KV_LENS):
    """One chunk's gathered buffers in live-group order, padded rope rows.

    Live keys hold finite bytes; keys past kv_len inside a gathered atom and
    every slot the gather leaves unwritten hold NaN bytes (the latter in the
    rope tail too)."""
    rng = np.random.default_rng(seed)
    tokens = kv_lens.shape[0]
    tab, num_live = sparse_mla.live_group_table(
        jnp.asarray(kv_lens),
        jnp.int32(tokens),
        chunk_size=tokens,
        kernel_batch_size=BZ,
        topk_cap=CAP,
        num_windows=WINDOWS,
    )
    tab = np.asarray(tab)
    slots = WINDOWS * tokens * CAP
    nope = np.full((slots, 4 * 128), NAN_BYTE, np.uint8)
    rope = np.full((slots, 128), NAN_BYTE, np.uint8)
    windows = -(-max(int(kv_lens.max()), 1) // CAP)
    for t in range(tokens):
        kv = int(kv_lens[t])
        reach = min(windows * CAP, TOPK) if kv > CAP else int(tab[t // BZ]) * ATOM
        for p in range(reach):
            o = _occurrence(t, p, tab, tokens)
            if o is None:
                continue
            rope[o, 64:] = 0
            if p < kv:
                nope[o] = _finite_bytes(rng, 512)
                rope[o, :64] = _finite_bytes(rng, 64)
    nope_words = native_sc_cache_layout.pack_nope(nope.reshape(slots, 4, 128))
    rope_words = native_sc_cache_layout.pack_selected_rope_for_tc(rope)
    return {
        "nope": nope_words.reshape(WINDOWS * tokens, CAP, 128),
        "rope": rope_words,
        "tab": tab,
        "num_live": np.asarray(num_live),
        "kv_lens": kv_lens,
    }


def _queries(seed, tokens, q_rope):
    rng = np.random.default_rng(seed + 1)
    q = rng.standard_normal((tokens, HEADS, NOPE), np.float32) * 0.5
    pe_dim = RAW_PE if q_rope else sparse_mla.Q_ROPE_DIM
    q_pe = rng.standard_normal((tokens, HEADS, pe_dim), np.float32) * 0.5
    table = None
    if q_rope:
        angle = rng.uniform(0, 2 * np.pi, (tokens, 32)).astype(np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        table = np.stack(
            [np.tile(cos, 4), np.concatenate([-sin, sin, -sin, sin], axis=1)], axis=1
        )
    return (
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(q_pe, jnp.bfloat16),
        None if table is None else jnp.asarray(table),
    )


def attend(chunk, *, packed, levers, widths, q_rope, seed=0, rope=None):
    """`sparse_attention_chunk` over the pre-gathered chunk; (out, lse)."""
    tokens = chunk["kv_lens"].shape[0]
    q, q_pe, table = _queries(seed, tokens, q_rope)
    rope_rows = chunk["rope"] if rope is None else rope
    if packed and rope is None:
        rope_rows = regroup_rope(rope_rows)
    keys_per_row = TOPK * tokens // rope_rows.shape[0]
    nope = jnp.asarray(chunk["nope"])
    rope_buf = jnp.asarray(rope_rows).reshape(
        WINDOWS * tokens, CAP // keys_per_row, 128
    )
    block = BZ * ATOM
    views = (
        nope.reshape(tokens * TOPK // block, BZ, ATOM, 128),
        rope_buf.reshape(tokens * TOPK // block, BZ, ATOM // keys_per_row, 128),
    )
    pages = jnp.zeros((1, 8, 128), jnp.uint32)

    @jax.jit
    def run(q, q_pe, table, nope, rope_buf, views, kv_lens, tab):
        return sparse_mla.sparse_attention_chunk(
            q,
            pages,
            pages,
            kv_lens,
            None,
            jnp.full((tokens, HEADS), -jnp.inf, jnp.float32),
            start_pos=0,
            end_pos=tokens,
            token_end=jnp.int32(tokens),
            sm_scale=0.0625,
            k_scale=0.75,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            topk_cap=CAP,
            topk_override=TOPK,
            is_dcp_mode=True,
            q_pe=q_pe,
            lse_lane_dense=True,
            lse_mask_empty=True,
            group_tab=tab,
            group_num_live=None,
            group_widths=widths,
            group_levers=levers,
            q_rope_table=table,
            pre_gathered=(nope, rope_buf, views),
            packed_gather=packed,
        )

    out, lse = run(
        q,
        q_pe,
        table,
        nope,
        rope_buf,
        views,
        jnp.asarray(chunk["kv_lens"]),
        jnp.asarray(chunk["tab"]),
    )
    return np.asarray(out.astype(jnp.float32)), np.asarray(lse)


@pytest.fixture
def interpret_on_cpu(monkeypatch):
    if jax.devices()[0].platform == "tpu":
        return
    pallas_call = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams()
        return pallas_call(*args, **kwargs)

    monkeypatch.setattr(sparse_mla.pl, "pallas_call", interpreted)


CONFIGS = {
    "served_levers_q_rope": dict(
        levers=SERVED_LEVERS + ("q128",), widths=(256, 384), q_rope=True
    ),
    "no_levers": dict(levers=(), widths=(), q_rope=False),
}


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_packed_rope_rows_attend_bit_identically(interpret_on_cpu, config):
    chunk = gathered_chunk(seed=20261025)
    kw = CONFIGS[config]
    out0, lse0 = attend(chunk, packed=False, **kw)
    out1, lse1 = attend(chunk, packed=True, **kw)
    live = chunk["kv_lens"] > 0
    assert np.isfinite(out0[live]).all() and np.isfinite(lse0[live]).all()
    np.testing.assert_array_equal(out1, out0)
    np.testing.assert_array_equal(lse1, lse0)


def test_swapped_quads_change_the_output(interpret_on_cpu):
    """Positive control: the comparison sees a key read from the wrong quad."""
    chunk = gathered_chunk(seed=7)
    kw = CONFIGS["no_levers"]
    packed = regroup_rope(chunk["rope"]).reshape(-1, 2, 64)[:, ::-1].reshape(-1, 128)
    out0, _ = attend(chunk, packed=False, **kw)
    out1, _ = attend(chunk, packed=True, rope=packed, **kw)
    assert not np.array_equal(out0, out1)


def test_expand_packed_rope_restores_the_padded_rows():
    rng = np.random.default_rng(3)
    rope = rng.integers(0, 2**32, (4 * native_sc_gather.QUADS_PER_ATOM, 128), np.uint32)
    rope[:, native_sc_gather.QUAD_ROPE_WORDS :] = 0
    got = jax.jit(native_sc_gather.expand_packed_rope)(jnp.asarray(regroup_rope(rope)))
    np.testing.assert_array_equal(np.asarray(got), rope)


def test_gather_packed_out_matches_the_padded_rows():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("native-SC gather requires a TPU SparseCore")
    pages, page_size = 64, 1024
    rng = np.random.default_rng(11)
    rope_u8 = rng.integers(0, 256, (pages, page_size // 4, 4, 128), np.uint8)
    rope_u8[..., 64:] = 0
    nope = jnp.asarray(
        native_sc_cache_layout.pack_nope(
            rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        )
    )
    caches = {
        False: jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8)),
        True: jnp.asarray(
            native_sc_cache_layout.pack_rope_packed(
                rope_u8[..., :64].reshape(pages, page_size, 64)
            )
        ),
    }
    query = 64
    indices = rng.integers(0, pages * page_size, (query, TOPK), dtype=np.int32)
    for packed_rope, rope in caches.items():
        for atoms, live in ((16, None), (1, 5 * TOPK + 1000)):
            kw = dict(out_size=query * TOPK, atoms_per_batch=atoms)
            if live is not None:
                kw["num_live_indices"] = jnp.int32(live)
            nope0, rope0 = native_sc_gather.dsa_gather_native_sc(
                nope, rope, jnp.asarray(indices), packed_rope=packed_rope, **kw
            )
            nope1, rope1 = native_sc_gather.dsa_gather_native_sc(
                nope,
                rope,
                jnp.asarray(indices),
                packed_rope=packed_rope,
                packed_out=True,
                **kw,
            )
            rows = (live or query * TOPK) // ATOM * ATOM
            np.testing.assert_array_equal(
                np.asarray(nope1)[:rows], np.asarray(nope0)[:rows]
            )
            np.testing.assert_array_equal(
                np.asarray(rope1)[: rows // 8],
                regroup_rope(np.asarray(rope0)[: rows // 4]),
            )
