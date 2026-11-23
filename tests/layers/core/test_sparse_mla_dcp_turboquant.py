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
"""TurboQuant 4-bit latent caches in sparse MLA under decode context
parallelism (DCP).

What is pinned:

* `sparse_mla_attention_dcp(..., tq_tables=..., tq_prerotated=True)` against
  the unsharded `sparse_mla_attention` on the same inputs, over two steps at
  DCP 2 and 4: every rank's NoPE and RoPE shard is byte-identical to the
  unsharded cache sliced by the documented ownership rule (position `p` lives
  on rank `(p // interleave) % dcp` at local index
  `(p // (dcp * interleave)) * interleave + p % interleave`); a token's
  row-mate (two 256-B TurboQuant tokens share a 512-B NoPE row) survives the
  next step's write of the other half; and the per-rank partials, merged by
  log-sum-exp, match the unsharded output, with `k_scale != 1`, a token of
  which one rank owns more than `TPU_DCP_MLA_TOPK_CAP` entries, ranks owning
  none of a token's top-k, and unwritten (NaN) slots behind the dummy entry
  such ranks are handed. Negative controls must fail the same checks.
* The adapter boundary: `_build_sparse_mla_dcp_op` hands over the latent
  tables, the rotation flag as it stands at trace time and the non-DCP build's
  `k_scale`; `forward` hands the DCP op an unquantized latent and an fp8
  `k_pe`; `sparse_mla_attention_dcp` refuses unrotated weights and a NoPE
  width that does not match the tables.

The SparseCore cache writers, gather and attention kernel are TPU-only. On CPU
the writers are their pure-XLA goldens and the kernel is a stand-in that reads
the caches through the layout contract, dequantizes the 4-bit rows and attends
in f32, honouring `topk_cap` (windowed online softmax over every live entry:
the kernel's overflow pass is exact), `return_lse` and the `-1` tail (it
attends the first `kv_len` entries, `kv_len` being the count of non-`-1`
ones). These tests therefore pin the DCP plumbing around the kernel; the
kernel itself at the DCP arguments is checked on TPU in
`tests/kernels/test_mla_sparse_turboquant.py`. A CPU process has several
devices only when XLA is told so before jax starts, so the multi-rank checks
run in a subprocess.
"""

import os
import re
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.layers.core.attention_interface as attention_interface
from vllm_torchtpu.distributed.dcp import DCP_LANE_AXIS_NAME
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_rank_as_data,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import turboquant as tq

NOPE = tq.MLA_TQ_LATENT_DIM
ROPE = 64
QK_NOPE = 192
V_DIM = 256
HEADS = 8
TOPK = 256
TOPK_CAP = 64
TOPK_CAP_CHUNK = 64
K_SCALE = 0.75
SM_SCALE = (NOPE + ROPE) ** -0.5
SHARD_PAGE = 64  # positions per page on one rank
PAGES = 12  # per rank; the unsharded cache has as many, each dcp times longer
PAGES_PER_SEQ = 6
TOKENS = 16  # padded step width
# Every byte 0x7F: each k_pe byte of an unwritten slot is an fp8 NaN.
POISON = np.uint32(0x7F7F7F7F)
# Context per request before the steps, then each step's new tokens. A spans
# several virtual pages; B (4 positions at most, inside one interleave chunk)
# lives on rank 0 alone, so the dummy entry the other ranks are handed for it
# points at an unwritten slot; C crosses a virtual page at DCP 2.
PRIOR = (600, 2, 150)
STEP1 = (3, 1, 6)
STEP2 = (1, 1, 1)
# Step 1's C tokens attend 1 (themselves), ... and all positions so far.
C_TOPK = (1, 37, 64, 65, 100, None)


def _owner(pos, dcp, interleave):
    return (np.asarray(pos) // interleave) % dcp


def _rank_local(pos, dcp, interleave):
    pos = np.asarray(pos)
    return (pos // (dcp * interleave)) * interleave + pos % interleave


def _tq_specs(page_size):
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE,
            KVCacheLayout.SPARSECORE,
            PAGES,
            page_size,
            tq.MLA_TQ_NOPE_BYTES,
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, KVCacheLayout.SPARSECORE, PAGES, page_size, ROPE + 2
        ),
    )


# ---------------------------------------------------------------------------
# The kernel stand-in
# ---------------------------------------------------------------------------

# The e4m3fn values of the SWAR bytes `3 * m + 2` (m = 0..7) a 4-bit code's
# magnitude index unpacks to.
_TQ_LEVELS = np.array([2, 5, 8, 11, 14, 18, 24, 30], np.float32) / 512


def _bytes_of_words(words):
    """u32 `[..., n]` -> u8 `[..., 4 n]`; byte `b` of word `j` is token byte
    `b * n + j` (the band interleave of `pack_tokens`)."""
    bands = [(words >> (8 * b)) & 0xFF for b in range(4)]
    return jnp.concatenate(bands, axis=-1).astype(jnp.uint8)


def _decode_tq_tokens(nope_words, rope_words):
    """A token's 64 NoPE and 32 RoPE words -> (latent in the rotated basis,
    dequantized k_pe), both f32.

    The NoPE row stores the code blocks as `[0, 2, 1, 3]`; byte `i` holds
    code `i` in its low nibble and code `256 + i` in its high one (bit 3
    sign, bits 0..2 level). The RoPE row is 64 fp8 k_pe bytes, then the bf16
    norm."""
    nope = _bytes_of_words(nope_words)
    blk = tq.MLA_TQ_NOPE_BYTES // 4
    codes = jnp.concatenate(
        [
            nope[..., :blk],
            nope[..., 2 * blk : 3 * blk],
            nope[..., blk : 2 * blk],
            nope[..., 3 * blk :],
        ],
        axis=-1,
    )
    nibbles = jnp.concatenate([codes & 0xF, codes >> 4], axis=-1).astype(jnp.int32)
    level = jnp.asarray(_TQ_LEVELS)[nibbles & 7]
    values = jnp.where((nibbles & 8) != 0, -level, level)
    rope = _bytes_of_words(rope_words)
    k_pe = jax.lax.bitcast_convert_type(rope[..., :ROPE], jnp.float8_e4m3fn)
    off = tq.MLA_TQ_NORM_BYTE_OFFSET
    norm_bits = (
        rope[..., off].astype(jnp.uint32) | (rope[..., off + 1].astype(jnp.uint32) << 8)
    ) << 16
    norm = jax.lax.bitcast_convert_type(norm_bits, jnp.float32)
    return values * norm[..., None], k_pe.astype(jnp.float32)


def _stand_in_kernel(calls, *, drop_overflow=False):
    """`sparse_ragged_paged_attention` for TurboQuant caches on CPU.

    Records each call's static arguments in `calls`. With `drop_overflow`,
    a capped call attends only its first `topk_cap` entries: a kernel
    without the overflow pass."""

    def kernel(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        *,
        sm_scale=1.0,
        k_scale=1.0,
        return_lse=False,
        cache_layout=None,
        topk_cap=None,
        q_pe=None,
        lse_lane_dense=False,
        turboquant=False,
        gather_and_attention_chunk_size=None,
    ):
        calls.append(
            dict(
                turboquant=turboquant,
                cache_layout=cache_layout,
                topk_cap=topk_cap,
                return_lse=return_lse,
                split_q=q_pe is not None,
                lse_lane_dense=lse_lane_dense,
                k_scale=k_scale,
                chunk=gather_and_attention_chunk_size,
            )
        )
        assert turboquant, "the stand-in reads TurboQuant rows only"
        assert cache_layout == sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT
        num_tokens, topk = topk_indices.shape
        if topk_cap is not None and topk_cap >= topk:
            topk_cap = None
        if topk_cap is not None:
            assert topk_cap % sparse_mla_kernel.GATHER_ATOM_ROWS == 0, topk_cap
            assert topk % topk_cap == 0, (topk, topk_cap)
        q_nope, q_rope = (q[..., :NOPE], q[..., NOPE:]) if q_pe is None else (q, q_pe)
        num_pages = cache_kv_nope.shape[0]
        page_size = cache_kv_nope.shape[1] * 2  # two 256-B tokens per NoPE row
        num_seqs = cu_q_lens.shape[0] - 1

        tok = jnp.arange(num_tokens, dtype=jnp.int32)
        seq = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], tok, side="right"), num_seqs - 1
        )
        kv_len = jnp.sum(topk_indices != -1, axis=-1)
        if drop_overflow and topk_cap is not None:
            kv_len = jnp.minimum(kv_len, topk_cap)
        live = jnp.arange(topk)[None, :] < kv_len[:, None]
        idx = jnp.where(live, topk_indices, 0)
        page = page_indices.reshape(num_seqs, -1)[seq[:, None], idx // page_size]
        slot = idx % page_size
        latent, k_pe = _decode_tq_tokens(
            cache_kv_nope.reshape(num_pages, page_size, -1)[page, slot],
            cache_kv_rope.reshape(num_pages, page_size, -1)[page, slot],
        )
        k_pe = k_pe * k_scale
        hi = jax.lax.Precision.HIGHEST
        scores = sm_scale * (
            jnp.einsum("thd,tkd->thk", q_nope.astype(jnp.float32), latent, precision=hi)
            + jnp.einsum("thd,tkd->thk", q_rope.astype(jnp.float32), k_pe, precision=hi)
        )
        values = latent if q_pe is not None else jnp.concatenate([latent, k_pe], -1)
        values = jnp.where(live[..., None], values, 0.0)

        # Online softmax over `topk_cap`-wide windows, as the overflow pass.
        width = topk if topk_cap is None else topk_cap
        windows = topk // width
        heads = q_nope.shape[1]
        s = jnp.where(live[:, None, :], scores, -jnp.inf)
        s = s.reshape(num_tokens, heads, windows, width)
        has = live.reshape(num_tokens, 1, windows, width).any(-1)
        m_w = jnp.max(s, axis=-1)
        p = jnp.exp(s - jnp.where(has, m_w, 0.0)[..., None])
        acc_w = jnp.einsum(
            "thwk,twkv->thwv",
            p,
            values.reshape(num_tokens, windows, width, -1),
            precision=hi,
        )
        m = jnp.max(jnp.where(has, m_w, -jnp.inf), axis=-1)
        e = jnp.where(
            has, jnp.exp(m_w - jnp.where(jnp.isfinite(m), m, 0.0)[..., None]), 0.0
        )
        denom = (e * p.sum(-1)).sum(-1)
        out = (e[..., None] * acc_w).sum(-2) / denom[..., None]
        lse = m + jnp.log(denom)

        # Rows past the step keep the query (the output aliases it) and the
        # kernel's initial LSE.
        reached = tok < cu_q_lens[distribution[2]]
        out = jnp.where(reached[:, None, None], out, q.astype(jnp.float32))
        lse = jnp.where(reached[:, None], lse, jnp.finfo(jnp.float32).min)
        out = out.astype(q.dtype)
        return (out, lse) if return_lse else out

    return kernel


def test_stand_in_decode_matches_the_reference_dequant():
    """The stand-in's cache decode inverts the pack and the SparseCore word
    layout: rotated back, it is `tq.dequant_mla_latent_ref`."""
    rng = np.random.default_rng(3)
    tables = tq.make_mla_latent_tables()
    latent = jnp.asarray(rng.standard_normal((8, NOPE)), jnp.bfloat16)
    k_pe = jnp.asarray(rng.standard_normal((8, ROPE)), jnp.float8_e4m3fn)
    codes, rope_row = attention_interface._tq_pack_sparse_mla_latent(
        latent, k_pe, tables
    )
    rotated, got_pe = _decode_tq_tokens(
        kv_cache_utils.pack_tokens(codes, tq.MLA_TQ_NOPE_BYTES),
        kv_cache_utils.pack_tokens(rope_row, 128),
    )
    natural, norm = tq.quantize_mla_latent(latent, tables)
    want = tq.dequant_mla_latent_ref(natural, norm, tables)
    got = np.asarray(rotated, np.float64) @ np.asarray(tables.hadamard, np.float64)
    np.testing.assert_allclose(got, np.asarray(want), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(got_pe), np.asarray(k_pe, np.float32))


# ---------------------------------------------------------------------------
# DCP vs unsharded: fixture, oracles and checks
# ---------------------------------------------------------------------------


def _choose(rng, pool, n):
    return rng.choice(pool, size=n, replace=False)


def _row(rng, pos, n, must=()):
    """`n` distinct positions `<= pos`, including `pos` and `must`."""
    must = np.asarray([*must, pos])
    pool = np.setdiff1d(np.arange(pos + 1), must)
    return np.concatenate([_choose(rng, pool, n - must.size), must])


def _build(dcp, interleave, *, poison, seed=0):
    """Latents for every position, the prior context packed into the
    unsharded cache and, owner-sliced, into the DCP shards."""
    rng = np.random.default_rng(seed)
    tables = tq.make_mla_latent_tables()
    vps = SHARD_PAGE * dcp
    full_specs, shard_specs = _tq_specs(vps), _tq_specs(SHARD_PAGE)
    for nope_spec, rope_spec in (full_specs, shard_specs):
        page = nope_spec.page_size
        assert nope_spec.jax_dtype == jnp.uint32
        assert nope_spec.shape == (PAGES, page // 2, 128), nope_spec.shape
        assert rope_spec.shape == (PAGES, page // 4, 128), rope_spec.shape

    totals = [sum(lens) for lens in zip(PRIOR, STEP1, STEP2)]
    pages = [-(-n // vps) for n in totals]
    assert sum(pages) < PAGES and max(pages) <= PAGES_PER_SEQ
    perm = rng.permutation(PAGES).astype(np.int32)
    bt = np.full((len(totals), PAGES_PER_SEQ), perm[-1], np.int32)
    first = 0
    for s, n in enumerate(pages):
        bt[s, :n] = perm[first : first + n]
        first += n
    # DCP block tables hold virtual ordinals; one past the shard, which every
    # rank must resolve mod PAGES.
    bt_dcp = bt.copy()
    bt_dcp[2, 0] += PAGES

    outliers = 1.0 + 3.0 * (rng.random(NOPE) < 0.02)
    latent = [
        jnp.asarray(rng.standard_normal((n, NOPE)) * outliers, jnp.bfloat16)
        for n in totals
    ]
    # Large enough that the RoPE half of the scores (and so k_scale) matters.
    k_pe = [
        jnp.asarray(2.0 * rng.standard_normal((n, ROPE)) / K_SCALE, jnp.float32).astype(
            jnp.float8_e4m3fn
        )
        for n in totals
    ]
    codes, rope_rows = attention_interface._tq_pack_sparse_mla_latent(
        jnp.concatenate([c[:n] for c, n in zip(latent, PRIOR)]),
        jnp.concatenate([k[:n] for k, n in zip(k_pe, PRIOR)]),
        tables,
    )
    fill = POISON if poison else np.uint32(0)
    nope_full, rope_full = kv_cache_utils.update_sparse_mla_kv_cache_jax(
        jnp.full(full_specs[0].shape, fill, jnp.uint32),
        jnp.full(full_specs[1].shape, fill, jnp.uint32),
        codes,
        rope_rows,
        jnp.asarray(PRIOR, jnp.int32),
        jnp.asarray(bt.reshape(-1)),
        jnp.asarray(np.cumsum((0, *PRIOR)), jnp.int32),
        nope_spec=full_specs[0],
        rope_spec=full_specs[1],
    )

    devices = jax.devices()
    mesh_dcp = Mesh(
        np.asarray(devices[:dcp]).reshape(1, dcp), (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME)
    )
    sharded = NamedSharding(mesh_dcp, P(DCP_AXIS_NAME))
    fx = SimpleNamespace(
        rng=rng,
        dcp=dcp,
        interleave=interleave,
        vps=vps,
        tables=tables,
        full_specs=full_specs,
        shard_specs=shard_specs,
        bt=bt,
        bt_dcp=bt_dcp,
        latent=latent,
        k_pe=k_pe,
        nope_full=nope_full,
        rope_full=rope_full,
        mesh1=Mesh(np.asarray(devices[:1]), ("x",)),
        mesh_dcp=mesh_dcp,
        fill=fill,
    )
    fx.nope_dcp, fx.rope_dcp = (
        jax.device_put(jnp.asarray(_owner_sliced(fx, cache, i)), sharded)
        for i, cache in enumerate((nope_full, rope_full))
    )
    # Dequantized straight from the latents, no cache: the direct reference.
    hadamard = np.asarray(tables.hadamard, np.float64)
    fx.ref_latent, fx.ref_k_pe = [], []
    for c, k in zip(latent, k_pe):
        natural, norm = tq.quantize_mla_latent(c, tables)
        deq = np.asarray(tq.dequant_mla_latent_ref(natural, norm, tables), np.float64)
        fx.ref_latent.append(deq @ hadamard)
        fx.ref_k_pe.append(np.asarray(k.astype(jnp.float32), np.float64) * K_SCALE)
    return fx


def _owner_sliced(fx, full, which):
    """The DCP shards (`[dcp * PAGES, ...]`) the ownership rule makes of an
    unsharded cache: within each page, offset `o` goes to rank `owner(o)` at
    local slot `rank_local(o)`. `which` 0 is NoPE, 1 RoPE."""
    full_spec, shard_spec = fx.full_specs[which], fx.shard_specs[which]
    tokens = np.asarray(full).reshape(PAGES, fx.vps, -1)
    offsets = np.arange(fx.vps)
    owner = _owner(offsets, fx.dcp, fx.interleave)
    local = _rank_local(offsets, fx.dcp, fx.interleave)
    shards = np.empty((fx.dcp, PAGES, SHARD_PAGE, tokens.shape[-1]), tokens.dtype)
    for r in range(fx.dcp):
        mine = owner == r
        np.testing.assert_array_equal(np.sort(local[mine]), np.arange(SHARD_PAGE))
        shards[r][:, local[mine]] = tokens[:, mine]
    assert full_spec.page_size == fx.vps and shard_spec.page_size == SHARD_PAGE
    return shards.reshape(fx.dcp * PAGES, *shard_spec.shape[1:])


def _step_inputs(fx, step, done):
    """One step's rows, queries and top-k (global and per rank). `done` is
    each request's context before the step."""
    q_lens = STEP1 if step == 1 else STEP2
    rng, dcp, interleave = fx.rng, fx.dcp, fx.interleave
    real = sum(q_lens)
    seq_of = [s for s, q in enumerate(q_lens) for _ in range(q)]
    positions = [p for s, q in enumerate(q_lens) for p in range(done[s], done[s] + q)]
    pad = TOKENS - real
    kv_c = jnp.concatenate(
        [c[d : d + q] for c, d, q in zip(fx.latent, done, q_lens)]
        + [jnp.asarray(rng.standard_normal((pad, NOPE)), jnp.bfloat16)]
    )
    k_pe = jnp.concatenate(
        [k[d : d + q] for k, d, q in zip(fx.k_pe, done, q_lens)]
        + [jnp.asarray(rng.standard_normal((pad, ROPE)), jnp.float8_e4m3fn)]
    )

    rows = np.full((TOKENS, TOPK), -1, np.int32)
    marks = {}
    for t, (s, pos) in enumerate(zip(seq_of, positions)):
        if s == 1:
            sel = np.arange(pos + 1)
            marks.setdefault("b", t)
        elif step == 2 and s == 0:
            sel = _row(rng, pos, 150, must=[pos - 1])
        elif step == 2:
            sel = _row(rng, pos, 100)
        elif s == 2:
            n = C_TOPK[pos - done[2]]
            sel = _row(rng, pos, pos + 1 if n is None else n)
        elif pos == done[0]:
            sel = _row(rng, pos, 200)
        elif pos == done[0] + 1:
            # One rank owns more than the cap: the overflow pass must run.
            past = np.arange(done[0])
            own = _owner(past, dcp, interleave)
            sel = np.concatenate(
                [
                    _choose(rng, past[own == 0], TOPK_CAP + 32),
                    _choose(rng, past[own != 0], 32),
                    [done[0], pos],
                ]
            )
            marks["skew"] = t
        else:
            # Only rows this step writes, on one rank.
            sel = np.arange(done[0], pos + 1)
            marks["new"] = t
        sel = rng.permutation(sel)
        rows[t, : sel.size] = sel
    rows[real:, 0] = 0  # the unsharded kernel needs one live entry per row

    local = np.full((dcp, TOKENS, TOPK), -1, np.int32)
    for t in range(real):
        row = rows[t][rows[t] >= 0]
        own = _owner(row, dcp, interleave)
        for r in range(dcp):
            mine = _rank_local(row[own == r], dcp, interleave)
            local[r, t, : mine.size] = mine

    ql_raw = rng.standard_normal((TOKENS, HEADS, NOPE))
    q_pe = 2.0 * rng.standard_normal((TOKENS, HEADS, ROPE))
    # Near-uniform scores for these two, so whichever entries a defect drops
    # (past the cap, or a dummy let in) carry a fair share of the softmax.
    gentle = [marks[k] for k in ("skew", "new") if k in marks]
    ql_raw[gentle] *= 0.1
    q_pe[gentle] *= 0.1
    ql_raw = jnp.asarray(ql_raw, jnp.bfloat16)
    seq_lens = [d + q for d, q in zip(done, q_lens)]
    num_decode = len(q_lens) if step == 2 else 0
    return SimpleNamespace(
        step=step,
        real=real,
        seq_of=seq_of,
        kv_c=kv_c,
        k_pe=k_pe,
        ql_raw=ql_raw,
        ql_rot=attention_interface._tq_rotate(ql_raw, fx.tables),
        q_pe=jnp.asarray(q_pe, jnp.bfloat16),
        global_topk=rows,
        local_topk=local,
        counts=(local >= 0).sum(-1),
        seq_lens_list=seq_lens,
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        qsl=jnp.asarray(np.cumsum((0, *q_lens)), jnp.int32),
        dist=jnp.asarray([num_decode, num_decode, len(q_lens)], jnp.int32),
        marks=marks,
    )


def _unsharded(fx, nope, rope, inp):
    return attention_interface.sparse_mla_attention(
        inp.ql_rot,
        inp.q_pe,
        inp.kv_c,
        inp.k_pe,
        nope,
        rope,
        jnp.asarray(inp.global_topk),
        inp.seq_lens,
        jnp.asarray(fx.bt.reshape(-1)),
        inp.qsl,
        inp.dist,
        fx.mesh1,
        *fx.full_specs,
        sm_scale=SM_SCALE,
        k_scale=K_SCALE,
        tq_tables=fx.tables,
        tq_prerotated=True,
    )


def _dcp(fx, nope, rope, inp, *, ql=None, k_scale=K_SCALE):
    """`sparse_mla_attention_dcp`; partials as f32 `[dcp, T, H, V]` and LSE
    `[dcp, T, H]`."""
    local = jax.device_put(
        jnp.asarray(inp.local_topk.reshape(fx.dcp * TOKENS, TOPK)),
        NamedSharding(fx.mesh_dcp, P(DCP_AXIS_NAME, None)),
    )
    nope, rope, out, lse = attention_interface.sparse_mla_attention_dcp(
        inp.ql_rot if ql is None else ql,
        inp.q_pe,
        inp.kv_c,
        inp.k_pe,
        nope,
        rope,
        local,
        inp.seq_lens,
        jnp.asarray(fx.bt_dcp.reshape(-1)),
        inp.qsl,
        inp.dist,
        fx.mesh_dcp,
        *fx.shard_specs,
        sm_scale=SM_SCALE,
        k_scale=k_scale,
        dcp_size=fx.dcp,
        interleave_size=fx.interleave,
        tq_tables=fx.tables,
        tq_prerotated=True,
    )
    return SimpleNamespace(
        nope=nope,
        rope=rope,
        out=np.asarray(out.astype(jnp.float32)).reshape(fx.dcp, TOKENS, HEADS, NOPE),
        lse=np.asarray(lse).reshape(fx.dcp, TOKENS, HEADS),
    )


def _shard_mismatches(fx, ref_caches, got_caches):
    """`(rank, "NoPE" | "RoPE")` of every shard that differs from the
    owner-sliced unsharded cache."""
    bad = []
    for which, (name, ref, got) in enumerate(
        zip(("NoPE", "RoPE"), ref_caches, got_caches)
    ):
        want = _owner_sliced(fx, ref, which).reshape(fx.dcp, -1)
        got = np.asarray(got).reshape(fx.dcp, -1)
        bad += [(r, name) for r in range(fx.dcp) if not np.array_equal(got[r], want[r])]
    return bad


def _direct_reference(fx, inp):
    """f64 dequantize-then-attend over each token's global top-k, from the
    latents (rotated basis), not from a cache."""
    q = np.asarray(inp.ql_rot.astype(jnp.float32), np.float64)
    q_pe = np.asarray(inp.q_pe.astype(jnp.float32), np.float64)
    out = np.zeros((TOKENS, HEADS, NOPE))
    for t in range(inp.real):
        s = inp.seq_of[t]
        sel = inp.global_topk[t][inp.global_topk[t] >= 0]
        latent, k_pe = fx.ref_latent[s][sel], fx.ref_k_pe[s][sel]
        scores = SM_SCALE * (q[t] @ latent.T + q_pe[t] @ k_pe.T)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        out[t] = (p / p.sum(-1, keepdims=True)) @ latent
    return out


def _lse_merge(outs, lses):
    """`sum_r softmax_r(lse) * o_r` over ranks in f64. A `-inf` rank weighs
    0, and `0 * NaN` stays NaN, as in any weighted sum."""
    out = np.asarray(outs, np.float64)
    lse = np.asarray(lses, np.float64)
    finite = np.isfinite(lse)
    shift = np.where(finite, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(finite, np.exp(np.where(finite, lse, 0.0) - shift), 0.0)
    denom = w.sum(0)
    w = w / np.where(denom > 0, denom, 1.0)
    return (w[..., None] * out).sum(0)


def _token_cos(a, b):
    """Per token, the smallest cosine over heads; NaN or zero rows give 0."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore"):
        cos = np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
    return np.nan_to_num(cos, nan=0.0).min(-1)


def _assert_close(got, want, *, min_cos=0.999, allclose=True):
    assert np.isfinite(got).all()
    cos = _token_cos(got, want)
    assert cos.min() > min_cos, f"per-token cosine {cos}"
    if allclose:
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


def _close(got, want):
    """`_assert_close`'s verdict, for the negative controls."""
    return bool(
        np.isfinite(got).all()
        and _token_cos(got, want).min() > 0.999
        and np.allclose(got, want, rtol=2e-2, atol=2e-2)
    )


def _assert_step_exercises_the_dcp_paths(inp):
    counts, real = inp.counts, inp.real
    # Ownership partitions every top-k row.
    np.testing.assert_array_equal(
        counts.sum(0)[:real], (inp.global_topk[:real] >= 0).sum(-1)
    )
    assert (counts[:, :real] > TOPK_CAP).any(), "the overflow pass would not run"
    assert (counts[:, :real] == 0).any(), "no rank owns nothing of a token"
    assert not counts[:, real:].any()


def _assert_partials(got, inp):
    """A rank owning none of a token's top-k: LSE `-inf` and output zero (so
    no NaN read through its dummy entry reaches the merge as `0 * NaN`)."""
    owns = inp.counts > 0
    assert np.isneginf(got.lse[~owns]).all()
    assert not np.any(got.out[~owns])
    assert np.isfinite(got.lse[owns]).all() and np.isfinite(got.out[owns]).all()


def _assert_row_mate_survives(fx, before, after):
    """Step 1 wrote A's last row into the even half of a NoPE row (and a
    quarter of a RoPE row); step 2 wrote only its mate."""
    a = PRIOR[0] + STEP1[0] - 1
    owner = _owner([a, a + 1], fx.dcp, fx.interleave)
    local = _rank_local([a, a + 1], fx.dcp, fx.interleave)
    assert owner[0] == owner[1] and local[0] % 2 == 0 and local[1] == local[0] + 1
    page = fx.bt[0, a // fx.vps] % PAGES
    slot = local[0] % SHARD_PAGE
    for which in (0, 1):
        words = fx.shard_specs[which].token_bytes // kv_cache_utils.WORD_BYTES

        def tokens(caches, words=words, which=which):
            cache = np.asarray(caches[which])
            return cache.reshape(fx.dcp, PAGES, SHARD_PAGE, words)[owner[0], page]

        b, f = tokens(before), tokens(after)
        np.testing.assert_array_equal(f[slot], b[slot])
        assert (b[slot + 1] == fx.fill).all()
        assert not np.array_equal(f[slot + 1], b[slot + 1])


def _set_up(monkeypatch, split_q, *, stand_in=True):
    monkeypatch.setenv("TPU_DCP_MLA_TOPK_CAP", str(TOPK_CAP))
    monkeypatch.setenv("TPU_DCP_MLA_TOPK_CAP_CHUNK", str(TOPK_CAP_CHUNK))
    monkeypatch.setenv("TPU_MLA_SPARSE_SPLIT_Q", "1" if split_q else "0")
    monkeypatch.delenv("TPU_SPARSE_MLA_CHUNK_TOKENS", raising=False)
    calls = []
    if stand_in:
        monkeypatch.setattr(
            attention_interface,
            "update_sparse_mla_kv_cache",
            kv_cache_utils.update_sparse_mla_kv_cache_jax,
        )
        monkeypatch.setattr(
            attention_interface,
            "update_sparse_mla_kv_cache_dcp",
            kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax,
        )
        kernel = _stand_in_kernel(calls)
        monkeypatch.setattr(
            attention_interface, "sparse_mla_ragged_paged_attention", kernel
        )
        monkeypatch.setattr(sparse_mla_kernel, "sparse_ragged_paged_attention", kernel)
    return calls


def _equivalence(monkeypatch, dcp, interleave, split_q, *, stand_in=True):
    calls = _set_up(monkeypatch, split_q, stand_in=stand_in)
    # The poison stands for stale cache contents; the real kernel's handling
    # of NaN in masked slots is not what this pins, so TPU starts from zeros.
    fx = _build(dcp, interleave, poison=stand_in)
    full = (fx.nope_full, fx.rope_full)
    shards = (fx.nope_dcp, fx.rope_dcp)
    done = PRIOR
    for step in (1, 2):
        inp = _step_inputs(fx, step, done)
        if step == 1:
            _assert_step_exercises_the_dcp_paths(inp)
        ref_nope, ref_rope, ref_out = _unsharded(fx, *full, inp)
        got = _dcp(fx, *shards, inp)
        assert _shard_mismatches(fx, (ref_nope, ref_rope), (got.nope, got.rope)) == []
        _assert_partials(got, inp)
        ref = np.asarray(ref_out.astype(jnp.float32))[: inp.real]
        _assert_close(_lse_merge(got.out, got.lse)[: inp.real], ref)
        _assert_close(
            ref,
            _direct_reference(fx, inp)[: inp.real],
            min_cos=0.999 if stand_in else 0.995,
            allclose=False,
        )
        if step == 2:
            _assert_row_mate_survives(fx, shards, (got.nope, got.rope))
        full, shards, done = (
            (ref_nope, ref_rope),
            (got.nope, got.rope),
            inp.seq_lens_list,
        )
    if stand_in:
        dcp_calls = [c for c in calls if c["return_lse"]]
        full_calls = [c for c in calls if not c["return_lse"]]
        assert dcp_calls and full_calls
        want = dict(
            turboquant=True,
            cache_layout=sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT,
            topk_cap=TOPK_CAP,
            return_lse=True,
            split_q=split_q,
            lse_lane_dense=split_q,
            k_scale=K_SCALE,
            chunk=TOPK_CAP_CHUNK,
        )
        assert all(c == want for c in dcp_calls), dcp_calls
        assert all(
            c["turboquant"] and c["topk_cap"] is None and c["k_scale"] == K_SCALE
            for c in full_calls
        ), full_calls


def _raw_partials(fx, inp, nope, rope, *, split_q):
    """What the stand-in returns on each rank, before the DCP function masks
    ranks owning none of a token's top-k (they get the dummy entry 0)."""
    kernel = _stand_in_kernel([])
    nope = np.asarray(nope).reshape(fx.dcp, *fx.shard_specs[0].shape)
    rope = np.asarray(rope).reshape(fx.dcp, *fx.shard_specs[1].shape)
    owns = inp.counts > 0
    if split_q:
        q, kw = inp.ql_rot, dict(q_pe=inp.q_pe, lse_lane_dense=True)
    else:
        q, kw = jnp.concatenate([inp.ql_rot, inp.q_pe], axis=-1), {}
    outs, lses = [], []
    for r in range(fx.dcp):
        rows = inp.local_topk[r].copy()
        rows[:, 0] = np.where(owns[r], rows[:, 0], 0)
        out, lse = kernel(
            q,
            jnp.asarray(nope[r]),
            jnp.asarray(rope[r]),
            jnp.asarray(rows),
            jnp.asarray(fx.bt_dcp.reshape(-1) % PAGES),
            inp.qsl,
            inp.dist,
            sm_scale=SM_SCALE,
            k_scale=K_SCALE,
            return_lse=True,
            cache_layout=sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT,
            topk_cap=TOPK_CAP,
            turboquant=True,
            **kw,
        )
        outs.append(np.asarray(out[..., :NOPE].astype(jnp.float32)))
        lses.append(np.asarray(lse))
    return np.stack(outs), np.stack(lses), owns


def _negative_controls(monkeypatch):
    dcp, interleave, split_q = 2, 8, True
    _set_up(monkeypatch, split_q)
    fx = _build(dcp, interleave, poison=True)
    inp = _step_inputs(fx, 1, PRIOR)
    real, t_new, t_skew, t_b = (inp.real, *(inp.marks[k] for k in ("new", "skew", "b")))
    new_rows = np.arange(PRIOR[0], PRIOR[0] + STEP1[0])
    assert set(_owner(new_rows, dcp, interleave)) == {1}
    assert inp.counts[0, t_skew] > TOPK_CAP

    ref_nope, ref_rope, ref_out = _unsharded(fx, fx.nope_full, fx.rope_full, inp)
    ref = np.asarray(ref_out.astype(jnp.float32))[:real]
    good = _dcp(fx, fx.nope_dcp, fx.rope_dcp, inp)
    assert _shard_mismatches(fx, (ref_nope, ref_rope), (good.nope, good.rope)) == []
    _assert_close(_lse_merge(good.out, good.lse)[:real], ref)

    def merged(bad):
        return _lse_merge(bad.out, bad.lse)[:real]

    # Rank 1 drops the norm when it packs: its RoPE shard differs, and the
    # token attending only rank 1's new rows collapses to zero.
    pack = attention_interface._tq_pack_sparse_mla_latent

    def pack_dropping_the_norm_on_rank_1(kv_c_normed, k_pe, tables):
        codes, rope_row = pack(kv_c_normed, k_pe, tables)
        off = tq.MLA_TQ_NORM_BYTE_OFFSET
        dropped = rope_row.at[:, off : off + 2].set(0)
        on_rank_1 = cp_rank_as_data(DCP_AXIS_NAME, dcp) == 1
        return codes, jnp.where(on_rank_1, dropped, rope_row)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            attention_interface,
            "_tq_pack_sparse_mla_latent",
            pack_dropping_the_norm_on_rank_1,
        )
        bad = _dcp(fx, fx.nope_dcp, fx.rope_dcp, inp)
    assert _shard_mismatches(fx, (ref_nope, ref_rope), (bad.nope, bad.rope)) == [
        (1, "RoPE")
    ]
    assert not _close(merged(bad)[t_new], ref[t_new])

    # A kernel without the overflow pass loses the skewed token's entries
    # past the cap.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            sparse_mla_kernel,
            "sparse_ragged_paged_attention",
            _stand_in_kernel([], drop_overflow=True),
        )
        bad = _dcp(fx, fx.nope_dcp, fx.rope_dcp, inp)
    assert not _close(merged(bad)[t_skew], ref[t_skew])

    # The function masks exactly the ranks owning nothing; their dummy entry
    # reads a real row for A and an unwritten (NaN) slot for B.
    raw_out, raw_lse, owns = _raw_partials(
        fx, inp, good.nope, good.rope, split_q=split_q
    )
    np.testing.assert_allclose(good.lse[owns], raw_lse[owns], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(good.out[owns], raw_out[owns], rtol=2e-2, atol=2e-2)
    assert not owns[0, t_new] and np.isfinite(raw_lse[0, t_new]).all()
    assert np.isfinite(raw_out[0, t_new]).all()
    assert not owns[1, t_b] and np.isnan(raw_out[1, t_b]).all()
    # Without the -inf, rank 0's dummy row joins the merge.
    unmasked = _lse_merge(raw_out, raw_lse)
    assert not _close(unmasked[t_new], ref[t_new])
    # With the -inf but without the zeroing, 0 * NaN poisons B's token.
    unzeroed = _lse_merge(raw_out, np.where(owns[..., None], raw_lse, -np.inf))
    assert not np.isfinite(unzeroed[t_b]).all()

    # An unrotated query under `tq_prerotated=True`, and a wrong k_scale.
    assert not _close(
        merged(_dcp(fx, fx.nope_dcp, fx.rope_dcp, inp, ql=inp.ql_raw)), ref
    )
    assert not _close(merged(_dcp(fx, fx.nope_dcp, fx.rope_dcp, inp, k_scale=1.0)), ref)


# ---------------------------------------------------------------------------
# DCP vs unsharded: the tests (multi-device, so in a CPU subprocess)
# ---------------------------------------------------------------------------

_WORKER_ENV = "SPARSE_MLA_DCP_TURBOQUANT_WORKER"
_WORKER_DEVICES = 4
_CASES = {
    "d2_i8_split_q": (2, 8, True),
    "d4_i4_concat_q": (4, 4, False),
}


def _run_worker(request, node):
    flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+",
        "",
        os.environ.get("XLA_FLAGS", ""),
    )
    count = f"--xla_force_host_platform_device_count={_WORKER_DEVICES}"
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} {count}".strip(),
        **{_WORKER_ENV: "1"},
    )
    plugins = [arg for p in request.config.option.plugins or () for arg in ("-p", p)]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *plugins,
            f"{__file__}::{node}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
    assert "1 passed" in proc.stdout, proc.stdout[-3000:]


@pytest.mark.parametrize("case", list(_CASES))
def test_dcp_matches_unsharded_on_turboquant_caches(request, case):
    """Over two steps, every rank's shard is the owner-sliced unsharded
    cache byte for byte, the row-mate of a half-row write survives, and the
    LSE-merged partials match the unsharded output (and both a direct
    dequantize-then-attend reference), with the overflow pass, ranks owning
    nothing and NaN behind their dummy entry all exercised."""
    _run_worker(request, f"test_equivalence_worker[{case}]")


def test_dcp_negative_controls_fail_the_same_checks(request):
    """A rank dropping the norm when it packs, a kernel without the overflow
    pass, a merge without the `-inf` or without the zeroing for ranks owning
    nothing, an unrotated query and a wrong `k_scale` each fail."""
    _run_worker(request, "test_negative_controls_worker")


if os.environ.get(_WORKER_ENV) == "1":

    @pytest.mark.parametrize("case", list(_CASES))
    def test_equivalence_worker(monkeypatch, case):
        assert len(jax.devices()) >= _WORKER_DEVICES, jax.devices()
        _equivalence(monkeypatch, *_CASES[case])

    def test_negative_controls_worker(monkeypatch):
        assert len(jax.devices()) >= _WORKER_DEVICES, jax.devices()
        _negative_controls(monkeypatch)


@pytest.mark.tpu_test
def test_dcp_matches_unsharded_on_tpu(monkeypatch):
    """The same two steps on two TPU devices with the real SparseCore
    writers, gather and kernel (no stand-ins)."""
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 2:
        pytest.skip("needs two TPU devices")
    _equivalence(monkeypatch, 2, 8, True, stand_in=False)


# ---------------------------------------------------------------------------
# The adapter boundary
# ---------------------------------------------------------------------------


class _RecordedOp:
    """What `pallas.jax_op` / `pcp_streaming_jax_op` were handed; called, it
    records its arguments and returns `outputs(*args)`."""

    def __init__(self, name, fn, outputs, **kwargs):
        self.name, self.fn, self.outputs, self.kwargs = name, fn, outputs, kwargs
        self.calls = []

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        self.calls.append(args)
        return self.outputs(*args)


def _record_ops(monkeypatch, outputs=None):
    from vllm_torchtpu.layers.adapter import attention as attention_adapter

    built = []

    def record(name, fn, **kwargs):
        built.append(_RecordedOp(name, fn, outputs, **kwargs))
        return built[-1]

    monkeypatch.setattr(attention_adapter, "pcp_streaming_jax_op", record)
    monkeypatch.setattr(attention_adapter.pallas, "jax_op", record)
    return built


def _impl(dcp_size=2, interleave=8):
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = dcp_size
    impl.dcp_interleave_size = interleave
    impl._dcp_mesh = object()
    return impl


def _layer(*, turboquant=True):
    """What the op builders and `forward` read from a sparse MLA layer."""
    if turboquant:
        specs = _tq_specs(SHARD_PAGE)
    else:
        specs = tuple(
            SparseMLAKVCacheSpec.create(
                cache_type, KVCacheLayout.SPARSECORE, PAGES, SHARD_PAGE, dim
            )
            for cache_type, dim in ((KVCacheType.NOPE, NOPE), (KVCacheType.ROPE, ROPE))
        )
    gen = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        mla_kv_spec=specs,
        kv_cache_dtype="turboquant_4bit_nc" if turboquant else "fp8",
        # A TurboQuant layer's quantized dtype is its uint8 code storage.
        kv_cache_quantized_dtype=torch.uint8 if turboquant else torch.float8_e4m3fn,
        _k_scale_float=K_SCALE,
        scale=SM_SCALE,
        num_heads=HEADS,
        kv_lora_rank=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        W_UK_T=(
            torch.randn(HEADS, QK_NOPE, NOPE, generator=gen) / QK_NOPE**0.5
        ).bfloat16(),
        W_UV=(torch.randn(HEADS, NOPE, V_DIM, generator=gen) / NOPE**0.5).bfloat16(),
    )


@pytest.mark.parametrize("turboquant", [True, False], ids=["turboquant", "fp8"])
def test_dcp_op_builder_hands_the_tables_and_the_traced_rotation_flag(
    monkeypatch, turboquant
):
    """The DCP op of a TurboQuant layer hands `sparse_mla_attention_dcp` the
    latent tables, the rotation flag as it stands when the op is traced (the
    load-time fold may set it after the op is built), and the `k_scale` the
    non-DCP op hands `sparse_mla_attention`. An fp8 layer's DCP op passes no
    TurboQuant arguments."""
    from vllm_torchtpu.layers.adapter import attention as attention_adapter

    built = _record_ops(monkeypatch)
    monkeypatch.setattr(
        attention_adapter,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh="mesh"),
    )
    seen = {"dcp": [], "full": []}
    monkeypatch.setattr(
        attention_interface,
        "sparse_mla_attention_dcp",
        lambda *args, **kwargs: seen["dcp"].append(kwargs),
    )
    monkeypatch.setattr(
        attention_interface,
        "sparse_mla_attention",
        lambda *args, **kwargs: seen["full"].append(kwargs),
    )
    impl = _impl()
    layer = _layer(turboquant=turboquant)
    impl._build_sparse_mla_dcp_op(layer, 2, 8)
    impl._build_sparse_mla_op(layer)
    dcp_op, full_op = built
    for rotated in (True, False):
        layer._tq_weights_rotated = rotated
        dcp_op.fn(*range(11))
        full_op.fn(*range(11))

    dcp_kw, full_kw = seen["dcp"], seen["full"]
    assert [kw["k_scale"] for kw in dcp_kw] == [K_SCALE] * 2
    assert [kw["k_scale"] for kw in full_kw] == [K_SCALE] * 2
    assert all(kw["dcp_size"] == 2 and kw["interleave_size"] == 8 for kw in dcp_kw)
    if not turboquant:
        assert not any("tq_tables" in kw or "tq_prerotated" in kw for kw in dcp_kw)
        return
    tables = tq.make_mla_latent_tables()
    assert all(kw["tq_tables"] is tables for kw in dcp_kw + full_kw)
    assert [kw["tq_prerotated"] for kw in dcp_kw] == [True, False]
    assert [kw["tq_prerotated"] for kw in full_kw] == [True, False]


def test_forward_hands_the_dcp_op_an_unquantized_latent_and_an_fp8_k_pe(
    monkeypatch,
):
    """Under DCP, `forward` on a TurboQuant layer quantizes only `k_pe` (fp8,
    with the layer's `k_scale`) and hands the DCP op the latent as it came:
    the TurboQuant branch of `_quantized_kv_latents` wins over the layer's
    uint8 `kv_cache_quantized_dtype`."""
    from vllm_torchtpu.layers.adapter import attention as attention_adapter
    from vllm_torchtpu.layers.core.quantization import quantize_kv

    dcp_size, tokens = 2, 4

    def outputs(*args):
        partial = torch.zeros(tokens, dcp_size * HEADS, NOPE, dtype=torch.bfloat16)
        return args[0], args[1], partial, torch.zeros(tokens, dcp_size * HEADS)

    built = _record_ops(monkeypatch, outputs)
    impl = _impl(dcp_size)
    layer = _layer()
    layer.sparse_mla_dcp_op = impl._build_sparse_mla_dcp_op(layer, dcp_size, 8)
    (op,) = built
    monkeypatch.setattr(
        attention_adapter, "all_gather_heads", lambda t: t.repeat(1, dcp_size, 1)
    )
    monkeypatch.setattr(
        attention_adapter,
        "merge_lse_partials_scatter_heads",
        lambda out, lse, **kwargs: out[:, :HEADS],
    )
    gen = torch.Generator().manual_seed(1)
    q_nope = torch.randn(tokens, HEADS, QK_NOPE, generator=gen).bfloat16()
    q_pe = torch.randn(tokens, HEADS, ROPE, generator=gen).bfloat16()
    kv_c = torch.randn(tokens, NOPE, generator=gen).bfloat16()
    k_pe = torch.randn(tokens, ROPE, generator=gen).bfloat16()
    md = SimpleNamespace(
        seq_lens=torch.ones(1, dtype=torch.int32),
        block_tables=torch.zeros(PAGES_PER_SEQ, dtype=torch.int32),
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    topk = torch.zeros(dcp_size * tokens, TOPK, dtype=torch.int32)
    out = impl.forward(
        layer,
        (q_nope, q_pe),
        kv_c,
        k_pe,
        (torch.zeros(1), torch.zeros(1)),
        md,
        topk_indices=topk,
    )

    (args,) = op.calls
    assert args[4].dtype == torch.bfloat16
    assert torch.equal(args[4], kv_c)
    want_pe, _ = quantize_kv(torch.float8_e4m3fn, k_pe, value=None, k_scale=K_SCALE)
    assert args[5].dtype == torch.float8_e4m3fn
    assert torch.equal(args[5].view(torch.uint8), want_pe.view(torch.uint8))
    assert args[6] is topk
    assert out.shape == (tokens, HEADS * V_DIM)


@pytest.mark.parametrize(
    ("nope_bytes", "with_tables", "prerotated", "error", "match"),
    [
        (tq.MLA_TQ_NOPE_BYTES, True, False, NotImplementedError, "pre-rotated"),
        (NOPE, True, True, ValueError, "does not match TurboQuant"),
        (tq.MLA_TQ_NOPE_BYTES, False, False, ValueError, "does not match an fp8"),
    ],
    ids=["unrotated_weights", "tables_on_an_fp8_cache", "turboquant_cache_no_tables"],
)
def test_dcp_refuses(nope_bytes, with_tables, prerotated, error, match):
    """Unrotated weights (the merge would need every partial unrotated
    first), and a NoPE token width that disagrees with the tables."""
    tokens, dcp = 4, 2
    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, PAGES, SHARD_PAGE, nope_bytes
    )
    rope_spec = _tq_specs(SHARD_PAGE)[1]
    # The checks run before the mesh is used; any object with the DCP axis
    # passes the mesh checks ahead of them.
    mesh = SimpleNamespace(axis_names=(DCP_AXIS_NAME,), shape={DCP_AXIS_NAME: dcp})
    with pytest.raises(error, match=match):
        attention_interface.sparse_mla_attention_dcp(
            jnp.zeros((tokens, HEADS, NOPE), jnp.bfloat16),
            jnp.zeros((tokens, HEADS, ROPE), jnp.bfloat16),
            jnp.zeros((tokens, NOPE), jnp.bfloat16),
            jnp.zeros((tokens, ROPE), jnp.float8_e4m3fn),
            None,
            None,
            jnp.zeros((dcp * tokens, TOPK), jnp.int32),
            None,
            None,
            None,
            None,
            mesh,
            nope_spec,
            rope_spec,
            k_scale=K_SCALE,
            dcp_size=dcp,
            interleave_size=8,
            tq_tables=tq.make_mla_latent_tables() if with_tables else None,
            tq_prerotated=prerotated,
        )
