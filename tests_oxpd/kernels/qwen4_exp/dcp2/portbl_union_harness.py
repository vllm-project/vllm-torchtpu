# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL UNION: CPU harness for the union RPA verify attention on the O1s attend
(VLLM_TPU_QSA_DECODE_RPA_UNION_O1S; ported from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
benliangw/q38fh-integ-pin-5b44702aea, whose CPU tests emulate the metadata kernel the same way).

Neither kernel the union path launches runs on CPU (PORT-BL-UNION probe1, j-bb3b29c2: batched RPA fails under
pltpu.InterpretParams with "'BitcastTransform' object has no attribute 'indices'" and under the legacy interpreter
with a dynamic-shape error; the SparseCore metadata kernel needs a SparseCore). Everything else is production code:

* the metadata kernel's entry ``qsa_union_meta.union_sc_metadata_staged`` is replaced by ``sc_standin``, which runs
  the kernel's OWN per-request code (``qsa_union_meta.request_body``) on numpy stand-ins for the SparseCore
  primitives (``test_qwen4_exp_union_meta._NumpyOps``), with the kernel's staging (garbage in every scratch buffer,
  buffers reused across a worker's requests) and its ``write="tile"`` output (slots past the chunk holding U are
  garbage), inside a ``jax.pure_callback``. ``union_sc_metadata``'s XLA staging (``stage_inputs``) is the real one.
* batched RPA (``wrapper.ragged_paged_attention``) is replaced by one of two stand-ins with the kernel's contract
  (per sequence: tokens ``t < kv_lens`` of its pages; with ``selection_bits`` query ``j`` attends token ``t`` when bit
  ``j`` of the word is set; rows past the sequences keep the aliased query and LSE -inf; a query with no key gets
  finite garbage):
  - ``keyset``: EXACT. Each query's output holds integer checksums of the set of virtual tokens it attends (count
    and six modular sums, each below 2^24 so f32 holds it exactly); LSE is the count. Union on and off must agree
    bit for bit: each row attends exactly the keys it attends on the per-row path.
  - ``tiled``: the kernel's online softmax (flash_attention_qk_softmax's order of operations) over KV tiles of the
    call's ``decode_block_sizes.bkv_sz`` (512 per-row, 640 union) in sequence order, masked keys at the f32 minimum.
    Union on and off then differ only by the order of the sums: the NB measurement (CPU float, an inferred proxy for
    the TPU's MXU accumulation order).

Counters record what ran on device (callbacks inside ``lax.cond`` branches fire only for the branch taken).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_meta as um

COUNTS = {"sc": 0, "union_rpa": 0, "rows_rpa": 0}
GARBAGE_OUT = 4242.0
PRIMES = (1999, 1997, 1993, 1987, 1979, 1973)
MULTS = (1, 31, 977, 7919, 104729, 1299709)


def reset_counts():
    for k in COUNTS:
        COUNTS[k] = 0


def _bump(key, n=1):
    COUNTS[key] += int(n)


# --------------------------------------------------------------------------------------------------------------
# SparseCore metadata kernel stand-in: the kernel's per-request code on numpy.


def emulate_staged(blocks, header, tab, *, cols, pages, world, own_list, write, workers=8, seed=0):
    """``union_sc_metadata_staged``'s outputs from the kernel's staging and ``request_body`` on numpy
    (``write`` "tile" / "live" / "full" as the kernel copies them; "tile" leaves garbage past U's chunk)."""
    from test_qwen4_exp_union_meta import GARBAGE, LANES, _NumpyOps

    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks, np.int32)
    header, tab = np.asarray(header, np.int32), np.asarray(tab, np.int32)
    reqs, cols16 = tab.shape
    pos_words = -(-cols * um.PAGE_GROUPS // LANES) * LANES
    ops = _NumpyOps(races=False)
    out_pages = rng.integers(-9, 1 << 20, (reqs, pages)).astype(np.int32)  # unwritten HBM: garbage
    out_words = rng.integers(-9, 1 << 30, (reqs, pages)).astype(np.int32)
    out_meta = np.full((reqs, um.META), GARBAGE, np.int32)
    for w in range(workers):
        def junk(n):
            return rng.integers(-5, 3000, n).astype(np.int32)

        rows_v, head_v, tab_v = junk(um.MAX_ROWS * um.TOPK), junk(2 * LANES), junk(cols16)
        pos_v, pages_v, words_v, meta_v = junk(pos_words), junk(pages + 32), junk(pages + 32), junk(um.META)
        for r in range(w, reqs, workers):
            head_v[: um.META] = header[r]
            tab_v[:] = tab[r]
            start, n = int(header[r, um.H_START]), int(header[r, um.H_ROWS])
            for j in range(n):
                rows_v[j * um.TOPK : (j + 1) * um.TOPK] = blocks[start + j]
            um.request_body(ops, rows_v, head_v, tab_v, pos_v, pages_v, words_v, meta_v, pages=pages,
                            pos_words=pos_words, world=world, own_list=own_list)
            out_meta[r] = meta_v
            if write == "full":
                count = 1
                chunk = pages
            else:
                chunk = um.WRITE_CHUNK
                u = int(meta_v[0])
                count = 1 + sum(1 for c in range(1, pages // chunk) if u > c * chunk)
            out_pages[r, : count * chunk] = pages_v[: count * chunk]
            out_words[r, : count * chunk] = words_v[: count * chunk]
    return out_pages, out_words, out_meta


def _widen(words):
    shifts = np.arange(um.RATIO, dtype=np.int32) * 8
    return ((words[..., None] >> shifts) & 0xFF).reshape(words.shape[0], -1).astype(np.int32)


def sc_standin(blocks, header, tab, *, cols, world=4, pages_per_seq=2080, own_list=False, clear="full",
               insert="store", unroll=4, out_dma="async", write="full", emit_sel=False):
    """Drop-in for ``qsa_union_meta.union_sc_metadata_staged`` (same argument checks, same outputs)."""
    del clear, insert, unroll, out_dma
    assert blocks.shape[1] == um.TOPK, blocks.shape
    reqs = header.shape[0]
    width = um.RATIO if emit_sel else 1

    def host(b, h, t):
        _bump("sc")
        pages, words, meta = emulate_staged(b, h, t, cols=cols, pages=pages_per_seq, world=world, own_list=own_list,
                                            write=write, seed=int(np.asarray(h).sum()) & 0xFFFF)
        second = _widen(words) if emit_sel else words
        if write == "live":
            live = np.arange(pages_per_seq)[None, :] < meta[:, :1]
            pages = np.where(live, pages, 0)
            second = np.where(np.repeat(live, width, axis=1), second, 0)
        return pages.astype(np.int32), second.astype(np.int32), meta.astype(np.int32)

    shapes = (jax.ShapeDtypeStruct((reqs, pages_per_seq), jnp.int32),
              jax.ShapeDtypeStruct((reqs, pages_per_seq * width), jnp.int32),
              jax.ShapeDtypeStruct((reqs, um.META), jnp.int32))
    return jax.pure_callback(host, shapes, blocks.astype(jnp.int32), header.astype(jnp.int32),
                             tab.astype(jnp.int32))


# --------------------------------------------------------------------------------------------------------------
# Batched RPA stand-ins.


def _sequences(queries, kv_cache, kv_lens, page_indices, cu_q_lens, decode_block_sizes, selection_bits, kv_slot,
               k_scale, v_scale):
    """Per sequence s and query r < R: K / V of its tokens, the attend mask, the virtual token ids and the query."""
    seqs = kv_lens.shape[0]
    pps = page_indices.shape[0] // seqs
    page = kv_cache.shape[1]
    tokens = pps * page
    combined = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
    slot = 0 if kv_slot is None else int(kv_slot)
    pages = page_indices.reshape(seqs, pps)
    gathered = combined[pages]  # [S, pps, page, slots, D]
    rows_k = gathered[:, :, :, 2 * slot]  # [S, pps, page, D]
    rows_v = gathered[:, :, :, 2 * slot + 1]
    key = rows_k.reshape(seqs, tokens, -1).astype(jnp.float32) * k_scale
    value = rows_v.reshape(seqs, tokens, -1).astype(jnp.float32) * v_scale
    t = jnp.arange(tokens, dtype=jnp.int32)
    ids = (pages[:, t // page] * page + t % page).astype(jnp.int32)  # [S, T] virtual token ids
    reach = decode_block_sizes.bq_sz if selection_bits is not None else 1
    r = jnp.arange(reach, dtype=jnp.int32)
    nq = cu_q_lens[1:] - cu_q_lens[:-1]
    valid_q = r[None, :] < nq[:, None]  # [S, R]
    rows = jnp.clip(cu_q_lens[:-1, None] + r[None, :], 0, queries.shape[0] - 1)  # [S, R]
    visible = t[None, :] < kv_lens[:, None]  # [S, T]
    if selection_bits is not None:
        sel = selection_bits.reshape(seqs, -1)[:, :tokens]
        bits = ((sel[:, None, :] >> r[None, :, None]) & 1) == 1  # [S, R, T]
        mask = visible[:, None, :] & bits
    else:
        mask = jnp.broadcast_to(visible[:, None, :], (seqs, reach, tokens))
    mask = mask & valid_q[:, :, None]
    q = queries[rows].astype(jnp.float32)  # [S, R, H, D]
    return key, value, ids, mask, valid_q, rows, q


def _write_rows(queries, rows, valid_q, values, lse_values):
    out = queries.astype(jnp.float32)
    lse = jnp.full(queries.shape[:2], -jnp.inf, jnp.float32)
    flat_rows = jnp.where(valid_q, rows, queries.shape[0]).reshape(-1)  # out of range = dropped
    out = out.at[flat_rows].set(values.reshape(-1, *values.shape[2:]), mode="drop")
    lse = lse.at[flat_rows].set(lse_values.reshape(-1, lse_values.shape[-1]), mode="drop")
    return out, lse


def _record(selection_bits, nseq):
    if selection_bits is not None:
        jax.debug.callback(functools.partial(_bump, "union_rpa"))
    else:
        jax.debug.callback(functools.partial(_bump, "rows_rpa"))
    del nseq


def keyset_rpa(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, *, sm_scale,
               k_scale, v_scale, out_dtype, kv_layout, skip_kv_update, return_lse=False, decode_block_sizes=None,
               selection_bits=None, kv_slot=None, read_only_cache=False, **_):
    """EXACT stand-in: per query, integer checksums of the attended virtual-token set."""
    del keys, values, distribution, sm_scale, kv_layout, out_dtype
    assert skip_kv_update
    _record(selection_bits, kv_lens.shape[0])
    _, _, ids, mask, valid_q, rows, q = _sequences(queries, kv_cache, kv_lens, page_indices, cu_q_lens,
                                                   decode_block_sizes, selection_bits, kv_slot, k_scale, v_scale)
    count = jnp.sum(mask, axis=-1).astype(jnp.int32)  # [S, R]
    sums = [count]
    for p, m in zip(PRIMES, MULTS):
        h = ((ids % p) * (m % p) + 7) % p  # [S, T], int32 without overflow
        sums.append(jnp.sum(jnp.where(mask, h[:, None, :], 0), axis=-1, dtype=jnp.int32))
    sig = jnp.stack(sums, axis=-1).astype(jnp.float32)  # [S, R, 7], each < 2^24
    heads, dim = q.shape[2], q.shape[3]
    vals = jnp.zeros((*count.shape, heads, dim), jnp.float32)
    vals = vals.at[..., 0, : sig.shape[-1]].set(sig)
    vals = vals.at[..., :, dim - 1].set(count[..., None].astype(jnp.float32))
    has = (count > 0)[..., None, None]
    vals = jnp.where(has, vals, GARBAGE_OUT)
    lse_vals = jnp.where(count > 0, count.astype(jnp.float32), GARBAGE_OUT)
    lse_vals = jnp.broadcast_to(lse_vals[..., None], (*count.shape, heads))
    out, lse = _write_rows(queries, rows, valid_q, vals, lse_vals)
    result = (out, kv_cache) if not return_lse else (out, kv_cache, lse)
    return result


def tiled_rpa(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, *, sm_scale,
              k_scale, v_scale, out_dtype, kv_layout, skip_kv_update, return_lse=False, decode_block_sizes=None,
              selection_bits=None, kv_slot=None, read_only_cache=False, mask_value=None, **_):
    """The kernel's online softmax over KV tiles of ``decode_block_sizes.bkv_sz`` tokens."""
    del keys, values, distribution, kv_layout, out_dtype
    assert skip_kv_update
    _record(selection_bits, kv_lens.shape[0])
    key, value, _, mask, valid_q, rows, q = _sequences(queries, kv_cache, kv_lens, page_indices, cu_q_lens,
                                                       decode_block_sizes, selection_bits, kv_slot, 1.0, v_scale)
    bkv = decode_block_sizes.bkv_sz
    seqs, tokens = key.shape[0], key.shape[1]
    ntiles = -(-tokens // bkv)
    pad = ntiles * bkv - tokens
    key = jnp.pad(key, ((0, 0), (0, pad), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, pad), (0, 0)))
    mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad)))
    mval = jnp.finfo(jnp.float32).min if mask_value is None else mask_value
    heads, dim = q.shape[2], q.shape[3]
    reach = q.shape[1]
    m = jnp.full((seqs, reach, heads), -jnp.inf, jnp.float32)
    l_ = jnp.zeros((seqs, reach, heads), jnp.float32)
    acc = jnp.zeros((seqs, reach, heads, dim), jnp.float32)
    hi = jax.lax.Precision.HIGHEST
    for k in range(ntiles):
        sl = slice(k * bkv, (k + 1) * bkv)
        qk = jnp.einsum("srhd,std->srht", q, key[:, sl], precision=hi) * sm_scale * k_scale
        qk = jnp.where(mask[:, :, None, sl], qk, mval)
        m_cur = jnp.max(qk, axis=-1)
        m_next = jnp.maximum(m, m_cur)
        alpha = jnp.exp(m - m_next)
        p = jnp.exp(qk - m_next[..., None])
        l_ = alpha * l_ + jnp.sum(p, axis=-1)
        acc = alpha[..., None] * acc + jnp.einsum("srht,std->srhd", p, value[:, sl], precision=hi)
        m = m_next
    vals = acc / l_[..., None]
    lse_vals = m + jnp.log(l_)
    has = jnp.any(mask, axis=-1)[..., None, None]
    vals = jnp.where(has, vals, GARBAGE_OUT)
    lse_vals = jnp.where(has[..., 0], lse_vals, GARBAGE_OUT)
    out, lse = _write_rows(queries, rows, valid_q, vals, lse_vals)
    return (out, kv_cache) if not return_lse else (out, kv_cache, lse)


STANDINS = {"keyset": keyset_rpa, "tiled": tiled_rpa}


# --------------------------------------------------------------------------------------------------------------
# One O1s verify step at served dims (main-KV DCP 4, 512-token pages, 512 selected 4-token groups a row).

PAGE, RATIO, WORLD, TOPK, TP = 512, 4, 4, 512, 8
DH = 256


def make_step(seed, tokens, slots=64, heads=12, pattern_mix=None, rows_per_request=4, max_query_len=4):
    """A verify step of ``tokens`` rows: requests of 4 rows (one of 1 row, one of 3), contexts crossing DCP pages,
    selections identical / partially overlapping / disjoint across a request's rows, one request whose
    selections all sit on one DCP rank (multi-tile per-row and union), and short requests that keep every group.
    Returns the operands of ``pooled_qsa_request_attend_dcp_step`` (but the rank) and a description."""
    rng = np.random.default_rng(seed)
    nreq = tokens // rows_per_request
    assert nreq <= slots
    query_lens = [rows_per_request] * nreq
    if rows_per_request == 4:
        query_lens[1] = 1  # a request of one row (query length 1) inside the verify step
        query_lens[2] = 3
    patterns = pattern_mix or ["identical", "partial", "disjoint", "onerank", "short", "partial", "partial",
                               "identical"]
    pats = [patterns[i % len(patterns)] for i in range(nreq)]
    pats[0] = "disjoint"  # every step holds a long request: the merged-selection branch of _shard_attend
    contexts = []
    for i, p in enumerate(pats):
        if p == "short":
            contexts.append(int(rng.integers(40, 2000)))
        elif p in ("disjoint", "onerank"):
            contexts.append(int(rng.integers(33000, 36000)))
        else:
            contexts.append(int(rng.integers(2600, 9000)))
    starts = np.zeros(slots + 1, np.int32)
    starts[1 : nreq + 1] = np.cumsum(query_lens)
    starts[nreq + 1 :] = starts[nreq]
    live = int(starts[nreq])
    lengths = np.zeros(tokens, np.int32)
    req = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    for i in range(nreq):
        a, n = int(starts[i]), query_lens[i]
        for j in range(n):
            lengths[a + j] = contexts[i] + j + 1
            req[a + j] = i
        completes = [int(lengths[a + j]) // RATIO for j in range(n)]
        c0 = completes[0]
        p = pats[i]
        if p == "short" and c0 <= TOPK:
            for j in range(n):
                cj = completes[j]
                row = rng.permutation(cj)
                junk = rng.integers(-1, max(cj, 1), TOPK - cj)
                blocks[a + j] = np.concatenate([row, junk])
            continue
        recent = np.arange(c0 - 64, c0)
        if p == "identical":
            base = np.concatenate([recent, rng.choice(c0 - 64, TOPK - 64, replace=False)])
            for j in range(n):
                blocks[a + j] = rng.permutation(base)
        elif p == "partial":
            core = np.concatenate([recent, rng.choice(c0 - 64, 384, replace=False)])
            rest = np.setdiff1d(np.arange(c0 - 64), core)
            for j in range(n):
                own = rng.choice(rest, TOPK - len(core), replace=False)
                blocks[a + j] = rng.permutation(np.concatenate([core, own]))
        elif p == "disjoint":
            perm = rng.permutation(c0)
            for j in range(n):
                blocks[a + j] = perm[j * TOPK : (j + 1) * TOPK]
        elif p == "onerank":
            rank = i % WORLD
            groups = np.arange(c0)
            owned = groups[(groups // (PAGE // RATIO)) % WORLD == rank]
            shared = rng.choice(owned, 256, replace=False)
            rest = np.setdiff1d(owned, shared)
            for j in range(n):
                blocks[a + j] = rng.permutation(np.concatenate([shared, rng.choice(rest, 256, replace=False)]))
        else:
            raise ValueError(p)
    span = WORLD * PAGE
    need = [int(-(-(c + 8) // span)) for c in contexts]  # table entries each request uses
    cols = max(need) + 1
    npages = 2 + sum(need)
    phys = iter((1 + rng.permutation(npages - 2)).tolist())
    table = np.zeros((slots, cols), np.int32)  # unused entries: page 0, past every request's tokens
    for i, n in enumerate(need):
        table[i, :n] = [next(phys) for _ in range(n)]
    q = jnp.asarray(rng.normal(0, 1, (tokens, heads, DH)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1, (tokens, heads, DH)), jnp.bfloat16)
    context = np.zeros(slots, np.int32)
    context[:nreq] = contexts
    # request_shard_blocks inverse: rank r's slice holds flat rows [r * w, r * w + w).
    rows_per = tokens // TP
    align = max_query_len
    w = int(np.clip(((live + TP - 1) // TP + align - 1) // align * align, 1, rows_per))
    gathered = np.full((TP, rows_per, TOPK), -1, np.int32)
    for t in range(live):
        r = min(t // w, TP - 1)
        gathered[r, t - r * w] = blocks[t]
    return dict(q=q, gate=gate, lengths=jnp.asarray(lengths), req=jnp.asarray(req), starts=jnp.asarray(starts),
                main_table=jnp.asarray(table), context=jnp.asarray(context), gathered=jnp.asarray(gathered),
                npages=npages, blocks=blocks, live=live, patterns=pats, query_lens=query_lens)


def random_pool(seed, npages):
    from dcp2_harness import random_fp8

    return random_fp8(np.random.default_rng(seed + 1000), (npages, PAGE, 1, 4, DH))
