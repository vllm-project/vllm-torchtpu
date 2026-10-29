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
"""DCP decode with the MLA v2 kernel, emulating `d` ranks on one device.

A global latent cache is split the way vLLM shards it (rank `r` owns positions
`p` with `(p // interleave) % d == r`, at its local slots in order, under the
same block-table ids on every rank). Each rank runs
`mla_decode_dcp_local` on its shard, the partials are merged by LSE in f32 and
compared with exact attention over the whole sequence.

The CPU tests run the per-rank logic against `_kernel_model`, a plain restatement
of the kernel's contract; the `tpu_test` runs the real kernel.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.v2 import dcp

PAGE, PACK, LKV, R, LANES = 256, 4, 512, 64, 640
SM_SCALE = 0.07
FP8 = jnp.float8_e4m3fn


def _fp8(rng, shape, scale):
    return np.array(
        jnp.asarray(rng.standard_normal(shape, np.float32) * scale).astype(FP8)
    )


def _stale(rng, shape):
    """Random finite fp8 bytes, mostly large, so a leak into the softmax is loud."""
    raw = rng.integers(0, 256, shape, dtype=np.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x7E  # e4m3fn's only NaN encodings
    return raw.view(FP8)


def make_case(
    seq_lens,
    *,
    d,
    interleave,
    heads,
    max_num_seqs=None,
    pages_per_seq=None,
    num_pages=None,
    seed=0,
):
    """A decode step over `seq_lens` (each including its new token), sharded.

    The sequence and query data depend only on `seq_lens` and `seed`, not on `d`.
    """
    seq_lens = np.asarray(seq_lens, np.int32)
    b = len(seq_lens)
    tokens = max(b, 8)
    rng = np.random.default_rng(seed)
    kv = [np.zeros((n, LANES), FP8) for n in seq_lens]
    for s, n in enumerate(seq_lens):
        kv[s][:, : LKV + R] = _fp8(rng, (n, LKV + R), 0.5)
    ql, qpe = _q(rng, (heads, tokens, LKV)), _q(rng, (tokens, heads, R))

    rng = np.random.default_rng((seed, d))
    max_num_seqs = max_num_seqs or b
    cycle = PAGE * d
    blocks = [-(-int(n) // cycle) for n in seq_lens]
    pages_per_seq = pages_per_seq or max(blocks) + 1
    table = np.zeros((max_num_seqs, pages_per_seq), np.int32)
    nxt = 1  # id 0 stays unused, like vLLM's null block
    for s, nb in enumerate(blocks):
        table[s, :nb] = rng.permutation(np.arange(nxt, nxt + nb))
        nxt += nb
    num_pages = num_pages or nxt + 1
    assert num_pages > nxt

    caches = [_stale(rng, (num_pages, PAGE // PACK, PACK, LANES)) for _ in range(d)]
    for s, n in enumerate(seq_lens):
        # The cache holds everything but the new token.
        ranks, pages, slots = locate(np.arange(n - 1), table[s], d, interleave)
        for r in range(d):
            mine = ranks == r
            rows = kv[s][:-1][mine]
            caches[r][pages[mine], slots[mine] // PACK, slots[mine] % PACK] = rows

    new = np.zeros((tokens, LANES), FP8)
    new[:b] = np.stack([kv[s][n - 1] for s, n in enumerate(seq_lens)])
    cu = np.full(max_num_seqs + 1, b, np.int32)
    cu[: b + 1] = np.arange(b + 1)
    lens = np.zeros(max_num_seqs, np.int32)
    lens[:b] = seq_lens
    return dict(
        kv=kv,
        caches=caches,
        ql=ql,
        qpe=qpe,
        new_kv_c=jnp.asarray(new[:, :LKV]),
        new_k_pe=jnp.asarray(new[:, LKV : LKV + R]),
        seq_lens=jnp.asarray(lens),
        page_indices=jnp.asarray(table.reshape(-1)),
        table=table,
        cu=jnp.asarray(cu),
        distribution=jnp.asarray([b, b, b], jnp.int32),
        b=b,
    )


def _q(rng, shape):
    return jnp.asarray(rng.standard_normal(shape, np.float32) * 0.5).astype(
        jnp.bfloat16
    )


def locate(pos, table_row, d, interleave):
    """(rank, page, slot) of global positions `pos`, as vLLM's slot mapping."""
    rank = (pos // interleave) % d
    local = (pos // (interleave * d)) * interleave + pos % interleave
    return rank, table_row[local // PAGE], local % PAGE


def reference(case):
    """Exact f32 decode attention over each whole sequence: `[H, b, LKV]`."""
    outs = []
    for s, kv in enumerate(case["kv"]):
        k = kv.astype(np.float32)
        q = np.concatenate(
            [
                np.asarray(case["ql"][:, s].astype(jnp.float32)),
                np.asarray(case["qpe"][s].astype(jnp.float32)),
            ],
            axis=-1,
        )
        scores = q @ k[:, : LKV + R].T * SM_SCALE
        p = np.exp(scores - scores.max(-1, keepdims=True))
        outs.append((p / p.sum(-1, keepdims=True)) @ k[:, :LKV])
    return np.stack(outs, axis=1)


def merge(outs, lses):
    """LSE merge in f32 of `[d, H, T, L]` partials with `[d, T, H]` lse."""
    lse = np.asarray(lses, np.float32).transpose(0, 2, 1)[..., None]
    w = np.exp(lse - lse.max(0, keepdims=True))
    return (w * np.asarray(outs, np.float32)).sum(0) / w.sum(0)


def _kernel_model(
    ql_nope,
    q_pe,
    new_kv_c,
    new_k_pe,
    cache_kv,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    sm_scale,
    return_lse,
    **_,
):
    """The kernel's contract: the last `q_len` of `kv_len` keys come from the new
    rows and are stored at their slots; queries are causal at the tail."""
    assert return_lse
    cache = np.array(cache_kv)
    kv_lens, cu = np.asarray(kv_lens), np.asarray(cu_q_lens)
    table = np.asarray(page_indices).reshape(kv_lens.shape[0], -1)
    new = np.zeros((new_kv_c.shape[0], LANES), FP8)  # k_pe is zero-padded to a tile
    new[:, : LKV + R] = np.concatenate([np.asarray(new_kv_c), np.asarray(new_k_pe)], -1)
    heads, tokens, _ = ql_nope.shape
    out = np.zeros((heads, tokens, LKV), np.float32)
    lse = np.zeros((tokens, heads), np.float32)
    for s in range(int(distribution[2])):
        k_len, q0, q1 = kv_lens[s], cu[s], cu[s + 1]
        for i, slot in enumerate(range(k_len - (q1 - q0), k_len)):
            row = cache[table[s, slot // PAGE], (slot % PAGE) // PACK, slot % PACK]
            row[:] = new[q0 + i]
        k = np.stack(
            [
                cache[table[s, j // PAGE], (j % PAGE) // PACK, j % PACK]
                for j in range(k_len)
            ]
        ).astype(np.float32)
        for i, t in enumerate(range(q0, q1)):
            q = np.concatenate(
                [
                    np.asarray(ql_nope[:, t], np.float32),
                    np.asarray(q_pe[t], np.float32),
                ],
                -1,
            )
            sc = q @ k[: k_len - (q1 - q0) + i + 1, : LKV + R].T * sm_scale
            m = sc.max(-1, keepdims=True)
            p = np.exp(sc - m)
            out[:, t] = (p / p.sum(-1, keepdims=True)) @ k[: len(p[0]), :LKV]
            lse[t] = (m + np.log(p.sum(-1, keepdims=True)))[:, 0]
    return jnp.asarray(out), jnp.asarray(cache), jnp.asarray(lse)


def run_ranks(case, *, d, interleave, local_fn):
    outs, lses, caches = [], [], []
    for r in range(d):
        out, cache, lse = local_fn(
            case["ql"],
            case["qpe"],
            case["new_kv_c"],
            case["new_k_pe"],
            jnp.asarray(case["caches"][r]),
            case["seq_lens"],
            case["page_indices"],
            case["cu"],
            case["distribution"],
            r,
        )
        outs.append(np.asarray(out.astype(jnp.float32)))
        lses.append(np.asarray(lse))
        caches.append(np.asarray(cache))
    return merge(outs, lses)[:, : case["b"]], caches, lses


def check_caches(case, caches, *, d, interleave):
    """Every rank's owned positions, new token included, hold the sequence."""
    for s, kv in enumerate(case["kv"]):
        ranks, pages, slots = locate(
            np.arange(len(kv)), case["table"][s], d, interleave
        )
        for r in range(d):
            mine = ranks == r
            got = caches[r][pages[mine], slots[mine] // PACK, slots[mine] % PACK]
            np.testing.assert_array_equal(
                got.view(np.uint8), kv[mine].view(np.uint8), err_msg=f"seq {s} rank {r}"
            )


SEQ_LENS = [1, 255, 256, 257, 511, 513, 1023, 1024, 1025, 2049, 3000, 77]


@pytest.mark.parametrize("d", [1, 2, 4])
@pytest.mark.parametrize("interleave", [256, 64])
def test_local_kv_lens_counts_owned_positions(d, interleave):
    n = np.arange(0, 3 * d * interleave + 7)
    for r in range(d):
        owned = (np.arange(n.max()) // interleave) % d == r
        want = np.concatenate([[0], np.cumsum(owned)])[n]
        got = dcp.local_kv_lens(
            jnp.asarray(n, jnp.int32), r, dcp_size=d, interleave_size=interleave
        )
        np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("d,interleave", [(1, 256), (4, 256), (4, 64), (3, 128)])
def test_decode_merge_matches_reference_with_kernel_model(monkeypatch, d, interleave):
    monkeypatch.setattr(dcp, "mla_ragged_paged_attention", _kernel_model)
    case = make_case(SEQ_LENS, d=d, interleave=interleave, heads=4, max_num_seqs=16)
    local_fn = functools.partial(
        dcp.mla_decode_dcp_local,
        dcp_size=d,
        interleave_size=interleave,
        sm_scale=SM_SCALE,
    )
    got, caches, lses = run_ranks(case, d=d, interleave=interleave, local_fn=local_fn)
    np.testing.assert_allclose(got, reference(case), atol=2e-5, rtol=2e-5)
    check_caches(case, caches, d=d, interleave=interleave)
    # Ranks with nothing of a sequence must be the merge identity.
    for r in range(d):
        local = np.asarray(
            dcp.local_kv_lens(
                case["seq_lens"], r, dcp_size=d, interleave_size=interleave
            )
        )
        empty = local[: case["b"]] == 0
        assert np.all(np.isneginf(lses[r][: case["b"]][empty]))


@pytest.mark.parametrize("d,interleave", [(4, 256), (2, 64)])
def test_update_kv_cache_dcp_writes_only_owned_rows(d, interleave):
    """A chunked-prefill step: each rank stores exactly its owned new rows."""
    rng = np.random.default_rng(1)
    q_lens = np.array([300, 1, 700, 0, 5])
    seq_lens = np.array([300, 900, 1500, 40, 2000])  # incl. this step's tokens
    case = make_case(seq_lens, d=d, interleave=interleave, heads=4)
    tokens = int(q_lens.sum()) + 3  # plus padding
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    rows = np.zeros((tokens, LANES), FP8)
    rows[:, : LKV + R] = _fp8(rng, (tokens, LKV + R), 0.5)
    before = [c.copy() for c in case["caches"]]
    expect = [c.copy() for c in case["caches"]]
    for s, (n, q) in enumerate(zip(seq_lens, q_lens)):
        ranks, pages, slots = locate(
            np.arange(n - q, n), case["table"][s], d, interleave
        )
        for i, (r, page, slot) in enumerate(zip(ranks, pages, slots)):
            expect[r][page, slot // PACK, slot % PACK] = rows[cu[s] + i]
    for r in range(d):
        got = dcp.update_kv_cache_dcp(
            jnp.asarray(before[r]),
            jnp.asarray(rows[:, :LKV]),
            jnp.asarray(rows[:, LKV : LKV + R]),
            jnp.asarray(seq_lens, jnp.int32),
            case["page_indices"],
            jnp.asarray(cu),
            jnp.int32(r),
            dcp_size=d,
            interleave_size=interleave,
        )
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint8), expect[r].view(np.uint8)
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("d", [8, 32])
@pytest.mark.parametrize("b", [1, 16])
def test_decode_merge_matches_reference_tpu(d, b):
    rng = np.random.default_rng(b)
    seq_lens = rng.integers(1, 3 * PAGE * d, b)
    seq_lens[0] = PAGE * d - 1
    case = make_case(seq_lens, d=d, interleave=PAGE, heads=96, max_num_seqs=128)
    local_fn = jax.jit(
        functools.partial(
            dcp.mla_decode_dcp_local,
            dcp_size=d,
            interleave_size=PAGE,
            sm_scale=SM_SCALE,
            num_kv_pages_per_block=(8, 1, 1),
            num_queries_per_block=(1, 16, 16),
            decode_batch_size=1,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        donate_argnums=(4,),
    )
    got, caches, _ = run_ranks(case, d=d, interleave=PAGE, local_fn=local_fn)
    ref = reference(case)
    assert np.abs(got - ref).max() < 2e-2, np.abs(got - ref).max()
    check_caches(case, caches, d=d, interleave=PAGE)


def make_prefill_case(ctx, q_lens, *, d, heads, max_model_len, seed=0):
    """A mixed step: `q_lens` new tokens after `ctx` cached ones per sequence,
    1-token sequences first, as the runner orders them."""
    ctx, q_lens = np.asarray(ctx), np.asarray(q_lens)
    seq_lens = ctx + q_lens
    case = make_case(
        seq_lens,
        d=d,
        interleave=PAGE,
        heads=heads,
        pages_per_seq=-(-max_model_len // (PAGE * d)),
        seed=seed,
    )
    # Keep only the context in the caches: overwrite the new tokens' slots.
    rng = np.random.default_rng(seed + 1)
    for s, (c, n) in enumerate(zip(ctx, seq_lens)):
        ranks, pages, slots = locate(np.arange(c, n), case["table"][s], d, PAGE)
        for r, page, slot in zip(ranks, pages, slots):
            case["caches"][r][page, slot // PACK, slot % PACK] = _stale(rng, LANES)
    rows = np.concatenate([kv[c:] for kv, c in zip(case["kv"], ctx)])
    tokens = len(rows)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    num_decode = int(np.sum(q_lens == 1))
    assert np.all(q_lens[:num_decode] == 1) and np.all(q_lens[num_decode:] > 1)
    case.update(
        ql=_q(rng, (heads, tokens, LKV)),
        qpe=_q(rng, (tokens, heads, R)),
        new_kv_c=jnp.asarray(rows[:, :LKV]),
        new_k_pe=jnp.asarray(rows[:, LKV : LKV + R]),
        cu=jnp.asarray(cu),
        distribution=jnp.asarray([num_decode, num_decode, len(ctx)], jnp.int32),
        ctx=ctx,
    )
    return case


def prefill_reference(case, chunk=512):
    """Exact f32 causal attention of every new token: `[H, T, LKV]`."""
    cu = np.asarray(case["cu"])
    ql = case["ql"].astype(jnp.float32)
    qpe = case["qpe"].astype(jnp.float32).transpose(1, 0, 2)
    q = jnp.concatenate([ql, qpe], -1)  # [H, T, LKV + R]
    out = np.zeros(ql.shape, np.float32)
    for s, (kv, c) in enumerate(zip(case["kv"], case["ctx"])):
        k = jnp.asarray(kv.astype(np.float32))
        for t0 in range(cu[s], cu[s + 1], chunk):
            t1 = min(t0 + chunk, cu[s + 1])
            sc = jnp.einsum(
                "htd,kd->htk", q[:, t0:t1], k[:, : LKV + R], precision="highest"
            )
            pos = c + (t0 - cu[s]) + jnp.arange(t1 - t0)
            visible = jnp.arange(k.shape[0])[None, :] <= pos[:, None]
            p = jax.nn.softmax(jnp.where(visible, sc * SM_SCALE, -jnp.inf), axis=-1)
            o = jnp.einsum("htk,kd->htd", p, k[:, :LKV], precision="highest")
            out[:, t0:t1] = np.asarray(o)
    return out


def run_prefill_ranks(case, *, d, max_model_len, kernel, **kernel_kwargs):
    """`mla_prefill_dcp_local` with the all-gather done by concatenation."""
    args = (case["seq_lens"], case["page_indices"], case["cu"], case["distribution"])
    pool = jnp.concatenate(
        [
            dcp.prefill_pool_pages(
                jnp.asarray(c), *args, dcp_size=d, max_model_len=max_model_len
            )
            for c in case["caches"]
        ]
    )
    table, kv_lens = dcp.prefill_pool_table(
        case["seq_lens"],
        case["cu"],
        case["distribution"],
        page_size=PAGE,
        dcp_size=d,
        max_model_len=max_model_len,
    )
    out = kernel(
        case["ql"],
        case["qpe"],
        case["new_kv_c"],
        case["new_k_pe"],
        pool,
        kv_lens,
        table,
        case["cu"],
        case["distribution"],
        sm_scale=SM_SCALE,
        **kernel_kwargs,
    )[0]
    first_prefill = int(np.asarray(case["cu"])[int(case["distribution"][0])])
    return np.asarray(out.astype(jnp.float32))[:, first_prefill:], first_prefill


@pytest.mark.parametrize("d", [1, 2, 4])
def test_prefill_pool_matches_reference_with_kernel_model(d):
    max_model_len = 4096
    g = -(-max_model_len // (PAGE * d))
    # Two decodes, a first chunk, and continuations whose contexts fill the
    # pool's g rows: 1 + (g - 1).
    big = (g - 2) * PAGE * d + 5
    case = make_prefill_case(
        [700, 40, 0, 200, big],
        [1, 1, 200, 77, 31],
        d=d,
        heads=4,
        max_model_len=max_model_len,
    )
    model = functools.partial(_kernel_model, return_lse=True)
    got, start = run_prefill_ranks(case, d=d, max_model_len=max_model_len, kernel=model)
    want = prefill_reference(case)[:, start:]
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)


@pytest.mark.tpu_test
@pytest.mark.parametrize("d", [8, 32])
def test_prefill_pool_matches_reference_tpu(d):
    from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention

    max_model_len = 82432
    case = make_prefill_case(
        [5000, 70000, 40960, 0],
        [1, 1, 2048, 2048],
        d=d,
        heads=3,
        max_model_len=max_model_len,
    )
    kernel = functools.partial(
        mla_ragged_paged_attention,
        num_kv_pages_per_block=(3, 1, 4),  # the K3 TP32 serving tiles
        num_queries_per_block=(1, 16, 512),
        mixed_q_split=2,
        vmem_limit_bytes=60 * 1024 * 1024,
    )
    got, start = run_prefill_ranks(
        case, d=d, max_model_len=max_model_len, kernel=kernel
    )
    err = np.abs(got - prefill_reference(case)[:, start:]).max()
    assert err < 2e-2, err
