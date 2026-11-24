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
"""TPU_MINIMAX_M3_PT16: 16-bit SMEM page tables (`kernels/page_table.py`)
give the int32 tables' bits in the MSA attention (decode, pair, one-call,
page-cache prefill) and indexer kernels (Pallas interpret mode); the SMEM
budget doubles max_num_seqs for MiniMax M3 only; and the packed tables
lower for tpu7x at the 1M served shapes that the int32 ones overflow."""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention_lse,
    msa_fp4,
    msa_indexer,
)
from vllm_torchtpu.models.minimax_m3.kernels import page_table as pt

ENV = "TPU_MINIMAX_M3_PT16"
D, HG, PAGE, TOPK = 128, 4, 128, 16


def test_pack_roundtrip_full_range():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, pt.MAX_PAGES, 1001).astype(np.int32)
    ids[:4] = [0, 65535, 32767, 32768]
    packed = pt.pack(jnp.asarray(ids))
    assert packed.dtype == jnp.uint32 and packed.shape == (501,)
    assert pt.num_keys(packed) == 1002
    np.testing.assert_array_equal(np.asarray(pt.unpack(packed))[:1001], ids)
    keys = jnp.arange(1001, dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(pt.load(packed, keys)), ids)
    for base in (0, 100, 501, 900):  # runs within one half (H = 501)
        for g in range(4):
            run = pt.run_start(packed, jnp.int32(base))
            assert int(pt.load_run(packed, run, g)) == ids[base + g]
    assert pt.run_safe(pt.pack(jnp.zeros(6 * 512, jnp.int32)), 512, 32)
    assert not pt.run_safe(pt.pack(jnp.zeros(5 * 512, jnp.int32)), 512, 32)
    assert not pt.run_safe(pt.pack(jnp.zeros(6 * 472, jnp.int32)), 472, 32)


def test_load_in_kernel_smem():
    """A scalar-prefetched packed table read in a kernel body and an index
    map, ids past 32767 included."""
    ids = np.array([5, 65535, 40000, 7, 32768, 3, 0, 12345], np.int32)
    keys = np.array([1, 2, 4, 0, 7, 6, 3, 5], np.int32)
    pages = jnp.arange(65536, dtype=jnp.int32).reshape(65536, 1) * 2

    def kernel(keys_ref, table_ref, page_ref, o_ref):
        i = pl.program_id(0)
        o_ref[0, 0] = pt.load(table_ref, keys_ref[i])
        o_ref[0, 1] = page_ref[0, 0]

    def run(table):
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(8,),
                in_specs=[
                    pl.BlockSpec(
                        (1, 1),
                        lambda i, k, t: (pt.load(t, k[i]), 0),
                        memory_space=pltpu.SMEM,
                    )
                ],
                out_specs=pl.BlockSpec(
                    (1, 2), lambda i, *_: (i, 0), memory_space=pltpu.SMEM
                ),
            ),
            out_shape=jax.ShapeDtypeStruct((8, 2), jnp.int32),
            interpret=True,
        )(jnp.asarray(keys), table, pages)

    want = np.stack([ids[keys], 2 * ids[keys]], axis=1)
    np.testing.assert_array_equal(np.asarray(run(jnp.asarray(ids))), want)
    np.testing.assert_array_equal(np.asarray(run(pt.pack(jnp.asarray(ids)))), want)


def test_prefetch_respects_flag_and_pool(monkeypatch):
    table = jnp.arange(8, dtype=jnp.int32)
    monkeypatch.delenv(ENV, raising=False)
    assert pt.prefetch(table, 100).dtype == jnp.int32
    monkeypatch.setenv(ENV, "1")
    # Tables that fit SMEM as int32 stay int32.
    assert pt.prefetch(table, 100).dtype == jnp.int32
    big = jnp.zeros(pt.INT32_MAX_BYTES // 4 + 2, jnp.int32)
    assert pt.prefetch(big, 100).dtype == jnp.uint32
    assert pt.prefetch(big, pt.MAX_PAGES + 1).dtype == jnp.int32
    monkeypatch.setattr(pt, "INT32_MAX_BYTES", 0)
    assert pt.prefetch(table, 100).dtype == jnp.uint32


def _attention_inputs(rng, heads, reqs, fp4):
    tokens = reqs * 8
    pps = 8
    # Every key has a table entry (an out-of-range key reads garbage).
    pages = reqs * pps + 4
    table = rng.permutation(pages)[: reqs * pps].astype(np.int32)
    seq = np.arange(tokens) // 8
    sel = np.full((tokens, heads, TOPK), -1, np.int32)
    for t in range(tokens):
        for h in range(heads):
            n = rng.integers(1, pps)
            sel[t, h, :n] = seq[t] * pps + rng.choice(pps, n, replace=False)
    local = (seq * pps + rng.integers(0, pps, tokens)).astype(np.int32)
    limit = rng.integers(1, PAGE + 1, tokens).astype(np.int32)
    q = rng.standard_normal((tokens, heads * HG, D)).astype(np.float32)
    if fp4:
        nib = rng.integers(0, 16, (pages, PAGE, heads * 2 * D), dtype=np.uint8)
        kv = msa_fp4.pack_rows(jnp.asarray(np.where(nib == 8, 0, nib).astype(np.uint8)))
        sc = jnp.asarray(
            rng.uniform(0.05, 0.2, (pages, msa_fp4.scale_rows(heads), PAGE)),
            jnp.float32,
        )
    else:
        kv = jnp.asarray(
            rng.standard_normal((pages, PAGE // 4, 4, heads * 2 * D)),
            jnp.float8_e4m3fn,
        )
        sc = None
    return (
        jnp.asarray(q, jnp.bfloat16),
        kv,
        jnp.asarray(sel),
        jnp.asarray(local),
        jnp.asarray(limit),
        jnp.asarray(table),
        sc,
    )


@pytest.mark.parametrize(
    "heads,reqs,fp4,envs,kw",
    [
        (1, 3, False, (), {}),
        (2, 4, True, (), {}),
        (2, 6, True, ("pair",), {}),
        (4, 3, True, ("one",), {}),
        (1, 3, False, ("work",), {}),
        # 4-bit prefill with the page cache (`_page_cache_tables` key bound).
        (2, 40, True, (), {"page_cache_slots": 32, "block_q": 8}),
    ],
)
def test_attention_same_bits(monkeypatch, heads, reqs, fp4, envs, kw):
    q, kv, sel, local, limit, table, sc = _attention_inputs(
        np.random.default_rng(heads * 100 + reqs), heads, reqs, fp4
    )
    for name, env in (
        ("pair", (msa_attention_lse.DECODE_OPT_ENV, "pair")),
        ("one", (msa_attention_lse.ONE_CALL_ENV, "1")),
        ("work", (msa_attention_lse.WORK_LIST_KERNEL_ENV, "1")),
    ):
        if name in envs:
            monkeypatch.setenv(*env)
        else:
            monkeypatch.delenv(env[0], raising=False)

    def run(t):
        jax.clear_caches()
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q,
            kv,
            sel,
            local,
            limit,
            t,
            sm_scale=D**-0.5,
            scale_cache=sc,
            interpret=True,
            head_major=True,
            **({"block_q": 8} | kw),
        )

    ref = run(table)
    got = run(pt.pack(table))
    for a, b in zip(ref, got, strict=False):
        assert np.isfinite(np.asarray(a, np.float32)).all()
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


S = 128
CTX = [128 * 300 + 5, 128 * 150, 128 * 33 + 1, 128 * 130, 128 * 470 + 77]


def _index_case(seed, ctx, max_num_seqs, pages_per_seq):
    rng = np.random.default_rng(seed)
    num = len(ctx)
    pool = max_num_seqs * pages_per_seq + 8
    bt = rng.permutation(pool)[: max_num_seqs * pages_per_seq].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num] = ctx
    cu = np.zeros(max_num_seqs + 1, np.int32)
    cu[1 : num + 1] = np.cumsum([8] * num)
    cu[num + 1 :] = cu[num]
    key = jax.random.PRNGKey(seed)
    nib = jax.random.randint(key, (pool, S, D), 0, 16, jnp.int32).astype(jnp.uint8)
    scl = jax.random.uniform(
        jax.random.fold_in(key, 1), (pool, D), jnp.float32, 0.01, 0.03
    )
    cache = msa_fp4.pack_rows(nib, scl)
    q = jax.random.normal(jax.random.fold_in(key, 2), (8 * num, 1, D), jnp.float32)
    args = (
        q.astype(jnp.bfloat16),
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(bt),
        jnp.asarray(cu),
        jnp.asarray(num, jnp.int32),
    )
    return args, cu


@pytest.mark.parametrize("pps", [471, 512])  # 471: per-entry loads; 512: runs
@pytest.mark.parametrize("spec", ["", "noclamp,defer,lanes", "noclamp,full"])
def test_indexer_same_bits(monkeypatch, spec, pps):
    """The served decode indexer (kernel v3, 4-bit folded cache): packing
    happens inside `block_scores` from the flag."""
    args, cu = _index_case(3, CTX, 6, pps)
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_DECODE_OPT", spec)
    monkeypatch.setattr(pt, "INT32_MAX_BYTES", 0)  # pack the small table

    def run(flag, lower=False):
        monkeypatch.setenv(ENV, flag)
        jax.clear_caches()
        f = jax.jit(
            functools.partial(
                msa_indexer.block_scores.__wrapped__,
                interpret=True,
                fp8_q=False,
                **msa.block_score_options(1),
            )
        )
        if lower:
            return f.lower(*args).as_text()
        return np.asarray(f(*args), np.float32)

    ref, got = run("0"), run("1")
    for i, c in enumerate(CTX):
        b = ref[cu[i] : cu[i + 1], 0, : -(-c // S)]
        g = got[cu[i] : cu[i + 1], 0, : -(-c // S)]
        assert np.isfinite(b).all()
        np.testing.assert_array_equal(g, b)
    assert "ui32" in run("1", lower=True) and run("1", True) != run("0", True)


def _config(arch, max_num_seqs, mml, dcp):
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=dcp
        ),
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=types.SimpleNamespace(max_model_len=mml, architectures=[arch]),
    )


@pytest.mark.parametrize(
    "arch,flag,seqs,dcp,page",
    [
        ("MiniMaxM3SparseForCausalLM", "0", 32, 2, 128),
        ("MiniMaxM3SparseForCausalLM", "0", 48, 2, 256),
        ("MiniMaxM3SparseForCausalLM", "1", 48, 2, 128),
        ("MiniMaxM3SparseForCausalLM", "1", 64, 2, 128),
        ("MiniMaxM3SparseForCausalLM", "1", 80, 2, 256),
        ("MiniMaxM3SparseForCausalLM", "0", 96, 4, 256),
        ("MiniMaxM3SparseForCausalLM", "1", 96, 4, 128),
        ("MiniMaxM3SparseForCausalLM", "1", 128, 4, 128),
        ("MiniMaxM3SparseForConditionalGeneration", "1", 128, 4, 128),
        ("Qwen3ForCausalLM", "1", 48, 2, 256),
    ],
)
def test_min_page_size_budget(monkeypatch, arch, flag, seqs, dcp, page):
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackend,
        get_tpu_min_page_size,
        page_table_entry_bytes,
    )

    monkeypatch.setenv(ENV, flag)
    cfg = _config(arch, seqs, 1 << 20, dcp)
    assert get_tpu_min_page_size(cfg) == page
    # The runner's per-step row cap (`num_reqs_max_model_len`) agrees.
    pages = (1 << 20) // dcp // 128
    cap = PallasAttentionBackend.get_max_num_seqs_for_pages(
        pages, page_table_entry_bytes(cfg)
    )
    assert (cap >= seqs) == (page == 128)
