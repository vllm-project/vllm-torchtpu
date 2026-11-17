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
"""MSA index reuse (IndexCache) under DCP, interpret mode: the layer
patterns, the default path unchanged, a Shared layer equal to its own
forward with the Full layer's selection forced in, and an accuracy proxy
(output cosine of reused vs own selection on correlated synthetic layers)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack, rounded, unpack
from tests.models.minimax_m3.test_msa_dcp import (
    HEAD_DIM,
    HEADS,
    PAGE,
    PAGES,
    SCALE,
    _batch,
    _merge_lse,
    _xla_write_rows,
)
from vllm_torchtpu.models.minimax_m3 import index_reuse
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_fp4, msa_write
from vllm_torchtpu.opt_in_config import DEFAULTS


def _xla_write_rows_many(
    caches, rows, pages, slots, *, max_runs, bits=None, interpret=False
):
    return tuple(
        _xla_write_rows(c, r, pages, slots, max_runs=max_runs)
        for c, r in zip(caches, rows)
    )


@pytest.fixture(autouse=True)
def _scatter_writes(monkeypatch):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)
    monkeypatch.setattr(msa_write, "write_rows_many", _xla_write_rows_many)


# ------------------------------------------------------------- patterns
def test_patterns():
    full = index_reuse.parse_pattern
    assert full("", 6) == full("1", 6) == frozenset(range(6))
    assert full("2", 6) == full("stride:2", 6) == {0, 2, 4}
    assert full("4", 10) == {0, 4, 8}
    assert full("full:0,1,5", 6) == {0, 1, 5}
    assert full("FFSFSS", 6) == {0, 1, 3}
    for bad in ("0", "stride:0", "full:1,2", "FSF", "SFFFFF", "x", "full:0,9"):
        with pytest.raises(ValueError, match=index_reuse.KEY):
            full(bad, 6)


def test_plan_keys_and_index_writes():
    keys = (
        index_reuse.KEY,
        index_reuse.PREFILL_KEY,
        index_reuse.DECODE_KEY,
        index_reuse.SKIP_INDEX_WRITE_KEY,
    )
    assert all(k in DEFAULTS for k in keys)
    off = index_reuse.plan_from_config({}, 8)
    assert not off.active and all(off.is_full(i, p) for i in range(8) for p in (0, 1))
    # Disabling one kind under an active base key: "1".
    plan = index_reuse.plan_from_config({keys[0]: "2", keys[2]: "1", keys[3]: True}, 8)
    assert plan.active
    assert plan.full_prefill == {0, 2, 4, 6} and plan.full_decode == set(range(8))
    assert all(plan.writes_index(i) for i in range(8))  # all Full in decode
    plan = index_reuse.plan_from_config({keys[1]: "2", keys[2]: "4", keys[3]: True}, 8)
    assert plan.full_decode == {0, 4}
    assert [plan.writes_index(i) for i in range(8)] == [1, 0, 1, 0, 1, 0, 1, 0]
    plan = index_reuse.plan_from_config({keys[0]: "4"}, 8)
    assert all(plan.writes_index(i) for i in range(8))  # skip key off


# ------------------------------------------------------- DCP layer helpers
_Q_LENS, _CTX_LENS = [1, 130, 3, 1], [1100, 200, 0, 40]
TOPK, CONFIG = 3, dict(init_blocks=0, local_blocks=1)


def _setup(seed, dcp, dtype=jnp.bfloat16):
    rng = np.random.default_rng(seed)
    logical_pages = -(-max(np.add(_Q_LENS, _CTX_LENS)) // (PAGE * dcp))
    num_tokens, seq_lens, cu_q, tables, _ = _batch(
        rng, _Q_LENS, _CTX_LENS, dcp, logical_pages
    )
    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    return rng, num_tokens, meta


def _layer_inputs(rng, num_tokens, dcp, dtype, base=None, eps=1.0, pages=PAGES):
    """One layer's caches and step inputs; with `base` (another layer's),
    its index keys / queries are ``base + eps * noise`` (correlated
    layers), the rest independent."""

    def normal(*shape):
        return rng.standard_normal(shape)

    def bf16(x):
        return jnp.asarray(x).astype(jnp.bfloat16)

    layer = {}
    index_rows = normal(dcp, pages, PAGE, HEAD_DIM)
    iq = normal(num_tokens, 1, HEAD_DIM)
    if base is not None:
        index_rows = base["index_rows_raw"] + eps * index_rows
        iq = base["iq_raw"] + eps * iq
    layer["index_rows_raw"], layer["iq_raw"] = index_rows, iq
    layer["kv"] = [
        pack(rounded(normal(pages, PAGE, 2 * HEAD_DIM), dtype), dtype)
        for _ in range(dcp)
    ]
    layer["index"] = [pack(rounded(r, dtype), dtype) for r in index_rows]
    layer["q"] = bf16(normal(num_tokens, HEADS, HEAD_DIM))
    layer["k"] = bf16(normal(num_tokens, 1, HEAD_DIM))
    layer["v"] = bf16(normal(num_tokens, 1, HEAD_DIM))
    layer["iq"] = bf16(iq)
    layer["ik"] = bf16(normal(num_tokens, HEAD_DIM))
    return layer


def _full_layer(layer, meta, dcp):
    """Every rank's write + candidates, the all-gather, and the merged
    attention: (caches per rank, gathered candidates, merged out)."""
    distribution = jnp.asarray([0, 0, len(_Q_LENS)], jnp.int32)
    caches, cands = [], []
    for rank in range(dcp):
        kv, index, cand = msa_dcp.msa_write_and_candidates_dcp(
            layer["k"],
            layer["v"],
            layer["iq"],
            layer["ik"],
            layer["kv"][rank],
            layer["index"][rank],
            *meta,
            distribution,
            topk=TOPK,
            rank=rank,
            dcp=dcp,
            interpret=True,
            **CONFIG,
        )
        caches.append((kv, index))
        cands.append(cand)
    return caches, jnp.stack(cands)


def _attend(q, caches, meta, dcp, candidates=None, selected=None):
    """The merged ``[T, H, D]`` output of every rank's partial over
    `candidates` (`msa_attend_dcp`) or a merged `selected`
    (`msa_attend_selected_dcp`)."""
    outs, lses = [], []
    for rank in range(dcp):
        common = dict(sm_scale=SCALE, rank=rank, dcp=dcp, interpret=True)
        if selected is None:
            out, lse = msa_dcp.msa_attend_dcp(
                q, caches[rank][0], candidates, *meta, topk=TOPK, **common
            )
        else:
            out, lse = msa_dcp.msa_attend_selected_dcp(
                q, caches[rank][0], selected, *meta, **common
            )
        outs.append(np.asarray(out.astype(jnp.float32)).transpose(1, 0, 2))
        lses.append(np.asarray(lse).T)
    return outs, lses


# ------------------------------------------------- key off: same results
def test_attend_split_is_bitwise():
    """`msa_attend_dcp` (the default path) = merge + `msa_attend_selected_dcp`,
    and `return_selected` adds the merge without changing the outputs."""
    dcp = 2
    rng, num_tokens, meta = _setup(1, dcp)
    layer = _layer_inputs(rng, num_tokens, dcp, jnp.bfloat16)
    caches, cands = _full_layer(layer, meta, dcp)
    selected = msa_dcp.merge_head_candidates(cands, TOPK)
    for rank in range(dcp):
        common = dict(sm_scale=SCALE, rank=rank, dcp=dcp, interpret=True)
        ref = msa_dcp.msa_attend_dcp(
            layer["q"], caches[rank][0], cands, *meta, topk=TOPK, **common
        )
        out, lse, sel = msa_dcp.msa_attend_dcp(
            layer["q"],
            caches[rank][0],
            cands,
            *meta,
            topk=TOPK,
            return_selected=True,
            **common,
        )
        np.testing.assert_array_equal(np.asarray(sel), np.asarray(selected))
        for want, got in zip(ref, (out, lse)):
            np.testing.assert_array_equal(np.asarray(want), np.asarray(got))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_shared_write_matches_full_write(dtype):
    """A Shared layer's writes (`msa_write_dcp`) are the Full forward's, byte
    for byte; without the index cache it writes only the K/V."""
    dcp = 2
    rng, num_tokens, meta = _setup(2, dcp, dtype)
    layer = _layer_inputs(rng, num_tokens, dcp, dtype)
    caches, _ = _full_layer(layer, meta, dcp)
    for rank in range(dcp):
        args = (layer["k"], layer["v"], layer["ik"], layer["kv"][rank])
        kv, index = msa_dcp.msa_write_dcp(
            *args, layer["index"][rank], *meta, rank=rank, dcp=dcp, interpret=True
        )
        (kv_only,) = msa_dcp.msa_write_dcp(
            *args, None, *meta, rank=rank, dcp=dcp, interpret=True
        )
        np.testing.assert_array_equal(unpack(kv), unpack(caches[rank][0]))
        np.testing.assert_array_equal(unpack(index), unpack(caches[rank][1]))
        np.testing.assert_array_equal(unpack(kv_only), unpack(caches[rank][0]))


def test_fp4_kv_write_matches_full_write():
    """`msa.write_fp4_kv` writes the K/V nibbles and scale rows of
    `msa.write_fp4` (G = 2 heads)."""
    rng = np.random.default_rng(5)
    g, d, t = 2, 128, 5
    kv_cache = msa_fp4.pack_rows(jnp.zeros((2, 128, 2 * d * g), jnp.uint8))
    index_cache = msa_fp4.pack_rows(
        jnp.zeros((2, 128, d), jnp.uint8), scales=jnp.zeros((2, 128), jnp.float32)
    )
    scale_cache = jnp.zeros((2, msa_fp4.scale_rows(g), 128), jnp.float32)
    k = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    ik = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    pages = jnp.asarray([1, 1, 1, 0, 0], jnp.int32)
    pos = jnp.asarray([126, 127, 128, 3, 4], jnp.int32)
    want_kv, _, want_sc = msa.write_fp4(
        kv_cache,
        index_cache,
        scale_cache,
        k,
        v,
        ik,
        pages,
        pos,
        max_num_seqs=2,
        interpret=True,
    )
    got_kv, got_sc = msa.write_fp4_kv(
        kv_cache, scale_cache, k, v, pages, pos, max_num_seqs=2, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(got_kv), np.asarray(want_kv))
    np.testing.assert_array_equal(np.asarray(got_sc), np.asarray(want_sc))


# --------------------------------------- key on: the Full layer's selection
@pytest.mark.parametrize("dcp", [2, 4])
def test_shared_layer_equals_forced_full_selection(dcp):
    """Layer B Shared after Full layer A: B's merged output equals B's own
    forward with A's gathered candidates forced into its attend, bit for
    bit, on every rank; B's own (unused) index scoring would differ."""
    rng, num_tokens, meta = _setup(10 + dcp, dcp)
    a = _layer_inputs(rng, num_tokens, dcp, jnp.bfloat16)
    b = _layer_inputs(rng, num_tokens, dcp, jnp.bfloat16)
    _, cands_a = _full_layer(a, meta, dcp)
    # Full layer A hands its merged selection on (the op's attend_and_select).
    selected_a = msa_dcp.merge_head_candidates(cands_a, TOPK)
    # Shared layer B: writes only, then attends over A's selection.
    shared_caches = []
    for rank in range(dcp):
        shared_caches.append(
            msa_dcp.msa_write_dcp(
                b["k"],
                b["v"],
                b["ik"],
                b["kv"][rank],
                b["index"][rank],
                *meta,
                rank=rank,
                dcp=dcp,
                interpret=True,
            )
        )
    got = _attend(b["q"], shared_caches, meta, dcp, selected=selected_a)
    # Reference: B's Full forward (its own writes) with A's candidates forced.
    full_caches_b, cands_b = _full_layer(b, meta, dcp)
    want = _attend(b["q"], full_caches_b, meta, dcp, candidates=cands_a)
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.stack(w), np.stack(g))
    own_b = msa_dcp.merge_head_candidates(cands_b, TOPK)
    num_real = sum(_Q_LENS)
    assert np.any(np.asarray(own_b)[:num_real] != np.asarray(selected_a)[:num_real])


# ------------------------------------------------------- accuracy proxy
def _cosine(x, y):
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    num = (x * y).sum(-1)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return num / np.maximum(den, 1e-30)


def test_accuracy_proxy_reused_vs_own_selection():
    """Correlated layers (B's index keys / queries = A's + eps * noise) over
    long decode contexts (~31 blocks per request on each of 2 ranks, top-3 +
    the local block): the cosine of B's output under A's selection against
    B's own selection, per real token, and the selections' overlap, fall as
    the layers decorrelate; identical index inputs give the same output."""
    dcp, ctx, pages = 2, 8000, 140
    q_lens = [1, 1, 1, 4]
    rng = np.random.default_rng(30)
    num_real = sum(q_lens)
    num_tokens = num_real + 3
    seq_lens = np.asarray(ctx + np.asarray(q_lens), np.int32)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    logical = -(-int(seq_lens.max()) // (PAGE * dcp))
    tables = rng.permutation(np.arange(1, pages))[: 4 * logical].reshape(4, logical)
    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    a = _layer_inputs(rng, num_tokens, dcp, jnp.bfloat16, pages=pages)
    _, cands_a = _full_layer(a, meta, dcp)
    sel_a = np.asarray(msa_dcp.merge_head_candidates(cands_a, TOPK))
    stats = {}
    for eps in (0.0, 0.3, 1.0, 3.0):
        b = _layer_inputs(
            np.random.default_rng(31),
            num_tokens,
            dcp,
            jnp.bfloat16,
            base=a,
            eps=eps,
            pages=pages,
        )
        caches_b, cands_b = _full_layer(b, meta, dcp)
        sel_b = np.asarray(msa_dcp.merge_head_candidates(cands_b, TOPK))
        own = _merge_lse(*_attend(b["q"], caches_b, meta, dcp, candidates=cands_b))
        reused = _merge_lse(*_attend(b["q"], caches_b, meta, dcp, candidates=cands_a))
        cos = _cosine(own[:num_real], reused[:num_real])
        overlap = np.mean(
            [
                len(set(x[x >= 0]) & set(y[y >= 0])) / max(1, len(set(y[y >= 0])))
                for x, y in zip(sel_a[:num_real, 0], sel_b[:num_real, 0])
            ]
        )
        stats[eps] = (round(float(cos.mean()), 4), round(float(overlap), 3))
    print("eps -> (mean output cosine, selection overlap):", stats)
    assert stats[0.0] == (1.0, 1.0)
    assert stats[0.3][1] >= stats[3.0][1] and stats[0.3][0] >= stats[3.0][0]
    assert stats[3.0][1] < 0.9  # decorrelated layers pick other blocks


# ------------------------------------------------------- build / lowering
def test_reuse_ops_build_only_with_the_key():
    """The DCP op factory builds the reuse ops only with ``index_reuse`` (off:
    the same op and program set as before)."""
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    built = []
    real = msa_ops._dcp_reuse_ops
    try:
        msa_ops._dcp_reuse_ops = lambda *a, **kw: built.append(kw) or real(*a, **kw)
        msa_ops.get_msa_dcp_op.cache_clear()
        for fp4 in (False, True):
            msa_ops.get_msa_dcp_op(SCALE, 16, 0, 1, 4, 2, fp4=fp4)
        assert built == []
        for fp4 in (False, True):
            msa_ops.get_msa_dcp_op(SCALE, 16, 0, 1, 4, 2, fp4=fp4, index_reuse=True)
        assert built == [{"fp4": False}, {"fp4": True}]
    finally:
        msa_ops._dcp_reuse_ops = real
        msa_ops.get_msa_dcp_op.cache_clear()


@pytest.mark.parametrize("fp4", [False, True])
def test_reuse_jax_ops_lower_for_tpu(fp4, monkeypatch):
    """A Shared layer's writes (with and without index keys) and its attend
    over a selection lower to Mosaic without a TPU (G = 2 KV heads)."""
    import jax
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    tokens, max_seqs, pages_per_seq, dcp, g, heads = 256, 16, 1024, 4, 2, 32
    i32, bf16 = jnp.int32, jnp.bfloat16
    sds = jax.ShapeDtypeStruct
    meta = (
        sds((max_seqs,), i32),
        sds((max_seqs * pages_per_seq,), i32),
        sds((max_seqs + 1,), i32),
    )
    rank = sds((1,), i32)
    if fp4:
        kv = sds((4096, PAGE // 8, 4, 2 * g * HEAD_DIM), jnp.uint8)
        index = sds((4096, PAGE // 8 + 1, 4, HEAD_DIM), jnp.uint8)
        scale = sds((4096, msa_fp4.scale_rows(g), PAGE), jnp.float32)
    else:
        kv = sds((4096, PAGE // 2, 2, 2 * g * HEAD_DIM), bf16)
        index = sds((4096, PAGE // 2, 2, HEAD_DIM), bf16)
        scale = None
    k = sds((tokens, g, HEAD_DIM), bf16)
    ik = sds((tokens, HEAD_DIM), bf16)

    def write(k, v, ik, kv, ix, sc, *m):
        *m, r = m
        return msa_dcp.msa_write_dcp(
            k, v, ik, kv, ix, *m, rank=r, dcp=dcp, scale_cache=sc
        )

    def write_kv(k, v, kv, sc, *m):
        *m, r = m
        return msa_dcp.msa_write_dcp(
            k, v, None, kv, None, *m, rank=r, dcp=dcp, scale_cache=sc
        )

    def attend(q, kv, sc, sel, *m):
        *m, r = m
        return msa_dcp.msa_attend_selected_dcp(
            q, kv, sel, *m, sm_scale=SCALE, rank=r, dcp=dcp, scale_cache=sc
        )

    cases = (
        (write, (k, k, ik, kv, index, scale, *meta, rank)),
        (write_kv, (k, k, kv, scale, *meta, rank)),
        (
            attend,
            (
                sds((tokens, heads, HEAD_DIM), bf16),
                kv,
                scale,
                sds((tokens, g, 16), i32),
                *meta,
                rank,
            ),
        ),
    )
    monkeypatch.undo()  # the Pallas writes, not the XLA scatters
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        for fn, args in cases:
            lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()


def test_attend_merge_builds_selection_programs_only_with_the_key(monkeypatch):
    """minimax_dcp_attend_merge (one KV head per DCP group): index reuse adds
    the Full ("return") and Shared ("given") attend+merge programs; off, only
    the default one is built."""
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    built = []
    monkeypatch.setattr(msa_ops, "ATTEND_MERGE", True)
    monkeypatch.setattr(
        msa_ops,
        "_get_msa_dcp_attend_merge_op",
        lambda *a, **kw: built.append((a[3], a[5] if len(a) > 5 else "candidates"))
        or object(),
    )
    try:
        for reuse in (False, True):
            for fp4 in (False, True):
                msa_ops.get_msa_dcp_op.cache_clear()
                built.clear()
                msa_ops.get_msa_dcp_op(
                    SCALE, 16, 0, 1, 2, 2, fp4=fp4, index_reuse=reuse
                )
                want = [(fp4, "candidates")]
                if reuse:
                    want += [(fp4, "return"), (fp4, "given")]
                assert built == want
    finally:
        msa_ops.get_msa_dcp_op.cache_clear()
