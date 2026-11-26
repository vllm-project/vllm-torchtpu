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
"""CPU JAX unit tests for DSA indexer qprep, DCP row compaction, and metadata."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import indexer_qprep, lse_rescale
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_topk_mod
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk_dcp as sc_dcp_mod
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import kv_cache_append, metadata
from vllm_torchtpu.kernels.mla import absorb_q as absorb_q_mod

streamindex_topk_mod = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


def _imported_names(mod) -> list[str]:
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    return [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for a in node.names
    ]


def test_qprep_kernel_interpret_matches_reference():
    kq, kc = jax.random.split(jax.random.PRNGKey(42))
    t, n_head, head_dim, rope_dim = 16, 8, 128, 64
    q = jax.random.normal(kq, (t, n_head * head_dim), dtype=jnp.bfloat16)
    angles = jax.random.uniform(kc, (t, rope_dim // 2), dtype=jnp.float32)
    cos, sin = (
        jnp.cos(angles).astype(jnp.bfloat16),
        jnp.sin(angles).astype(jnp.bfloat16),
    )
    c = jnp.concatenate(
        [cos, cos, jnp.ones((t, head_dim - rope_dim), jnp.bfloat16)], axis=-1
    )
    s = jnp.concatenate(
        [-sin, sin, jnp.zeros((t, head_dim - rope_dim), jnp.bfloat16)], axis=-1
    )
    q_ref, scale_ref = indexer_qprep.reference(
        q, c, s, n_head=n_head, rope_dim=rope_dim
    )
    q_got, scale_got = indexer_qprep.indexer_q_prep(
        q, c, s, n_head=n_head, rope_dim=rope_dim, block_tokens=8, interpret=True
    )
    assert q_got.shape == (t, n_head, head_dim) and scale_got.shape == (t, n_head)
    np.testing.assert_allclose(
        np.asarray(scale_got), np.asarray(scale_ref), rtol=1e-2, atol=5e-5
    )
    np.testing.assert_allclose(
        np.asarray(q_got.astype(jnp.float32)),
        np.asarray(q_ref.astype(jnp.float32)),
        rtol=0.13,
        atol=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(q_got.astype(jnp.float32)) * np.asarray(scale_got)[..., None],
        np.asarray(q_ref.astype(jnp.float32)) * np.asarray(scale_ref)[..., None],
        rtol=0.13,
        atol=2e-2,
    )


def test_compact_owned_rows_preserves_order_and_respects_bound():
    dcp_size, interleave_size, num_seqs, num_tokens = 4, 4, 3, 48
    src = jnp.arange(num_tokens * 8, dtype=jnp.uint8).reshape(num_tokens, 8)
    pattern = np.array(
        [
            10 + i if (i // interleave_size) % dcp_size == 1 else -1
            for i in range(num_tokens)
        ],
        dtype=np.int32,
    )
    kw = dict(num_seqs=num_seqs, dcp_size=dcp_size, interleave_size=interleave_size)
    bound = kv_cache_append.owned_rows_bound(num_tokens, **kw)
    comp_src, comp_dst = kv_cache_append.compact_owned_rows(
        src, jnp.asarray(pattern), **kw
    )
    assert comp_src.shape == (bound, 8) and comp_dst.shape == (bound,)
    valid_idx = np.flatnonzero(pattern >= 0)
    got_dst = np.asarray(comp_dst)
    np.testing.assert_array_equal(got_dst[: len(valid_idx)], pattern[valid_idx])
    assert np.all(got_dst[len(valid_idx) :] == -1)
    np.testing.assert_array_equal(
        np.asarray(comp_src)[: len(valid_idx)], np.asarray(src)[valid_idx]
    )


def test_metadata_tighter_ub_and_compression_ratio_preserved():
    seq_lens = jnp.array([260, 512, 0], dtype=jnp.int32)
    cu_q = jnp.array([0, 1, 2, 2], dtype=jnp.int32)
    dist = jnp.array([2, 2, 2], dtype=jnp.int32)
    for ratio, want in ((1, [260, 512, 0, 0]), (4, [65, 128, 0, 0])):
        eff = streamindex_topk_mod._effective_row_lengths(
            seq_lens,
            cu_q,
            dist,
            num_tokens=4,
            num_positions=1024,
            compression_ratio=ratio,
            cp_rank=0,
            cp_size=1,
            interleave_c=1,
        )
        np.testing.assert_array_equal(np.asarray(eff), want)
    meta = metadata.compute_batched_seq_metadata(
        seq_lens=jnp.array([1024, 512, 0, 0], jnp.int32),
        cu_q_lens=jnp.array([0, 1, 2, 2, 2], jnp.int32),
        start_seq_idx=jnp.int32(0),
        end_seq_idx=jnp.int32(2),
        bq_sz=4,
        bkv_sz=128,
        pages_per_seq=8,
        page_size=64,
        max_num_tokens=8,
        compression_ratio=1,
        static_q_len=1,
        seq_batch_size=4,
        cp_rank=jnp.int32(0),
        cp_size=4,
        interleave_c=4,
    )
    assert (
        int(np.asarray(meta.num_steps).reshape(-1)[0]) > 0
        and meta.bq_idx.shape == meta.bkv_idx.shape
    )


@pytest.mark.parametrize(
    "fn_name,want_dtype",
    [
        ("lse_rescale_head_major_f32", jnp.float32),
        ("lse_rescale_head_major_bf16", jnp.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "tokens,dim", [(1, 128), (7, 128), (8, 128), (130, 128), (16, 64)]
)
def test_lse_rescale_head_major_jit_shapes_and_nan_inf_safety(
    fn_name, want_dtype, tokens, dim
):
    fn = jax.jit(getattr(lse_rescale, fn_name))
    k_out, k_lse = jax.random.split(jax.random.PRNGKey(tokens * 31 + dim))
    out = jax.random.normal(k_out, (tokens, 8, dim), jnp.bfloat16).at[0, 0].set(jnp.nan)
    lse = jax.random.normal(k_lse, (tokens, 8), jnp.float32)
    merged = (lse + 0.5).at[0, 0].set(-jnp.inf)
    lse = lse.at[0, 0].set(-jnp.inf)
    if tokens > 1:
        lse, out = lse.at[-1, 1].set(-jnp.inf), out.at[-1, 1].set(jnp.nan)
    got = fn(out, lse, merged)
    assert (
        got.shape == (8, tokens, dim)
        and got.dtype == want_dtype
        and not bool(jnp.any(jnp.isnan(got)))
    )
    assert float(jnp.max(jnp.abs(got[0, 0].astype(jnp.float32)))) == 0.0
    if tokens > 1:
        assert float(jnp.max(jnp.abs(got[1, -1].astype(jnp.float32)))) == 0.0
    w_thd = jnp.exp(jnp.where(jnp.isfinite(merged), lse - merged, -jnp.inf))[..., None]
    ref = jnp.swapaxes(
        jnp.where(w_thd > 0, out.astype(jnp.float32) * w_thd, 0.0).astype(want_dtype),
        0,
        1,
    )
    tol = 2e-2 if want_dtype == jnp.bfloat16 else 2e-5
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(ref.astype(jnp.float32)),
        rtol=tol,
        atol=1e-2 if want_dtype == jnp.bfloat16 else 2e-5,
    )


@pytest.mark.parametrize(
    "tokens,heads", [(1, 16), (130, 16), (260, 16), (256, 4), (256, 8)]
)
def test_absorb_q_jit_fallback_on_unaligned_tokens_and_low_heads(tokens, heads):
    k1, k2 = jax.random.split(jax.random.PRNGKey(tokens * 7 + heads))
    q_nope = jax.random.normal(k1, (tokens, heads, 128), dtype=jnp.bfloat16)
    w_uk_t = jax.random.normal(k2, (heads, 128, 512), dtype=jnp.bfloat16) * 0.1
    got = jax.jit(absorb_q_mod.absorb_q)(q_nope, w_uk_t)
    ref = jnp.einsum(
        "thd,hdq->thq",
        q_nope.astype(jnp.float32),
        w_uk_t.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    assert got.shape == (tokens, heads, 512) and got.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(ref.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )


def test_deepseek_v4_topk_kernels_stay_pure_and_default_false():
    for mod in (sc_topk_mod, streamindex_topk_mod):
        assert not any("glm_dsa_dispatch" in n for n in _imported_names(mod))
    for fn, flags in (
        (sc_topk_mod.sparsecore_topk, ("coop_v2", "double_slice_budget")),
        (sc_topk_mod.sparsecore_topk_dcp_merge, ("coop_merge", "rank_major")),
        (streamindex_topk_mod.streamindex_topk, ("coop_v2", "double_slice_budget")),
        (
            streamindex_topk_mod.streamindex_topk_dcp,
            (
                "a2a_merge",
                "flat_merge",
                "small_all_gather",
                "coop_v2",
                "double_slice_budget",
                "coop_merge",
            ),
        ),
    ):
        sig = inspect.signature(fn)
        assert all(sig.parameters[f].default is False for f in flags)
    n_big = 4 * sc_topk_mod.MAX_SLICE_WORDS
    assert sc_topk_mod._pick_partition(32, n_big, 2048) == 4
    assert sc_topk_mod._pick_partition(32, n_big, 2048, double_slice_budget=True) == 2


def test_sparsecore_topk_dcp_peer_vmem_bounds_and_public_imports(monkeypatch):
    from jax._src.pallas import core as pallas_core

    for mod in (sc_topk_mod, sc_dcp_mod, streamindex_topk_mod):
        imps = _imported_names(mod)
        assert not any("envs" in n or "jax._src" in n for n in imps), imps

    class _DummyMesh(pallas_core.Mesh):
        supported_memory_spaces = (
            sc_dcp_mod.pltpu.MemorySpace.VMEM,
            sc_dcp_mod.pltpu.MemorySpace.SMEM,
            sc_dcp_mod.pltpu.MemorySpace.SEMAPHORE,
        )

        @property
        def shape(self):
            return {"core": 2}

    sc_info = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
    monkeypatch.setattr(
        sc_dcp_mod.pltpu, "get_tpu_info", lambda: SimpleNamespace(sparse_core=sc_info)
    )
    monkeypatch.setattr(
        sc_dcp_mod.plsc, "VectorSubcoreMesh", lambda **_kw: _DummyMesh()
    )
    monkeypatch.setattr(
        sc_dcp_mod.plsc, "ScalarSubcoreMesh", lambda **_kw: _DummyMesh()
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        sc_dcp_mod.pl,
        "kernel",
        lambda _b, *, out_type, scratch_types, **_kw: (
            captured.setdefault("scratch", scratch_types)
            and (lambda *_a: jnp.zeros(out_type.shape, out_type.dtype))
        ),
    )
    dcp_size, b, n, k = 16, 8, 256, 64
    out = sc_dcp_mod.sparsecore_topk_dcp(
        jnp.zeros((b, n), jnp.float32),
        k,
        jnp.full((b,), n, jnp.int32),
        jnp.full((b,), n, jnp.int32),
        dcp_size=dcp_size,
        axis_name="dcp",
    )
    assert out.shape == (b, k)
    expected = (
        dcp_size * sc_dcp_mod.MAX_ROWS * sc_dcp_mod.NUM_BUCKETS + sc_dcp_mod.LANES
    )
    assert captured["scratch"]["peer_vmem"].shape == (expected,)  # type: ignore[index]


def test_streamindex_topk_dcp_chunking_omits_scheduling_group_ids(monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config as s_cfg

    seen_groups: list[int | None] = []
    monkeypatch.setattr(
        streamindex_topk_mod,
        "sparsecore_topk_dcp",
        lambda scores, k, *_a, scheduling_group_id=None, **_kw: (
            seen_groups.append(scheduling_group_id),
            jnp.zeros((scores.shape[0], k), jnp.int32),
        )[1],
    )
    s_cfg.reset_scheduling_group_ids()
    mesh = jax.sharding.AbstractMesh(
        (2,), ("dcp",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(mesh):
        fn = lambda *a: streamindex_topk_mod.streamindex_topk(  # noqa: E731
            *a,
            k=64,
            compression_ratio=1,
            num_kv_pages_per_block=4,
            num_queries_per_block=8,
            cp_size=2,
            interleave_size=4,
            dcp_axis_name="dcp",
            chunk_tokens=16,
            decode_req_batch_size=8,
            decode_as_mixed=True,
        )
        jax.eval_shape(
            fn,
            jax.ShapeDtypeStruct((32, 8, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((32, 8), jnp.float32),
            jax.ShapeDtypeStruct((8, 64, 4, 256), jnp.uint8),
            jax.ShapeDtypeStruct((8,), jnp.int32),
            jax.ShapeDtypeStruct((32,), jnp.int32),
            jax.ShapeDtypeStruct((9,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )
    assert seen_groups == [None, None]
    assert s_cfg.reserve_scheduling_group_ids(1) == 1
