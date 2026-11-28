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
"""Executable specification tests for deepseek_v41.indexer.paged_indexer."""

import ast
import contextlib
import dataclasses
import functools
import inspect
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.deepseek_v41 import indexer as indexer_pkg
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    DIM,
    MAX_N,
    MAX_SCORING_TOKENS,
    MAX_TOPK,
    MIN_SPARSECORE_KEYS,
    ROW_WORDS,
    SC_LANES,
    SLOTS,
    benchmark_paged_indexer,
    paged_indexer_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    paged_indexer as paged_indexer_mod,
)


@contextlib.contextmanager
def _isolated_jit_patch(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    attr: str,
    value: Any,
) -> Iterator[None]:
    paged_indexer_topk.clear_cache()
    with monkeypatch.context() as mp:
        mp.setattr(target, attr, value)
        try:
            yield
        finally:
            paged_indexer_topk.clear_cache()


@pytest.fixture(autouse=True)
def _setup_tpu7x_abstract_mesh() -> Iterator[None]:
    if jax.default_backend() == "tpu":
        yield
        return
    abstract_device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    abstract_mesh = jax.sharding.AbstractMesh(
        (), (), (), abstract_device=abstract_device
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
        yield


def _split_elements(x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-1], 64, 2).swapaxes(-1, -2).reshape(x.shape)


def _pack_keys_to_words(keys: jax.Array) -> jax.Array:
    pages, page_keys, _ = keys.shape
    page_rows = page_keys // SLOTS
    paired = keys.reshape(pages, page_rows, SLOTS, 64, 2)
    bits = jax.lax.bitcast_convert_type(paired, jnp.uint16).astype(jnp.uint32)
    packed = bits[..., 0] | (bits[..., 1] << 16)
    return packed.reshape(pages, page_rows, ROW_WORDS).astype(jnp.int32)


def _collect_imported_modules(module: object) -> set[str]:
    source = inspect.getsource(module)  # type: ignore[arg-type] -- module object
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _ref_paged_indexer_topk(
    q: np.ndarray,
    weights: np.ndarray,
    keys_all: np.ndarray,
    block_tables: np.ndarray,
    seq_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    *,
    topk: int,
    candidate_indices: np.ndarray | None = None,
    compress_ratio: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, _, _ = q.shape
    num_seqs, max_pages = block_tables.shape
    num_pages, page_keys, _ = keys_all.shape
    max_keys = max_pages * page_keys
    keys_f64 = keys_all.astype(np.float64)

    ref_indices = np.full((tokens, topk), -1, dtype=np.int32)
    full_scores = np.full((tokens, max_keys), -np.inf, dtype=np.float64)

    for seq_idx in range(num_seqs):
        q_start = max(0, int(cu_q_lens[seq_idx]))
        q_end = min(tokens, int(cu_q_lens[seq_idx + 1]))
        seq_len = int(seq_lens[seq_idx])
        bt_seq = block_tables[seq_idx]
        for token_idx in range(q_start, q_end):
            pos_t = seq_len - (int(cu_q_lens[seq_idx + 1]) - token_idx)
            valid_count = max(0, (pos_t + 1) // compress_ratio)
            eff_len = min(valid_count, max_keys)
            if eff_len == 0:
                continue

            if candidate_indices is not None:
                cands = np.unique(candidate_indices[token_idx])
                key_idxs = cands[(cands >= 0) & (cands < eff_len)].astype(np.int64)
            else:
                key_idxs = np.arange(eff_len, dtype=np.int64)
            if key_idxs.size == 0:
                continue

            logical_pages = key_idxs // page_keys
            page_offsets = key_idxs % page_keys
            phys_pages = bt_seq[logical_pages]
            mapped_mask = (phys_pages >= 0) & (phys_pages < num_pages)
            mapped_keys = key_idxs[mapped_mask]
            if mapped_keys.size == 0:
                continue

            q_t = q[token_idx].astype(np.float64)
            w_t = weights[token_idx, :, 0].astype(np.float64)
            mapped_phys = phys_pages[mapped_mask]
            used_phys = np.unique(mapped_phys)
            dots = np.maximum(keys_f64[used_phys] @ q_t.T, 0.0)
            used_scores = dots @ w_t
            phys_pos = np.searchsorted(used_phys, mapped_phys)
            full_scores[token_idx, mapped_keys] = used_scores[
                phys_pos, page_offsets[mapped_mask]
            ]

            valid_cols = np.where(np.isfinite(full_scores[token_idx]))[0]
            if valid_cols.size > 0:
                order = valid_cols[
                    np.argsort(-full_scores[token_idx, valid_cols], kind="stable")
                ]
                chosen = order[:topk]
                ref_indices[token_idx, : chosen.size] = chosen.astype(np.int32)

    return ref_indices, full_scores


def _assert_topk_set_parity(
    actual: np.ndarray,
    ref_indices: np.ndarray,
    full_scores: np.ndarray,
    *,
    rtol: float = 1e-4,
) -> None:
    for token_idx in range(actual.shape[0]):
        act_valid = [int(x) for x in actual[token_idx] if x >= 0]
        ref_valid = [int(x) for x in ref_indices[token_idx] if x >= 0]
        assert len(act_valid) == len(ref_valid)
        assert len(set(act_valid)) == len(act_valid)
        assert int(np.sum(actual[token_idx] == -1)) == int(
            np.sum(ref_indices[token_idx] == -1)
        )
        if act_valid:
            scores_t = full_scores[token_idx]
            assert all(np.isfinite(scores_t[c]) for c in act_valid)
            min_sel = min(float(scores_t[c]) for c in act_valid)
            kth_ref = float(scores_t[ref_valid[-1]])
            assert min_sel >= kth_ref - rtol * max(1.0, abs(kth_ref))


def test_should_export_single_bank_api_and_stay_pure_jax() -> None:
    # Act
    sig = inspect.signature(paged_indexer_topk)
    paged_imports = _collect_imported_modules(paged_indexer_mod)
    init_imports = _collect_imported_modules(indexer_pkg)
    bench_imports = _collect_imported_modules(benchmark_paged_indexer)

    # Assert
    assert (MAX_SCORING_TOKENS, MIN_SPARSECORE_KEYS, MAX_TOPK, MAX_N, SC_LANES) == (
        288,
        2048,
        4096,
        1_048_576,
        16,
    )
    assert list(sig.parameters) == [
        "q",
        "weights",
        "k_pool",
        "block_tables",
        "seq_lens",
        "cu_q_lens",
        "topk",
        "candidate_indices",
        "compress_ratio",
        "interpret",
        "enable_named_scope",
    ]
    assert {
        "BITS_NEG_INF",
        "MAX_N",
        "MAX_SCORING_TOKENS",
        "MAX_TOPK",
        "MIN_SPARSECORE_KEYS",
        "SC_LANES",
        "paged_indexer_topk",
        "sparsecore_topk",
    }.issubset(set(indexer_pkg.__all__))
    forbidden_runtime = {"torch", "torch_tpu", "vllm"}
    for mods in (paged_imports, init_imports, bench_imports):
        roots = {mod.split(".")[0] for mod in mods}
        assert roots.isdisjoint(forbidden_runtime)
        assert all("deepseek_v4." not in mod for mod in mods)
    assert {mod.split(".")[0] for mod in paged_imports}.isdisjoint({"os"})
    assert {mod.split(".")[0] for mod in init_imports}.isdisjoint({"os"})


@pytest.mark.parametrize("dyadic_pool", [False, True])
def test_should_match_reference_with_single_bank_paging_and_dyadic_layout(
    dyadic_pool: bool,
) -> None:
    # Arrange: keep identical static shapes across paging tests to reuse the compiled
    # CPU interpret kernel.
    rng = np.random.default_rng(101)
    tokens, heads, num_pages, page_rows, topk = 6, 4, 5, 64, 8
    page_keys = SLOTS * page_rows

    if dyadic_pool:
        q_nat = jnp.ones((tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jnp.ones((tokens, heads, 1), dtype=jnp.float32)
        page_scale = (np.arange(num_pages, dtype=np.float32)[:, None, None] + 1.0) / 8.0
        key_offset = np.arange(page_keys, dtype=np.float32)[None, :, None] / 128.0
        keys_np = np.broadcast_to(
            page_scale + key_offset, (num_pages, page_keys, DIM)
        ).astype(np.float32)
        keys_all = jnp.asarray(keys_np, dtype=jnp.bfloat16)
        block_tables = jnp.array([[3, 1, -1], [4, 2, 0]], dtype=jnp.int32)
    else:
        q_nat = jnp.array(
            rng.standard_normal((tokens, heads, DIM), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        weights = jnp.array(
            rng.uniform(0.1, 1.5, size=(tokens, heads, 1)).astype(np.float32)
        )
        keys_all = jnp.array(
            rng.standard_normal((num_pages, page_keys, DIM), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        block_tables = jnp.array([[0, 3, -1], [2, 4, 99]], dtype=jnp.int32)

    q_split = _split_elements(q_nat)
    k_pool = _pack_keys_to_words(keys_all)
    seq_lens = jnp.array([7, 350], dtype=jnp.int32)
    cu_q_lens = jnp.array([1, 3, 5], dtype=jnp.int32)

    # Act
    actual = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_pool,
            block_tables,
            seq_lens,
            cu_q_lens,
            topk=topk,
            compress_ratio=1,
            interpret=True,
        )
    )
    expected, _ = _ref_paged_indexer_topk(
        np.asarray(q_nat),
        np.asarray(weights),
        np.asarray(keys_all),
        np.asarray(block_tables),
        np.asarray(seq_lens),
        np.asarray(cu_q_lens),
        topk=topk,
        compress_ratio=1,
    )

    # Assert
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], -1)
    np.testing.assert_array_equal(actual[-1], -1)


def test_should_mask_interior_unmapped_oob_and_negative_weight_zero_scores() -> None:
    # Arrange: verify unmapped (-1) and out-of-bounds pages stay excluded even when
    # valid mapped keys produce negative or zero scores.
    tokens, heads, num_pages, page_rows, topk = 6, 4, 5, 64, 8
    page_keys = SLOTS * page_rows
    q_np = np.ones((tokens, heads, DIM), dtype=np.float32)
    q_np[1] = -1.0
    q_nat = jnp.asarray(q_np, dtype=jnp.bfloat16)
    q_split = _split_elements(q_nat)
    weights_np = -np.ones((tokens, heads, 1), dtype=np.float32)
    weights_np[1] = 1.0
    weights = jnp.asarray(weights_np)
    keys_all = jnp.ones((num_pages, page_keys, DIM), dtype=jnp.bfloat16)
    k_pool = _pack_keys_to_words(keys_all)
    bt = jnp.array(
        [[0, -1, num_pages - 1], [num_pages - 1, num_pages, 1]],
        dtype=jnp.int32,
    )
    sl = jnp.array([3 * page_keys, 3 * page_keys], dtype=jnp.int32)
    cu = jnp.array([0, 2, 3], dtype=jnp.int32)

    # Act
    actual = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_pool,
            bt,
            sl,
            cu,
            topk=topk,
            compress_ratio=1,
            interpret=True,
        )
    )
    expected, _ = _ref_paged_indexer_topk(
        np.asarray(q_nat),
        np.asarray(weights),
        np.asarray(keys_all),
        np.asarray(bt),
        np.asarray(sl),
        np.asarray(cu),
        topk=topk,
        compress_ratio=1,
    )

    # Assert: unmapped page 1 keys excluded against both < 0.0 and == 0.0 valid keys.
    np.testing.assert_array_equal(actual, expected)
    unmapped_keys = set(range(page_keys, 2 * page_keys))
    assert unmapped_keys.isdisjoint(set(actual.ravel().tolist()))
    np.testing.assert_array_equal(actual[3:], -1)


def test_should_exclude_oob_candidates_when_raw_rows_aligned_and_seq_full() -> None:
    # Arrange: raw_rows == SUB_BLOCK_ROWS (256) so total_scored_keys == max_keys (512),
    # compress_ratio=2 with full-length seqs (eff_len == 512), topk=12 > num_cand=9,
    # enable_named_scope=True, and invalid/duplicate candidates (< 0 or >= 512).
    rng = np.random.default_rng(707)
    tokens, heads, max_pages, page_rows, topk = 2, 2, 4, 64, 12
    max_keys = max_pages * page_rows * SLOTS
    q_nat = jnp.array(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32), jnp.bfloat16
    )
    q_split = _split_elements(q_nat)
    weights = jnp.array(
        rng.uniform(0.2, 1.5, size=(tokens, heads, 1)).astype(np.float32)
    )
    keys_all = jnp.array(
        rng.standard_normal((2 * max_pages, SLOTS * page_rows, DIM), dtype=np.float32),
        jnp.bfloat16,
    )
    k_pool = _pack_keys_to_words(keys_all)
    bt = jnp.arange(2 * max_pages, dtype=jnp.int32).reshape(2, max_pages)
    sl = jnp.array([max_keys * 2, max_keys * 2], dtype=jnp.int32)
    cu = jnp.array([0, 1, 2], dtype=jnp.int32)
    cand = jnp.array(
        [
            [-3, -1, 0, 17, 17, 511, max_keys, max_keys + 88, 9999],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )

    # Act
    actual = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_pool,
            bt,
            sl,
            cu,
            topk=topk,
            candidate_indices=cand,
            compress_ratio=2,
            interpret=True,
            enable_named_scope=True,
        )
    )
    ref_indices, full_scores = _ref_paged_indexer_topk(
        np.asarray(q_nat),
        np.asarray(weights),
        np.asarray(keys_all),
        np.asarray(bt),
        np.asarray(sl),
        np.asarray(cu),
        topk=topk,
        candidate_indices=np.asarray(cand),
        compress_ratio=2,
    )

    # Assert: sentinel max_keys never leaks, cand_k < topk pads with -1.
    assert np.all((actual == -1) | ((actual >= 0) & (actual < max_keys)))
    np.testing.assert_array_equal(
        np.sum(actual == -1, axis=1), np.sum(ref_indices == -1, axis=1)
    )
    np.testing.assert_array_equal(actual[1], -1)
    _assert_topk_set_parity(actual, ref_indices, full_scores)


def test_should_iterate_dynamic_fori_loop_tiles_and_break_candidate_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: lower MAX_TILE_ROWS to 256 so 5 pages (320 rows) span two dynamic
    # loop tiles; higher tile-1 scores verify exact tile-offset placement.
    tokens, heads, pages, page_rows, max_pages, topk = 2, 3, 6, 64, 5, 4
    bt_np = np.full((2, max_pages), -1, dtype=np.int32)
    bt_np[0, :2] = [0, 1]
    bt_np[1, :3] = [3, 4, 2]
    bt_np[1, 4] = 5
    bt = jnp.asarray(bt_np)
    cu = jnp.array([0, 1, 2], dtype=jnp.int32)
    cand = jnp.array(
        [
            [5, 3, 0, 5, 3, 1, 2, -1],
            [517, 515, 512, 515, 260, 16, 17, 9999],
        ],
        dtype=jnp.int32,
    )

    # Act & Assert: compile patched MAX_TILE_ROWS=256 once, verify while lowering,
    # and run both 1-tile (16, 200) and 2-tile (16, 620) active sequence lengths.
    with _isolated_jit_patch(monkeypatch, paged_indexer_mod, "MAX_TILE_ROWS", 256):
        compiled_while: Any = None
        for seq_lens_pair in ((16, 200), (16, 620)):
            rng = np.random.default_rng(1200 + seq_lens_pair[1])
            q_nat = jnp.array(
                np.abs(rng.standard_normal((tokens, heads, DIM), dtype=np.float32))
                + 0.5,
                dtype=jnp.bfloat16,
            )
            q_split = _split_elements(q_nat)
            weights = jnp.array(
                rng.uniform(0.5, 1.5, size=(tokens, heads, 1)).astype(np.float32)
            )
            keys_np = np.abs(
                rng.standard_normal((pages, SLOTS * page_rows, DIM)).astype(np.float32)
            )
            # Deliberate ties on page 0 (tile 0) and high-score page 5 (tile 1).
            keys_np[0, 3] = keys_np[0, 0]
            keys_np[0, 5] = keys_np[0, 0]
            keys_np[5] = keys_np[5] + 2.5
            keys_np[5, 3] = keys_np[5, 0]
            keys_np[5, 5] = keys_np[5, 0]
            keys_all = jnp.array(keys_np, dtype=jnp.bfloat16)
            k_pool = _pack_keys_to_words(keys_all)
            sl = jnp.array(list(seq_lens_pair), dtype=jnp.int32)

            if compiled_while is None:
                traced_while = paged_indexer_topk.trace(
                    q_split,
                    weights,
                    k_pool,
                    bt,
                    sl,
                    cu,
                    topk=topk,
                    candidate_indices=cand,
                    compress_ratio=1,
                    interpret=True,
                )
                assert any(
                    eqn.primitive.name == "while"
                    for eqn in traced_while.jaxpr.jaxpr.eqns
                )
                compiled_while = traced_while.lower().compile()

            actual = np.asarray(
                compiled_while(
                    q_split, weights, k_pool, bt, sl, cu, candidate_indices=cand
                )
            )
            expected, _ = _ref_paged_indexer_topk(
                np.asarray(q_nat),
                np.asarray(weights),
                np.asarray(keys_all),
                bt_np,
                np.asarray(sl),
                np.asarray(cu),
                topk=topk,
                candidate_indices=np.asarray(cand),
                compress_ratio=1,
            )
            np.testing.assert_array_equal(actual, expected)
            if seq_lens_pair[1] > 512:
                assert set(actual[1, :3].tolist()) == {512, 515, 517}


def _token_chunk_scan_shapes(closed_jaxpr: Any) -> list[tuple[int, ...]]:
    pjit_jaxpr = closed_jaxpr.jaxpr.eqns[0].params["jaxpr"].jaxpr
    return [
        tuple(v.aval.shape)
        for eqn in pjit_jaxpr.eqns
        if eqn.primitive.name == "scan"
        for v in eqn.outvars
    ]


def test_should_budget_lane_padded_weights_in_max_scoring_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: topk=8 exercises the exact _max_scoring_tokens scan boundary cleanly.
    fn = functools.partial(paged_indexer_topk, topk=8, interpret=False)
    k_pool_256 = jax.ShapeDtypeStruct((4, 64, ROW_WORDS), jnp.int32)
    bt_256 = jax.ShapeDtypeStruct((1, 4), jnp.int32)
    sl_1 = jax.ShapeDtypeStruct((1,), jnp.int32)
    cu_2 = jax.ShapeDtypeStruct((2,), jnp.int32)

    def trace_shapes(tokens: int, heads: int) -> list[tuple[int, ...]]:
        return _token_chunk_scan_shapes(
            jax.make_jaxpr(fn)(
                jax.ShapeDtypeStruct((tokens, heads, DIM), jnp.bfloat16),
                jax.ShapeDtypeStruct((tokens, heads, 1), jnp.float32),
                k_pool_256,
                bt_256,
                sl_1,
                cu_2,
            )
        )

    # Act & Assert: default TPU7x VMEM hits MAX_SCORING_TOKENS = 288 boundary.
    assert trace_shapes(288, 32) == []
    assert trace_shapes(289, 32) == [(1, 288, 8)]

    # Constrained VMEM (usable = 114_688 B) with heads=5 (padded_heads=8 -> 8192 B/tok)
    # caps max_tokens_per_chunk at 14, proving (8, 128) sublane/lane tiling + 7/16 cap.
    orig_get_tpu_info = paged_indexer_mod.pltpu.get_tpu_info
    with _isolated_jit_patch(
        monkeypatch,
        paged_indexer_mod.pltpu,
        "get_tpu_info",
        lambda: dataclasses.replace(orig_get_tpu_info(), vmem_capacity_bytes=262_144),
    ):
        assert trace_shapes(14, 5) == []
        assert trace_shapes(15, 5) == [(1, 14, 8)]
        assert trace_shapes(28, 5) == [(2, 14, 8)]


def test_should_scan_token_chunks_with_remainder_and_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: lower MAX_SCORING_TOKENS to 2 so 5 tokens exercise both 2x2 scan chunks
    # and a 1-token remainder tail with candidate filtering.
    rng = np.random.default_rng(917)
    tokens, heads, pages, page_rows, topk = 5, 2, 4, 64, 4

    q_nat = jnp.array(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32), jnp.bfloat16
    )
    q_split = _split_elements(q_nat)
    weights = jnp.array(
        rng.uniform(0.1, 1.2, size=(tokens, heads, 1)).astype(np.float32)
    )
    keys_all = jnp.array(
        rng.standard_normal((pages, SLOTS * page_rows, DIM), dtype=np.float32),
        jnp.bfloat16,
    )
    k_pool = _pack_keys_to_words(keys_all)
    bt = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    sl = jnp.array([15, 28], dtype=jnp.int32)
    cu = jnp.array([0, tokens - 2, tokens], dtype=jnp.int32)
    cand = jnp.broadcast_to(jnp.arange(12, dtype=jnp.int32) - 1, (tokens, 12))

    # Act: verify patched MAX_SCORING_TOKENS=2 emits scan (2 chunks x 2 tok) + tail.
    with _isolated_jit_patch(monkeypatch, paged_indexer_mod, "MAX_SCORING_TOKENS", 2):
        traced_scan = paged_indexer_topk.trace(
            q_split,
            weights,
            k_pool,
            bt,
            sl,
            cu,
            topk=topk,
            candidate_indices=cand,
            compress_ratio=1,
            interpret=True,
        )
        scan_shapes = [
            tuple(v.aval.shape)
            for eqn in traced_scan.jaxpr.jaxpr.eqns
            if eqn.primitive.name == "scan"
            for v in eqn.outvars
        ]
        assert scan_shapes == [(2, 2, topk)]
        actual = np.asarray(
            traced_scan.lower().compile()(
                q_split, weights, k_pool, bt, sl, cu, candidate_indices=cand
            )
        )
    expected, _ = _ref_paged_indexer_topk(
        np.asarray(q_nat),
        np.asarray(weights),
        np.asarray(keys_all),
        np.asarray(bt),
        np.asarray(sl),
        np.asarray(cu),
        topk=topk,
        candidate_indices=np.asarray(cand),
        compress_ratio=1,
    )

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_should_short_circuit_zero_dimensions_to_minus_one() -> None:
    # Arrange
    q = jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16)
    w = jnp.ones((2, 4, 1), dtype=jnp.float32)
    k_pool = jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    sl = jnp.array([8], dtype=jnp.int32)
    cu = jnp.array([0, 2], dtype=jnp.int32)

    # Act
    out_0tok = paged_indexer_topk(
        q[:0],
        w[:0],
        k_pool,
        bt,
        sl,
        jnp.array([0, 0], dtype=jnp.int32),
        topk=4,
        interpret=True,
    )
    out_0seq = paged_indexer_topk(
        q,
        w,
        k_pool,
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        topk=4,
        interpret=True,
    )
    out_0page = paged_indexer_topk(
        q, w, k_pool, jnp.zeros((1, 0), dtype=jnp.int32), sl, cu, topk=4, interpret=True
    )
    out_0cand = paged_indexer_topk(
        q,
        w,
        k_pool,
        bt,
        sl,
        cu,
        topk=4,
        candidate_indices=jnp.zeros((2, 0), dtype=jnp.int32),
        interpret=True,
    )

    # Assert
    assert out_0tok.shape == (0, 4)
    assert out_0tok.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out_0seq), -1)
    np.testing.assert_array_equal(np.asarray(out_0page), -1)
    np.testing.assert_array_equal(np.asarray(out_0cand), -1)


@pytest.mark.parametrize(
    ("bad_arg", "match"),
    [
        ("q_dim", "Expected bf16 q"),
        ("q_rank", "Expected bf16 q"),
        ("q_dtype", "Expected bf16 q"),
        ("weights_shape", "Expected bf16 q"),
        ("weights_dtype", "Expected bf16 q"),
        ("k_pool_words", "Expected int32 k_pool"),
        ("k_pool_empty_pages", "Expected int32 k_pool"),
        ("k_pool_empty_rows", "Expected int32 k_pool"),
        ("k_pool_unaligned_rows", "multiple of 8"),
        ("k_pool_dtype", "Expected int32 k_pool"),
        ("bt_rank", "Expected int32 block_tables"),
        ("bt_dtype", "Expected int32 block_tables"),
        ("sl_shape", "Expected int32 block_tables"),
        ("cu_shape", "Expected int32 block_tables"),
        ("topk_zero", "Expected topk > 0"),
        ("ratio_invalid", "Expected topk > 0"),
        ("cand_shape", "Expected topk > 0"),
        ("cand_dtype", "Expected topk > 0"),
    ],
)
def test_should_fail_fast_on_invalid_inputs(bad_arg: str, match: str) -> None:
    # Arrange
    call_kwargs: dict[str, Any] = {
        "q": jax.ShapeDtypeStruct((2, 4, DIM), jnp.bfloat16),
        "weights": jax.ShapeDtypeStruct((2, 4, 1), jnp.float32),
        "k_pool": jax.ShapeDtypeStruct((2, 8, ROW_WORDS), jnp.int32),
        "block_tables": jax.ShapeDtypeStruct((1, 2), jnp.int32),
        "seq_lens": jax.ShapeDtypeStruct((1,), jnp.int32),
        "cu_q_lens": jax.ShapeDtypeStruct((2,), jnp.int32),
        "topk": 4,
        "candidate_indices": None,
        "compress_ratio": 1,
        "interpret": True,
    }
    overrides: dict[str, tuple[str, Any]] = {
        "q_dim": ("q", jax.ShapeDtypeStruct((2, 4, 64), jnp.bfloat16)),
        "q_rank": ("q", jax.ShapeDtypeStruct((2, DIM), jnp.bfloat16)),
        "q_dtype": ("q", jax.ShapeDtypeStruct((2, 4, DIM), jnp.float32)),
        "weights_shape": ("weights", jax.ShapeDtypeStruct((2, 4, 2), jnp.float32)),
        "weights_dtype": ("weights", jax.ShapeDtypeStruct((2, 4, 1), jnp.bfloat16)),
        "k_pool_words": ("k_pool", jax.ShapeDtypeStruct((2, 8, 64), jnp.int32)),
        "k_pool_empty_pages": (
            "k_pool",
            jax.ShapeDtypeStruct((0, 8, ROW_WORDS), jnp.int32),
        ),
        "k_pool_empty_rows": (
            "k_pool",
            jax.ShapeDtypeStruct((2, 0, ROW_WORDS), jnp.int32),
        ),
        "k_pool_unaligned_rows": (
            "k_pool",
            jax.ShapeDtypeStruct((2, 5, ROW_WORDS), jnp.int32),
        ),
        "k_pool_dtype": (
            "k_pool",
            jax.ShapeDtypeStruct((2, 8, ROW_WORDS), jnp.float32),
        ),
        "bt_rank": ("block_tables", jax.ShapeDtypeStruct((2,), jnp.int32)),
        "bt_dtype": ("block_tables", jax.ShapeDtypeStruct((1, 2), jnp.float32)),
        "sl_shape": ("seq_lens", jax.ShapeDtypeStruct((2,), jnp.int32)),
        "cu_shape": ("cu_q_lens", jax.ShapeDtypeStruct((3,), jnp.int32)),
        "topk_zero": ("topk", 0),
        "ratio_invalid": ("compress_ratio", 4),
        "cand_shape": (
            "candidate_indices",
            jax.ShapeDtypeStruct((3, 4), jnp.int32),
        ),
        "cand_dtype": (
            "candidate_indices",
            jax.ShapeDtypeStruct((2, 4), jnp.float32),
        ),
    }
    key, val = overrides[bad_arg]
    call_kwargs[key] = val

    # Act & Assert
    with pytest.raises(ValueError, match=match):
        paged_indexer_topk.trace(**call_kwargs)


def test_should_reject_non_tpu7x_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    q = jax.ShapeDtypeStruct((2, 4, DIM), jnp.bfloat16)
    weights = jax.ShapeDtypeStruct((2, 4, 1), jnp.float32)
    k_pool = jax.ShapeDtypeStruct((2, 8, ROW_WORDS), jnp.int32)
    bt = jax.ShapeDtypeStruct((1, 2), jnp.int32)
    sl = jax.ShapeDtypeStruct((1,), jnp.int32)
    cu = jax.ShapeDtypeStruct((2,), jnp.int32)

    # Act & Assert
    if jax.default_backend() != "tpu":
        empty_mesh = jax.sharding.AbstractMesh((), (), ())
        with (
            jax.sharding.use_abstract_mesh(empty_mesh),
            pytest.raises(
                RuntimeError, match="paged_indexer_topk requires TPU7x, got backend="
            ),
        ):
            paged_indexer_topk.trace(
                q, weights, k_pool, bt, sl, cu, topk=4, interpret=True
            )

    v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"),
    ):
        paged_indexer_topk.trace(q, weights, k_pool, bt, sl, cu, topk=4, interpret=True)

    orig_get_tpu_info = paged_indexer_mod.pltpu.get_tpu_info
    for field_override in ({"num_lanes": 64}, {"num_sublanes": 4}):
        with (
            _isolated_jit_patch(
                monkeypatch,
                paged_indexer_mod.pltpu,
                "get_tpu_info",
                lambda ov=field_override: dataclasses.replace(
                    orig_get_tpu_info(), **ov
                ),
            ),
            pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"),
        ):
            paged_indexer_topk.trace(
                q, weights, k_pool, bt, sl, cu, topk=4, interpret=True
            )


def _has_named_scope(jaxpr: Any, target: str) -> bool:
    stack = [jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr]
    while stack:
        cur = stack.pop()
        for eqn in cur.eqns:
            if target in str(eqn.source_info.name_stack).split("/"):
                return True
            for val in eqn.params.values():
                if hasattr(val, "eqns"):
                    stack.append(val)
                elif hasattr(val, "jaxpr"):
                    stack.append(val.jaxpr)
                elif isinstance(val, (tuple, list)):
                    for item in val:
                        if hasattr(item, "jaxpr"):
                            stack.append(item.jaxpr)
                        elif hasattr(item, "eqns"):
                            stack.append(item)
    return False


def _collect_jaxpr_call_names(jaxpr: Any) -> list[str]:
    names: list[str] = []
    stack = [jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr]
    while stack:
        cur = stack.pop()
        for eqn in cur.eqns:
            name = eqn.params.get("name")
            if isinstance(name, str):
                names.append(name)
            for val in eqn.params.values():
                if hasattr(val, "eqns"):
                    stack.append(val)
                elif hasattr(val, "jaxpr"):
                    stack.append(val.jaxpr)
                elif isinstance(val, (tuple, list)):
                    for item in val:
                        if hasattr(item, "jaxpr"):
                            stack.append(item.jaxpr)
                        elif hasattr(item, "eqns"):
                            stack.append(item)
    return names


def test_should_lower_sparsecore_topk_with_hbm_bounds_and_named_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    mock_logger = MagicMock()
    q = jax.ShapeDtypeStruct((64, 32, DIM), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((64, 32, 1), jnp.float32)
    k_pool_large = jax.ShapeDtypeStruct((16, 64, ROW_WORDS), jnp.int32)
    bt_large = jax.ShapeDtypeStruct((2, 16), jnp.int32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)

    # Act & Assert: SparseCore path (n >= 2048, topk=16) emits sc_topk_direct.
    with _isolated_jit_patch(monkeypatch, paged_indexer_mod, "logger", mock_logger):
        traced_sc = paged_indexer_topk.trace(
            q, w, k_pool_large, bt_large, sl, cu, topk=16, compress_ratio=2
        )
        hlo_sc = traced_sc.lower(lowering_platforms=("tpu",)).as_text()
        assert "dsv41_gather_paged_index_rows" in hlo_sc
        assert "dsv41_index_word_scores" in hlo_sc
        assert "sc_topk_direct" in hlo_sc
        mock_logger.warning_once.assert_not_called()

        # Lock host chunk-index and Pallas DMA HBM bounds: [0, P) & [0, total_rows).
        outer_cmps = [
            (eqn.primitive.name, int(eqn.invars[1].val))
            for eqn in traced_sc.jaxpr.jaxpr.eqns
            if eqn.primitive.name in ("ge", "gt", "le", "lt")
            and eqn.outvars[0].aval.shape == (2, 16)
            and hasattr(eqn.invars[1], "val")
        ]
        assert outer_cmps.count(("ge", 0)) == 2
        assert outer_cmps.count(("lt", 16)) == 2
        gather_eqns = [
            eqn
            for eqn in traced_sc.jaxpr.jaxpr.eqns
            if eqn.params.get("name") == "dsv41_gather_paged_index_rows"
        ]
        assert len(gather_eqns) == 1
        dma_cmps = [
            (eqn.primitive.name, int(eqn.invars[1].val))
            for eqn in gather_eqns[0].params["jaxpr"].eqns
            if eqn.primitive.name in ("ge", "gt", "le", "lt")
            and hasattr(eqn.invars[1], "val")
        ]
        assert dma_cmps.count(("ge", 0)) == 16
        assert dma_cmps.count(("lt", 1024)) == 16
        assert len(dma_cmps) == 32

        traced_scoped = paged_indexer_topk.trace(
            q,
            w,
            k_pool_large,
            bt_large,
            sl,
            cu,
            topk=16,
            compress_ratio=2,
            enable_named_scope=True,
        )
        assert _has_named_scope(traced_scoped.jaxpr, "gather_paged_index_rows")
        assert not _has_named_scope(traced_sc.jaxpr, "gather_paged_index_rows")


def test_should_fallback_to_lax_topk_and_warn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    mock_logger = MagicMock()
    q = jax.ShapeDtypeStruct((64, 32, DIM), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((64, 32, 1), jnp.float32)
    k_pool_small = jax.ShapeDtypeStruct((4, 16, ROW_WORDS), jnp.int32)
    bt_small = jax.ShapeDtypeStruct((2, 4), jnp.int32)
    k_pool_large = jax.ShapeDtypeStruct((16, 64, ROW_WORDS), jnp.int32)
    bt_large = jax.ShapeDtypeStruct((2, 16), jnp.int32)
    k_pool_8k = jax.ShapeDtypeStruct((64, 64, ROW_WORDS), jnp.int32)
    bt_8k = jax.ShapeDtypeStruct((2, 64), jnp.int32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)

    # Act & Assert: fallback paths (n < 2048, topk % 16 != 0, topk > 4096,
    # topk > width, and topk_width > MAX_N) use jax.lax.top_k and warn once.
    with _isolated_jit_patch(monkeypatch, paged_indexer_mod, "logger", mock_logger):
        names_fb = _collect_jaxpr_call_names(
            paged_indexer_topk.trace(
                q, w, k_pool_small, bt_small, sl, cu, topk=16, compress_ratio=2
            ).jaxpr
        )
        assert "dsv41_gather_paged_index_rows" in names_fb
        assert not any(name.startswith("sc_topk_direct") for name in names_fb)
        mock_logger.warning_once.assert_called_once()

        names_unaligned_k = _collect_jaxpr_call_names(
            paged_indexer_topk.trace(
                q, w, k_pool_large, bt_large, sl, cu, topk=12, compress_ratio=2
            ).jaxpr
        )
        assert not any(name.startswith("sc_topk_direct") for name in names_unaligned_k)

        names_over_k = _collect_jaxpr_call_names(
            paged_indexer_topk.trace(
                q, w, k_pool_8k, bt_8k, sl, cu, topk=4112, compress_ratio=2
            ).jaxpr
        )
        assert not any(name.startswith("sc_topk_direct") for name in names_over_k)

        mock_logger.reset_mock()
        cand_2k = jax.ShapeDtypeStruct((64, 2048), jnp.int32)
        names_k_gt_w = _collect_jaxpr_call_names(
            paged_indexer_topk.trace(
                q,
                w,
                k_pool_8k,
                bt_8k,
                sl,
                cu,
                topk=4096,
                candidate_indices=cand_2k,
                compress_ratio=2,
            ).jaxpr
        )
        assert not any(name.startswith("sc_topk_direct") for name in names_k_gt_w)
        mock_logger.warning_once.assert_called_once()

        mock_logger.reset_mock()
        cand_over_max_n = jax.ShapeDtypeStruct((64, MAX_N + SC_LANES), jnp.int32)
        names_over_max_n = _collect_jaxpr_call_names(
            paged_indexer_topk.trace(
                q,
                w,
                k_pool_large,
                bt_large,
                sl,
                cu,
                topk=16,
                candidate_indices=cand_over_max_n,
                compress_ratio=2,
            ).jaxpr
        )
        assert not any(name.startswith("sc_topk_direct") for name in names_over_max_n)
        mock_logger.warning_once.assert_called_once()


def test_should_trace_128k_prefill_and_decode_without_scatter() -> None:
    # Arrange
    k_pool_128k = jax.ShapeDtypeStruct((4096, 64, ROW_WORDS), jnp.int32)
    bt_128k = jax.ShapeDtypeStruct((2, 1024), jnp.int32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)

    # Act & Assert: 128K prefill (T=289, R=65536) traces bounded lax.scan + fori_loop.
    q_128k = jax.ShapeDtypeStruct((289, 32, DIM), jnp.bfloat16)
    w_128k = jax.ShapeDtypeStruct((289, 32, 1), jnp.float32)
    names_128k = _collect_jaxpr_call_names(
        paged_indexer_topk.trace(
            q_128k, w_128k, k_pool_128k, bt_128k, sl, cu, topk=16, compress_ratio=1
        ).jaxpr
    )
    assert names_128k.count("dsv41_index_word_scores") == 2
    assert sum(name.startswith("sc_topk_direct") for name in names_128k) == 2

    # Candidate-filtered 128K decode (T=288, C=16384, K=2048): 1 chunk, 0 XLA scatter.
    q_dec = jax.ShapeDtypeStruct((288, 32, DIM), jnp.bfloat16)
    w_dec = jax.ShapeDtypeStruct((288, 32, 1), jnp.float32)
    cand_16k = jax.ShapeDtypeStruct((288, 16384), jnp.int32)
    traced_cand = paged_indexer_topk.trace(
        q_dec,
        w_dec,
        k_pool_128k,
        bt_128k,
        sl,
        cu,
        topk=2048,
        candidate_indices=cand_16k,
        compress_ratio=1,
    )
    names_cand = _collect_jaxpr_call_names(traced_cand.jaxpr)
    assert any("n16384_k2048" in name for name in names_cand)
    assert sum(name.startswith("sc_topk_direct") for name in names_cand) == 1
    outer_eqns = traced_cand.jaxpr.jaxpr.eqns
    assert all(eqn.primitive.name != "scatter" for eqn in outer_eqns)


def test_should_fail_fast_when_running_benchmark_outside_tpu7x_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if jax.default_backend() != "tpu":
        with pytest.raises(
            RuntimeError, match="run_benchmarks requires physical TPU7x execution"
        ):
            benchmark_paged_indexer.main(["--warmup", "1", "--repeat", "1"])

    monkeypatch.setattr(benchmark_paged_indexer.jax, "default_backend", lambda: "tpu")
    v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(
            RuntimeError, match=r"run_benchmarks requires TPU7x \(8x128 lanes\)"
        ),
    ):
        benchmark_paged_indexer.run_benchmarks(warmup=1, repeat=1)

    orig_get_tpu_info = benchmark_paged_indexer.pltpu.get_tpu_info
    for field_override in ({"num_lanes": 64}, {"num_sublanes": 4}):
        monkeypatch.setattr(
            benchmark_paged_indexer.pltpu,
            "get_tpu_info",
            lambda ov=field_override: dataclasses.replace(orig_get_tpu_info(), **ov),
        )
        with pytest.raises(
            RuntimeError, match=r"run_benchmarks requires TPU7x \(8x128 lanes\)"
        ):
            benchmark_paged_indexer.run_benchmarks(warmup=1, repeat=1)


def _patch_benchmark_for_xplane_unit_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    xplane_files: list[str],
    planes: list[Any],
) -> None:
    fake_profile = SimpleNamespace(planes=planes)
    orig_get_tpu_info = benchmark_paged_indexer.pltpu.get_tpu_info
    monkeypatch.setattr(benchmark_paged_indexer.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        benchmark_paged_indexer.pltpu,
        "get_tpu_info",
        lambda: dataclasses.replace(orig_get_tpu_info(), vmem_capacity_bytes=65536),
    )
    monkeypatch.setattr(
        benchmark_paged_indexer,
        "TPU_WORKLOADS",
        (
            ("smoke_no_cand", 1, 1, 32, 1, 16, 0, 2),
            ("smoke_with_cand", 1, 1, 32, 1, 16, 16, 2),
        ),
    )
    monkeypatch.setattr(
        benchmark_paged_indexer,
        "paged_indexer_topk",
        lambda *_a, **_k: jnp.zeros((1, 16), dtype=jnp.int32),
    )
    monkeypatch.setattr(
        benchmark_paged_indexer.jax.profiler,
        "trace",
        lambda _dir: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        benchmark_paged_indexer.glob, "glob", lambda *_a, **_k: xplane_files
    )
    monkeypatch.setattr(
        benchmark_paged_indexer.jax.profiler.ProfileData,
        "from_file",
        lambda _path: fake_profile,
    )


def test_should_extract_xplane_median_duration_in_benchmark_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    fake_event = SimpleNamespace(duration_ns=4500.0)
    other_line = SimpleNamespace(name="Other Line", events=[])
    fake_line = SimpleNamespace(
        name=benchmark_paged_indexer.XLA_MODULES_LINE, events=[fake_event]
    )
    other_plane = SimpleNamespace(name="/host:CPU", lines=[])
    fake_plane = SimpleNamespace(
        name=benchmark_paged_indexer.TPU_DEVICE_PLANE,
        lines=[other_line, fake_line],
    )
    _patch_benchmark_for_xplane_unit_test(
        monkeypatch,
        xplane_files=["fake.pb"],
        planes=[other_plane, fake_plane],
    )

    # Act
    exit_code = benchmark_paged_indexer.main(["--warmup", "1", "--repeat", "1"])
    results = benchmark_paged_indexer.run_benchmarks(warmup=1, repeat=1)

    # Assert
    assert exit_code == 0
    assert len(results) == 2
    assert results[0][0] == "smoke_no_cand"
    assert results[0][1] == pytest.approx(4.5)
    assert results[0][2] > 0.0
    assert results[0][3] > 0.0
    assert results[0][4] > 0.0


@pytest.mark.parametrize(
    ("xplane_files", "planes", "match"),
    [
        ([], [], r"No \.xplane\.pb"),
        (
            ["fake.pb"],
            [SimpleNamespace(name="/host:CPU", lines=[])],
            "Missing profiler plane",
        ),
        (
            ["fake.pb"],
            [
                SimpleNamespace(
                    name=benchmark_paged_indexer.TPU_DEVICE_PLANE,
                    lines=[SimpleNamespace(name="Other Line", events=[])],
                )
            ],
            "Missing profiler line",
        ),
        (
            ["fake.pb"],
            [
                SimpleNamespace(
                    name=benchmark_paged_indexer.TPU_DEVICE_PLANE,
                    lines=[
                        SimpleNamespace(
                            name=benchmark_paged_indexer.XLA_MODULES_LINE, events=[]
                        )
                    ],
                )
            ],
            "Empty device event samples",
        ),
    ],
)
def test_should_raise_runtime_error_when_benchmark_xplane_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    xplane_files: list[str],
    planes: list[Any],
    match: str,
) -> None:
    # Arrange
    _patch_benchmark_for_xplane_unit_test(
        monkeypatch,
        xplane_files=xplane_files,
        planes=planes,
    )

    # Act & Assert
    with pytest.raises(RuntimeError, match=match):
        benchmark_paged_indexer.run_benchmarks(warmup=1, repeat=1)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("max_pages", "compress_ratio"),
    [(2, 2), (16, 2), (32, 2), (80, 2), (16, 1)],
)
def test_should_match_reference_on_tpu7x_hardware(
    max_pages: int, compress_ratio: int
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware required")
    # Arrange
    rng = np.random.default_rng(77 + max_pages * 10 + compress_ratio)
    tokens, heads, page_rows, topk = 4, 32, 64, 16
    num_pages = 2 * max_pages
    q_nat = jnp.array(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    q_split = _split_elements(q_nat)
    weights_np = rng.uniform(0.1, 1.0, size=(tokens, heads, 1)).astype(np.float32)
    if max_pages >= 16:
        # Negative weights on seq-1 token 3 make mapped keys score < 0.0 so unmasked
        # 0.0 unmapped/OOB pages would beat mapped keys if page_ok masking failed.
        weights_np[3] = -weights_np[3]
    weights = jnp.asarray(weights_np)
    keys_all = jnp.array(
        rng.standard_normal((num_pages, SLOTS * page_rows, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    k_pool = _pack_keys_to_words(keys_all)
    bt_np = np.arange(2 * max_pages, dtype=np.int32).reshape(2, max_pages)
    if max_pages >= 16:
        bt_np[1, 2] = -1
        bt_np[1, 5] = num_pages
    bt = jnp.asarray(bt_np)
    # Full seq_len exercises every logical page, including tile 1 (keys >= 8192)
    # when max_pages=80 (raw_rows=5120 > 4096).
    sl = jnp.array(
        [20, max_pages * page_rows * SLOTS * compress_ratio], dtype=jnp.int32
    )
    cu = jnp.array([0, 2, 4], dtype=jnp.int32)

    # Act
    out_tpu = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_pool,
            bt,
            sl,
            cu,
            topk=topk,
            compress_ratio=compress_ratio,
            interpret=False,
        )
    )
    ref_indices, full_scores = _ref_paged_indexer_topk(
        np.asarray(q_nat),
        np.asarray(weights),
        np.asarray(keys_all),
        np.asarray(bt),
        np.asarray(sl),
        np.asarray(cu),
        topk=topk,
        compress_ratio=compress_ratio,
    )

    # Assert
    unmapped_keys: set[int] = set()
    if max_pages >= 16:
        page_keys = page_rows * SLOTS
        unmapped_keys = set(range(2 * page_keys, 3 * page_keys)) | set(
            range(5 * page_keys, 6 * page_keys)
        )
        assert unmapped_keys.isdisjoint(set(out_tpu[2:].ravel().tolist()))
    _assert_topk_set_parity(out_tpu, ref_indices, full_scores)

    if max_pages == 32 and compress_ratio == 2:
        cand_np = np.full((tokens, 2048), -1, dtype=np.int32)
        cand_np[:, :512] = np.arange(512, dtype=np.int32)[None, :]
        cand_np[:, 512:768] = np.arange(256, dtype=np.int32)[None, :]
        cand_np[:, 768:896] = np.arange(640, 768, dtype=np.int32)[None, :]
        cand_tpu = jnp.asarray(cand_np)
        out_cand = np.asarray(
            paged_indexer_topk(
                q_split,
                weights,
                k_pool,
                bt,
                sl,
                cu,
                topk=topk,
                candidate_indices=cand_tpu,
                compress_ratio=compress_ratio,
                interpret=False,
            )
        )
        ref_cand, scores_cand = _ref_paged_indexer_topk(
            np.asarray(q_nat),
            np.asarray(weights),
            np.asarray(keys_all),
            np.asarray(bt),
            np.asarray(sl),
            np.asarray(cu),
            topk=topk,
            candidate_indices=cand_np,
            compress_ratio=compress_ratio,
        )
        assert unmapped_keys.isdisjoint(set(out_cand[2:].ravel().tolist()))
        _assert_topk_set_parity(out_cand, ref_cand, scores_cand)


@pytest.mark.tpu_test
def test_should_reuse_single_compile_and_skip_trailing_tiles_at_128k_on_tpu7x() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware required")
    # Arrange: fixed static 128K block_tables (max_pages=1024 -> 131072 keys, 16 tiles)
    # with higher-scoring sentinel pages placed in tile 1 (8192..16383) and tile 15
    # (122880..131071) so shorter seq_lens must exclude unvisited trailing tiles.
    rng = np.random.default_rng(1280)
    num_seqs, query_len, heads, page_rows, max_pages, topk = 2, 1, 4, 64, 1024, 64
    num_pages = 18
    page_keys = SLOTS * page_rows
    q_nat = jnp.array(
        np.abs(rng.standard_normal((num_seqs * query_len, heads, DIM), np.float32))
        + 0.5,
        dtype=jnp.bfloat16,
    )
    q_split = _split_elements(q_nat)
    weights = jnp.array(
        rng.uniform(0.5, 1.5, size=(num_seqs * query_len, heads, 1)).astype(np.float32)
    )
    keys_np = np.abs(rng.standard_normal((num_pages, page_keys, DIM), np.float32)) * 0.5
    keys_np[16] = 3.0
    keys_np[17] = 8.0
    keys_all = jnp.array(keys_np, dtype=jnp.bfloat16)
    k_pool = _pack_keys_to_words(keys_all)

    bt_np = np.empty((num_seqs, max_pages), dtype=np.int32)
    bt_np[:, :64] = np.arange(64, dtype=np.int32) % 16
    bt_np[:, 64:128] = 16
    bt_np[:, 128:960] = np.arange(832, dtype=np.int32) % 16
    bt_np[:, 960:1024] = 17
    bt_np[1, 3] = -1
    bt = jnp.asarray(bt_np)
    cu = jnp.arange(num_seqs + 1, dtype=jnp.int32) * query_len

    # Act & Assert: dynamic fori_loop traces once and serves 2K..128K without recompile.
    sl_probe = jnp.full((num_seqs,), 2048, dtype=jnp.int32)
    traced = paged_indexer_topk.trace(
        q_split, weights, k_pool, bt, sl_probe, cu, topk=topk, compress_ratio=1
    )
    assert any(eqn.primitive.name == "while" for eqn in traced.jaxpr.jaxpr.eqns)

    paged_indexer_topk.clear_cache()
    compiled_count: int | None = None
    unmapped_keys = set(range(3 * page_keys, 4 * page_keys))
    for ctx_len in (2048, 8192, 32768, 131072):
        sl = jnp.full((num_seqs,), ctx_len, dtype=jnp.int32)
        actual = np.asarray(
            paged_indexer_topk(
                q_split, weights, k_pool, bt, sl, cu, topk=topk, compress_ratio=1
            )
        )
        # JAX PjitFunction exposes compilation count only via _cache_size().
        cur_count = paged_indexer_topk._cache_size()  # noqa: SLF001 -- JAX jit API
        if compiled_count is None:
            compiled_count = cur_count
        else:
            assert cur_count == compiled_count

        ref_indices, full_scores = _ref_paged_indexer_topk(
            np.asarray(q_nat),
            np.asarray(weights),
            np.asarray(keys_all),
            bt_np,
            np.asarray(sl),
            np.asarray(cu),
            topk=topk,
            compress_ratio=1,
        )
        _assert_topk_set_parity(actual, ref_indices, full_scores)
        assert unmapped_keys.isdisjoint(set(actual[1].tolist()))
        if ctx_len <= 8192:
            assert np.all((actual >= 0) & (actual < ctx_len))
        elif ctx_len == 32768:
            assert np.all((actual >= 8192) & (actual < 16384))
        else:
            assert np.all((actual >= 122880) & (actual < 131072))
