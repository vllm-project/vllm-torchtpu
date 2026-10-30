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
"""Unit tests for `paged_indexer_topk`."""

from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    DIM,
    ROW_WORDS,
    SLOTS,
    benchmark_paged_indexer,
    paged_indexer_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    paged_indexer as paged_indexer_mod,
)


@pytest.fixture(autouse=True)
def _setup_tpu7x_abstract_mesh() -> Iterator[None]:
    """Provide an abstract TPU7x mesh on CPU so `pltpu.get_tpu_info()` succeeds."""
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
    """Reorder the last axis `[..., 128]` into `[even_elements, odd_elements]`."""
    return x.reshape(*x.shape[:-1], 64, 2).swapaxes(-1, -2).reshape(x.shape)


def _pack_keys_to_words(keys: jax.Array) -> jax.Array:
    """Pack `[pages, 2 * page_rows, 128]` BF16 keys into `int32` word rows."""
    pages, page_keys, _ = keys.shape
    page_rows = page_keys // SLOTS
    paired = keys.reshape(pages, page_rows, SLOTS, 64, 2)
    bits = jax.lax.bitcast_convert_type(paired, jnp.uint16).astype(jnp.uint32)
    packed = bits[..., 0] | (bits[..., 1] << 16)
    return packed.reshape(pages, page_rows, ROW_WORDS).astype(jnp.int32)


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
    """Independent float64 NumPy reference returning `(ref_indices, full_scores)`."""
    tokens, _, _ = q.shape
    num_seqs, max_pages = block_tables.shape
    total_pages, page_keys, _ = keys_all.shape
    max_keys = max_pages * page_keys

    ref_indices = np.full((tokens, topk), -1, dtype=np.int32)
    full_scores = np.full((tokens, max_keys), -np.inf, dtype=np.float64)

    for seq_idx in range(num_seqs):
        q_start = max(0, int(cu_q_lens[seq_idx]))
        q_end = min(tokens, int(cu_q_lens[seq_idx + 1]))
        seq_len = int(seq_lens[seq_idx])
        for token_idx in range(q_start, q_end):
            pos_t = seq_len - (int(cu_q_lens[seq_idx + 1]) - token_idx)
            valid_count = max(0, (pos_t + 1) // compress_ratio)
            eff_len = min(valid_count, max_keys)

            cand_set: set[int] | None = None
            if candidate_indices is not None:
                cand_set = {
                    int(cand)
                    for cand in candidate_indices[token_idx]
                    if 0 <= int(cand) < max_keys
                }

            q_t = q[token_idx].astype(np.float64)
            w_t = weights[token_idx, :, 0].astype(np.float64)
            for key_idx in range(eff_len):
                if cand_set is not None and key_idx not in cand_set:
                    continue
                logical_page = key_idx // page_keys
                page_offset = key_idx % page_keys
                phys_page = int(block_tables[seq_idx, logical_page])
                if 0 <= phys_page < total_pages:
                    k_vec = keys_all[phys_page, page_offset].astype(np.float64)
                    dots = np.maximum(q_t @ k_vec, 0.0)
                    full_scores[token_idx, key_idx] = float(np.sum(dots * w_t))

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
    """Assert order-insensitive exact top-k set and score optimality."""
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


@pytest.mark.parametrize(
    ("tokens", "two_bank", "compress_ratio", "use_candidates", "named_scope"),
    [
        (6, False, 1, False, False),
        (6, True, 2, False, True),
        (6, True, 2, True, False),
        (6, False, 2, True, False),
        (300, True, 2, False, False),
    ],
)
def test_paged_indexer_topk_cpu_parity(
    tokens: int,
    two_bank: bool,
    compress_ratio: int,
    use_candidates: bool,
    named_scope: bool,
) -> None:
    """Verify `paged_indexer_topk` matches independent NumPy reference on CPU."""
    rng = np.random.default_rng(100 + tokens + int(two_bank) * 10 + compress_ratio)
    heads = 2 if tokens > 6 else 4
    pages_lo, pages_hi = 3, (2 if two_bank else 0)
    total_pages = pages_lo + pages_hi
    page_rows = 8
    page_keys = SLOTS * page_rows
    topk = 8

    q_nat = jnp.array(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    q_split = _split_elements(q_nat)
    weights = jnp.array(
        rng.uniform(0.1, 1.5, size=(tokens, heads, 1)).astype(np.float32)
    )
    keys_all = jnp.array(
        rng.standard_normal((total_pages, page_keys, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    words_all = _pack_keys_to_words(keys_all)
    k_lo = words_all[:pages_lo]
    k_hi = words_all[pages_lo:] if two_bank else None

    block_tables = jnp.array(
        [
            [0, pages_lo if two_bank else 1, -1],
            [2, pages_lo + 1 if two_bank else 0, 99],
        ],
        dtype=jnp.int32,
    )
    seq_lens = (
        jnp.array([200, 200], dtype=jnp.int32)
        if tokens > 6
        else jnp.array([7, 45], dtype=jnp.int32)
    )
    cu_q_lens = (
        jnp.array([2, 180, 298], dtype=jnp.int32)
        if tokens > 6
        else jnp.array([1, 3, 5], dtype=jnp.int32)
    )

    candidate_indices: jax.Array | None = None
    if use_candidates:
        cand_np = np.array(
            [
                [0, 1, 2, -1, 999, 0],
                [0, 1, 2, 3, -1, 1],
                [0, 2, 4, 5, -1, 999],
                [1, 3, 5, 16, 17, 18],
                [0, 1, 2, 16, 17, 20],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        candidate_indices = jnp.asarray(cand_np)

    actual = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_lo,
            block_tables,
            seq_lens,
            cu_q_lens,
            topk=topk,
            k_pool_hi=k_hi,
            candidate_indices=candidate_indices,
            compress_ratio=compress_ratio,
            interpret=True,
            enable_named_scope=named_scope,
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
        candidate_indices=(
            None if candidate_indices is None else np.asarray(candidate_indices)
        ),
        compress_ratio=compress_ratio,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], -1)
    np.testing.assert_array_equal(actual[-1], -1)


def test_zero_dimension_short_circuits() -> None:
    """Verify zero-token, zero-seq, zero-page, and zero-candidate short-circuits."""
    q = jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16)
    w = jnp.ones((2, 4, 1), dtype=jnp.float32)
    k_lo = jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    sl = jnp.array([8], dtype=jnp.int32)
    cu = jnp.array([0, 2], dtype=jnp.int32)

    out_0tok = paged_indexer_topk(
        q[:0],
        w[:0],
        k_lo,
        bt,
        sl,
        jnp.array([0, 0], dtype=jnp.int32),
        topk=4,
        interpret=True,
    )
    assert out_0tok.shape == (0, 4)

    out_0seq = paged_indexer_topk(
        q,
        w,
        k_lo,
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        topk=4,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(out_0seq), -1)

    out_0page = paged_indexer_topk(
        q, w, k_lo, jnp.zeros((1, 0), dtype=jnp.int32), sl, cu, topk=4, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(out_0page), -1)

    out_0cand = paged_indexer_topk(
        q,
        w,
        k_lo,
        bt,
        sl,
        cu,
        topk=4,
        candidate_indices=jnp.zeros((2, 0), dtype=jnp.int32),
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(out_0cand), -1)


@pytest.mark.parametrize(
    ("bad_arg", "match"),
    [
        ("q", "Expected bf16 q"),
        ("weights", "Expected bf16 q"),
        ("k_lo", "Expected int32 k_pool_lo/k_pool_hi"),
        ("k_hi", "Expected int32 k_pool_lo/k_pool_hi"),
        ("bt", "Expected int32 block_tables"),
        ("sl", "Expected int32 block_tables"),
        ("cu", "Expected int32 block_tables"),
        ("topk", "Expected topk > 0"),
        ("ratio", "Expected topk > 0"),
        ("cand", "Expected topk > 0"),
    ],
)
def test_paged_indexer_validation_errors(bad_arg: str, match: str) -> None:
    """Verify fail-fast ValueError checks on `paged_indexer_topk`."""
    call_kwargs: dict[str, Any] = {
        "q": jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16),
        "weights": jnp.ones((2, 4, 1), dtype=jnp.float32),
        "k_pool_lo": jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32),
        "block_tables": jnp.zeros((1, 2), dtype=jnp.int32),
        "seq_lens": jnp.array([8], dtype=jnp.int32),
        "cu_q_lens": jnp.array([0, 2], dtype=jnp.int32),
        "topk": 4,
        "k_pool_hi": None,
        "candidate_indices": None,
        "compress_ratio": 1,
        "interpret": True,
    }
    if bad_arg == "q":
        call_kwargs["q"] = jnp.zeros((2, 4, 64), dtype=jnp.bfloat16)
    elif bad_arg == "weights":
        call_kwargs["weights"] = jnp.ones((2, 4, 2), dtype=jnp.float32)
    elif bad_arg == "k_lo":
        call_kwargs["k_pool_lo"] = jnp.zeros((2, 8, 64), dtype=jnp.int32)
    elif bad_arg == "k_hi":
        call_kwargs["k_pool_hi"] = jnp.zeros((2, 4, ROW_WORDS), dtype=jnp.int32)
    elif bad_arg == "bt":
        call_kwargs["block_tables"] = jnp.zeros((2,), dtype=jnp.int32)
    elif bad_arg == "sl":
        call_kwargs["seq_lens"] = jnp.zeros((2,), dtype=jnp.int32)
    elif bad_arg == "cu":
        call_kwargs["cu_q_lens"] = jnp.zeros((3,), dtype=jnp.int32)
    elif bad_arg == "topk":
        call_kwargs["topk"] = 0
    elif bad_arg == "ratio":
        call_kwargs["compress_ratio"] = 4
    elif bad_arg == "cand":
        call_kwargs["candidate_indices"] = jnp.zeros((3, 4), dtype=jnp.int32)

    with pytest.raises(ValueError, match=match):
        paged_indexer_topk(**call_kwargs)


def test_paged_indexer_rejects_non_tpu7x_generation() -> None:
    """Verify `paged_indexer_topk` raises `RuntimeError` on non-TPU7x devices."""
    v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
    q = jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16)
    weights = jnp.ones((2, 4, 1), dtype=jnp.float32)
    k_lo = jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    sl = jnp.array([8], dtype=jnp.int32)
    cu = jnp.array([0, 2], dtype=jnp.int32)
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(RuntimeError, match="requires TPU7x"),
    ):
        paged_indexer_topk(q, weights, k_lo, bt, sl, cu, topk=4, interpret=True)


def test_tpu7x_sparsecore_topk_and_fallback_lowering_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `interpret=False` lowers `sparsecore_topk` (`n>=2048`) and fallback."""
    mock_logger = MagicMock()
    monkeypatch.setattr(paged_indexer_mod, "logger", mock_logger)
    q = jax.ShapeDtypeStruct((512, 32, DIM), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((512, 32, 1), jnp.float32)
    k_lo_large = jax.ShapeDtypeStruct((16, 64, ROW_WORDS), jnp.int32)
    bt_large = jax.ShapeDtypeStruct((2, 16), jnp.int32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)
    lowered_sc = paged_indexer_topk.trace(
        q,
        w,
        k_lo_large,
        bt_large,
        sl,
        cu,
        topk=16,
        compress_ratio=2,
        interpret=False,
    ).lower(lowering_platforms=("tpu",))
    hlo_sc = lowered_sc.as_text()
    assert "dsv41_gather_paged_index_rows" in hlo_sc
    assert "dsv41_index_word_scores" in hlo_sc
    assert "sc_topk_direct" in hlo_sc
    mock_logger.warning_once.assert_not_called()

    k_lo_small = jax.ShapeDtypeStruct((4, 16, ROW_WORDS), jnp.int32)
    bt_small = jax.ShapeDtypeStruct((2, 4), jnp.int32)
    lowered_fb = paged_indexer_topk.trace(
        q,
        w,
        k_lo_small,
        bt_small,
        sl,
        cu,
        topk=16,
        compress_ratio=2,
        interpret=False,
    ).lower(lowering_platforms=("tpu",))
    hlo_fb = lowered_fb.as_text()
    assert "dsv41_gather_paged_index_rows" in hlo_fb
    assert "sc_topk_direct" not in hlo_fb
    mock_logger.warning_once.assert_called_once()

    # 128k-context prefill (T=2048, R=65536) must lower to a single scan loop
    # rather than unrolling 40+ full-score tensors in HBM (HBM transient guard).
    q_128k = jax.ShapeDtypeStruct((2048, 32, DIM), jnp.bfloat16)
    w_128k = jax.ShapeDtypeStruct((2048, 32, 1), jnp.float32)
    k_lo_128k = jax.ShapeDtypeStruct((4096, 64, ROW_WORDS), jnp.int32)
    bt_128k = jax.ShapeDtypeStruct((2, 1024), jnp.int32)
    hlo_128k = (
        paged_indexer_topk.trace(
            q_128k,
            w_128k,
            k_lo_128k,
            bt_128k,
            sl,
            cu,
            topk=2048,
            compress_ratio=1,
            interpret=False,
        )
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    assert hlo_128k.count("dsv41_index_word_scores") == 2
    assert hlo_128k.count("sc_topk_direct") == 2
    assert "131072,2]" not in hlo_128k

    # Candidate-filtered 128k DSpark decode (T=288, C=16384, K=2048) must
    # size chunk_t by min(padded_rows, tile_rows)=4096 (chunk_t=288, single
    # chunk) and run SparseCore top-k directly on [288, 16384] with 0 scatter.
    q_dec = jax.ShapeDtypeStruct((288, 32, DIM), jnp.bfloat16)
    w_dec = jax.ShapeDtypeStruct((288, 32, 1), jnp.float32)
    cand_16k = jax.ShapeDtypeStruct((288, 16384), jnp.int32)
    hlo_cand = (
        paged_indexer_topk.trace(
            q_dec,
            w_dec,
            k_lo_128k,
            bt_128k,
            sl,
            cu,
            topk=2048,
            candidate_indices=cand_16k,
            compress_ratio=1,
            interpret=False,
        )
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    assert "n16384_k2048" in hlo_cand
    assert hlo_cand.count("sc_topk_direct") == 1
    assert "scatter" not in hlo_cand


@pytest.mark.parametrize("seq_lens_pair", [(16, 200), (16, 620)])
def test_paged_indexer_multi_tile_fori_loop_and_candidate_ties(
    monkeypatch: pytest.MonkeyPatch, seq_lens_pair: tuple[int, int]
) -> None:
    """Verify multi-tile `fori_loop` and candidate tie-breaking."""
    monkeypatch.setattr(paged_indexer_mod, "MAX_TILE_ROWS", 256)
    jax.clear_caches()
    rng = np.random.default_rng(1200 + seq_lens_pair[1])
    tokens, heads, pages, page_rows, max_pages, topk = 4, 2, 6, 8, 40, 4
    q_nat = jnp.ones((tokens, heads, DIM), dtype=jnp.bfloat16)
    q_split = _split_elements(q_nat)
    weights = jnp.ones((tokens, heads, 1), dtype=jnp.float32)
    keys_np = rng.standard_normal((pages, SLOTS * page_rows, DIM)).astype(np.float32)
    # Create exact score ties across keys (0, 3, 5) to verify ascending tie-breaking.
    keys_np[0, 3] = keys_np[0, 0]
    keys_np[0, 5] = keys_np[0, 0]
    keys_all = jnp.array(keys_np, dtype=jnp.bfloat16)
    k_lo = _pack_keys_to_words(keys_all)

    bt_np = np.full((2, max_pages), -1, dtype=np.int32)
    bt_np[0, :3] = [0, 1, 2]
    bt_np[1, :3] = [3, 4, 5]
    bt_np[1, 32:35] = [0, 1, 2]
    bt = jnp.asarray(bt_np)
    sl = jnp.array(list(seq_lens_pair), dtype=jnp.int32)
    cu = jnp.array([0, 2, 4], dtype=jnp.int32)
    cand = jnp.array(
        [
            [5, 3, 0, 5, 3, 1, 2, -1],
            [5, 0, 3, 3, 5, 2, 1, 9999],
            [512, 515, 517, 512, 0, 1, 2, 3],
            [517, 515, 512, 515, 16, 17, 18, -1],
        ],
        dtype=jnp.int32,
    )
    for cand_arg in (None, cand):
        actual = np.asarray(
            paged_indexer_topk(
                q_split,
                weights,
                k_lo,
                bt,
                sl,
                cu,
                topk=topk,
                candidate_indices=cand_arg,
                compress_ratio=1,
                interpret=True,
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
            candidate_indices=None if cand_arg is None else np.asarray(cand_arg),
            compress_ratio=1,
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("tokens", "use_cand"), [(6, False), (8, True)])
def test_paged_indexer_multi_chunk_scan_and_cli(
    monkeypatch: pytest.MonkeyPatch, tokens: int, use_cand: bool
) -> None:
    """Verify fused per-chunk scan (exact and remainder) + CLI benchmark smoke."""
    monkeypatch.setattr(paged_indexer_mod, "MAX_SCORING_TOKENS", 3)
    jax.clear_caches()
    rng = np.random.default_rng(909 + tokens)
    heads, pages, page_rows, topk = 4, 4, 8, 4
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
    k_lo = _pack_keys_to_words(keys_all)
    bt = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    sl = jnp.array([15, 28], dtype=jnp.int32)
    cu = jnp.array([0, tokens - 3, tokens], dtype=jnp.int32)
    cand = (
        jnp.broadcast_to(jnp.arange(12, dtype=jnp.int32) - 1, (tokens, 12))
        if use_cand
        else None
    )
    actual = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            k_lo,
            bt,
            sl,
            cu,
            topk=topk,
            candidate_indices=cand,
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
        candidate_indices=None if cand is None else np.asarray(cand),
        compress_ratio=1,
    )
    np.testing.assert_array_equal(actual, expected)
    if tokens == 6:
        rc = benchmark_paged_indexer.main(
            ["--interpret", "--warmup", "1", "--repeat", "1"]
        )
        assert rc == 0
        fake_event = SimpleNamespace(duration_ns=4500.0)
        fake_line = SimpleNamespace(
            name=benchmark_paged_indexer.XLA_MODULES_LINE, events=[fake_event]
        )
        fake_plane = SimpleNamespace(
            name=benchmark_paged_indexer.TPU_DEVICE_PLANE, lines=[fake_line]
        )
        fake_profile = SimpleNamespace(planes=[fake_plane])
        monkeypatch.setattr(
            benchmark_paged_indexer.jax, "default_backend", lambda: "tpu"
        )
        monkeypatch.setattr(
            benchmark_paged_indexer.jax.profiler, "trace", lambda _dir: nullcontext()
        )
        monkeypatch.setattr(
            benchmark_paged_indexer.glob, "glob", lambda *_a, **_k: ["fake.pb"]
        )
        monkeypatch.setattr(
            benchmark_paged_indexer.jax.profiler.ProfileData,
            "from_file",
            lambda _path: fake_profile,
        )
        dev_us, host_us = benchmark_paged_indexer._measure_device_and_host_us(
            lambda: jnp.zeros((1,), dtype=jnp.int32),
            warmup=1,
            repeat=1,
            interpret=False,
        )
        assert dev_us == pytest.approx(4.5)
        assert host_us > 0.0


@pytest.mark.tpu_test
@pytest.mark.parametrize("max_pages", [2, 16, 32])
def test_paged_indexer_topk_tpu7x_hardware(max_pages: int) -> None:
    """Verify `paged_indexer_topk` on physical TPU7x against float64 reference."""
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware required")
    rng = np.random.default_rng(77 + max_pages)
    tokens, heads, page_rows, topk = 4, 32, 64, 16
    pages_lo = max_pages
    pages_hi = max_pages
    total_pages = pages_lo + pages_hi
    q_nat = jnp.array(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    q_split = _split_elements(q_nat)
    weights = jnp.array(
        rng.uniform(0.1, 1.0, size=(tokens, heads, 1)).astype(np.float32)
    )
    keys_all = jnp.array(
        rng.standard_normal((total_pages, SLOTS * page_rows, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    words_all = _pack_keys_to_words(keys_all)
    bt = jnp.arange(2 * max_pages, dtype=jnp.int32).reshape(2, max_pages)
    sl = jnp.array([20, max_pages * page_rows * 2], dtype=jnp.int32)
    cu = jnp.array([0, 2, 4], dtype=jnp.int32)

    out_tpu = np.asarray(
        paged_indexer_topk(
            q_split,
            weights,
            words_all[:pages_lo],
            bt,
            sl,
            cu,
            topk=topk,
            k_pool_hi=words_all[pages_lo:],
            compress_ratio=2,
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
        compress_ratio=2,
    )
    _assert_topk_set_parity(out_tpu, ref_indices, full_scores)

    if max_pages == 32:
        cand_np = np.full((tokens, 2048), -1, dtype=np.int32)
        cand_np[:, :512] = np.arange(512, dtype=np.int32)[None, :]
        cand_np[:, 512:768] = np.arange(256, dtype=np.int32)[None, :]
        cand_tpu = jnp.asarray(cand_np)
        out_cand = np.asarray(
            paged_indexer_topk(
                q_split,
                weights,
                words_all[:pages_lo],
                bt,
                sl,
                cu,
                topk=topk,
                k_pool_hi=words_all[pages_lo:],
                candidate_indices=cand_tpu,
                compress_ratio=2,
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
            compress_ratio=2,
        )
        _assert_topk_set_parity(out_cand, ref_cand, scores_cand)
