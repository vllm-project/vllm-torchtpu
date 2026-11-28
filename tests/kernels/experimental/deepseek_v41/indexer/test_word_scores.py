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
"""Executable specification tests for deepseek_v41.indexer.word_scores."""

import ast
import contextlib
import dataclasses
import inspect
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from parameterized import parameterized

from vllm_torchtpu.kernels.experimental.deepseek_v41 import indexer as indexer_pkg
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    BF16_BYTES,
    DIM,
    MAX_BLOCK_ROWS,
    MIN_ROWS,
    ROW_WORDS,
    SLOTS,
    SUB_BLOCK_ROWS,
    WEIGHT_BYTES_PER_HEAD,
    WORD_BYTES,
    benchmark_word_scores,
    score_word_rows,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    word_scores as word_scores_mod,
)


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


def _rand_inputs(
    seed: int,
    tokens: int,
    groups: int,
    rows: int,
    heads: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    num_keys = SLOTS * rows
    q = jnp.asarray(
        rng.standard_normal((tokens, heads, DIM), dtype=np.float32), dtype=jnp.bfloat16
    )
    weights = jnp.asarray(
        rng.standard_normal((tokens, heads, 1), dtype=np.float32), dtype=jnp.float32
    )
    keys = jnp.asarray(
        rng.standard_normal((groups, num_keys, DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    return q, weights, keys


def _split_elements(x: jax.Array) -> jax.Array:
    x_np = np.asarray(x)
    return jnp.asarray(
        x_np.reshape(*x_np.shape[:-1], 64, 2).swapaxes(-1, -2).reshape(x_np.shape)
    )


def _pack_keys_to_words(keys: jax.Array) -> jax.Array:
    groups, num_keys, _ = keys.shape
    rows = num_keys // SLOTS
    paired_u16 = (
        np.asarray(keys)
        .view(np.uint16)
        .reshape(groups, rows, SLOTS, 64, 2)
        .astype(np.uint32)
    )
    packed = paired_u16[..., 0] | (paired_u16[..., 1] << 16)
    return jnp.asarray(packed.reshape(groups, rows, ROW_WORDS).astype(np.int32))


def _uniform_cu_q_lens(tokens: int, groups: int) -> jax.Array:
    return jnp.arange(groups + 1, dtype=jnp.int32) * (tokens // groups)


def _reference_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    tokens, heads, _ = q.shape
    groups, num_keys, _ = keys.shape
    if tokens == 0 or heads == 0:
        return jnp.zeros((tokens, num_keys), dtype=jnp.float32)

    out = np.zeros((tokens, num_keys), dtype=np.float32)
    cu_np = np.asarray(cu_q_lens)
    q_np = np.asarray(q.astype(jnp.float32))
    w_np = np.asarray(weights)
    k_np = np.asarray(keys.astype(jnp.float32))
    for group_idx in range(groups):
        t0 = int(np.clip(cu_np[group_idx], 0, tokens))
        t1 = int(np.clip(cu_np[group_idx + 1], t0, tokens))
        for token_idx in range(t0, t1):
            logits = np.matmul(q_np[token_idx], k_np[group_idx].T)
            out[token_idx] = (np.maximum(logits, 0.0) * w_np[token_idx]).sum(axis=0)
    return jnp.asarray(out)


def _collect_imported_roots(module: object) -> set[str]:
    source = inspect.getsource(module)  # type: ignore[arg-type] -- module object
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots


class TestScoreWordRows:
    def test_should_export_public_api_and_stay_pure_jax(self) -> None:
        # Act
        word_scores_imports = _collect_imported_roots(word_scores_mod)
        init_imports = _collect_imported_roots(indexer_pkg)
        bench_imports = _collect_imported_roots(benchmark_word_scores)
        sig = inspect.signature(score_word_rows)
        cu_param = sig.parameters["cu_q_lens"]

        # Assert
        assert (
            DIM,
            ROW_WORDS,
            SLOTS,
            MIN_ROWS,
            SUB_BLOCK_ROWS,
            MAX_BLOCK_ROWS,
        ) == (128, 128, 2, 1, 256, 512)
        assert (BF16_BYTES, WORD_BYTES, WEIGHT_BYTES_PER_HEAD) == (2, 4, 4)
        assert cu_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert cu_param.default is inspect.Parameter.empty
        assert set(indexer_pkg.__all__) == {
            "BF16_BYTES",
            "BITS_NEG_INF",
            "DIM",
            "MAX_BLOCK_ROWS",
            "MAX_N",
            "MAX_SCORING_TOKENS",
            "MAX_TOPK",
            "MIN_ROWS",
            "MIN_SPARSECORE_KEYS",
            "ROW_WORDS",
            "SC_LANES",
            "SLOTS",
            "SUB_BLOCK_ROWS",
            "WEIGHT_BYTES_PER_HEAD",
            "WORD_BYTES",
            "paged_indexer_topk",
            "score_word_rows",
            "sparsecore_topk",
        }
        forbidden_runtime = {"torch", "torch_tpu", "vllm"}
        assert word_scores_imports.isdisjoint(forbidden_runtime | {"os"})
        assert init_imports.isdisjoint(forbidden_runtime | {"os"})
        assert bench_imports.isdisjoint(forbidden_runtime)

    def test_should_reject_swapped_key_slots_in_reference_oracle(self) -> None:
        # Arrange
        q, weights, keys = _rand_inputs(17, 2, 1, 8, 4)
        cu_q_lens = _uniform_cu_q_lens(2, 1)
        swapped_keys = jnp.asarray(
            np.asarray(keys)
            .reshape(1, 8, SLOTS, DIM)[:, :, ::-1, :]
            .reshape(1, SLOTS * 8, DIM)
        )

        # Act
        ref_orig = _reference_scores(q, weights, keys, cu_q_lens)
        ref_swapped = _reference_scores(q, weights, swapped_keys, cu_q_lens)

        # Assert
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                np.asarray(ref_orig), np.asarray(ref_swapped), rtol=1e-2, atol=1e-2
            )

    @parameterized.expand(
        [
            ("single_seq_min_rows_1", 2, 1, 1, 4, [0, 2], False, False),
            ("unaligned_rows_3", 2, 2, 3, 4, [0, 1, 2], False, False),
            ("unaligned_rows_10", 2, 2, 10, 4, [0, 1, 2], False, False),
            ("exact_sub_block_256_scope", 2, 1, 256, 4, [0, 2], True, True),
            ("exact_max_block_512", 2, 1, 512, 4, [0, 2], False, True),
            ("multi_block_unaligned_520", 2, 1, 520, 4, [0, 2], False, False),
            ("ragged_padded_spans", 6, 3, 8, 4, [1, 3, 3, 5], False, True),
            ("clipped_oob_and_empty_spans", 6, 3, 8, 4, [-2, 2, 2, 10], False, True),
            ("all_empty_groups", 6, 3, 8, 4, [2, 2, 2, 2], False, False),
            (
                "raw_non_monotonic_clipped_monotonic",
                6,
                3,
                8,
                4,
                [-1, -5, 9, 7],
                False,
                True,
            ),
            (
                "raw_non_monotonic_tail_clipped_end",
                6,
                3,
                8,
                4,
                [0, 8, 7, 9],
                False,
                False,
            ),
            ("empty_tokens_0", 0, 2, 8, 4, [0, 0, 0], False, True),
            ("empty_heads_0", 2, 2, 8, 0, [0, 1, 2], False, False),
        ]
    )
    def test_should_match_reference_and_zero_padded_tokens_with_cu_q_lens(
        self,
        _name: str,
        tokens: int,
        groups: int,
        rows: int,
        heads: int,
        cu_q_list: list[int],
        enable_named_scope: bool,
        pass_as_keyword: bool,
    ) -> None:
        # Arrange
        num_keys = SLOTS * rows
        q, weights, keys = _rand_inputs(
            99 + rows + tokens + groups, tokens, groups, rows, heads
        )
        cu_q_lens = jnp.array(cu_q_list, dtype=jnp.int32)
        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)

        # Act
        if pass_as_keyword:
            actual = score_word_rows(
                q_split,
                weights,
                words,
                cu_q_lens=cu_q_lens,
                interpret=True,
                enable_named_scope=enable_named_scope,
            )
        else:
            actual = score_word_rows(
                q_split,
                weights,
                words,
                cu_q_lens,
                interpret=True,
                enable_named_scope=enable_named_scope,
            )
        expected = _reference_scores(q, weights, keys, cu_q_lens)

        # Assert
        assert actual.shape == (tokens, num_keys)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=1e-2
        )
        first_active = int(np.clip(cu_q_list[0], 0, tokens))
        last_active = int(np.clip(cu_q_list[-1], first_active, tokens))
        if first_active > 0:
            np.testing.assert_array_equal(np.asarray(actual[:first_active]), 0.0)
        if last_active < tokens:
            np.testing.assert_array_equal(np.asarray(actual[last_active:]), 0.0)

    def test_should_clamp_all_negative_dots_to_exact_zero(self) -> None:
        # Arrange
        rng = np.random.default_rng(73)
        tokens, groups, rows, heads = 6, 3, 8, 4
        q = jnp.asarray(
            -(
                np.abs(rng.standard_normal((tokens, heads, DIM), dtype=np.float32))
                + 0.5
            ),
            dtype=jnp.bfloat16,
        )
        weights = jnp.asarray(
            rng.standard_normal((tokens, heads, 1), dtype=np.float32) * 5.0,
            dtype=jnp.float32,
        )
        keys = jnp.asarray(
            np.abs(rng.standard_normal((groups, SLOTS * rows, DIM), dtype=np.float32))
            + 0.5,
            dtype=jnp.bfloat16,
        )
        cu_q_lens = _uniform_cu_q_lens(tokens, groups)

        # Act
        actual = score_word_rows(
            _split_elements(q),
            weights,
            _pack_keys_to_words(keys),
            cu_q_lens,
            interpret=True,
        )

        # Assert
        assert actual.shape == (tokens, SLOTS * rows)
        np.testing.assert_array_equal(np.asarray(actual), 0.0)

    def test_should_preserve_cold_warm_jit_and_derived_seam_invariants(self) -> None:
        # Arrange
        tokens, groups, rows, heads = 6, 3, 8, 4
        q, w1, keys = _rand_inputs(211, tokens, groups, rows, heads)
        rng = np.random.default_rng(212)
        w2 = jnp.asarray(
            rng.standard_normal((tokens, heads, 1), dtype=np.float32), dtype=jnp.float32
        )
        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)
        cu_q_lens = _uniform_cu_q_lens(tokens, groups)
        jit_fn = jax.jit(
            lambda q_in, w_in, k_in, cu_in: score_word_rows(
                q_in, w_in, k_in, cu_in, interpret=True
            )
        )

        # Act
        w_combo = jnp.asarray(
            1.75 * np.asarray(w1) - 0.5 * np.asarray(w2), dtype=jnp.float32
        )
        q_scaled = jnp.asarray(
            2.0 * np.asarray(q_split, dtype=np.float32), dtype=jnp.bfloat16
        )
        out_eager = score_word_rows(q_split, w1, words, cu_q_lens, interpret=True)
        out_jit_cold = jit_fn(q_split, w1, words, cu_q_lens)
        out_jit_warm = jit_fn(q_split, w1, words, cu_q_lens)
        out_w2 = score_word_rows(q_split, w2, words, cu_q_lens, interpret=True)
        out_combo = score_word_rows(q_split, w_combo, words, cu_q_lens, interpret=True)
        out_scaled_q = score_word_rows(q_scaled, w1, words, cu_q_lens, interpret=True)

        # Assert
        np.testing.assert_allclose(
            np.asarray(out_jit_cold), np.asarray(out_eager), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(
            np.asarray(out_jit_cold), np.asarray(out_jit_warm)
        )
        np.testing.assert_allclose(
            np.asarray(out_combo),
            1.75 * np.asarray(out_eager) - 0.5 * np.asarray(out_w2),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(out_scaled_q),
            2.0 * np.asarray(out_eager),
            rtol=1e-2,
            atol=1e-2,
        )

    @parameterized.expand(
        [
            (
                "q_rank_2d",
                (4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected 3D tensors",
            ),
            (
                "weights_rank_2d",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected 3D tensors",
            ),
            (
                "words_rank_2d",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected 3D tensors",
            ),
            (
                "q_bad_dim",
                (4, 4, 64),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "q_bad_dtype",
                (4, 4, DIM),
                jnp.float32,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "weights_bad_shape",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 2),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "weights_bad_dtype",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.bfloat16,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "words_bad_row_words",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, 64),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "words_bad_dtype",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.float32,
                ([0, 2, 4], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "words_zero_groups",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (0, 8, ROW_WORDS),
                jnp.int32,
                ([0], jnp.int32),
                True,
                "Expected BF16 q",
            ),
            (
                "cu_q_lens_bad_shape",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2], jnp.int32),
                True,
                "Expected int32 cu_q_lens",
            ),
            (
                "cu_q_lens_bad_dtype",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.float32),
                True,
                "Expected int32 cu_q_lens",
            ),
            (
                "cu_q_lens_non_monotonic",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([0, 3, 1], jnp.int32),
                True,
                "monotonically non-decreasing",
            ),
            (
                "cu_q_lens_non_monotonic_after_clip",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ([5, 2, 6], jnp.int32),
                True,
                "monotonically non-decreasing",
            ),
            (
                "rows_below_min_1",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 0, ROW_WORDS),
                jnp.int32,
                ([0, 2, 4], jnp.int32),
                True,
                "Word rows per group",
            ),
        ]
    )
    def test_should_reject_invalid_inputs(
        self,
        _name: str,
        q_shape: tuple[int, ...],
        q_dtype: jnp.dtype,
        w_shape: tuple[int, ...],
        w_dtype: jnp.dtype,
        words_shape: tuple[int, ...],
        words_dtype: jnp.dtype,
        cu_spec: tuple[list[int], jnp.dtype],
        interpret: bool,
        match: str,
    ) -> None:
        # Arrange
        q = jnp.zeros(q_shape, dtype=q_dtype)
        weights = jnp.ones(w_shape, dtype=w_dtype)
        words = jnp.zeros(words_shape, dtype=words_dtype)
        cu_q_lens = jnp.array(cu_spec[0], dtype=cu_spec[1])

        # Act & Assert
        with pytest.raises(ValueError, match=match):
            score_word_rows(q, weights, words, cu_q_lens, interpret=interpret)

    def test_should_reject_non_tpu7x_generation_and_lane_count(self) -> None:
        # Arrange
        v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
        v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
        q_empty = jnp.zeros((0, 4, DIM), dtype=jnp.bfloat16)
        w_empty = jnp.ones((0, 4, 1), dtype=jnp.float32)
        words = jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32)
        cu_empty = jnp.zeros((3,), dtype=jnp.int32)
        orig_get_tpu_info = word_scores_mod.pltpu.get_tpu_info

        # Act & Assert
        with (
            jax.sharding.use_abstract_mesh(v6e_mesh),
            pytest.raises(RuntimeError, match="requires TPU7x"),
        ):
            score_word_rows(q_empty, w_empty, words, cu_empty, interpret=True)

        with (
            patch.object(
                word_scores_mod.pltpu,
                "get_tpu_info",
                side_effect=lambda: dataclasses.replace(
                    orig_get_tpu_info(), num_lanes=64
                ),
            ),
            pytest.raises(RuntimeError, match="requires TPU7x"),
        ):
            score_word_rows(
                jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16),
                jnp.ones((2, 4, 1), dtype=jnp.float32),
                words,
                jnp.array([0, 1, 2], dtype=jnp.int32),
                interpret=True,
            )

    def test_should_chunk_across_vmem_cap_with_exact_and_remainder_batches(
        self,
    ) -> None:
        # Arrange: R=16 is unique to this test so both forward/reverse orders miss cache
        # cleanly under the single patched VMEM limit without jax.clear_caches().
        rows, heads = 16, 4
        cases = (
            (6, 2, [0, 3, 6]),
            (7, 2, [1, 4, 6]),
        )
        orig_get_tpu_info = word_scores_mod.pltpu.get_tpu_info
        # Force max_tokens_per_chunk = 3 < tokens (padded_heads=8, padded_rows=256 ->
        # bytes_per_token=8192, fixed_scratch=786432; usable=24583 // 8192 = 3).
        with patch.object(
            word_scores_mod.pltpu,
            "get_tpu_info",
            side_effect=lambda: dataclasses.replace(
                orig_get_tpu_info(), vmem_capacity_bytes=1_853_748
            ),
        ):
            for tokens, groups, cu_list in cases:
                q, weights, keys = _rand_inputs(
                    303 + tokens + groups, tokens, groups, rows, heads
                )
                cu = jnp.array(cu_list, dtype=jnp.int32)
                q_split, words = _split_elements(q), _pack_keys_to_words(keys)

                # Act
                hlo_text = (
                    jax.jit(score_word_rows)
                    .trace(
                        jax.ShapeDtypeStruct(q_split.shape, q_split.dtype),
                        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
                        jax.ShapeDtypeStruct(words.shape, words.dtype),
                        jax.ShapeDtypeStruct(cu.shape, cu.dtype),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text()
                )
                actual = score_word_rows(q_split, weights, words, cu, interpret=True)
                expected = _reference_scores(q, weights, keys, cu)

                # Assert: prove chunk_cap < total_tokens triggered jax.lax.scan
                assert "while(" in hlo_text
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=1e-2
                )

    @parameterized.expand(
        [
            ("single_call_within_budget_1024x1x256x32", 1024, 1, 256, 32, False),
            ("scan_lane_padded_weights_2048x1x256x32", 2048, 1, 256, 32, True),
            ("scan_lane_padded_weights_4096x1x1024x32", 4096, 1, 1024, 32, True),
            ("scan_lane_padded_weights_1024x1x256x64", 1024, 1, 256, 64, True),
            ("scan_lane_padded_groups_4096x16x64x32", 4096, 16, 64, 32, True),
            ("scan_sublane_padded_heads_4096x1x256x1", 4096, 1, 256, 1, True),
        ]
    )
    def test_should_trigger_scan_chunking_from_tpu_vmem_layout_budget(
        self,
        _name: str,
        tokens: int,
        groups: int,
        rows: int,
        heads: int,
        expect_scan_loop: bool,
    ) -> None:
        # Arrange
        q = jax.ShapeDtypeStruct((tokens, heads, DIM), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((tokens, heads, 1), jnp.float32)
        words = jax.ShapeDtypeStruct((groups, rows, ROW_WORDS), jnp.int32)
        cu_q_lens = jax.ShapeDtypeStruct((groups + 1,), jnp.int32)

        # Act
        hlo_text = (
            jax.jit(score_word_rows)
            .trace(q, weights, words, cu_q_lens)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )

        # Assert
        assert "dsv41_index_word_scores" in hlo_text
        assert ("while(" in hlo_text) is expect_scan_loop

    @parameterized.expand(
        [
            ("unaligned_12x4x3", 12, 4, 3, 32, False),
            ("decode_48x48x4096", 48, 48, 4096, 32, False),
            ("mtp_288x48x6144", 288, 48, 6144, 32, False),
            ("prefill_512x1x4096", 512, 1, 4096, 32, True),
            ("nonpow2_24x4x520", 24, 4, 520, 32, False),
            ("ctx128k_scan_512x4x65536", 512, 4, 65536, 32, False),
        ]
    )
    def test_should_lower_pallas_mosaic_kernel_for_abstract_tpu7x_workloads(
        self,
        _name: str,
        tokens: int,
        groups: int,
        rows: int,
        heads: int,
        enable_named_scope: bool,
    ) -> None:
        # Arrange
        q = jax.ShapeDtypeStruct((tokens, heads, DIM), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((tokens, heads, 1), jnp.float32)
        words = jax.ShapeDtypeStruct((groups, rows, ROW_WORDS), jnp.int32)
        cu_q_lens = jax.ShapeDtypeStruct((groups + 1,), jnp.int32)

        # Act
        lowered = (
            jax.jit(
                score_word_rows, static_argnames=("interpret", "enable_named_scope")
            )
            .trace(
                q,
                weights,
                words,
                cu_q_lens,
                interpret=False,
                enable_named_scope=enable_named_scope,
            )
            .lower(lowering_platforms=("tpu",))
        )
        hlo_text = lowered.as_text()

        # Assert
        assert "dsv41_index_word_scores" in hlo_text
        assert lowered.out_info == jax.ShapeDtypeStruct(
            (tokens, SLOTS * rows), jnp.float32
        )
        assert f"tensor<{tokens}x{SLOTS * rows}xf32>" in hlo_text

    def test_should_reject_non_tpu7x_in_benchmark_main(self) -> None:
        # Arrange
        v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
        v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)

        # Act & Assert
        with (
            patch.object(
                benchmark_word_scores.jax, "default_backend", return_value="cpu"
            ),
            pytest.raises(RuntimeError, match="requires physical TPU7x hardware"),
        ):
            benchmark_word_scores.main([])

        with (
            patch.object(
                benchmark_word_scores.jax, "default_backend", return_value="tpu"
            ),
            jax.sharding.use_abstract_mesh(v6e_mesh),
            pytest.raises(RuntimeError, match="requires TPU7x"),
        ):
            benchmark_word_scores.main([])

    def test_should_extract_median_device_and_host_us_from_xplane(self) -> None:
        # Arrange
        fake_events = [
            SimpleNamespace(duration_ns=3200.0),
            SimpleNamespace(duration_ns=4800.0),
            SimpleNamespace(duration_ns=3600.0),
        ]
        fake_line = SimpleNamespace(
            name=benchmark_word_scores.XLA_MODULES_LINE, events=fake_events
        )
        fake_plane = SimpleNamespace(
            name=benchmark_word_scores.TPU_DEVICE_PLANE, lines=[fake_line]
        )
        fake_profile = SimpleNamespace(planes=[fake_plane])

        # Act
        with (
            patch.object(
                benchmark_word_scores.jax.profiler,
                "trace",
                side_effect=lambda _dir: contextlib.nullcontext(),
            ),
            patch.object(benchmark_word_scores.glob, "glob", return_value=["fake.pb"]),
            patch.object(
                benchmark_word_scores.jax.profiler.ProfileData,
                "from_file",
                return_value=fake_profile,
            ),
        ):
            dev_us, host_us = benchmark_word_scores.measure_device_and_host_us(
                lambda: jnp.zeros((1,), dtype=jnp.float32),
                warmup=1,
                iters=3,
            )

        # Assert
        assert dev_us == pytest.approx(3.6)
        assert host_us > 0.0

    @parameterized.expand(
        [
            ("missing_xplane_file", [], [], "No .xplane.pb profile file"),
            (
                "missing_tpu_plane",
                ["fake.pb"],
                [SimpleNamespace(name="/host:CPU", lines=[])],
                "Missing profiler plane '/device:TPU:0'",
            ),
            (
                "missing_xla_modules_line",
                ["fake.pb"],
                [
                    SimpleNamespace(
                        name=benchmark_word_scores.TPU_DEVICE_PLANE,
                        lines=[SimpleNamespace(name="Other Line", events=[])],
                    )
                ],
                "Missing profiler line 'XLA Modules'",
            ),
            (
                "empty_device_events",
                ["fake.pb"],
                [
                    SimpleNamespace(
                        name=benchmark_word_scores.TPU_DEVICE_PLANE,
                        lines=[
                            SimpleNamespace(
                                name=benchmark_word_scores.XLA_MODULES_LINE, events=[]
                            )
                        ],
                    )
                ],
                "Empty device event samples",
            ),
        ]
    )
    def test_should_raise_runtime_error_when_benchmark_xplane_is_incomplete(
        self,
        _name: str,
        xplane_files: list[str],
        planes: list[Any],
        match: str,
    ) -> None:
        # Arrange
        fake_profile = SimpleNamespace(planes=planes)

        # Act & Assert
        with (
            patch.object(
                benchmark_word_scores.jax.profiler,
                "trace",
                side_effect=lambda _dir: contextlib.nullcontext(),
            ),
            patch.object(benchmark_word_scores.glob, "glob", return_value=xplane_files),
            patch.object(
                benchmark_word_scores.jax.profiler.ProfileData,
                "from_file",
                return_value=fake_profile,
            ),
            pytest.raises(RuntimeError, match=match),
        ):
            benchmark_word_scores.measure_device_and_host_us(
                lambda: jnp.zeros((1,), dtype=jnp.float32),
                warmup=1,
                iters=1,
            )

    @pytest.mark.tpu_test
    def test_should_clip_oob_and_raw_non_monotonic_cu_q_lens_on_tpu7x_hardware(
        self,
    ) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        # Arrange & Act & Assert
        cases = [
            (4, 3, 128, 32, [-2, 2, 2, 10]),
            (8, 4, 256, 32, [-1, -5, 2, 15, 13]),
            (4, 2, 64, 32, [5, 4, 6]),
        ]
        for idx, (tokens, groups, rows, heads, cu_list) in enumerate(cases):
            rng = jax.random.key(611 + idx)
            k_q, k_w, k_k = jax.random.split(rng, 3)
            q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
            weights = jax.random.normal(k_w, (tokens, heads, 1), dtype=jnp.float32)
            keys = jax.random.normal(
                k_k, (groups, SLOTS * rows, DIM), dtype=jnp.bfloat16
            )
            cu_q_lens = jnp.array(cu_list, dtype=jnp.int32)

            actual = score_word_rows(
                _split_elements(q),
                weights,
                _pack_keys_to_words(keys),
                cu_q_lens,
                interpret=False,
            )
            expected = _reference_scores(q, weights, keys, cu_q_lens)
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=2e-2
            )

    @parameterized.expand(
        [
            ("prefill_2048x1x256x32", 2048, 1, 256, 32),
            ("prefill_4096x1x256x32", 4096, 1, 256, 32),
            ("prefill_2048x1x512x32", 2048, 1, 512, 32),
            ("prefill_4096x1x1024x32", 4096, 1, 1024, 32),
            ("prefill_1024x1x256x64", 1024, 1, 256, 64),
            ("prefill_2048x1x256x64", 2048, 1, 256, 64),
            ("batched_4096x16x64x32", 4096, 16, 64, 32),
            ("sublane_heads_4096x1x256x1", 4096, 1, 256, 1),
        ]
    )
    @pytest.mark.tpu_test
    def test_should_bound_vmem_across_large_token_and_head_prefill_on_tpu7x_hardware(
        self,
        _name: str,
        tokens: int,
        groups: int,
        rows: int,
        heads: int,
    ) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        # Arrange
        rng = jax.random.key(809 + tokens + rows + heads)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.normal(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        cu_q_lens = _uniform_cu_q_lens(tokens, groups)

        # Act
        actual = score_word_rows(
            _split_elements(q),
            weights,
            _pack_keys_to_words(keys),
            cu_q_lens,
            interpret=False,
        )
        expected = _reference_scores(q, weights, keys, cu_q_lens)

        # Assert
        assert actual.shape == (tokens, num_keys)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=3e-2
        )

    @parameterized.expand(
        [
            ("single_block_min_rows_1", 1),
            ("single_block_unaligned_3", 3),
            ("single_block_non_pow2_24", 24),
            ("single_block_504", 504),
            ("single_block_512", 512),
            ("multi_block_unaligned_520", 520),
            ("multi_block_non_pow2_576", 576),
            ("multi_block_1024", 1024),
            ("multi_block_non_pow2_3072", 3072),
            ("multi_block_non_pow2_6144", 6144),
        ]
    )
    @pytest.mark.tpu_test
    def test_should_match_reference_on_tpu7x_hardware(
        self, _name: str, rows: int
    ) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        # Arrange
        rng = jax.random.key(7 + rows)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        tokens, groups, heads = 12, 4, 32
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.normal(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)

        # Act & Assert: uniform-spaced cu_q_lens
        cu_uni = _uniform_cu_q_lens(tokens, groups)
        actual_uni = score_word_rows(q_split, weights, words, cu_uni, interpret=False)
        expected_uni = _reference_scores(q, weights, keys, cu_uni)
        np.testing.assert_allclose(
            np.asarray(actual_uni), np.asarray(expected_uni), rtol=1e-2, atol=2e-2
        )

        # Act & Assert: ragged cu_q_lens
        cu_rag = jnp.array([0, 2, 5, 9, 11], dtype=jnp.int32)
        actual_rag = score_word_rows(q_split, weights, words, cu_rag, interpret=False)
        expected_rag = _reference_scores(q, weights, keys, cu_rag)
        np.testing.assert_allclose(
            np.asarray(actual_rag), np.asarray(expected_rag), rtol=1e-2, atol=2e-2
        )

    @pytest.mark.tpu_test
    def test_should_handle_128k_vmem_chunking_on_tpu7x_hardware(self) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        # Arrange
        rng = jax.random.key(515)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        tokens, groups, rows, heads = 192, 4, 32768, 32
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.normal(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        q_split, words = _split_elements(q), _pack_keys_to_words(keys)
        cu_q_lens = _uniform_cu_q_lens(tokens, groups)

        # Act
        actual = score_word_rows(q_split, weights, words, cu_q_lens, interpret=False)
        expected = _reference_scores(q, weights, keys, cu_q_lens)

        # Assert
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=3e-2
        )
