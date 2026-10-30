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
"""Unit tests for score_word_rows and benchmark_word_scores."""

import dataclasses
from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from parameterized import parameterized

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    DIM,
    ROW_WORDS,
    SLOTS,
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


def _split_elements(x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-1], 64, 2).swapaxes(-1, -2).reshape(x.shape)


def _pack_keys_to_words(keys: jax.Array) -> jax.Array:
    groups, num_keys, _ = keys.shape
    rows = num_keys // SLOTS
    paired = keys.reshape(groups, rows, SLOTS, 64, 2)
    bits = jax.lax.bitcast_convert_type(paired, jnp.uint16).astype(jnp.uint32)
    packed = bits[..., 0] | (bits[..., 1] << 16)
    return packed.reshape(groups, rows, ROW_WORDS).astype(jnp.int32)


def _reference_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    *,
    cu_q_lens: jax.Array | None = None,
) -> jax.Array:
    tokens, heads, _ = q.shape
    groups, num_keys, _ = keys.shape
    if tokens == 0 or heads == 0:
        return jnp.zeros((tokens, num_keys), dtype=jnp.float32)
    if cu_q_lens is None:
        group_tokens = tokens // groups
        q_grouped = q.reshape(groups, group_tokens, heads, DIM).astype(jnp.float32)
        w_grouped = weights.reshape(groups, group_tokens, heads, 1)
        logits = jnp.einsum("gqhd,gkd->gqhk", q_grouped, keys.astype(jnp.float32))
        scores = (jnp.maximum(logits, 0.0) * w_grouped).sum(axis=2)
        return scores.reshape(tokens, num_keys)

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


class TestScoreWordRows:
    @parameterized.expand(
        [
            ("single_block_min_rows_1", 2, 2, 1, 4, False),
            ("single_block_unaligned_rows_3", 2, 2, 3, 4, False),
            ("single_block_rows_8", 2, 2, 8, 4, False),
            ("single_block_non_pow2_24", 4, 2, 24, 8, False),
            ("single_block_sub_block_128", 2, 2, 128, 4, True),
            ("single_block_non_sub_div_136", 2, 2, 136, 4, False),
            ("single_block_multi_sub_256", 6, 2, 256, 4, False),
            ("single_block_padded_504", 2, 1, 504, 4, False),
            ("single_block_max_512", 2, 1, 512, 4, False),
            ("multi_block_unaligned_520", 2, 1, 520, 4, False),
            ("multi_block_non_pow2_576", 2, 1, 576, 4, False),
            ("multi_block_256_div_768", 2, 1, 768, 4, False),
            ("multi_block_1024", 4, 2, 1024, 4, False),
            ("loop_group_tokens_9", 18, 2, 128, 4, False),
            ("interpret_unaligned_rows_10", 2, 2, 10, 4, False),
            ("empty_tokens_0", 0, 2, 16, 4, False),
            ("empty_heads_0", 2, 2, 16, 0, False),
        ]
    )
    def test_should_match_reference_on_uniform_groups(
        self,
        _name: str,
        tokens: int,
        groups: int,
        rows: int,
        heads: int,
        enable_named_scope: bool,
    ) -> None:
        rng = jax.random.key(42 + rows + tokens + heads)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)

        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)

        actual = score_word_rows(
            q_split,
            weights,
            words,
            interpret=True,
            enable_named_scope=enable_named_scope,
        )
        expected = _reference_scores(q, weights, keys)

        assert actual.shape == (tokens, num_keys)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=1e-2
        )

    @parameterized.expand(
        [
            ("single_block_padded_spans", 7, 3, 16, 4, [1, 3, 3, 5], True),
            ("multi_block_1024_rows", 6, 2, 1024, 4, [1, 3, 5], False),
            ("clipped_and_empty_spans", 4, 3, 128, 4, [-2, 2, 2, 10], False),
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
    ) -> None:
        num_keys = SLOTS * rows
        rng = jax.random.key(99 + rows + tokens)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        cu_q_lens = jnp.array(cu_q_list, dtype=jnp.int32)

        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)

        actual = score_word_rows(
            q_split,
            weights,
            words,
            cu_q_lens=cu_q_lens,
            interpret=True,
            enable_named_scope=enable_named_scope,
        )
        expected = _reference_scores(q, weights, keys, cu_q_lens=cu_q_lens)

        assert actual.shape == (tokens, num_keys)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=1e-2
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
                None,
                True,
                "Expected 3D tensors",
            ),
            (
                "weights_rank_2d",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
                True,
                "Expected BF16 q",
            ),
            (
                "tokens_not_divisible_by_groups",
                (5, 4, DIM),
                jnp.bfloat16,
                (5, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                None,
                True,
                "divisible by groups",
            ),
            (
                "cu_q_lens_bad_shape",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 8, ROW_WORDS),
                jnp.int32,
                ((2,), jnp.int32),
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
                ((3,), jnp.float32),
                True,
                "Expected int32 cu_q_lens",
            ),
            (
                "rows_below_min_1",
                (4, 4, DIM),
                jnp.bfloat16,
                (4, 4, 1),
                jnp.float32,
                (2, 0, ROW_WORDS),
                jnp.int32,
                None,
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
        cu_spec: tuple[tuple[int, ...], jnp.dtype] | None,
        interpret: bool,
        match: str,
    ) -> None:
        q = jnp.zeros(q_shape, dtype=q_dtype)
        weights = jnp.ones(w_shape, dtype=w_dtype)
        words = jnp.zeros(words_shape, dtype=words_dtype)
        cu_q_lens = None if cu_spec is None else jnp.zeros(cu_spec[0], dtype=cu_spec[1])
        with pytest.raises(ValueError, match=match):
            score_word_rows(q, weights, words, cu_q_lens=cu_q_lens, interpret=interpret)

    def test_should_reject_non_tpu7x_generation(self) -> None:
        v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
        v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
        q = jnp.zeros((2, 4, DIM), dtype=jnp.bfloat16)
        weights = jnp.ones((2, 4, 1), dtype=jnp.float32)
        words = jnp.zeros((2, 8, ROW_WORDS), dtype=jnp.int32)
        with (
            jax.sharding.use_abstract_mesh(v6e_mesh),
            pytest.raises(RuntimeError, match="requires TPU7x"),
        ):
            score_word_rows(q, weights, words, interpret=True)

    @parameterized.expand(
        [
            ("single_block_504", 504),
            ("single_block_512", 512),
            ("multi_block_unaligned_520", 520),
            ("multi_block_576", 576),
            ("multi_block_1024", 1024),
        ]
    )
    @pytest.mark.tpu_test
    def test_should_match_reference_on_tpu7x_hardware(
        self, _name: str, rows: int
    ) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        rng = jax.random.key(7 + rows)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        tokens, groups, heads = 12, 4, 32
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        q_split = _split_elements(q)
        words = _pack_keys_to_words(keys)

        actual_uni = score_word_rows(q_split, weights, words, interpret=False)
        expected_uni = _reference_scores(q, weights, keys)
        np.testing.assert_allclose(
            np.asarray(actual_uni), np.asarray(expected_uni), rtol=1e-2, atol=1e-2
        )

        cu_q_lens = jnp.array([0, 2, 5, 9, 11], dtype=jnp.int32)
        actual_rag = score_word_rows(
            q_split, weights, words, cu_q_lens=cu_q_lens, interpret=False
        )
        expected_rag = _reference_scores(q, weights, keys, cu_q_lens=cu_q_lens)
        np.testing.assert_allclose(
            np.asarray(actual_rag), np.asarray(expected_rag), rtol=1e-2, atol=1e-2
        )

    @parameterized.expand(
        [
            ("uniform_grp_exact", 4, 2, None),
            ("uniform_grp_rem", 5, 5, None),
            ("uniform_intra_grp_exact", 6, 1, None),
            ("uniform_intra_grp_rem", 10, 2, None),
            ("ragged_exact", 6, 2, [0, 3, 6]),
            ("ragged_rem", 7, 3, [1, 3, 5, 6]),
        ]
    )
    def test_should_chunk_across_vmem_cap_with_exact_and_remainder_batches(
        self,
        _name: str,
        tokens: int,
        groups: int,
        cu_list: list[int] | None,
    ) -> None:
        rows, heads = 128, 4
        num_keys = SLOTS * rows
        rng = jax.random.key(303 + tokens + groups)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        cu = None if cu_list is None else jnp.array(cu_list, dtype=jnp.int32)
        q_split, words = _split_elements(q), _pack_keys_to_words(keys)

        orig_get_tpu_info = word_scores_mod.pltpu.get_tpu_info
        with patch.object(
            word_scores_mod.pltpu,
            "get_tpu_info",
            side_effect=lambda: dataclasses.replace(
                orig_get_tpu_info(), vmem_capacity_bytes=49152
            ),
        ):
            assert word_scores_mod._max_vmem_tokens(heads, 256) == 3
            jax.clear_caches()
            actual = score_word_rows(
                q_split, weights, words, cu_q_lens=cu, interpret=True
            )
        jax.clear_caches()

        expected = _reference_scores(q, weights, keys, cu_q_lens=cu)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.tpu_test
    def test_should_handle_128k_vmem_chunking_on_tpu7x_hardware(self) -> None:
        if jax.default_backend() != "tpu":
            pytest.skip("TPU7x hardware required")
        rng = jax.random.key(515)
        k_q, k_w, k_k = jax.random.split(rng, 3)
        tokens, groups, rows, heads = 192, 4, 32768, 32
        num_keys = SLOTS * rows
        q = jax.random.normal(k_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(k_w, (tokens, heads, 1), dtype=jnp.float32)
        keys = jax.random.normal(k_k, (groups, num_keys, DIM), dtype=jnp.bfloat16)
        q_split, words = _split_elements(q), _pack_keys_to_words(keys)
        actual = score_word_rows(q_split, weights, words, interpret=False)
        expected = _reference_scores(q, weights, keys)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=2e-2
        )

    def test_benchmark_word_scores_cli_interpret(self) -> None:
        assert (
            benchmark_word_scores.main(["--interpret", "--warmup", "1", "--iters", "1"])
            == 0
        )
        fake_event = SimpleNamespace(duration_ns=3200.0)
        fake_line = SimpleNamespace(
            name=benchmark_word_scores.XLA_MODULES_LINE, events=[fake_event]
        )
        fake_plane = SimpleNamespace(
            name=benchmark_word_scores.TPU_DEVICE_PLANE, lines=[fake_line]
        )
        fake_profile = SimpleNamespace(planes=[fake_plane])
        with (
            patch.object(
                benchmark_word_scores.jax, "default_backend", return_value="tpu"
            ),
            patch.object(
                benchmark_word_scores.jax.profiler,
                "trace",
                side_effect=lambda _dir: nullcontext(),
            ),
            patch.object(benchmark_word_scores.glob, "glob", return_value=["fake.pb"]),
            patch.object(
                benchmark_word_scores.jax.profiler.ProfileData,
                "from_file",
                return_value=fake_profile,
            ),
        ):
            dev_us, host_us = benchmark_word_scores._measure_device_and_host_us(
                lambda: jnp.zeros((1,), dtype=jnp.float32),
                warmup=1,
                iters=1,
                interpret=False,
            )
        assert dev_us == pytest.approx(3.2)
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
        fake_profile = SimpleNamespace(planes=planes)
        with (
            patch.object(
                benchmark_word_scores.jax, "default_backend", return_value="tpu"
            ),
            patch.object(
                benchmark_word_scores.jax.profiler,
                "trace",
                side_effect=lambda _dir: nullcontext(),
            ),
            patch.object(benchmark_word_scores.glob, "glob", return_value=xplane_files),
            patch.object(
                benchmark_word_scores.jax.profiler.ProfileData,
                "from_file",
                return_value=fake_profile,
            ),
            pytest.raises(RuntimeError, match=match),
        ):
            benchmark_word_scores._measure_device_and_host_us(
                lambda: jnp.zeros((1,), dtype=jnp.float32),
                warmup=1,
                iters=1,
                interpret=False,
            )
