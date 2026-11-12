# SPDX-License-Identifier: Apache-2.0
"""Selection contract and backend injection through the real scoring pipeline."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)


@dataclass(frozen=True)
class ReferenceTopK:
    """Test-only JAX selector; not a proposed TensorCore implementation."""

    overlaps_scoring: ClassVar[bool] = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        assert scheduling_group_id is None
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None],
            scores,
            -jnp.inf,
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids


@pytest.mark.parametrize("return_scores", [False, True])
def test_reference_contract_masks_rows_and_pads(return_scores):
    scores = np.array([[2, 2, 1, 100], [1, 2, 3, 4]], np.float32)
    selected = ReferenceTopK().select(
        jnp.asarray(scores.view(np.int32)),
        3,
        jnp.array([2, 0], jnp.int32),
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    ids = selected[0] if return_scores else selected
    np.testing.assert_array_equal(ids, [[0, 1, -1], [-1, -1, -1]])
    if return_scores:
        np.testing.assert_array_equal(
            np.asarray(selected[1]).view(np.float32),
            [[2, 2, -np.inf], [-np.inf, -np.inf, -np.inf]],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("return_scores", [False, True])
def test_default_selector_preserves_sparsecore_results(return_scores):
    rng = np.random.default_rng(42)
    scores = rng.integers(-4, 5, (32, 512)).astype(np.float32)
    scores[:, 31] = -np.inf
    bits = jnp.asarray(scores.view(np.int32))
    lengths = jnp.array([0, 17, 128, 511] * 8, jnp.int32)
    expected = sparsecore_topk(bits, 128, lengths, return_scores=return_scores)
    actual = SparseCoreTopK().select(
        bits,
        128,
        lengths,
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)
    ids = np.asarray(actual[0] if return_scores else actual)
    for row, length in enumerate(np.asarray(lengths)):
        live = ids[row][ids[row] >= 0]
        assert np.all(live < length)
        assert len(live) == len(np.unique(live))
        eligible = scores[row, :length]
        eligible = eligible[eligible > -np.inf]
        np.testing.assert_array_equal(
            np.sort(scores[row, live]),
            np.sort(eligible)[-128:],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_tokens", [None, 128])
@pytest.mark.parametrize("return_scores", [False, True])
def test_backend_replacement_keeps_real_streamindex_scores(chunk_tokens, return_scores):
    # All scores are zero: varying causal row lengths exercise ties, masks,
    # short rows and padding without a tolerance hiding selection errors.
    q = jnp.zeros((256, 4, 128), jnp.bfloat16)
    weights = jnp.zeros((256, 4), jnp.float32)
    cache = jnp.zeros((4, 32, 4, 256), jnp.uint8)
    args = (
        q,
        weights,
        cache,
        jnp.array([256, 257]),
        jnp.array([0, 1, 2, 3]),
        jnp.array([0, 128, 256]),
        jnp.array([0, 0, 2]),
    )
    kwargs = dict(
        k=128,
        compression_ratio=2,
        num_kv_pages_per_block=2,
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
        return_scores=return_scores,
    )
    default = streamindex_topk(*args, **kwargs)
    alternate = streamindex_topk(*args, topk_backend=ReferenceTopK(), **kwargs)
    for result in (default, alternate):
        ids = np.asarray(result[0] if return_scores else result)
        positions = np.r_[np.arange(128, 256), np.arange(129, 257)]
        lengths = np.minimum(positions // 2 + 1, np.repeat([128, 128], 128))
        for row, length in enumerate(lengths):
            live = ids[row][ids[row] >= 0]
            assert len(live) == min(length, 128)
            assert len(live) == len(np.unique(live))
            assert np.all(live < length)
            assert np.all(ids[row, len(live) :] == -1)
        if return_scores:
            scores = np.asarray(result[1]).view(np.float32)
            np.testing.assert_array_equal(scores[ids >= 0], 0)
            np.testing.assert_array_equal(scores[ids < 0], -np.inf)


@pytest.mark.parametrize("bounded", [False, True])
def test_sparsecore_backend_forwards_bounded_copy(monkeypatch, bounded):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import topk_backend

    seen = {}

    def capture(score_bits, k, **kwargs):
        seen.update(kwargs)
        return jnp.zeros((score_bits.shape[0], k), jnp.int32)

    monkeypatch.setattr(topk_backend, "sparsecore_topk", capture)
    SparseCoreTopK(bounded_copy=bounded).select(
        jnp.zeros((4, 128), jnp.int32),
        16,
        jnp.full((4,), 128, jnp.int32),
        scheduling_group_id=None,
        return_scores=False,
    )
    assert seen["bounded_copy"] is bounded
    assert SparseCoreTopK() == SparseCoreTopK(bounded_copy=False)
    assert hash(SparseCoreTopK(bounded_copy=True)) != hash(SparseCoreTopK())


def test_default_topk_backend_routing_and_env_overrides(monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import topk_backend
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        TensorCoreTopK,
        default_topk_backend,
    )

    monkeypatch.setattr(topk_backend, "has_usable_sparse_core", lambda: False)
    monkeypatch.delenv("VLLM_TPU_TOPK_BACKEND", raising=False)
    assert default_topk_backend() == TensorCoreTopK()

    monkeypatch.setattr(topk_backend, "has_usable_sparse_core", lambda: True)
    monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "auto")
    assert default_topk_backend() == SparseCoreTopK()

    monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "tensorcore")
    assert default_topk_backend() == TensorCoreTopK()

    monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "sparsecore")
    monkeypatch.setattr(topk_backend, "has_usable_sparse_core", lambda: False)
    assert default_topk_backend() == SparseCoreTopK()

    monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "invalid_backend")
    with pytest.raises(ValueError, match="Invalid VLLM_TPU_TOPK_BACKEND"):
        default_topk_backend()


@pytest.mark.parametrize("return_scores", [False, True])
def test_tensorcore_topk_tie_certified_subbatch_matches_reference(return_scores):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        TensorCoreTopK,
    )

    rng = np.random.default_rng(123)
    # 64 rows (> 32 sub_batch) with an empty second sub-tile exercises
    # VMEM sub-batching and both certified fast-path and fallback tournament.
    scores = rng.integers(-3, 4, (64, 8192)).astype(np.float32)
    scores[:, 13] = -np.inf
    scores[0, :16] = 0.0
    scores[0, 16:32] = -0.0
    bits = jnp.asarray(scores.view(np.int32))
    lengths = jnp.asarray(
        [0, 7, 512, 1025, 4096, 8192] * 5 + [19, 256] + [0] * 32,
        dtype=jnp.int32,
    )
    expected = ReferenceTopK().select(
        bits,
        128,
        lengths,
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    for backend in (
        TensorCoreTopK(enable_prefix_bucketing=False),
        TensorCoreTopK(enable_prefix_bucketing=True),
    ):
        actual = backend.select(
            bits,
            128,
            lengths,
            scheduling_group_id=None,
            return_scores=return_scores,
        )
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_streamindex_topk_resolves_default_backend_outside_jit(monkeypatch):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        streamindex_topk,
    )

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    shape = jax.ShapeDtypeStruct
    args = (
        shape((8, 4, 128), jnp.bfloat16),
        shape((8, 4), jnp.float32),
        shape((4, 32, 4, 256), jnp.uint8),
        shape((8,), jnp.int32),
        shape((64,), jnp.int32),
        shape((9,), jnp.int32),
        shape((3,), jnp.int32),
    )
    kw = dict(
        k=128,
        compression_ratio=2,
        num_kv_pages_per_block=8,
        num_queries_per_block=1,
    )
    mesh = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(mesh):
        monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "tensorcore")
        hlo = (
            streamindex_topk.trace(*args, **kw)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        assert "StreamIdxTC" in hlo
        # Subsequent call with topk_backend=None must re-read env (R-4.3)
        monkeypatch.setenv("VLLM_TPU_TOPK_BACKEND", "invalid_backend")
        with pytest.raises(ValueError, match="Invalid VLLM_TPU_TOPK_BACKEND"):
            streamindex_topk.trace(*args, **kw)
