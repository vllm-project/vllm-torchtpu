# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing tests; numerical kernels are covered separately on TPU."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak


@pytest.mark.cpu_test
@pytest.mark.parametrize("with_mesh", [False, True])
def test_prefill_bridge_preserves_full_record_mask(monkeypatch, with_mesh):
    from jax.sharding import Mesh

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as context

    def wrapper_context():
        assert with_mesh
        return SimpleNamespace(mesh=Mesh(np.asarray(jax.devices()[:1]), ("test",)))

    def score(q, weights, pool, pages, limit, *, records, **kwargs):
        # Observe both inputs through the bridge, including a partial-page hole.
        return jnp.where(records >= 0, records, -1) + jnp.repeat(
            pages, kwargs["page_records"], axis=1
        )

    monkeypatch.setattr(context, "get_vllm_model_wrapper_context", wrapper_context)
    monkeypatch.setattr(ak.prefill_index_scores, "prefill_index_scores", score)
    records = jnp.arange(2048, dtype=jnp.int32)[None].at[0, 7].set(-1)
    actual = ak._prefill_score(
        jnp.zeros((128, 32, 128), jnp.bfloat16),
        jnp.ones((128, 32, 1), jnp.float32),
        (jnp.zeros((128, 128), jnp.int32), jnp.zeros((128, 128), jnp.int32)),
        records,
        jnp.array([2048], jnp.int32),
        page_records=128,
    )
    expected = records + jnp.repeat(records[:, ::128], 128, axis=1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows,live,mixed,mode,marker,width,length",
    [
        (128, 128, False, None, 2, 8192, 8191),
        (256, 252, False, None, 2, 8192, 8191),
        (1024, 1020, False, None, 2, 8192, 8191),
        (4096, 4092, False, None, 2, 8192, 8191),
        (256, 6, False, None, 1, 8192, 8191),
        (256, 252, True, None, 1, 8192, 8191),
        (288, 284, False, None, 1, 8192, 8191),
        (2048, 2044, False, None, 2, 131072, 50176),
        (2048, 2044, False, None, 2, 16384, 8193),
        (2048, 2044, False, None, 2, 65536, 65536),
        (2048, 2044, False, None, 2, 131072, 65537),
        (2048, 2044, False, None, 2, 131200, 131199),
        (2048, 2044, False, None, 2, 524288, 524288),
        (2048, 2044, False, None, 2, 1048576, 1048576),
        (2048, 2044, False, "streaming", 2, 131072, 50176),
        (2048, 2044, False, "baseline", 1, 131072, 50176),
        (384, 288, True, None, 1, 131072, 50176),
        (384, 288, True, "streaming", 1, 131072, 50176),
        (2048, 6, False, "streaming", 1, 131072, 50176),
        (2048, 2044, True, "streaming", 1, 131072, 50176),
    ],
)
def test_prefill_route_excludes_decode_and_mixed(
    monkeypatch, rows, live, mixed, mode, marker, width, length
):
    # Run graph plumbing on CPU with distinct scorer markers. This is not
    # SparseCore emulation or a numerical/performance qualification.
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    records = jnp.arange(width, dtype=jnp.int32)[None]
    records = jnp.concatenate((records, records + width), axis=0)
    q = jnp.zeros((rows, 32, 128), jnp.bfloat16)
    weights = jnp.ones((rows, 32), jnp.float32)
    lengths = jnp.where(jnp.arange(rows) < live, length, 0).astype(jnp.int32)
    requests = (jnp.arange(rows) % 2 if mixed else jnp.ones(rows)).astype(jnp.int32)
    candidates = jnp.zeros((rows, 0), jnp.int32)
    pool = jnp.zeros((128, 128), jnp.int32)

    def old_score(q, pool, records, weights, lengths, requests, page_records):
        return jnp.ones((q.shape[0], records.shape[1]), jnp.float32)

    def new_score(q, weights, pool, shared_records, limit, *, page_records):
        assert page_records == 128
        # The active request need not be row zero. Pass all records to preserve
        # per-record validity, rather than just the first address of each page.
        assert shared_records.shape[0] == 1
        np.testing.assert_array_equal(
            np.asarray(shared_records[0, :3]), np.arange(3) + width
        )
        assert weights.shape == (128, 32, 1)
        assert int(limit[0]) == length
        return jnp.full((q.shape[0], shared_records.shape[1]), 2, jnp.float32)

    def select(scores, lengths):
        return jnp.where(lengths[:, None] > 0, scores[:, :512].astype(jnp.int32), -1)

    monkeypatch.setattr(ak, "paged_index_scores_jax", old_score)
    monkeypatch.setattr(ak, "_prefill_score", new_score)
    monkeypatch.setattr(ak, "_prefill_top_k", lambda s, n, **kw: select(s, n))
    monkeypatch.setattr(
        ak,
        "select_indices_jax",
        lambda s, n, c, *args: (select(s, n), c),
    )
    with jax.disable_jit():
        selected, published = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            1,
            False,
            False,
            True,
            page_records=128,
            **({} if mode is None else {"prefill_mode": mode}),
        )
    np.testing.assert_array_equal(np.asarray(selected[:live]), marker)
    np.testing.assert_array_equal(np.asarray(selected[live:]), -1)
    assert published.shape == (rows, 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("length", [0, 1, 127, 511, 512])
def test_short_prefill_returns_all_valid_indices(monkeypatch, length):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    def unexpected(*args, **kwargs):
        raise AssertionError("All visible keys fit: no scoring or TopK required")

    monkeypatch.setattr(ak, "_prefill_score", unexpected)
    monkeypatch.setattr(ak, "paged_index_scores_jax", unexpected)
    records = jnp.arange(2048, dtype=jnp.int32)[None]
    records = records.at[0, 128:256].set(-1)
    lengths = jnp.full((2048,), length, jnp.int32).at[-4:].set(0)
    with jax.disable_jit():
        selected, _ = ak.paged_indexer_jax(
            jnp.zeros((2048, 32, 128), jnp.bfloat16),
            jnp.zeros((128, 128), jnp.int32),
            records,
            jnp.ones((2048, 32), jnp.float32),
            lengths,
            jnp.zeros((2048, 0), jnp.int32),
            jnp.zeros(2048, jnp.int32),
            512,
            1,
            False,
            False,
            True,
            page_records=128,
        )
    valid = np.arange(length)
    valid = valid[(valid < 128) | (valid >= 256)]
    expected = np.pad(valid, (0, 512 - len(valid)), constant_values=-1)
    np.testing.assert_array_equal(
        np.asarray(selected[:-4]), np.tile(expected, (2044, 1))
    )
    np.testing.assert_array_equal(np.asarray(selected[-4:]), -1)
