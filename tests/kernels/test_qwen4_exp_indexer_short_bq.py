# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_INDEXER_SHORT_BQ: the prefill indexer runs the leading
short (MTP verify) requests in their own mixed pass with a small query block."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.request_types import (
    decode_prefix,
    short_request_prefix,
)


@pytest.mark.parametrize(
    "lengths,count",
    [
        ([1, 1, 4, 4, 128, 0], 4),
        ([4, 4, 4], 3),
        ([1, 16, 17, 4], 2),
        ([128, 4], 0),
        ([0, 4], 0),
        ([4, 0, 4], 1),
        ([1, 1], 2),
    ],
)
def test_short_request_prefix(lengths, count):
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    assert int(short_request_prefix(starts)) == count
    assert int(short_request_prefix(starts)) >= int(decode_prefix(starts))


def _select(monkeypatch, short_bq, lengths):
    seen = {}

    def transport(q, weights, cache, lengths_, table, starts, distribution, **kw):
        seen.update(kw, distribution=np.asarray(distribution))
        return jnp.zeros((len(q), kw["k"]), jnp.int32)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", transport)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_INDEXER_SHORT_BQ", short_bq)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_INDEXER_BQ", 128)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    tokens = int(starts[-1])
    pooled_qsa.select_blocks(
        jnp.zeros((tokens, 4, 128), jnp.bfloat16),
        jnp.zeros((len(lengths), 128, 1, 4, 256), jnp.float8_e4m3fn),
        jnp.zeros(tokens, jnp.int32),
        jnp.zeros((len(lengths), 1), jnp.int32),
        jnp.zeros(tokens, jnp.int32),
        jnp.asarray(starts),
        jnp.full(len(lengths), 4096, jnp.int32),
        ratio=4,
        block_topk=512,
    )
    return seen


LENGTHS = [1, 1, 4, 4, 4, 98]  # 2 decodes, 3 verify requests, 1 prefill chunk


def test_off_keeps_one_mixed_pass(monkeypatch):
    seen = _select(monkeypatch, 0, LENGTHS)
    assert "short_seq_end" not in seen
    assert seen["num_queries_per_block"] == (1, 128, 128)
    np.testing.assert_array_equal(seen["distribution"], [2, 2, 6])


def test_on_splits_the_verify_requests(monkeypatch):
    seen = _select(monkeypatch, 32, LENGTHS)
    assert int(seen["short_seq_end"]) == 5
    assert seen["num_queries_per_block"] == (1, 32, 128)
    # Decode / mixed boundaries are unchanged.
    np.testing.assert_array_equal(seen["distribution"], [2, 2, 6])


def test_not_smaller_than_the_long_block_is_inert(monkeypatch):
    seen = _select(monkeypatch, 128, LENGTHS)
    assert "short_seq_end" not in seen
    assert seen["num_queries_per_block"] == (1, 128, 128)


@pytest.mark.parametrize("bq", [8, 16, 48])
def test_rejects_non_multiple_of_32(monkeypatch, bq):
    with pytest.raises(ValueError, match="multiple of 32"):
        _select(monkeypatch, bq, LENGTHS)


@pytest.mark.parametrize("chunk_tokens", [None, 512])
def test_short_pass_lowers_for_tpu7x_with_one_more_scoring_launch(chunk_tokens):
    """The split mixed pass traces and lowers for TPU7x (served tiles: BKV 1024,
    BQ 128, short BQ 32) and adds exactly one scoring kernel per chunk."""
    import functools
    import importlib

    import jax

    st = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk")
    from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing

    shape = jax.ShapeDtypeStruct
    tokens = 1024
    args = (
        shape((tokens, 4, 256), jnp.bfloat16),
        shape((tokens, 4), jnp.float32),
        shape((640, 16, 4, 256), jnp.uint8),
        shape((3,), jnp.int32),
        shape((960,), jnp.int32),
        shape((4,), jnp.int32),
        shape((3,), jnp.int32),
    )
    mesh = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )

    def lower(short):
        fn = functools.partial(
            st.streamindex_topk,
            k=512,
            compression_ratio=4,
            num_kv_pages_per_block=(32, 32, 32),
            num_queries_per_block=(1, 32 if short else 128, 128),
            chunk_tokens=chunk_tokens,
            components=qsa_indexing(3),
            enable_early_exit=True,
        )
        extra = (shape((), jnp.int32),) if short else ()
        body = (lambda *a: fn(*a[:7], short_seq_end=a[7])) if short else fn
        with jax.sharding.use_abstract_mesh(mesh):
            return jax.jit(body).trace(*args, *extra).lower(
                lowering_platforms=("tpu",)).as_text()

    base, split = lower(False), lower(True)
    chunks = 1 if chunk_tokens is None else tokens // chunk_tokens
    assert split.count("tpu_custom_call") > base.count("tpu_custom_call")
    # One extra scoring kernel per chunk; the 32-row query block appears.
    assert split.count("tpu_custom_call") - base.count("tpu_custom_call") == chunks
