# SPDX-License-Identifier: Apache-2.0
"""Serving reuses the pipeline's gathered KV with one attention launch per layer."""

import functools
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels, csa2_adapter
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


@pytest.mark.cpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_bridge_accepts_normalized_static_keywords(fp8):
    """Pallas signature normalization supplies a partial's keyword defaults."""

    def quantize(values: jax.Array, multiplier: float, *, fp8: bool = False):
        values = values * multiplier
        return values.astype(jnp.float8_e4m3fn) if fp8 else values

    with jax.default_device(jax.devices("cpu")[0]):
        mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("x",))
        bridge = csa2_adapter._manual_bridge(
            functools.partial(quantize, fp8=fp8), mesh, 1, P()
        )
        values = jnp.arange(8, dtype=jnp.float32)
        normalized = inspect.signature(bridge).bind(values, 2.0)
        normalized.apply_defaults()
        result = jax.jit(bridge, static_argnums=(1,), static_argnames=("fp8",))(
            *normalized.args, **normalized.kwargs
        )
        assert result.dtype == (jnp.float8_e4m3fn if fp8 else jnp.float32)
        np.testing.assert_array_equal(result.astype(jnp.float32), values * 2)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [1, 513])
def test_chunk_cache_reuse_and_stage_invalidation(monkeypatch, tokens):
    gathered, consumed = [], []

    def gather(query, raw, high, scales, high_scales, selected, *args):
        gathered.append(selected)
        return (
            query * 2,
            raw.new_empty((tokens, 512, 128)),
            scales.new_empty((tokens, 512)),
            (selected >= 0).sum(1).int(),
        )

    def attend(query, words, scales, raw, high, starts, *args):
        consumed.append((words, scales, starts, args[-1]))
        return query * 2

    monkeypatch.setattr(
        csa2_adapter, "_ops", {"gather_attend": gather, "attend": attend}
    )
    raw = torch.zeros((128, 128), dtype=torch.int32)
    main = PagedCacheRows(
        raw, ((4, 32, 512), torch.uint8, 32 * 512, 0), scale=torch.ones(128)
    )
    swa = PagedCacheRows(
        raw, ((2, 32, 512), torch.bfloat16, 32 * 1024, 0), swa_words=True
    )
    q = torch.ones((tokens, 8, 512), dtype=torch.bfloat16)
    records = torch.zeros((tokens, 512), dtype=torch.int32)
    starts = torch.tensor([0, min(17, tokens - 1), tokens - 1], dtype=torch.int32)

    def attention(selected):
        return csa2_adapter.attention(
            q,
            swa,
            main,
            selected,
            None,
            None,
            starts,
            torch.arange(tokens),
            torch.zeros(8),
            torch.empty((tokens, 64)),
            torch.empty((2, 4)),
            torch.empty((2, 2)),
            2,
            512**-0.5,
            False,
        )

    with attention_kernels.gathered_kv_cache_scope():
        torch.testing.assert_close(attention(records), q * 2)
        cache = attention_kernels._active_gathered_kv
        assert cache is not None and cache.values is not None
        cached = cache.values
        torch.testing.assert_close(attention(records), q * 2)
        assert len(gathered) == 1
        assert len(consumed) == 1
        words, scales, reused_starts, bq = consumed[0]
        assert words is cached[0] and scales is cached[1]
        torch.testing.assert_close(reused_starts, starts)
        assert bq == (6 if tokens <= 384 else 16)
        attention(records.clone())
        assert len(gathered) == 2
    with attention_kernels.gathered_kv_cache_scope():
        attention(records)
        assert len(gathered) == 3
    assert attention_kernels._active_gathered_kv is None
