# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end record addressing and FP8/RoPE ownership contracts."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.mla.cache_layout import packed_record_shape
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _csa_gather_pool,
    unpack_union_rope_words,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    PagedCacheRows,
    encode_fp8_kv,
    gather_cache_values,
    scatter_cache_values,
)


@pytest.mark.parametrize(
    "dim,dtype",
    [
        (64, torch.bfloat16),
        (128, torch.bfloat16),
        (512, torch.bfloat16),
        (1024, torch.float32),
    ],
)
@pytest.mark.parametrize("split", [False, True])
def test_paged_native_record_roundtrip(dim, dtype, split):
    _, width = packed_record_shape(dim, dtype.itemsize)
    raw = torch.full((32 * width // 128, 128), 0x12345678, dtype=torch.int32)
    owner = tuple(raw.chunk(2)) if split else raw
    # One page of this layer, one page of another layer in each block stride.
    cache = PagedCacheRows(owner, ((4, 4, dim), dtype, 8 * width * 4, 4 * width * 4))
    slots = torch.tensor([0, 3, -1, 8, 15])
    values = torch.randn(5, dim, generator=torch.Generator().manual_seed(dim)).to(dtype)
    cache.write(slots, values)
    valid = slots >= 0
    got = cache[slots[valid] // 4, slots[valid] % 4]
    torch.testing.assert_close(got, values[valid], rtol=0, atol=0)
    physical = cache.row_indices(slots[valid])
    jax_pool = jnp.asarray(raw.numpy())
    jax_values = gather_cache_values(
        jax_pool, jnp.asarray(physical.numpy()), dim, dtype == torch.float32
    )
    np.testing.assert_array_equal(
        np.asarray(jax_values, np.float32), values[valid].float().numpy()
    )
    untouched = sorted(set(range(32)) - set(physical.tolist()))
    assert torch.all(raw.view(32, width)[untouched] == 0x12345678)


@pytest.mark.parametrize("split", [False, True])
def test_combined_fp8_records_gather_nope_and_rope(split):
    count, topk = 1024, 128
    latent = jnp.asarray(
        np.random.default_rng(5).normal(size=(count, 512)), jnp.bfloat16
    )
    nope, rope = encode_fp8_kv(latent)
    assert rope.shape == (count, 128)
    words = jnp.concatenate((nope, rope), -1)
    pool = scatter_cache_values(
        jnp.zeros((count * 2, 128), jnp.int32), jnp.arange(count), words
    )
    pages = pool.reshape(16, 128, 128)  # 64 logical records per physical page
    banks = (pages[:8], pages[8:]) if split else pages
    ids = jnp.asarray(np.random.default_rng(1).permutation(count), jnp.int32)
    got_nope, got_rope = _csa_gather_pool(banks, banks, ids, top_k=topk)
    np.testing.assert_array_equal(got_nope, nope[ids])
    unpacked = unpack_union_rope_words(got_rope, topk)
    expected = csa_cache_layout.rope_words(latent[ids, 448:])
    np.testing.assert_array_equal(unpacked, expected)


def test_fp8_cpu_writer_owns_only_its_combined_records():
    raw = torch.zeros((32, 128), dtype=torch.int32)
    cache = PagedCacheRows(raw, ((2, 4, 512), torch.uint8, 8192, 4096), raw)
    values = torch.randn(3, 512, generator=torch.Generator().manual_seed(8)).bfloat16()
    slots = torch.tensor([0, -1, 7])
    cache.write(slots, values)
    nope, rope = encode_fp8_kv(jnp.asarray(values.float().numpy(), jnp.bfloat16))
    records = cache.row_indices(slots[[0, 2]])
    expected = np.concatenate((np.asarray(nope), np.asarray(rope)), -1)[[0, 2]]
    np.testing.assert_array_equal(raw.view(16, 256)[records].numpy(), expected)
    assert not raw.view(16, 256)[[i for i in range(16) if i not in records]].any()


@pytest.mark.parametrize(
    "dim,dtype,record_bytes",
    [
        (128, torch.bfloat16, 512),
        (512, torch.bfloat16, 1024),
        (1024, torch.float32, 4096),
        (512, torch.uint8, 1024),
    ],
)
def test_allocator_budgets_native_records(dim, dtype, record_bytes):
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        materialize_upstream_kv_cache,
    )

    spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=dim,
        dtype=dtype,
        state_content_bytes=record_bytes,
    )
    page_bytes = spec.page_size_bytes
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * page_bytes,
                layers=["cache"],
                block_stride=page_bytes,
                layer_stride=4 * page_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["cache"], kv_cache_spec=spec)],
    )
    result = materialize_upstream_kv_cache(
        cfg, torch.device("cpu"), KVCacheLayout.BLHNC, word_rows=True
    )
    assert sum(t.nbytes for t in result.raw_tensors) == 16 * record_bytes
    cache = PagedCacheRows(
        result.native_raw_cache["cache"],
        result.native_cache_layouts["cache"],
        result.rope_pools.get("cache"),
    )
    assert cache.shape == (4, 4, dim)
    assert cache.row_bytes == record_bytes
    assert cache.page_stride == 4
    cache.write(
        torch.tensor([0, 15]),
        torch.ones(2, dim, dtype=torch.bfloat16 if dtype == torch.uint8 else dtype),
    )
    if dtype != torch.uint8:
        torch.testing.assert_close(
            cache[torch.tensor([0, 3]), torch.tensor([0, 3])],
            torch.ones(2, dim, dtype=dtype),
            rtol=0,
            atol=0,
        )


def test_native_binding_keeps_engram_history_until_rebind():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import NativeCacheView
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import NgramHashState

    class Base:
        def bind_kv_cache(self, cache):
            self._bound_cache = cache

    class Binding(NativeCacheView, Base):
        pass

    binding = Binding()
    raw = torch.zeros(32, 128, dtype=torch.int32)
    layout = ((4, 4, 512), torch.bfloat16, 4096, 0)
    binding.bind_kv_cache(torch.empty(0), raw_cache=raw, layout=layout)
    state = object.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.swa_cache_module = binding
    state.use_slot_cache = True
    state.block_size = 4
    state._kv_cache_ref = None
    state._cache = None
    assert state.ensure_cache()
    state._cache[3] = 123
    assert state.ensure_cache()
    assert state._cache[3] == 123
    binding.bind_kv_cache(torch.empty(0), raw_cache=raw.clone(), layout=layout)
    assert state.ensure_cache()
    assert state._cache[3] == 0
