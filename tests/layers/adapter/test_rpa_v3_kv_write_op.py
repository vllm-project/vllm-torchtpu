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
"""The attention backend's write-only KV path.

`tests/kernels/test_rpa_v3_kv_write.py` checks the write itself against RPA
v3. These tests cover what the backend adds around it: which pools may take
the write, quantization as in forward(), and the torch bridge storing into
the caller's cache.
"""

from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kv_write import write_kv
from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv

NUM_KV_HEADS = 4
HEAD_SIZE = 128
PAGE_SIZE = 16
NUM_PAGES = 9
# Two 16-token verify windows, then 3 padding tokens.
KV_LENS = [40, 20]
NUM_TOKENS = 35
JAX_DTYPE = {torch.bfloat16: jnp.bfloat16, torch.float8_e4m3fn: jnp.float8_e4m3fn}


def _impl(**overrides):
    """A PallasAttentionBackendImpl carrying only what the write path reads."""
    impl = object.__new__(PallasAttentionBackendImpl)
    impl.__dict__.update(
        head_size=HEAD_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        kv_sharing_target_layer_name=None,
        _pool_is_seq_along_lane=False,
        _bm_kernel=False,
        _pcp_streaming=False,
        dcp_world_size=1,
        kv_cache_quantized_dtype=None,
    )
    impl.__dict__.update(overrides)
    return impl


def _metadata(device="cpu"):
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return AttentionMetadata(
        input_positions=None,
        block_tables=ints(list(range(1, NUM_PAGES))),
        seq_lens=ints(KV_LENS),
        query_start_loc=ints([0, 16, 32]),
        request_distribution=ints([0, 0, 2]),
    )


def _cache(dtype):
    shape = rpa_v3.get_kv_cache_shape(
        NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_SIZE, JAX_DTYPE[dtype]
    )
    return torch.randn(shape).to(dtype)


def test_supports_kv_cache_write_on_the_layer_major_v3_pool():
    assert _impl().supports_kv_cache_write()


@pytest.mark.parametrize(
    "override",
    [
        {"head_size": 64},
        {"kv_sharing_target_layer_name": "model.layers.0.self_attn.attn"},
        {"_pool_is_seq_along_lane": True},
        {"_bm_kernel": True},
        {"_pcp_streaming": True},
        {"dcp_world_size": 2},
    ],
    ids=lambda o: next(iter(o)),
)
def test_supports_kv_cache_write_rejects_other_layouts(override):
    assert not _impl(**override).supports_kv_cache_write()


def test_supports_kv_cache_write_rejects_the_bundled_pool(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True, raising=False)
    assert not _impl().supports_kv_cache_write()


def test_supports_kv_cache_write_rejects_the_batched_rpa_pool():
    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    impl.__dict__.update(_impl().__dict__)
    assert not impl.supports_kv_cache_write()


def test_write_kv_cache_quantizes_like_forward(monkeypatch):
    written = {}
    monkeypatch.setattr(
        attention,
        "write_new_kv_",
        lambda kv_cache, key, value, md: written.update(key=key, value=value, md=md),
    )
    impl = _impl(kv_cache_quantized_dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=2.0)
    key = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    value = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    md = _metadata()

    impl.write_kv_cache(
        layer, key.flatten(1), value.flatten(1), _cache(torch.float8_e4m3fn), md
    )

    q_key, q_value = quantize_kv(torch.float8_e4m3fn, key, value, 0.5, 2.0)
    assert q_value is not None
    assert torch.equal(written["key"].view(torch.uint8), q_key.view(torch.uint8))
    assert torch.equal(written["value"].view(torch.uint8), q_value.view(torch.uint8))
    assert written["md"] is md


def test_write_kv_cache_skips_the_memory_profiling_placeholder(monkeypatch):
    """Like forward(), the write is a no-op on the empty cache that memory
    profiling passes before the pools exist."""
    monkeypatch.setattr(
        attention,
        "write_new_kv_",
        mock.Mock(side_effect=AssertionError("must not write an empty cache")),
    )
    key = torch.zeros(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    empty_cache = torch.empty(0, dtype=torch.bfloat16)

    _impl().write_kv_cache(SimpleNamespace(), key, key, empty_cache, _metadata())


def test_write_kv_cache_rejects_a_dtype_mismatch():
    key = torch.zeros(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32)
    with pytest.raises(AssertionError, match="does not match the KV cache"):
        _impl().write_kv_cache(
            SimpleNamespace(), key, key, _cache(torch.bfloat16), _metadata()
        )


def _as_jax(tensor):
    if tensor.dtype == torch.int32:
        return jnp.asarray(tensor.numpy())
    bits = tensor.view(torch.uint8 if tensor.element_size() == 1 else torch.int16)
    return jnp.asarray(bits.numpy()).view(JAX_DTYPE[tensor.dtype])


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_write_new_kv_stores_into_the_callers_cache(dtype):
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.layers.adapter.rpa_v3_kv_write_op import write_new_kv_

    cache = _cache(dtype)
    key = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE).to(dtype)
    value = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE).to(dtype)
    md = _metadata()
    expected = write_kv(
        _as_jax(cache),
        _as_jax(key),
        _as_jax(value),
        _as_jax(md.seq_lens),
        _as_jax(md.block_tables),
        _as_jax(md.query_start_loc),
        _as_jax(md.request_distribution),
    )

    tpu_cache = cache.to("tpu")
    write_new_kv_(tpu_cache, key.to("tpu"), value.to("tpu"), _metadata("tpu"))
    np.testing.assert_array_equal(
        tpu_cache.cpu().view(torch.uint8).numpy(), np.asarray(expected).view(np.uint8)
    )


@pytest.mark.tpu_test
def test_write_new_kv_compiles_into_one_graph():
    """DFlash calls the write inside torch.compile(backend="tpu",
    fullgraph=True, dynamic=False); it must trace into that one graph."""
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.layers.adapter.rpa_v3_kv_write_op import write_new_kv_

    cache = _cache(torch.bfloat16)
    key = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE).to(torch.bfloat16)
    value = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE).to(torch.bfloat16)
    eager_cache = cache.to("tpu")
    write_new_kv_(eager_cache, key.to("tpu"), value.to("tpu"), _metadata("tpu"))

    compiled = torch.compile(
        write_new_kv_, backend="tpu", fullgraph=True, dynamic=False
    )
    compiled_cache = cache.to("tpu")
    compiled(compiled_cache, key.to("tpu"), value.to("tpu"), _metadata("tpu"))
    assert torch.equal(
        compiled_cache.cpu().view(torch.int16), eager_cache.cpu().view(torch.int16)
    )
