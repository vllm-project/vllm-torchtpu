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
"""TurboQuant sparse-MLA caches through the Raiden host tier's registration.

`TPURaidenOffloadingConnector.register_kv_caches` registers each cache
storage as an opaque int8 `(num_blocks, page_bytes)` view, so the host tier
moves raw page bytes whatever the KV dtype. This pins, for GLM-5.2's
TurboQuant (NoPE, RoPE) caches at block size 256:

* the registered page bytes are the ones the KV cache spec publishes
  (`get_sparse_kv_cache_page_size_bytes`), which size the host store;
* a block's bytes copied out through the registered views and back into
  zeroed caches restore every TurboQuant row of that block and nothing else.

CPU only: the registration is storage arithmetic; the native store is not run.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import turboquant_attention
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

NUM_BLOCKS = 4
BLOCK = 256
HEAD_SIZE = 576  # kv_lora_rank 512 + rope 64
INDEXER_PAGE_BYTES = 132 * BLOCK


@pytest.fixture
def turboquant_preset(monkeypatch):
    monkeypatch.setattr(turboquant_attention, "_active_preset", "turboquant_4bit_nc")
    return "turboquant_4bit_nc"


def _fake_connector():
    captured: list = []
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=NUM_BLOCKS,
            kv_cache_groups=[object()],
            hisparse_host_num_blocks=None,
            prefix_cacheable_group_ids=(0,),
        ),
        _init_worker=captured.append,
    )
    return SimpleNamespace(connector_worker=worker), captured


def _glm_caches(cache_dtype, seed=0):
    specs = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        NUM_BLOCKS, BLOCK, HEAD_SIZE, cache_dtype
    )
    gen = torch.Generator().manual_seed(seed)
    caches = {}
    for layer in range(2):
        nope, rope = (
            torch.randint(
                0, 256, (NUM_BLOCKS * _page_bytes(s),), generator=gen, dtype=torch.uint8
            )
            .view(s.torch_dtype)
            .view(s.shape)
            for s in specs
        )
        caches[f"model.layers.{layer}.self_attn.attn"] = (nope, rope)
        caches[f"model.layers.{layer}.self_attn.indexer.k_cache"] = torch.randint(
            0, 256, (NUM_BLOCKS, INDEXER_PAGE_BYTES), generator=gen, dtype=torch.uint8
        )
    return specs, caches


def _bytes_per_block(cache):
    """`[num_blocks, page_bytes]` uint8 view of a cache (any dtype)."""
    return cache.reshape(-1).view(torch.uint8).view(NUM_BLOCKS, -1)


def _page_bytes(spec):
    n = 1
    for d in spec.shape[1:]:
        n *= int(d)
    return n * spec.torch_dtype.itemsize


def test_turboquant_pages_register_at_the_spec_page_bytes(turboquant_preset):
    specs, caches = _glm_caches(turboquant_preset)
    fake, captured = _fake_connector()
    TPURaidenOffloadingConnector.register_kv_caches(fake, caches)
    (canonical,) = captured

    nope_bytes, rope_bytes = (_page_bytes(s) for s in specs)
    # 2 tokens per 512 B NoPE row (256 B codes each); 128 B per token RoPE row.
    assert (nope_bytes, rope_bytes) == (BLOCK * 256, BLOCK * 128)
    # Per layer: NoPE, RoPE, indexer, in first-seen order.
    assert [t.page_size_bytes for t in canonical.tensors] == [
        nope_bytes,
        rope_bytes,
        INDEXER_PAGE_BYTES,
    ] * 2
    # What the KV cache spec publishes, and so what sizes the host store.
    assert nope_bytes + rope_bytes == (
        PallasMLAttentionBackend.get_sparse_kv_cache_page_size_bytes(
            BLOCK, HEAD_SIZE, turboquant_preset
        )
    )
    for entry in canonical.tensors:
        assert entry.tensor.dtype == torch.int8
        assert entry.tensor.shape == (NUM_BLOCKS, entry.page_size_bytes)


def test_turboquant_pages_are_two_thirds_of_fp8(turboquant_preset):
    tq = PallasMLAttentionBackend.get_sparse_kv_cache_page_size_bytes(
        BLOCK, HEAD_SIZE, turboquant_preset
    )
    fp8 = PallasMLAttentionBackend.get_sparse_kv_cache_page_size_bytes(
        BLOCK, HEAD_SIZE, "fp8"
    )
    # 384 B vs 576 B per token for the MLA caches (indexer cache unchanged).
    assert (tq, fp8) == (BLOCK * 384, BLOCK * 576)


@pytest.mark.parametrize("block", [0, 2])
def test_a_block_round_trips_through_the_registered_views(turboquant_preset, block):
    """Copy one block out through the int8 views and back into zeroed caches:
    exactly that block's TurboQuant rows come back."""
    _, caches = _glm_caches(turboquant_preset, seed=3)
    fake, captured = _fake_connector()
    TPURaidenOffloadingConnector.register_kv_caches(fake, caches)
    host = [entry.tensor[block].clone() for entry in captured[0].tensors]

    restored = {
        name: tuple(torch.zeros_like(t) for t in value)
        if isinstance(value, tuple)
        else torch.zeros_like(value)
        for name, value in caches.items()
    }
    fake2, captured2 = _fake_connector()
    TPURaidenOffloadingConnector.register_kv_caches(fake2, restored)
    for entry, page in zip(captured2[0].tensors, host):
        entry.tensor[block].copy_(page)

    for name, value in caches.items():
        pairs = (
            zip(value, restored[name])
            if isinstance(value, tuple)
            else [(value, restored[name])]
        )
        for want, got in pairs:
            want_b, got_b = _bytes_per_block(want), _bytes_per_block(got)
            assert torch.equal(got_b[block], want_b[block])
            others = [b for b in range(NUM_BLOCKS) if b != block]
            assert not got_b[others].any()
