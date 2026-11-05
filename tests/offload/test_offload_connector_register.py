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
"""Unit test for TPURaidenOffloadingConnector.register_kv_caches.

The override canonicalizes the TPU unified KV pool itself instead of
delegating to OffloadingConnectorWorker.register_kv_caches, whose per-layer
canonicalization assumes GPU layouts (since vLLM v0.26.1rc0 a mamba layer
must be one flat int8 byte-tensor) and crashes on the TPU runner's hybrid
registration, where a mamba layer is a list of state tensors that are
strided views into shared pool tensors.

Runs on CPU tensors: the canonicalization is pure storage arithmetic.
"""

from types import SimpleNamespace

import torch

from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

NUM_BLOCKS = 4


def _fake_connector(num_groups: int, offloaded_group_ids=None):
    """Connector stand-in with just the attributes the override touches."""
    captured: list = []
    if offloaded_group_ids is None:
        offloaded_group_ids = tuple(range(num_groups))
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=NUM_BLOCKS,
            kv_cache_groups=[object() for _ in range(num_groups)],
            hisparse_host_num_blocks=None,
            prefix_cacheable_group_ids=offloaded_group_ids,
        ),
        _init_worker=captured.append,
        spec=SimpleNamespace(pool_sources=[]),
    )
    return SimpleNamespace(connector_worker=worker), captured


def test_register_kv_caches_canonicalizes_unified_pool():
    # Two pool allocations of 512 bytes each: 4 scheduler blocks x 128-byte
    # pages. Layer entries mimic the TPU unified layout: attention layers are
    # tensor views sharing pool0's storage; the mamba layer is a LIST of two
    # strided state views into pool1 (the second at a non-zero offset).
    pool0 = torch.zeros(8, 16, dtype=torch.float32)
    pool1 = torch.zeros(16, 8, dtype=torch.float32)
    kv_caches = {
        "model.layers.0.attn": pool0.view(4, 2, 16),
        "model.layers.1.attn": pool0.view(8, 4, 4),
        "model.layers.2.mamba": [pool1[:, :4], pool1[:, 4:]],
    }

    fake, captured = _fake_connector(num_groups=2)
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)

    assert len(captured) == 1
    canonical = captured[0]

    # One canonical tensor per unique pool storage, in first-seen layer order.
    assert len(canonical.tensors) == 2
    for entry, pool in zip(canonical.tensors, (pool0, pool1)):
        assert entry.tensor.dtype == torch.int8
        assert entry.tensor.shape == (NUM_BLOCKS, 128)
        assert entry.page_size_bytes == 128
        assert (
            entry.tensor.untyped_storage().data_ptr()
            == pool.untyped_storage().data_ptr()
        )

    # The canonical view aliases the pool: bytes written through the pool
    # tensor are visible through the registered int8 view.
    pool1.view(torch.int8)[0, 0] = 42
    assert canonical.tensors[1].tensor[0, 0].item() == 42

    # One ref per canonical tensor for each KV cache group.
    assert len(canonical.group_data_refs) == 2
    for refs in canonical.group_data_refs:
        assert [(r.tensor_idx, r.page_size_bytes) for r in refs] == [(0, 128), (1, 128)]


def test_register_kv_caches_builds_refs_for_offloaded_groups_only():
    # vLLM pairs the refs with the offloaded groups only; a group that is not
    # prefix-cacheable (the DSv4 HCA compressor state) gets no ref list.
    pool = torch.zeros(8, 16, dtype=torch.float32)
    fake, captured = _fake_connector(num_groups=3, offloaded_group_ids=(0, 2))
    TPURaidenOffloadingConnector.register_kv_caches(fake, {"attn": pool})

    assert len(captured[0].group_data_refs) == 2


def test_register_kv_caches_ds_v4_alias_groups():
    """DSv4's overlay: SWA and compressor-state caches alias MLA arrays.

    What the Raiden worker consumes is `tensors`, so this pins the shape that
    matters: four layer entries -- one of them a (NoPE, RoPE) tuple, two of
    them aliases of arrays an earlier layer already bound -- collapse to three
    canonical arrays of three different page sizes, in first-seen order, each
    aliasing the model's own storage rather than copying it.
    """
    # uint8, [num_blocks, ...]: NoPE 256 B/page, RoPE 64 B/page,
    # indexer 128 B/page -- the shape family _initialize_ds_v4_kv_cache builds.
    nope = torch.zeros(NUM_BLOCKS, 8, 4, 8, dtype=torch.uint8)
    rope = torch.zeros(NUM_BLOCKS, 2, 4, 8, dtype=torch.uint8)
    indexer = torch.zeros(NUM_BLOCKS, 2, 4, 16, dtype=torch.uint8)

    kv_caches = {
        # A CSA layer binds a (NoPE, RoPE) pair under one block table.
        "csa.attn": (nope, rope),
        "csa.attn.indexer_k_cache": indexer,
        # The SWA cache overlays the CSA NoPE array.
        "swa.0.attn": nope,
        # The compressor state is written through its own KV layer's buffer.
        "csa.attn.compressor_state": nope,
    }

    # Groups: (CSA main + indexer), (SWA), (compressor state).
    fake, captured = _fake_connector(num_groups=3)
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)

    canonical = captured[0]

    # Aliases collapse: four layer entries, three distinct arrays.
    assert len(canonical.tensors) == 3
    assert [t.page_size_bytes for t in canonical.tensors] == [256, 64, 128]


def test_registration_follows_pools_rebound_in_place():
    """The connector hands the worker and its save fences the runner's own pool
    tensors, so registered views follow pools re-bound with `Tensor.set_` (see
    `view_on_current_storage`)."""
    from vllm_torchtpu.offload.raiden_store import view_on_current_storage

    pool = torch.zeros(8, 16, dtype=torch.float32)
    kv_caches = {
        "model.layers.0.attn": pool,
        "model.layers.1.mamba": [pool],
    }
    fake, captured = _fake_connector(num_groups=2)
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)

    assert fake._pool_sync_tensors == [pool]
    assert fake._pool_sync_tensors[0] is pool
    assert fake.connector_worker.spec.pool_sources[0] is pool
    registered = captured[0].tensors[0].tensor

    fresh = torch.ones(8, 16, dtype=torch.float32)
    pool.set_(fresh)
    assert registered.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()

    rebuilt = view_on_current_storage(registered, pool)
    assert rebuilt.shape == registered.shape
    assert rebuilt.dtype == registered.dtype
    assert rebuilt.untyped_storage().data_ptr() == fresh.untyped_storage().data_ptr()
    pool.view(torch.int8)[0, 0] = 7
    assert rebuilt[0, 0].item() == 7
    assert view_on_current_storage(registered, None) is registered


def _kimi_connector(rows: int, layers: int, pp_size: int = 1, architecture=None):
    """A Kimi-K3.1 connector stand-in: the spec sized for an index-key cache
    of ``layers`` MLA layers and ``rows`` rows per page."""
    from vllm_torchtpu.offload.raiden_store import KimiIndexGeometry

    fake, captured = _fake_connector(num_groups=2)
    fake._vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture or "KimiK31ForConditionalGeneration"
        )
    )
    fake.connector_worker.spec = SimpleNamespace(
        pool_sources=[],
        kimi_index=KimiIndexGeometry(rows=rows, layers=layers),
        pp_size=pp_size,
    )
    return fake, captured


def _kimi_caches(layers: int, rows: int):
    """Two unified pools, then the index-key cache under each MLA layer's
    index name, as the megakernel runner binds them."""
    pool0 = torch.zeros(NUM_BLOCKS, 32, dtype=torch.float32)
    pool1 = torch.zeros(NUM_BLOCKS, 32, dtype=torch.float32)
    index = torch.zeros(NUM_BLOCKS, layers, rows, 128, dtype=torch.uint8)
    kv_caches = {
        "model.layers.3.self_attn.mla_attn.attn": pool0,
        "model.layers.0.self_attn": [pool0],
        "model.layers.7.self_attn.mla_attn.attn": pool1,
        "model.layers.4.self_attn": [pool1],
    }
    for layer in (3, 7)[:layers]:
        kv_caches[f"model.layers.{layer}.self_attn.indexer.k_cache"] = index
    return kv_caches, (pool0, pool1, index)


def test_register_kv_caches_adds_kimi_index_array():
    # Kimi-K3.1: the index-key cache is one more canonical array of its own
    # row (every layer's page of a block), under the same block ids, and its
    # writes are fenced before a store like the pools'.
    kv_caches, (pool0, pool1, index) = _kimi_caches(layers=2, rows=3)
    fake, captured = _kimi_connector(rows=3, layers=2)
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)

    canonical = captured[0]
    assert [t.page_size_bytes for t in canonical.tensors] == [128, 128, 2 * 3 * 128]
    assert fake.connector_worker.spec.index_arrays == [2]
    entry = canonical.tensors[2]
    assert entry.tensor.shape == (NUM_BLOCKS, 2 * 3 * 128)
    assert (
        entry.tensor.untyped_storage().data_ptr() == index.untyped_storage().data_ptr()
    )
    index[1, 1, 2, 5] = 9  # block 1, layer slot 1, row 2, byte 5
    assert entry.tensor[1, (1 * 3 + 2) * 128 + 5].item() == 9
    assert any(t is index for t in fake._pool_sync_tensors)
    for refs in canonical.group_data_refs:
        assert [r.tensor_idx for r in refs] == [0, 1, 2]


def test_register_kv_caches_kimi_stage_holds_its_layers():
    # A prefill stage holds the index pages of its own MLA layers: the spec's
    # model-wide layer count over the pipeline stages.
    kv_caches, _ = _kimi_caches(layers=1, rows=4)
    fake, captured = _kimi_connector(rows=4, layers=4, pp_size=4)
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)
    assert captured[0].tensors[2].page_size_bytes == 4 * 128


def test_register_kv_caches_refuses_an_unsized_kimi_index():
    import pytest

    kv_caches, _ = _kimi_caches(layers=2, rows=3)
    # The spec sized the host pool for pages of 4 rows, the runner made 3.
    fake, _ = _kimi_connector(rows=4, layers=2)
    with pytest.raises(ValueError, match="index-key cache must be"):
        TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)
    fake, _ = _kimi_connector(rows=3, layers=2)
    fake.connector_worker.spec.kimi_index = None
    with pytest.raises(ValueError, match="sized by the spec"):
        TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)


def test_register_kv_caches_other_indexers_are_not_index_arrays():
    # GLM's indexer shares the cache name but is a layer of its own geometry.
    kv_caches, _ = _kimi_caches(layers=2, rows=3)
    fake, captured = _kimi_connector(
        rows=3, layers=2, architecture="GlmMoeDsaForCausalLM"
    )
    TPURaidenOffloadingConnector.register_kv_caches(fake, kv_caches)
    assert fake.connector_worker.spec.index_arrays == []
    assert len(captured[0].tensors) == 3
