# SPDX-License-Identifier: Apache-2.0
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


def _fake_connector(num_groups: int):
    """Connector stand-in with just the attributes the override touches."""
    captured: list = []
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=NUM_BLOCKS,
            kv_cache_groups=[object() for _ in range(num_groups)]),
        _init_worker=captured.append,
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
        assert entry.tensor.untyped_storage().data_ptr() == \
            pool.untyped_storage().data_ptr()

    # The canonical view aliases the pool: bytes written through the pool
    # tensor are visible through the registered int8 view.
    pool1.view(torch.int8)[0, 0] = 42
    assert canonical.tensors[1].tensor[0, 0].item() == 42

    # One ref per canonical tensor for each KV cache group.
    assert len(canonical.group_data_refs) == 2
    for refs in canonical.group_data_refs:
        assert [(r.tensor_idx, r.page_size_bytes) for r in refs] == \
            [(0, 128), (1, 128)]
