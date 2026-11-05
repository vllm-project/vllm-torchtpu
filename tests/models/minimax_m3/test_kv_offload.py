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
"""KV offload (TPURaidenOffloadingConnector) over the MiniMax M3 caches.

Each layer owns a ``[K | V]`` cache and an index cache whose rows differ in
width, so the worker KV cache config wraps both in one
``UniformTypeKVCacheSpecs`` group. The connector must take its multi-shape
path for that layout, in the workers and in the scheduler, which sees the
group flattened to one per-layer spec.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.models import minimax_m3
from vllm_torchtpu.offload import raiden_store
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_LAYERS = 3
NUM_BLOCKS = 4
DTYPE = torch.float8_e4m3fn


class _StubVllmConfig:
    """What `get_kv_cache_config_from_groups` reads, nothing more."""

    def __init__(self) -> None:
        self.cache_config = CacheConfig()
        self.cache_config.kv_cache_layout = "LBNHC"
        self.parallel_config = SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
        self.attention_config = SimpleNamespace(hisparse_config=None)
        self.model_config = None

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None


def _layer_specs() -> dict[str, MLAAttentionSpec]:
    """The two specs `MiniMaxM3Cache.get_kv_cache_spec` declares per layer."""
    specs = {}
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}.self_attn.attn"
        specs[prefix] = MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=2 * HEAD_DIM, dtype=DTYPE
        )
        specs[f"{prefix}.index_cache"] = MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=HEAD_DIM, dtype=DTYPE
        )
    return specs


def _worker_kv_cache_config():
    specs = _layer_specs()
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs))
    page_bytes = group.kv_cache_spec.page_size_bytes
    return get_kv_cache_config_from_groups(
        _StubVllmConfig(), [group], NUM_BLOCKS * page_bytes
    )


def _vllm_config(architecture: str):
    return SimpleNamespace(model_config=SimpleNamespace(architecture=architecture))


@pytest.fixture(scope="module", autouse=True)
def _registered():
    minimax_m3.register()


@pytest.mark.parametrize("architecture", minimax_m3._ARCHITECTURES)
def test_register_declares_multi_shapes_kv(architecture):
    assert raiden_store.has_multi_shapes_kv_caches(_vllm_config(architecture))
    assert not raiden_store.has_multi_shapes_kv_caches(_vllm_config("LlamaForCausalLM"))


def test_geometry_agrees_between_workers_and_scheduler():
    worker_config = _worker_kv_cache_config()
    assert worker_config.num_blocks == NUM_BLOCKS
    # Without the multi-shape path this group fails the full-attention check.
    assert isinstance(
        worker_config.kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    )
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    vllm_config = _vllm_config(minimax_m3._ARCHITECTURES[1])

    kernel, shapes, dtypes, device_block = raiden_store.resolve_kernel_geometry(
        vllm_config, worker_config
    )
    assert (kernel, device_block) == (BLOCK_SIZE, BLOCK_SIZE)
    assert raiden_store.is_multi_shapes_geometry(shapes)
    assert shapes == {(BLOCK_SIZE, 1, 2 * HEAD_DIM), (BLOCK_SIZE, 1, HEAD_DIM)}
    assert dtypes == {DTYPE}

    kernel, shapes, _, device_block = raiden_store.resolve_kernel_geometry(
        vllm_config, scheduler_config
    )
    assert (kernel, device_block) == (BLOCK_SIZE, BLOCK_SIZE)
    assert raiden_store.is_multi_shapes_geometry(shapes)

    # Host capacity is sized from every layer of both caches, in both views.
    row_bytes = NUM_LAYERS * (2 * HEAD_DIM + HEAD_DIM) * DTYPE.itemsize
    for config in (worker_config, scheduler_config):
        assert raiden_store._worker_kv_bytes_per_block(config) == (
            BLOCK_SIZE * row_bytes
        )


def test_worker_registers_each_cache_array():
    worker_config = _worker_kv_cache_config()
    # Allocate the caches as the TPU runner does for MLAAttentionSpec layers:
    # one array per layer, leading dimension num_blocks.
    kv_caches = {}
    for name in worker_config.kv_cache_groups[0].layer_names:
        spec = worker_config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[name]
        shape = PallasMLAttentionBackend.get_kv_cache_shape(
            NUM_BLOCKS, spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype
        )
        kv_caches[name] = torch.zeros(shape, dtype=spec.dtype)

    captured = []
    connector = SimpleNamespace(
        connector_worker=SimpleNamespace(
            kv_cache_config=worker_config, _init_worker=captured.append
        )
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)
    (canonical,) = captured
    assert len(canonical.tensors) == 2 * NUM_LAYERS
    assert sorted({t.page_size_bytes for t in canonical.tensors}) == [
        BLOCK_SIZE * HEAD_DIM * DTYPE.itemsize,
        BLOCK_SIZE * 2 * HEAD_DIM * DTYPE.itemsize,
    ]

    registered = []
    with patch.object(
        raiden_store.RaidenStoreOffloadingWorker,
        "_init_registration",
        lambda self, tensors, *args: registered.append(tensors),
    ):
        raiden_store.RaidenStoreOffloadingWorker(
            canonical,
            vllm_config=_vllm_config(minimax_m3._ARCHITECTURES[1]),
            kv_cache_config=worker_config,
            host_blocks_to_allocate=8,
            controller_address="127.0.0.1:1",
            rank=0,
        )
    (device_tensors,) = registered
    assert [t.shape[0] for t in device_tensors] == [NUM_BLOCKS] * (2 * NUM_LAYERS)
    assert [t.untyped_storage().data_ptr() for t in device_tensors] == [
        cache.untyped_storage().data_ptr() for cache in kv_caches.values()
    ]
