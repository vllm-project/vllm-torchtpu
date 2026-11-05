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

With a DSpark drafter and the hybrid manager off, the drafter's
sliding-window K/V layers are promoted to full attention and join that same
group, so the multi-shape path also registers their fused K/V arrays.
"""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.models import minimax_m3
from vllm_torchtpu.offload import raiden_store
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_LAYERS = 3
NUM_BLOCKS = 4
DTYPE = torch.float8_e4m3fn
# DSpark drafter of MiniMax-M3.1-Flash (dspark/config.json): 5 sliding-window
# layers, 4 KV heads of 128 at window 4096; at TP4 each rank holds one head.
NUM_DRAFT_LAYERS = 2
DRAFT_KV_HEADS_PER_RANK = 1
DRAFT_WINDOW = 4096


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


def _draft_layer_specs() -> dict[str, SlidingWindowSpec]:
    """The specs the TPU runner declares for the drafter's `Attention` layers:
    sliding window, page padded to the fused K/V page the backend allocates."""
    page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, DRAFT_KV_HEADS_PER_RANK, HEAD_DIM, DTYPE
    )
    return {
        f"draft_model.layers.{i}.self_attn.attn": SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=DRAFT_KV_HEADS_PER_RANK,
            head_size=HEAD_DIM,
            dtype=DTYPE,
            page_size_padded=page,
            sliding_window=DRAFT_WINDOW,
        )
        for i in range(NUM_DRAFT_LAYERS)
    }


def _dspark_worker_kv_cache_config(*, draft_first: bool = False):
    """Target + drafter config through the runner's normalizer and vLLM's
    grouping with `--disable-hybrid-kv-cache-manager`."""
    target, draft = _layer_specs(), _draft_layer_specs()
    specs = {**draft, **target} if draft_first else {**target, **draft}
    specs = normalize_kv_cache_specs_for_tpu(
        specs, DTYPE, attention_backend=PallasAttentionBackend
    )
    stub = _StubVllmConfig()
    stub.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=True)
    groups = get_kv_cache_groups(stub, specs)
    page_bytes = sum(g.kv_cache_spec.page_size_bytes for g in groups)
    return get_kv_cache_config_from_groups(stub, groups, NUM_BLOCKS * page_bytes)


def _allocate_like_tpu_runner(worker_config) -> dict[str, torch.Tensor]:
    """One array per layer, leading dimension num_blocks: the MLA row layout
    for the owned target caches, the fused K/V layout for the drafter."""
    kv_caches = {}
    group_spec = worker_config.kv_cache_groups[0].kv_cache_spec
    for name in worker_config.kv_cache_groups[0].layer_names:
        spec = group_spec.kv_cache_specs[name]
        backend = (
            PallasMLAttentionBackend
            if isinstance(spec, MLAAttentionSpec)
            else PallasAttentionBackend
        )
        shape = backend.get_kv_cache_shape(
            NUM_BLOCKS, spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype
        )
        kv_caches[name] = torch.zeros(shape, dtype=spec.dtype)
    return kv_caches


def _register(worker_config, kv_caches):
    """`register_kv_caches` then the Raiden worker, up to the C++ handoff."""
    captured = []
    connector = SimpleNamespace(
        connector_worker=SimpleNamespace(
            kv_cache_config=worker_config, _init_worker=captured.append
        )
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)
    (canonical,) = captured
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
    return canonical, device_tensors


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

    canonical, device_tensors = _register(worker_config, kv_caches)
    assert len(canonical.tensors) == 2 * NUM_LAYERS
    assert sorted({t.page_size_bytes for t in canonical.tensors}) == [
        BLOCK_SIZE * HEAD_DIM * DTYPE.itemsize,
        BLOCK_SIZE * 2 * HEAD_DIM * DTYPE.itemsize,
    ]
    assert [t.shape[0] for t in device_tensors] == [NUM_BLOCKS] * (2 * NUM_LAYERS)
    assert [t.untyped_storage().data_ptr() for t in device_tensors] == [
        cache.untyped_storage().data_ptr() for cache in kv_caches.values()
    ]


def _draft_page_bytes() -> int:
    """Bytes of one drafter page as the TPU backend allocates it: K and V
    fused along the head axis, padded to the dtype packing."""
    shape = PallasAttentionBackend.get_kv_cache_shape(
        1, BLOCK_SIZE, DRAFT_KV_HEADS_PER_RANK, HEAD_DIM, DTYPE
    )
    return math.prod(shape) * DTYPE.itemsize


def test_dspark_draft_layers_join_the_target_group():
    worker_config = _dspark_worker_kv_cache_config()
    (group,) = worker_config.kv_cache_groups
    assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    leaves = group.kv_cache_spec.kv_cache_specs
    draft = {n: s for n, s in leaves.items() if n.startswith("draft_model.")}
    assert len(draft) == NUM_DRAFT_LAYERS
    for spec in draft.values():
        # Promoted from SlidingWindowSpec; not MLA, so the old path rejected it.
        assert type(spec) is FullAttentionSpec
        assert spec.page_size_bytes == _draft_page_bytes()


@pytest.mark.parametrize("draft_first", [False, True])
def test_dspark_geometry_agrees_between_workers_and_scheduler(draft_first):
    worker_config = _dspark_worker_kv_cache_config(draft_first=draft_first)
    # The scheduler keeps one leaf of the group: a target or a drafter one.
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    vllm_config = _vllm_config(minimax_m3._ARCHITECTURES[1])

    kernel, shapes, dtypes, device_block = raiden_store.resolve_kernel_geometry(
        vllm_config, worker_config
    )
    assert (kernel, device_block) == (BLOCK_SIZE, BLOCK_SIZE)
    assert shapes == {
        (BLOCK_SIZE, 1, 2 * HEAD_DIM),
        (BLOCK_SIZE, 1, HEAD_DIM),
        (BLOCK_SIZE, 2, DRAFT_KV_HEADS_PER_RANK, HEAD_DIM),
    }
    assert dtypes == {DTYPE}

    kernel, shapes, _, device_block = raiden_store.resolve_kernel_geometry(
        vllm_config, scheduler_config
    )
    assert (kernel, device_block) == (BLOCK_SIZE, BLOCK_SIZE)
    assert raiden_store.is_multi_shapes_geometry(shapes)

    # Host capacity covers the drafter's padded K/V pages too, in both views.
    target_bytes = NUM_LAYERS * (2 * HEAD_DIM + HEAD_DIM) * DTYPE.itemsize
    expected = BLOCK_SIZE * target_bytes + NUM_DRAFT_LAYERS * _draft_page_bytes()
    for config in (worker_config, scheduler_config):
        assert raiden_store._worker_kv_bytes_per_block(config) == expected


def test_dspark_worker_registers_target_and_draft_arrays():
    worker_config = _dspark_worker_kv_cache_config()
    kv_caches = _allocate_like_tpu_runner(worker_config)
    canonical, device_tensors = _register(worker_config, kv_caches)

    leaves = worker_config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    num_arrays = 2 * NUM_LAYERS + NUM_DRAFT_LAYERS
    assert len(canonical.tensors) == num_arrays
    # Each registered array is one layer's cache in whole pages, sized as
    # vLLM budgeted it; together they are the bytes the host pool holds.
    assert sorted(t.page_size_bytes for t in canonical.tensors) == sorted(
        spec.page_size_bytes for spec in leaves.values()
    )
    assert sum(t.page_size_bytes for t in canonical.tensors) == (
        raiden_store._worker_kv_bytes_per_block(worker_config)
    )
    assert [tuple(t.shape) for t in device_tensors] == [
        (NUM_BLOCKS, leaves[name].page_size_bytes) for name in kv_caches
    ]
    assert [t.untyped_storage().data_ptr() for t in device_tensors] == [
        cache.untyped_storage().data_ptr() for cache in kv_caches.values()
    ]

    # A drafter page row covers the whole fused K/V block: a block written
    # through the cache is the block the byte view moves.
    name = next(n for n in kv_caches if n.startswith("draft_model."))
    index = list(kv_caches).index(name)
    kv_caches[name].view(torch.int8)[2].fill_(7)
    rows = device_tensors[index]
    assert bool((rows[2] == 7).all())
    assert not bool((rows[1] == 7).any()) and not bool((rows[3] == 7).any())


def test_sliding_window_leaf_needs_the_hybrid_manager_off():
    spec = next(iter(_draft_layer_specs().values()))
    with pytest.raises(AssertionError, match="disable-hybrid-kv-cache-manager"):
        raiden_store.multi_shapes_leaf_block_shape(spec)


def _offloading_spec(kv_cache_config, *, world_size: int = 4, extra=None):
    """The real connector spec over stub engine configs; only the geometry,
    host sizing and namespace inputs are real."""
    from unittest.mock import MagicMock

    raiden_store._all_ranks_kv_bytes_per_block = None
    bytes_per_block = raiden_store._worker_kv_bytes_per_block(kv_cache_config)
    config = SimpleNamespace(
        extra_config={
            "cpu_bytes_to_use": 100 * world_size * bytes_per_block,
            **(extra or {}),
        },
        enable_kv_cache_events=False,
        groups=[SimpleNamespace(group_id=0, tokens_per_block=BLOCK_SIZE)],
        cache=SimpleNamespace(tokens_per_hash=BLOCK_SIZE, blocks_per_chunk=1),
        parallel=SimpleNamespace(
            tp_size=world_size, pp_size=1, pcp_size=1, dcp_size=1, world_size=world_size
        ),
        worker_kv_bytes_per_block=bytes_per_block,
        engine_id="engine",
    )
    vllm_config = MagicMock()
    vllm_config.kv_events_config.enable_kv_cache_events = False
    vllm_config.model_config.is_hybrid = False
    vllm_config.model_config.architecture = minimax_m3._ARCHITECTURES[1]
    vllm_config.model_config.model = "MiniMaxAI/MiniMax-M3.1-Flash"
    vllm_config.model_config.revision = None
    vllm_config.model_config.quantization = "tpu-mxfp8"
    vllm_config.cache_config.prefix_caching_hash_algo = "sha256"
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_rank_local = 0
    vllm_config.parallel_config.data_parallel_index = 0
    return raiden_store.TPURaidenStoreOffloadingSpec(
        config, vllm_config, kv_cache_config
    )


def test_dspark_connector_spec_builds_in_scheduler_and_workers():
    worker_config = _dspark_worker_kv_cache_config()
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    specs = [_offloading_spec(cfg) for cfg in (worker_config, scheduler_config)]
    for spec in specs:
        assert spec.device_block_size_factor == 1
        assert spec.offload_logical_blocks_capacity == 100
        assert spec.kernel_physical_blocks_capacity == 100
    # Every restart of the same deployment lands in the same key namespace.
    assert _offloading_spec(scheduler_config).key_namespace == specs[1].key_namespace
    # A deployment without the drafter keeps different arrays per block, so it
    # must not share keys with this one.
    assert (
        _offloading_spec(
            generate_scheduler_kv_cache_config([_worker_kv_cache_config()])
        ).key_namespace
        != specs[1].key_namespace
    )


def test_draft_kv_layer_names_are_the_promoted_drafter_leaves():
    worker_config = _dspark_worker_kv_cache_config()
    names = raiden_store.draft_kv_layer_names(worker_config)
    assert names == set(_draft_layer_specs())
    # No drafter, nothing to skip.
    assert raiden_store.draft_kv_layer_names(_worker_kv_cache_config()) == set()
    assert raiden_store._worker_draft_kv_bytes_per_block(worker_config) == (
        NUM_DRAFT_LAYERS * _draft_page_bytes()
    )


def test_dspark_worker_skips_draft_arrays_with_skip_draft_kv():
    worker_config = _dspark_worker_kv_cache_config()
    kv_caches = _allocate_like_tpu_runner(worker_config)
    captured = []
    connector = SimpleNamespace(
        connector_worker=SimpleNamespace(
            kv_cache_config=worker_config, _init_worker=captured.append
        ),
        _skip_draft_kv=True,
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)
    (canonical,) = captured
    assert len(canonical.tensors) == 2 * NUM_LAYERS
    assert sum(t.page_size_bytes for t in canonical.tensors) == (
        raiden_store._worker_kv_bytes_per_block(worker_config)
        - raiden_store._worker_draft_kv_bytes_per_block(worker_config)
    )
    target_ptrs = {
        cache.untyped_storage().data_ptr()
        for name, cache in kv_caches.items()
        if not name.startswith("draft_model.")
    }
    assert {t.tensor.untyped_storage().data_ptr() for t in canonical.tensors} == (
        target_ptrs
    )


def test_skip_draft_kv_sizes_host_pool_without_the_drafter():
    worker_config = _dspark_worker_kv_cache_config()
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    world = 4
    full = raiden_store._worker_kv_bytes_per_block(worker_config)
    draft = raiden_store._worker_draft_kv_bytes_per_block(worker_config)
    for cfg in (worker_config, scheduler_config):
        base = _offloading_spec(cfg, world_size=world)
        # Record as the engine core / workers do before the spec is built.
        raiden_store.record_all_ranks_kv_bytes_per_block([worker_config] * world)
        spec = _offloading_spec_recorded(cfg, world, {"skip_draft_kv": True})
        assert base.skip_draft_kv is False and spec.skip_draft_kv is True
        assert spec.offload_logical_blocks_capacity == (
            100 * world * full // (world * (full - draft))
        )
        assert (
            spec.key_namespace
            != _offloading_spec_recorded(cfg, world, {}).key_namespace
        )


def test_skip_draft_kv_needs_a_drafter():
    worker_config = _worker_kv_cache_config()
    raiden_store.record_all_ranks_kv_bytes_per_block([worker_config] * 4)
    with pytest.raises(ValueError, match="no drafter"):
        _offloading_spec_recorded(worker_config, 4, {"skip_draft_kv": "true"})


def _offloading_spec_recorded(kv_cache_config, world_size, extra):
    """`_offloading_spec` without resetting the recorded all-ranks bytes."""
    saved = raiden_store._all_ranks_kv_bytes_per_block
    saved_draft = raiden_store._all_ranks_draft_kv_bytes_per_block
    orig = raiden_store.TPURaidenStoreOffloadingSpec

    def build(config, vllm_config, kv_cache_config):
        raiden_store._all_ranks_kv_bytes_per_block = saved
        raiden_store._all_ranks_draft_kv_bytes_per_block = saved_draft
        return orig(config, vllm_config, kv_cache_config)

    from unittest.mock import patch

    with patch.object(raiden_store, "TPURaidenStoreOffloadingSpec", build):
        spec = _offloading_spec(kv_cache_config, world_size=world_size, extra=extra)
    if extra.get("skip_draft_kv") not in (None, False, "false"):
        assert spec.vllm_config.speculative_config is not None
    return spec
