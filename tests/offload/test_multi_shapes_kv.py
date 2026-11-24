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
"""Multi-shape KV layouts, decided and keyed from the recorded per-layer specs.

`has_multi_shapes_kv_caches` used to match the architecture against
DeepSeek-V4 and GLM-5.2. It now decides from the per-layer specs of all
workers recorded at the engine-core / worker hook, and refuses to answer
before they are recorded. These tests pin the decision and the geometry
`resolve_kernel_geometry` resolves on the worker and the flattened scheduler
configs of each layout, over KV cache configs planned as vLLM plans them.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from tests.offload.test_offloading_config_build_patch import (
    _kv_cache_config as _qwen_hybrid_kv_cache_config,
)
from tests.offload.test_offloading_config_build_patch import (
    _mixed_mla_kv_cache_config,
)
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.offload import raiden_store

NUM_BLOCKS = 4


def _config(groups: list[tuple[str, object]]) -> KVCacheConfig:
    """A worker's KV cache config as vLLM v0.33 plans it
    (`get_kv_cache_config_from_groups`): per group, one tensor for each
    distinct spec, each sized to the whole backing allocation and placed at
    its byte offset. vLLM needs a block-outermost layout only when two or
    more groups have different page sizes; TPU then sets BLHNC (DeepSeek-V4),
    and otherwise resolves the layer-outermost LBNHC (GLM-5.2, DeepSeek-V3.2,
    MiniMax M3)."""
    kv_cache_groups = [
        KVCacheGroupSpec(names.split(","), spec) for names, spec in groups
    ]
    page_sizes = {
        leaf.page_size_bytes
        for group in kv_cache_groups
        for leaf in raiden_store.iter_leaf_kv_cache_specs(group.kv_cache_spec)
    }
    block_outermost = len(kv_cache_groups) > 1 and len(page_sizes) > 1
    cache_config = CacheConfig(num_gpu_blocks_override=NUM_BLOCKS)
    cache_config.kv_cache_layout = "BLHNC" if block_outermost else "LBNHC"
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=cache_config,
    )
    return get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, 1 << 40)


def _uniform(block_size: int, leaves: dict[str, object]) -> tuple[str, object]:
    return ",".join(leaves), UniformTypeKVCacheSpecs(
        block_size=block_size, kv_cache_specs=leaves
    )


def _minimax_m3(*, int4: bool, layers: int = 3) -> KVCacheConfig:
    """MiniMax M3 at TP8 / DCP4: per layer a ``[K | V]`` cache of 4 KV heads,
    an index cache and, for 4-bit caches (profile agentx-int4), a scale
    cache; the DSpark drafter's K/V layers, promoted to full attention with
    the hybrid manager off, join the same group."""
    block, head_dim, kv_heads = 128, 128, 4
    dtype = torch.uint8 if int4 else torch.float8_e4m3fn
    kv_width = 2 * head_dim * kv_heads // (2 if int4 else 1)
    index_width = head_dim // 2 + 4 if int4 else head_dim
    leaves: dict[str, object] = {}
    for i in range(layers):
        prefix = f"model.layers.{i}.self_attn.attn"
        leaves[prefix] = MLAAttentionSpec(
            block_size=block, num_kv_heads=1, head_size=kv_width, dtype=dtype
        )
        leaves[f"{prefix}.index_cache"] = MLAAttentionSpec(
            block_size=block, num_kv_heads=1, head_size=index_width, dtype=dtype
        )
        if int4:
            leaves[f"{prefix}.scale_cache"] = MLAAttentionSpec(
                block_size=block, num_kv_heads=1, head_size=8, dtype=torch.float32
            )
    for i in range(2):
        # The runner gives each K/V leaf at least its backend page: one fp8
        # KV head's K and V (two rows) padded to four.
        leaves[f"draft_model.layers.{i}.self_attn.attn"] = FullAttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=torch.float8_e4m3fn,
            sliding_window=4096,
            page_size_padded=PallasAttentionBackend.get_kv_cache_page_size_bytes(
                block, 1, head_dim, torch.float8_e4m3fn
            ),
        )
    return _config([_uniform(block, leaves)])


def _deepseek_v4() -> KVCacheConfig:
    """DeepSeek-V4: compressed (CSA, HCA) and sliding-window MLA caches next
    to indexer keys, each kind in its own group."""
    block = 256
    csa = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=584,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    hca = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.uint8,
        tokens_per_state=128,
    )
    indexer = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    swa = SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.uint8,
        sliding_window=128,
    )
    return _config(
        [
            _uniform(block, {"l0.attn": csa, "l0.indexer": indexer, "l1.attn": hca}),
            ("l0.swa,l1.swa", swa),
        ]
    )


def _mla_only(*, draft: bool = False) -> KVCacheConfig:
    """DeepSeek-V3 / Kimi: one MLA latent shape (the MTP layer's too)."""
    mla = MLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    leaves: dict[str, object] = {"l0": mla, "l1": mla, "mtp": mla}
    if draft:
        # An EAGLE drafter's K/V next to the target's MLA caches.
        leaves["draft"] = FullAttentionSpec(
            block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
        )
        return _config([_uniform(256, leaves)])
    return _config([("l0,l1,mtp", mla)])


def _dense() -> KVCacheConfig:
    attn = FullAttentionSpec(
        block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    return _config([("l0,l1", attn)])


def _deepseek_v3_2() -> KVCacheConfig:
    """DeepSeek-V3.2: the MLA latent and the DSA indexer keys (fp8 keys plus
    scales, declared by `DeepseekV32IndexerCache` as an MLA spec) in one
    group. Main sent it down the single-shape path, which stops at an
    assertion; it now takes the per-array path."""
    block = 256
    return _config(
        [
            _uniform(
                block,
                {
                    "l0.attn": MLAAttentionSpec(
                        block_size=block,
                        num_kv_heads=1,
                        head_size=576,
                        dtype=torch.uint8,
                    ),
                    "l0.indexer": MLAAttentionSpec(
                        block_size=block,
                        num_kv_heads=1,
                        head_size=132,
                        dtype=torch.uint8,
                    ),
                },
            )
        ]
    )


_LAYOUTS = {
    "minimax_m3_int4": _minimax_m3(int4=True),
    "minimax_m3_fp8": _minimax_m3(int4=False),
    "deepseek_v4": _deepseek_v4(),
    "glm_5_2": _mixed_mla_kv_cache_config(num_blocks=NUM_BLOCKS),
    "deepseek_v3_2": _deepseek_v3_2(),
    "deepseek_v3": _mla_only(),
    "deepseek_v3_eagle": _mla_only(draft=True),
    "llama": _dense(),
    "qwen3_5_hybrid": _qwen_hybrid_kv_cache_config(),
}
_MULTI_SHAPES_LAYOUTS = (
    "minimax_m3_int4",
    "minimax_m3_fp8",
    "deepseek_v4",
    "glm_5_2",
    "deepseek_v3_2",
)

# `resolve_kernel_geometry` on each multi-shape layout, the same on the
# worker and the flattened scheduler configs: (kernel block, one
# (per-block shape, page bytes, KV cache layout) entry per array kind, KV
# dtypes, device block). A full-attention leaf (the MiniMax M3 DSpark
# drafter's K/V layers, one fp8 KV head of 128) has its backend page, K/V
# padded from two rows to four, and that padded page's bytes.
_LAYOUT = "LBNHC"
_DRAFTER_PAGE = 65536
_EXPECTED_GEOMETRY = {
    "minimax_m3_int4": (
        128,
        {
            ((128, 1, 512), 65536, _LAYOUT),
            ((128, 1, 68), 8704, _LAYOUT),
            ((128, 1, 8), 4096, _LAYOUT),
            ((128, 1, 4, 128), _DRAFTER_PAGE, _LAYOUT),
        },
        {torch.uint8, torch.float32, torch.float8_e4m3fn},
        128,
    ),
    "minimax_m3_fp8": (
        128,
        {
            ((128, 1, 1024), 131072, _LAYOUT),
            ((128, 1, 128), 16384, _LAYOUT),
            ((128, 1, 4, 128), _DRAFTER_PAGE, _LAYOUT),
        },
        {torch.float8_e4m3fn},
        128,
    ),
    "deepseek_v4": (
        256,
        {
            ((64, 1, 584), 37376, _LAYOUT),
            ((64, 1, 132), 8448, _LAYOUT),
            ((2, 1, 1024), 2048, _LAYOUT),
        },
        {torch.uint8},
        256,
    ),
    "glm_5_2": (
        256,
        {((256, 1, 576), 147456, _LAYOUT), ((256, 1, 256), 65536, _LAYOUT)},
        {torch.uint8},
        256,
    ),
    "deepseek_v3_2": (
        256,
        {((256, 1, 576), 147456, _LAYOUT), ((256, 1, 132), 33792, _LAYOUT)},
        {torch.uint8},
        256,
    ),
}


@pytest.fixture(autouse=True)
def _nothing_recorded():
    with patch.object(raiden_store, "_all_ranks_leaf_kv_cache_specs", None):
        yield


def test_vllm_planned_tensors_span_the_whole_allocation():
    # `_worker_kv_bytes_per_block` reads the first tensor's size, which vLLM
    # sets to the whole allocation: every leaf's page for one block, the
    # drafter's padded backend page included.
    worker = _LAYOUTS["minimax_m3_int4"]
    leaves = worker.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values()
    drafter_pages = [
        leaf.page_size_bytes for leaf in leaves if type(leaf) is FullAttentionSpec
    ]
    assert drafter_pages == [_DRAFTER_PAGE] * 2
    assert {t.size for t in worker.kv_cache_tensors} == {
        NUM_BLOCKS * sum(leaf.page_size_bytes for leaf in leaves)
    }
    assert raiden_store._worker_kv_bytes_per_block(worker) == sum(
        leaf.page_size_bytes for leaf in leaves
    )


def test_decision_needs_the_recorded_specs():
    with pytest.raises(RuntimeError, match="were not recorded"):
        raiden_store.has_multi_shapes_kv_caches()
    with pytest.raises(RuntimeError, match="were not recorded"):
        raiden_store.resolve_kernel_geometry(SimpleNamespace(), _LAYOUTS["glm_5_2"])


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_decision_from_the_recorded_specs(layout):
    # Recorded at the engine-core / worker hook from the per-worker configs,
    # so the scheduler, whose config keeps one spec per group, and the
    # workers decide alike.
    # deepseek_v3_eagle (one MLA geometry and a drafter K/V leaf) is not
    # multi-shape: the decision counts MLA-family geometries only.
    raiden_store.record_all_ranks_leaf_kv_cache_specs([_LAYOUTS[layout]] * 8)
    assert raiden_store.has_multi_shapes_kv_caches() is (
        layout in _MULTI_SHAPES_LAYOUTS
    )


def _vllm_config(kv_cache_layout: str = _LAYOUT):
    vllm_config = MagicMock()
    vllm_config.model_config.model = "org/model"
    vllm_config.model_config.revision = None
    vllm_config.model_config.quantization = None
    vllm_config.cache_config.prefix_caching_hash_algo = "sha256"
    vllm_config.cache_config.kv_cache_layout = kv_cache_layout
    return vllm_config


def _geometry_and_namespace(vllm_config, kv_cache_config):
    with (
        # The TPU full-attention backend, as the drafter's layers select it
        # in the engine core (no model).
        patch(
            "vllm_torchtpu.platforms.tpu_platform.TpuPlatform."
            "_find_dense_attention_backend",
            return_value=None,
        ),
        patch(
            "vllm.v1.attention.selector.get_attn_backend",
            return_value=PallasAttentionBackend,
        ),
        patch("vllm.config.set_current_vllm_config", return_value=nullcontext()),
    ):
        geometry = raiden_store.resolve_kernel_geometry(vllm_config, kv_cache_config)
    kernel_block_size, per_block_shape, kv_dtype, device_block_size = geometry
    namespace = raiden_store.derive_offload_namespace(
        vllm_config,
        kernel_block_size=kernel_block_size,
        per_block_shape=per_block_shape,
        kv_dtype=kv_dtype,
        device_block_size=device_block_size,
        world_size=8,
        num_kv_cache_groups=len(kv_cache_config.kv_cache_groups),
        num_kv_cache_tensors=len(kv_cache_config.kv_cache_tensors),
        offloaded_group_ids=tuple(range(len(kv_cache_config.kv_cache_groups))),
    )
    return geometry, namespace


@pytest.mark.parametrize("layout", _MULTI_SHAPES_LAYOUTS)
def test_geometry_on_worker_and_flattened_scheduler_config(layout):
    worker = _LAYOUTS[layout]
    scheduler = generate_scheduler_kv_cache_config([worker])
    raiden_store.record_all_ranks_leaf_kv_cache_specs([worker] * 8)
    vllm_config = _vllm_config()
    worker_geometry, worker_namespace = _geometry_and_namespace(vllm_config, worker)
    scheduler_geometry, scheduler_namespace = _geometry_and_namespace(
        vllm_config, scheduler
    )
    assert worker_geometry == scheduler_geometry == _EXPECTED_GEOMETRY[layout]
    # The tensor counts match too, so both views key storage alike.
    assert scheduler_namespace == worker_namespace


def _glm_namespace(*, indexer_page_size_padded=None, kv_cache_layout=_LAYOUT):
    """Scheduler namespace of a GLM-5.2 group (MLA latent + indexer)."""
    block = 256
    mla = MLAAttentionSpec(
        block_size=block, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )
    indexer = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        page_size_padded=indexer_page_size_padded,
    )
    worker = _config([_uniform(block, {"attn": mla, "indexer": indexer})])
    scheduler = generate_scheduler_kv_cache_config([worker])
    raiden_store.record_all_ranks_leaf_kv_cache_specs([worker])
    return _geometry_and_namespace(_vllm_config(kv_cache_layout), scheduler)[1]


def test_namespace_differs_when_a_non_representative_leaf_changes():
    # The scheduler keeps the latent leaf; the indexer differs only in the
    # recorded specs.
    assert _glm_namespace(indexer_page_size_padded=256 * 384) != _glm_namespace()


def test_namespace_differs_when_page_size_padded_changes():
    padded = _glm_namespace(indexer_page_size_padded=256 * 384)
    assert _glm_namespace(indexer_page_size_padded=256 * 512) != padded
    assert _glm_namespace(indexer_page_size_padded=256 * 384) == padded


def test_namespace_differs_when_kv_cache_layout_changes():
    assert _glm_namespace(kv_cache_layout="LBNHC") != _glm_namespace(
        kv_cache_layout="BLNHC"
    )
