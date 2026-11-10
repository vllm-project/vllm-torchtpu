# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 v2 KV cache groups under the block-outer (BLHNC) layout.

SWA pages stay 64 tokens while the compressed caches use a 256-token block,
so upstream's packed grouping fits several SWA layers per group. The block
stride is rounded to whole SWA pages and pages are placed largest first, so
every page of every group starts on a multiple of its own size.
"""

import collections
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    replace_as,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_torchtpu import _order_pages_by_size
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    BLOCK_STRIDE_ALIGNMENT,
    SWA_PAGE_BYTES,
    SWA_PAGE_TOKENS,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache import (
    PackedFP8IndexerSpec,
)

pytestmark = pytest.mark.cpu_test

# DeepSeek-V4.1 Flash: 40 target layers plus 3 DSpark draft layers.
RATIOS = [0, 0] + [2] * 18 + [1] * 20
KV_SOURCES = (2, 8, 14, 20)
COMPRESSED_BLOCK = 256
BUDGET = 30299652096


def _specs():
    align = BLOCK_STRIDE_ALIGNMENT
    specs = {}
    for i in range(43):
        prefix = f"model.layers.{i}.self_attn"
        ratio = RATIOS[i] if i < 40 else 0
        if i in KV_SOURCES:
            specs[prefix] = MLAAttentionSpec(
                block_size=COMPRESSED_BLOCK,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                tokens_per_state=ratio,
                cache_dtype_str="fp8",
                alignment=512,
                model_version="deepseek_v4",
                block_stride_alignment=align,
            )
            specs[f"{prefix}.indexer.k_cache"] = replace_as(
                MLAAttentionSpec(
                    block_size=COMPRESSED_BLOCK,
                    num_kv_heads=1,
                    head_size=132,
                    dtype=torch.uint8,
                    tokens_per_state=ratio,
                    alignment=512,
                    model_version="deepseek_v4",
                    block_stride_alignment=align,
                ),
                PackedFP8IndexerSpec,
            )
            if ratio == 2:
                specs[f"{prefix}.compressor.state_cache"] = CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1024,
                    head_size_v=0,
                    dtype=torch.float32,
                    block_stride_alignment=align,
                )
        specs[f"{prefix}.swa_cache"] = SlidingWindowMLASpec(
            bounded_replay=True,
            block_size=SWA_PAGE_TOKENS,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
            cache_dtype_str="fp8",
            alignment=512,
            model_version="deepseek_v4",
            block_stride_alignment=align,
        )
    return specs


def _config():
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "BLHNC"
    cache_config.block_size = COMPRESSED_BLOCK
    cache_config.enable_prefix_caching = True
    cache_config.num_gpu_blocks_override = None
    config = MagicMock()
    config.cache_config = cache_config
    config.scheduler_config = SimpleNamespace(
        disable_hybrid_kv_cache_manager=False,
        max_num_seqs=48,
        max_num_batched_tokens=2048,
    )
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41"), max_model_len=524288
    )
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=5,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: True,
    )
    config.parallel_config = SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1
    )
    config.kv_transfer_config = None
    config.attention_config = SimpleNamespace(hisparse_config=None)
    return config


def _kind(name):
    return name.rsplit(".", 1)[-1]


def test_swa_layers_share_block_outer_groups():
    groups = get_kv_cache_groups(_config(), _specs())
    swa = [g for g in groups if {_kind(n) for n in g.layer_names} == {"swa_cache"}]
    assert len(groups) == 9
    assert sorted(len(g.layer_names) for g in swa) == [6] * 6 + [7]
    # Every SWA group shares one 64-token page geometry.
    assert {g.kv_cache_spec.block_size for g in swa} == {SWA_PAGE_TOKENS}


def test_pages_largest_first_are_aligned_to_their_own_size():
    config = _config()
    specs = _specs()
    groups = _order_pages_by_size(get_kv_cache_groups(config, specs))
    kv_cache_config = get_kv_cache_config_from_groups(config, groups, BUDGET)
    views = allocate_kv_cache(
        kv_cache_config, torch.device("meta"), KVCacheLayout.BLHNC
    )
    strides = set()
    offsets = collections.defaultdict(set)
    for name, view in views.items():
        item = view.element_size()
        stride = view.stride(0) * item
        offset = view.storage_offset() * item % stride
        # The spec page, padded for the indexer's 132-byte records.
        page = specs[name].page_size_bytes
        strides.add(stride)
        assert offset % page == 0, (name, offset, page)
        offsets[_kind(name), view.shape[2]].add(offset // 1024)
    # 480 KiB of main + indexer pages, rounded to whole SWA pages.
    assert strides == {8 * SWA_PAGE_BYTES}
    assert offsets["self_attn", 256] == {0}
    assert offsets["self_attn", 128] == {128, 192, 256}


def test_order_pages_by_size_keeps_layers_and_specs_paired():
    groups = get_kv_cache_groups(_config(), _specs())
    ordered = _order_pages_by_size(groups)
    assert [len(g.layer_names) for g in ordered] == [len(g.layer_names) for g in groups]
    for before, after in zip(groups, ordered, strict=True):
        assert sorted(before.layer_names) == sorted(after.layer_names)
        specs = getattr(after.kv_cache_spec, "kv_cache_specs", None)
        if specs is None:
            continue
        assert list(specs) == after.layer_names
        sizes = [spec.page_size_bytes for spec in specs.values()]
        assert sizes == sorted(sizes, reverse=True)


# Layer 8's block-outer geometry: an 8-row state ring at row 8 of a 128-row
# block, main records at row 384 of a 1024-row block.
LAYER_8_GEOMETRY = dict(
    state_capacity=8,
    state_stride=128,
    state_offset=8,
    window=128,
    main_capacity=128,
    main_stride=1024,
    main_offset=384,
    index_capacity=256,
)


def test_compressor_layout_compares_with_the_bound_layout():
    from vllm_torchtpu.kernels.experimental import decode_preparation as prep

    bound = prep.CompressorLayout(**LAYER_8_GEOMETRY)
    assert prep.compressor_layout_problem(bound, bound) is None
    # Against the default layout, the block-outer fields are reported.
    problem = prep.compressor_layout_problem(bound)
    assert problem is not None
    assert "main_offset=384" in problem and "state_stride=128" in problem


def test_compressor_admission_uses_the_layer_layout(monkeypatch):
    import dataclasses

    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.kernels.experimental import decode_preparation as prep
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import preprocess_ops

    monkeypatch.setattr(prep, "_mode", prep.FusionMode.AUTO)
    monkeypatch.setattr(prep, "_reported", set())
    fields = dataclasses.astuple(prep.CompressorLayout(**LAYER_8_GEOMETRY))
    assert preprocess_ops.compressor_layout_admitted(
        "compressor_addresses", fields, **LAYER_8_GEOMETRY
    )
    assert not preprocess_ops.compressor_layout_admitted(
        "compressor_addresses", None, **LAYER_8_GEOMETRY
    )
