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
"""Native pool allocation, prefix alignment and state ownership contracts."""

import math
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


def pool_config(tp=1, kv="fp8_e4m3", ssm="bfloat16"):
    cache = CacheConfig(
        block_size=128,
        cache_dtype=kv,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype=ssm,
    )
    cache.kv_cache_layout = "BLHNC"
    return NS(
        additional_config={},
        attention_config=AttentionConfig(),
        cache_config=cache,
        model_config=NS(
            hf_text_config=q4exp_text_config(), max_model_len=9216, dtype=torch.bfloat16
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_seqs=8),
        speculative_config=None,
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("mode", ["align", "none"])
def test_pool_initialize_builds_mamba_plan_from_materialized_layers(monkeypatch, mode):
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.kv_cache_materializer import MaterializedKVCache
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_cache
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    raw = torch.zeros((4, 8, 8), dtype=torch.bfloat16)
    spec = MambaSpec(
        shapes=((3, 4), (2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=128,
        page_size_padded=256,
    )
    groups = [
        NS(kv_cache_spec=spec, layer_names=["gdn", "gdn.alias"]),
        NS(kv_cache_spec=spec, layer_names=["ple"]),
    ]
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    groups = [KVCacheGroupSpec(g.layer_names, g.kv_cache_spec) for g in groups]
    config = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=groups)
    runner = NS(
        device=torch.device("cpu"),
        vllm_config=NS(),
        cache_config=NS(mamba_cache_mode=mode),
        enforce_eager=True,
        _pool_block_split=1,
        may_reinitialize_input_batch=lambda *args: None,
        _precompile_mamba_state_seed_copies=lambda: None,
    )
    runner._build_mamba_copy_plan = TPUModelRunner._build_mamba_copy_plan.__get__(
        runner
    )
    cache = object.__new__(pooled_cache.Qwen4ExpPooledCache)
    cache.runner = runner
    cache.groups = groups
    cache.native_config = config
    cache.layout = NS(main_page_bytes=256)
    cache.kernel_block_sizes = [128, 128]
    cache.attn_groups = [
        [
            NS(
                kv_cache_spec=g.kv_cache_spec,
                layer_names=g.layer_names,
                metadata_builders=[],
            )
        ]
        for g in groups
    ]
    cache.materialized = MaterializedKVCache(
        kv_caches={},
        raw_tensors=[raw],
        layer_to_raw={"gdn": raw, "gdn.alias": raw, "ple": raw},
    )
    cache.allocate = lambda: None
    monkeypatch.setattr(pooled_cache, "AttentionMetadataBuilder", lambda *a, **kw: None)

    cache.initialize(config)

    assert runner.bm_pool_layout is None
    assert runner.kv_cache_raw_tensors[0] is raw
    if mode == "align":
        assert runner._mamba_state_block_size == 128
        assert runner._pool_block_split == 2
        assert [gid for gid, tensors in runner._mamba_copy_plan] == [0, 1]
        assert all(
            len(tensors) == 1 and tensors[0] is raw
            for _, tensors in runner._mamba_copy_plan
        )
    else:
        assert runner._mamba_copy_plan == []
        assert runner._mamba_state_block_size is None


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("kv", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("ssm", ["bfloat16", "float32"])
@pytest.mark.parametrize("prefix", [False, True])
def test_native_pool_preserves_prefix_alignment_and_qsa_owner_aliases(
    tp, kv, ssm, prefix
):
    from vllm.v1.kv_cache_interface import MambaSpec

    cfg = pool_config(tp, kv, ssm)
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = (
        layout.block_size if prefix else cfg.model_config.max_model_len
    )
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(cfg, groups, 1024**3)
    placements = layer_to_pool_index(native)
    scheduler = generate_scheduler_kv_cache_config([native])
    scheduler_block, _ = resolve_kv_cache_block_sizes(scheduler, cfg)
    assert scheduler_block == (
        layout.block_size
        if prefix
        else math.lcm(layout.block_size, cfg.model_config.max_model_len)
    )
    assert layout.page_bytes >= max(layout.gdn_state_bytes, layout.ple_state_bytes)
    assert layout.block_size % 128 == 0
    assert {s.page_size_bytes for s in specs.values()} == {layout.page_bytes}
    for spec in specs.values():
        if isinstance(spec, MambaSpec):
            assert spec.mamba_cache_mode == cfg.cache_config.mamba_cache_mode
            assert spec.block_size == cfg.cache_config.mamba_block_size
    for name in specs:
        if name.endswith(".self_attn"):
            assert placements[name] == placements[name + ".raw_ring"]
            assert name + ".compressed_k" not in specs
            assert specs[name].block_size == layout.block_size
            assert (
                layout.index_page_bytes
                >= layout.block_size
                // cfg.model_config.hf_text_config.indexer_compress_ratio
                * cfg.model_config.hf_text_config.indexer_head_dim
                * 2
            )


def test_ple_geometry_can_determine_the_pool_page():
    cfg = pool_config(tp=8)
    cfg.model_config.hf_text_config.ple_conv_kernel_size = 65
    layout, _ = pooled_cache_specs(cfg)
    assert layout.ple_state_bytes > layout.gdn_state_bytes
    assert layout.page_bytes >= layout.ple_state_bytes
    # One fewer 128-token interval would not fit. No literal GDN/head/page
    # constant may silently determine capacity when PLE is larger.
    assert layout.main_page_bytes - 128 * layout.token_bytes < layout.ple_state_bytes


def test_pool_warmup_tables_match_native_ring_padding():
    """The first request supplied 256 ring integers to an 8-integer warmup."""
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.block_table import MultiGroupBlockTable, SlotMappingMode

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    cfg = pool_config()
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = cfg.cache_config.mamba_block_size = layout.block_size
    runner = NS(vllm_config=cfg, max_model_len=9216, device=torch.device("cpu"))
    cache = Qwen4ExpPooledCache(runner, model=None)
    specs = [g.kv_cache_spec for g in cache.groups]
    native = MultiGroupBlockTable(
        max_num_reqs=8,
        max_num_batched_tokens=128,
        pin_memory=False,
        device=runner.device,
        block_sizes=[s.block_size for s in specs],
        max_num_blocks=[s.max_num_blocks_per_req(cfg, 9216) for s in specs],
        slot_mapping_modes=[
            SlotMappingMode.NONE
            if isinstance(s, MambaSpec)
            else SlotMappingMode.TOKEN_TO_KV_SLOT
            for s in specs
        ],
    )
    metadata = AttentionMetadata(
        input_positions=torch.arange(128, dtype=torch.int32, device="cpu"),
        query_start_loc=torch.tensor([0] + [128] * 8, dtype=torch.int32, device="cpu"),
        seq_lens=torch.tensor([128] + [0] * 7, dtype=torch.int32, device="cpu"),
    )
    warmup = cache.dummy_metadata(metadata, active_rows=1)
    for gid, group in enumerate(cache.groups):
        item = warmup[group.layer_names[0]]
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert item.mamba_state_indices.shape == (8,)
        else:
            assert (
                item.block_tables.shape == native[gid].get_cpu_tensor().flatten().shape
            )
        if group.layer_names[0].endswith(".raw_ring"):
            table = item.block_tables.reshape(8, -1)
            assert table[0, 0] > 0
            assert torch.count_nonzero(table) == 1


@pytest.mark.parametrize("symbolic", [False, True])
def test_pooled_ops_register_and_dispatch_without_allocating_a_device(symbolic):
    """Direct JAX tests miss schema/fake dispatch failures during model boot."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
        build_pooled_short_conv_op,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
        build_compact_pooled_qsa_op,
    )

    qgate_cpu = torch.empty((8, 12288), dtype=torch.bfloat16, device="cpu")
    with FakeTensorMode(shape_env=ShapeEnv()) as mode:

        def bf(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device="cpu")

        def ids(*shape):
            return torch.empty(shape, dtype=torch.int32, device="cpu")

        pool = torch.empty(
            (12, 1664, 1, 4, 256), dtype=torch.float8_e4m3fn, device="cpu"
        )
        output = build_pooled_short_conv_op(3, 9)(
            bf(8, 10240),
            bf(10240, 4),
            pool,
            ids(3),
            ids(2),
            torch.empty((2,), dtype=torch.bool, device="cpu"),
        )
        assert output.shape == (8, 10240)
        qgate = mode.from_tensor(qgate_cpu, static_shapes=not symbolic)
        operation = build_compact_pooled_qsa_op()
        index = bf(12, 208, 1, 2, 128)
        output, selected = operation(
            qgate,
            bf(8, 512),
            bf(8, 512),
            bf(8, 640),
            bf(256),
            bf(256),
            bf(128),
            bf(128),
            ids(8),
            bf(9216, 128),
            pool,
            index,
            ids(3),
            ids(2),
            ids(2),
            ids(2, 6),
            ids(2, 6),
            compress_ratio=4,
            token_topk=2048,
            norm_eps=1e-6,
            rotary_dim=64,
        )
        if symbolic:
            # len(FakeTensor) concretizes the batch and breaks vLLM's host
            # split with an integer-only FX output during model compilation.
            assert isinstance(output.shape[0], torch.SymInt)
            assert isinstance(selected.shape[0], torch.SymInt)
        assert output.shape == (8, 6144)
        assert selected.shape == (8, 2051)
        assert selected.dtype == torch.int32
