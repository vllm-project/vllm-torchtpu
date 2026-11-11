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

import contextlib
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    resolve_kv_cache_block_sizes,
)

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


def pool_config(tp=1, kv="fp8_e4m3", ssm="bfloat16"):
    from vllm.model_executor.models.interfaces import supports_kv_cache_grouping

    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM

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
            hf_text_config=q4exp_text_config(),
            max_model_len=9216,
            dtype=torch.bfloat16,
            supports_kv_cache_grouping=supports_kv_cache_grouping(Qwen4ExpForCausalLM),
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            data_parallel_size=1,
            enable_expert_parallel=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_seqs=8),
        speculative_config=None,
        num_prefill_lookahead_tokens=0,
        kv_transfer_config=None,
    )


def test_load_model_binds_target_then_draft_for_mtp(monkeypatch):
    """Check pool binding with an admitted ordinary MTP configuration."""
    from tests.models.vllm.test_qwen4_exp_parallel import config as model_config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.spec_decode import eagle3

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setattr(
        tpu_runner.torch, "device", Mock(return_value=torch.device("cpu"))
    )
    for name in (
        "TPU_ROPE_CACHE_TRUNCATE",
        "TPU_ROPE_CACHE_ROW_MAJOR",
        "TPU_MOE_HASH_TABLE_ROW_MAJOR",
        "TPU_GREEDY_TOP_TOKENS",
        "FUSED_MOE_DECODE_KERNEL",
        "TPU_MOE_SKIP_PADDED_TOKENS",
        "USE_MOE_TP_DECODE_KERNEL",
    ):
        monkeypatch.setattr(tpu_runner.envs, name, False)
    config = model_config()
    config.attention_config = AttentionConfig()
    config.cache_config = pool_config().cache_config
    config.model_config.architecture = "Qwen4ExpForCausalLM"
    config.model_config.supports_kv_cache_grouping = False
    config.num_prefill_lookahead_tokens = 1
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = NS(
        method="mtp",
        num_speculative_tokens=2,
        draft_tensor_parallel_size=1,
        draft_model_config=None,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: False,
    )
    validate_config(config)
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.get_language_model = lambda: target
    for model in (target, draft):
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(3, 2)
        model.model.embed_tokens.org_vocab_size = 3
        model.lm_head = torch.nn.Linear(2, 3, bias=False)
    draft.logits_processor = NS(_gather_logits=lambda logits: logits)
    runner = NS(
        vllm_config=config,
        model_config=config.model_config,
        parallel_config=config.parallel_config,
        cache_config=config.cache_config,
        device_config=NS(),
        load_config=NS(),
        lora_config=None,
        mesh=None,
        is_pooling_model=False,
        _is_async_drafter=True,
        _pcp_enabled=False,
        _mtp_enabled=True,
        _mtp_prefill_only=False,
        _pp_is_last=True,
        _relayout_tables=Mock(),
        _initialize_pallas_kernels=Mock(),
        _refresh_smem_request_caps=Mock(),
    )
    runner.kv_cache_manager = NS()
    proposer = eagle3.Eagle3Proposer(runner, config)
    runner.drafter = runner._async_drafter = proposer
    cache_during_load = []

    def load_draft(_target):
        cache_during_load.append(runner.model_adapter.cache)
        proposer.draft_model = draft

    monkeypatch.setattr(proposer, "_load_draft_model", load_draft)
    draft_name = (
        f"mtp.layers.{config.model_config.hf_text_config.num_hidden_layers}.self_attn"
    )
    layers = iter([{}, {draft_name: NS()}])
    monkeypatch.setattr(
        eagle3, "get_layers_from_vllm_config", lambda *args: next(layers)
    )
    monkeypatch.setattr(tpu_runner, "get_tpu_quantization_config", lambda cfg: None)
    monkeypatch.setattr(
        tpu_runner,
        "get_model_loader",
        lambda cfg: NS(load_model=lambda **kwargs: target),
    )
    monkeypatch.setattr(tpu_runner, "validate_online_fp8", lambda *args: None)
    monkeypatch.setattr(
        tpu_runner,
        "set_vllm_model_wrapper_context",
        lambda **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        tpu_runner, "set_current_vllm_config", lambda cfg: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache",
        lambda: None,
    )

    tpu_runner.TPUModelRunner.load_model(runner)

    (target_cache,) = cache_during_load
    assert target_cache.model is target
    assert target_cache.draft_model is None
    assert draft_name in target_cache.get_specs()
    assert runner.model_adapter.cache.draft_model is draft
    assert runner.model_adapter.cache.get_specs() == target_cache.get_specs()
    assert runner.model_adapter.cache.materialized is None
    assert proposer._draft_attn_layer_names == {draft_name}


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
        vllm_config=NS(speculative_config=None),
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
        kv_caches={"gdn": [raw], "gdn.alias": [raw], "ple": [raw]},
        raw_tensors=[raw],
        layer_to_raw={"gdn": raw, "gdn.alias": raw, "ple": raw},
    )
    cache.allocate = lambda: None
    monkeypatch.setattr(pooled_cache, "AttentionMetadataBuilder", lambda *a, **kw: None)

    transfer_caches = cache.initialize(config)

    assert transfer_caches is cache.materialized.kv_caches
    assert all(tensors[0] is raw for tensors in transfer_caches.values())
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
    runner = NS(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        max_model_len=9216,
        device=torch.device("cpu"),
    )
    cache = Qwen4ExpPooledCache(runner, model=None)
    cache.get_specs()
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


@pytest.mark.parametrize("k", [0, 2])
@pytest.mark.parametrize("symbolic", [False, True])
def test_pooled_ops_register_and_dispatch_without_allocating_a_device(symbolic, k):
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
            ids(2, k + 1) if k else None,
            ids(pool.shape[0]) if k else None,
            ids(3) if k else None,
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
            ids(3),
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


@pytest.mark.parametrize("k", [1, 2, 4, 5])
def test_mtp_prefix_cache_uses_draft_attention_for_block_drop(k):
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM

    cfg = pool_config()
    cfg.num_prefill_lookahead_tokens = 1
    cfg.model_config.architectures = [Qwen4ExpForCausalLM.__name__]
    cfg.model_config.registry = NS(
        resolve_model_cls=lambda *args: (Qwen4ExpForCausalLM, None)
    )
    cfg.speculative_config = NS(
        method="mtp",
        num_speculative_tokens=k,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: True,
    )
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = layout.block_size
    groups = get_global_kv_cache_groups(cfg, [specs])
    draft = f"mtp.layers.{cfg.model_config.hf_text_config.num_hidden_layers}.self_attn"
    assert {name for name, spec in specs.items() if spec.is_draft} == {draft}
    drop_groups = [group for group in groups if group.is_eagle_group]
    assert len(drop_groups) == 1
    assert draft in drop_groups[0].layer_names
    assert draft + ".raw_ring" not in drop_groups[0].layer_names
    assert draft + ".compressed_k" not in specs
    assert layout.index_rows >= layout.block_size // 4
    for spec in specs.values():
        if isinstance(spec, MambaSpec):
            assert spec.num_speculative_blocks == k
            assert spec.mamba_cache_mode == "align"
            assert spec.block_size == layout.block_size
