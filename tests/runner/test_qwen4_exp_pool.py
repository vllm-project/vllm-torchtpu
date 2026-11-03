# SPDX-License-Identifier: Apache-2.0
"""Native pool allocation, prefix alignment and state ownership contracts."""

import math
from dataclasses import replace
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
    config = NS(kv_cache_groups=groups)
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
    cache.hbm_sizing = False
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
@pytest.mark.parametrize("pack", [False, True])
def test_native_pool_preserves_prefix_alignment_and_qsa_owner_aliases(
    monkeypatch, tp, kv, ssm, prefix, pack
):
    from vllm.v1.kv_cache_interface import MambaSpec

    monkeypatch.setenv("QWEN4_EXP_PACK_COMPRESSED_K", "1" if pack else "0")
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
            owner, _ = layout.compressed_owner(name)
            assert (
                placements[name] == placements[owner] == placements[name + ".raw_ring"]
            )
            assert specs[name].block_size == specs[owner].block_size


def test_ple_geometry_can_determine_the_pool_page():
    cfg = pool_config(tp=8)
    cfg.model_config.hf_text_config.ple_conv_kernel_size = 65
    layout, _ = pooled_cache_specs(cfg)
    assert layout.ple_state_bytes > layout.gdn_state_bytes
    assert layout.page_bytes >= layout.ple_state_bytes
    # One fewer 128-token interval would not fit. No literal GDN/head/page
    # constant may silently determine capacity when PLE is larger.
    assert layout.page_bytes - 128 * layout.token_bytes < layout.ple_state_bytes


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
        kernel_block_sizes=[s.block_size for s in specs],
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
    # Exactly the pages the profiling scratch pool is sized to.
    cache.pages = cache.dummy_pages([128], active_rows=1)
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
@pytest.mark.parametrize("slot", [0, 3])
def test_pooled_ops_register_and_dispatch_without_allocating_a_device(symbolic, slot):
    """Direct JAX tests miss schema/fake dispatch failures during model boot."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
        build_pooled_short_conv_op,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_pooled_qsa_op

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
        output, selected = build_pooled_qsa_op()(
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
            ids(3),
            ids(2),
            ids(2),
            ids(2, 6),
            ids(2, 6),
            compress_ratio=4,
            token_topk=2048,
            norm_eps=1e-6,
            rotary_dim=64,
            **({"compressed_slot": slot} if slot else {}),
        )
        if symbolic:
            # len(FakeTensor) concretizes the batch and breaks vLLM's host
            # split with an integer-only FX output during model compilation.
            assert isinstance(output.shape[0], torch.SymInt)
            assert isinstance(selected.shape[0], torch.SymInt)
        assert output.shape == (8, 6144)
        assert selected.shape == (8, 2051)
        assert selected.dtype == torch.int32


def packed_config(monkeypatch, pack, *, spec_tokens=3, tp=8):
    monkeypatch.setenv("QWEN4_EXP_PACK_COMPRESSED_K", "1" if pack else "0")
    cfg = pool_config(tp)
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1
    cfg.speculative_config = NS(
        num_speculative_tokens=spec_tokens,
        method="mtp",
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: True,
    )
    return cfg


def test_packed_compressed_keys_share_owner_pages_including_mtp(monkeypatch):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import (
        check_packed_compressed_groups,
    )

    unpacked_layout, unpacked = pooled_cache_specs(packed_config(monkeypatch, False))
    cfg = packed_config(monkeypatch, True)
    layout, specs = pooled_cache_specs(cfg)
    attention = [n for n in unpacked if n.endswith(".self_attn")]
    assert attention[-1].startswith("mtp.layers.")
    slots = layout.page_bytes // (
        layout.main_kernel_pages * layout.compressed_kernel_bytes
    )
    assert layout.compressed_slots == slots == 8
    assert unpacked_layout.compressed_slots == 1
    assert unpacked_layout.compressed_owners == ()
    # Consecutive layers fill a page's slots in order; the draft's included.
    for index, name in enumerate(attention):
        owner = attention[index // slots * slots] + ".compressed_k"
        assert layout.compressed_owner(name) == (owner, index % slots)
        assert unpacked_layout.compressed_owner(name) == (name + ".compressed_k", 0)
    owners = [n for n in specs if n.endswith(".compressed_k")]
    assert owners == [
        attention[i] + ".compressed_k" for i in range(0, len(attention), slots)
    ]
    # Only the members other than owners disappear; every other spec is equal.
    assert set(unpacked) - set(specs) == {n + ".compressed_k" for n in attention} - set(
        owners
    )
    assert all(specs[n] == unpacked[n] for n in specs)
    assert all(isinstance(specs[n], MLAAttentionSpec) for n in owners)
    # Other than the layout's packing fields, the geometry is unchanged.
    assert replace(layout, compressed_slots=1, compressed_owners=()) == (
        unpacked_layout
    )
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = layout.block_size
    groups = get_kv_cache_groups(cfg, specs)
    check_packed_compressed_groups(layout, groups)
    token_groups = [
        g for g in groups if g.layer_names[0].endswith((".self_attn", ".compressed_k"))
    ]
    assert len(token_groups) == len(attention) + -(-len(attention) // slots)
    with pytest.raises(ValueError, match="one scheduler group per owner"):
        check_packed_compressed_groups(
            layout, [NS(layer_names=owners[:2])] + list(groups)
        )


def test_unpacked_specs_are_unchanged_by_default(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PACK_COMPRESSED_K", raising=False)
    default = pooled_cache_specs(pool_config(8))
    monkeypatch.setenv("QWEN4_EXP_PACK_COMPRESSED_K", "0")
    assert pooled_cache_specs(pool_config(8)) == default
    assert default[0].compressed_owners == ()


def test_packed_pool_binds_every_qsa_layer_to_its_owner_slot(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    cfg = packed_config(monkeypatch, True)
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = cfg.cache_config.mamba_block_size = layout.block_size
    cache = object.__new__(Qwen4ExpPooledCache)
    cache.layout = layout
    carrier = object()
    caches = {name: carrier for name in specs}
    bound = {}

    class Attention:
        def __init__(self, name):
            self.name = name

        def bind_pool(self, pool, owner):
            assert pool is carrier
            bound[self.name] = owner

    for name in specs:
        if name.endswith(".self_attn"):
            cache._bind_qsa(Attention(name), name, caches)
    assert bound == {
        name: layout.compressed_owner(name)
        for name in specs
        if name.endswith(".self_attn")
    }
    assert sorted({slot for _, slot in bound.values()}) == list(range(8))


def test_qsa_bind_pool_threads_owner_table_and_static_slot():
    from functools import partial

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    attention = object.__new__(Qwen4ExpQSA)
    torch.nn.Module.__init__(attention)
    attention.prefix = "model.layers.7.self_attn"

    def op(*args, **kwargs):
        return kwargs

    attention._op = partial(op, compress_ratio=4)
    pool = torch.zeros(1)
    attention.bind_pool(pool, ("model.layers.3.self_attn.compressed_k", 1))
    assert attention.cache_owners == (pool,)
    assert attention.compressed_cache_name == "model.layers.3.self_attn.compressed_k"
    assert attention._op.keywords == dict(compress_ratio=4, compressed_slot=1)
    # Rebinding (profiling scratch -> planned pool) replaces the slot, and slot
    # zero / unpacked pools pass no extra static argument at all.
    attention.bind_pool(pool, ("model.layers.3.self_attn.compressed_k", 0))
    assert attention._op.keywords == dict(compress_ratio=4)
    attention.bind_pool(pool)
    assert attention.compressed_cache_name == "model.layers.7.self_attn.compressed_k"
    assert attention._op.keywords == dict(compress_ratio=4)


def paired_config(monkeypatch, *, compressed=False, tp=8, kv="fp8_e4m3"):
    cfg = packed_config(monkeypatch, compressed, tp=tp)
    cfg.cache_config.cache_dtype = kv
    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "1")
    return cfg


@pytest.mark.parametrize("compressed", [False, True])
def test_paired_main_kv_pairs_target_layers_and_keeps_mtp_pages(
    monkeypatch, compressed
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import (
        check_packed_compressed_groups,
    )

    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "0")
    unpaired_layout, unpaired = pooled_cache_specs(
        packed_config(monkeypatch, compressed)
    )
    cfg = paired_config(monkeypatch, compressed=compressed)
    layout, specs = pooled_cache_specs(cfg)
    attention = [n for n in unpaired if n.endswith(".self_attn")]
    targets = [n for n in attention if n.startswith("model.layers.")]
    assert attention == targets + ["mtp.layers.48.self_attn"]
    assert len(targets) % 2 == 0
    assert unpaired_layout.main_owners == ()
    # Consecutive target layers pair up; the draft keeps whole rows.
    for index, name in enumerate(targets):
        assert layout.main_owner(name) == (targets[index // 2 * 2], index % 2)
    assert layout.main_owner(attention[-1]) == (attention[-1], -1)
    assert unpaired_layout.main_owner(targets[1]) == (targets[1], -1)
    owners = [n for n in specs if n.endswith(".self_attn")]
    assert owners == targets[::2] + attention[-1:]
    assert set(unpaired) - set(specs) == set(targets[1::2])
    assert all(specs[n] == unpaired[n] for n in specs)
    assert all(isinstance(specs[n], FullAttentionSpec) for n in owners)
    assert replace(layout, main_owners=()) == unpaired_layout
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = layout.block_size
    groups = get_kv_cache_groups(cfg, specs)
    check_packed_compressed_groups(layout, groups)
    main_groups = [g for g in groups if g.layer_names[0].endswith(".self_attn")]
    assert [g.layer_names for g in main_groups] == [[n] for n in owners]
    with pytest.raises(ValueError, match="main KV need one scheduler group"):
        check_packed_compressed_groups(
            layout, [NS(layer_names=owners[:2])] + list(groups)
        )


def test_paired_main_kv_halves_token_pages_with_packed_compressed_keys(monkeypatch):
    """TP8 + MTP: 13 main + 2 compressed pages per span become 7 + 2."""
    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "0")
    _, before = pooled_cache_specs(packed_config(monkeypatch, True))
    _, after = pooled_cache_specs(paired_config(monkeypatch, compressed=True))

    def token_pages(specs):
        return sum(n.endswith((".self_attn", ".compressed_k")) for n in specs)

    attention = sum(n.endswith(".self_attn") for n in before)
    assert token_pages(before) - token_pages(after) == attention // 2


@pytest.mark.parametrize("tp,kv", [(1, "fp8_e4m3"), (8, "bfloat16"), (8, "auto")])
def test_paired_main_kv_needs_free_fp8_head_slots(monkeypatch, tp, kv):
    with pytest.raises(ValueError, match="QWEN4_EXP_PACK_MAIN_KV=1"):
        pooled_cache_specs(paired_config(monkeypatch, tp=tp, kv=kv))


def test_unpaired_specs_are_unchanged_by_default(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PACK_MAIN_KV", raising=False)
    default = pooled_cache_specs(pool_config(8))
    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "0")
    assert pooled_cache_specs(pool_config(8)) == default
    assert default[0].main_owners == ()


def test_paired_pool_binds_main_owner_only_for_paired_layers(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "0")
    _, unpaired = pooled_cache_specs(packed_config(monkeypatch, True))
    cfg = paired_config(monkeypatch, compressed=True)
    layout, specs = pooled_cache_specs(cfg)
    cache = object.__new__(Qwen4ExpPooledCache)
    cache.layout = layout
    carrier = object()
    caches = {name: carrier for name in specs}
    bound = {}

    class Attention:
        def __init__(self, name):
            self.name = name

        def bind_pool(self, pool, owner, **main):
            assert pool is carrier
            bound[self.name] = (owner, main)

    attention = [n for n in unpaired if n.endswith(".self_attn")]
    for name in attention:
        cache._bind_qsa(Attention(name), name, caches)
    for name in attention:
        owner, main = bound[name]
        assert owner == layout.compressed_owner(name)
        paired = layout.main_owner(name)
        assert main == ({} if paired[1] < 0 else {"main_owner": paired})
    assert sum(bool(main) for _, main in bound.values()) == len(attention) - 1
    with pytest.raises(KeyError):
        caches.pop(attention[0])
        cache._bind_qsa(Attention(attention[1]), attention[1], caches)


def test_qsa_bind_pool_threads_main_owner_table_and_slot():
    from functools import partial

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    attention = object.__new__(Qwen4ExpQSA)
    torch.nn.Module.__init__(attention)
    attention.prefix = "model.layers.7.self_attn"

    def op(*args, **kwargs):
        return kwargs

    attention._op = partial(op, compress_ratio=4)
    pool = torch.zeros(1)
    attention.bind_pool(
        pool,
        ("model.layers.3.self_attn.compressed_k", 1),
        main_owner=("model.layers.3.self_attn", 1),
    )
    assert attention.main_cache_name == "model.layers.3.self_attn"
    assert attention._op.keywords == dict(
        compress_ratio=4, compressed_slot=1, main_slot=1
    )
    # Rebinding without pairing restores the layer's own pages and program.
    attention.bind_pool(pool)
    assert attention.main_cache_name == attention.prefix
    assert attention._op.keywords == dict(compress_ratio=4)
    attention.bind_pool(pool, main_owner=("model.layers.7.self_attn", 0))
    assert attention._op.keywords == dict(compress_ratio=4, main_slot=0)
