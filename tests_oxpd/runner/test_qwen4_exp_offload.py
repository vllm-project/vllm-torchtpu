# SPDX-License-Identifier: Apache-2.0
"""Host-DRAM prefix-cache offload (TPURaidenOffloadingConnector) for Qwen4Exp.

CPU-only: admission, the pool row geometry Raiden registers, a byte-exact
page round trip for every cache group kind (QSA main KV, compressed keys
packed or not, GDN and PLE align checkpoints), and the scheduler-side store
and load plan on the real Qwen4Exp group structure (MTP K=3) against an
in-memory Raiden store, including the external hits reported for metrics.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from tests.models.vllm.test_qwen4_exp_mtp import mtp_config, spec_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig
from vllm_torchtpu.models.vllm.qwen4_exp.offload import offload_pool_geometry
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

GiB = 1024**3
ARCH = "Qwen4ExpForConditionalGeneration"
NUM_BLOCKS = 24


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.delenv("QWEN4_EXP_PACK_COMPRESSED_K", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PACK_MAIN_KV", raising=False)


def offload_transfer(**extra):
    return NS(
        kv_connector=qconfig.OFFLOAD_CONNECTOR,
        kv_connector_module_path=qconfig.OFFLOAD_CONNECTOR_MODULE,
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 600 * GiB, **extra},
    )


def offload_config():
    """The AgentX serving shape: TP8/EP8, FP8 KV, APC, MTP K=3, offload."""
    cfg = mtp_config(k=3, tp=8)
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.cache_dtype = "fp8"
    cfg.kv_transfer_config = offload_transfer()
    return cfg


# -- admission ------------------------------------------------------------------


def test_admits_offload_with_prefix_caching_mtp_and_tp8():
    cfg = offload_config()
    qconfig.validate_config(cfg)
    assert qconfig.host_offload_enabled(cfg)
    assert not qconfig.pd_transfer_enabled(cfg)


def test_offload_keeps_hbm_pool_sizing_and_pd_keeps_static_sizing():
    cfg = offload_config()
    assert qconfig.hbm_pool_sizing(cfg)
    cfg.kv_transfer_config = None
    assert qconfig.hbm_pool_sizing(cfg)
    cfg.kv_transfer_config = NS(kv_connector=qconfig.PD_CONNECTOR)
    assert not qconfig.hbm_pool_sizing(cfg)


def test_offload_module_path_may_be_omitted():
    cfg = offload_config()
    cfg.kv_transfer_config.kv_connector_module_path = None
    qconfig.validate_config(cfg)


def _set(path, value):
    def mutate(cfg):
        *parents, leaf = path.split(".")
        target = cfg
        for name in parents:
            target = getattr(target, name)
        setattr(target, leaf, value)

    return mutate


def _extra(**values):
    def mutate(cfg):
        cfg.kv_transfer_config.kv_connector_extra_config.update(values)

    return mutate


def _drop_cpu_bytes(cfg):
    del cfg.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (_set("kv_transfer_config.kv_role", "kv_producer"), "kv_both"),
        (_set("kv_transfer_config.kv_connector_module_path", "x.y"), "kv_both"),
        (_set("cache_config.enable_prefix_caching", False), "prefix caching"),
        (_set("cache_config.mamba_cache_mode", "none"), "align"),
        (_drop_cpu_bytes, "cpu_bytes_to_use"),
        (_extra(cpu_bytes_to_use=0), "cpu_bytes_to_use"),
        (_extra(cpu_bytes_to_use="lots"), "cpu_bytes_to_use"),
        (_extra(blocks_per_chunk=2), "one pool page"),
        (_extra(block_size=1024), "one pool page"),
    ],
)
def test_rejects_unsupported_offload(mutate, error):
    cfg = offload_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=error):
        qconfig.validate_config(cfg)


def test_rejects_offload_without_unified_pool(monkeypatch):
    cfg = offload_config()
    cfg.speculative_config = None
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    with pytest.raises(ValueError):
        qconfig.validate_config(cfg)


def test_pd_still_rejects_mtp_and_other_connectors():
    cfg = offload_config()
    cfg.kv_transfer_config = NS(
        kv_connector=qconfig.PD_CONNECTOR,
        kv_connector_module_path=None,
        kv_role="kv_producer",
    )
    with pytest.raises(ValueError, match="MTP off"):
        qconfig.validate_config(cfg)
    cfg = offload_config()
    cfg.kv_transfer_config.kv_connector = "TPURaidenConnector"
    with pytest.raises(ValueError, match="PD requires"):
        qconfig.validate_config(cfg)


# -- pool geometry --------------------------------------------------------------


def qwen_pool(monkeypatch, packed=False, num_blocks=NUM_BLOCKS, paired=False):
    """Native (worker) and scheduler KV cache configs of the TP8 MTP pool."""
    if packed:
        monkeypatch.setenv("QWEN4_EXP_PACK_COMPRESSED_K", "1")
    if paired:
        monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "1")
    # FP32 SSM state (the checkpoint default) gives the 512-token page.
    cfg = pool_config(tp=8, kv="fp8_e4m3", ssm="float32")
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1
    cfg.speculative_config = spec_config(3)
    layout, specs = pooled_cache_specs(cfg)
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(
        cfg, groups, num_blocks * layout.page_bytes
    )
    assert native.num_blocks == num_blocks
    scheduler = generate_scheduler_kv_cache_config([native])
    return cfg, layout, native, scheduler


@pytest.mark.parametrize("packed", [False, True])
def test_pool_rows_are_whole_carrier_pages(monkeypatch, packed):
    from vllm_torchtpu.offload import raiden_store

    _, layout, native, _ = qwen_pool(monkeypatch, packed)
    assert layout.compressed_slots == (8 if packed else 1)
    geometry = offload_pool_geometry(None, native)
    assert geometry == (
        layout.block_size,
        (layout.block_size, 1, 4, 256),
        torch.float8_e4m3fn,
        layout.block_size,
    )
    # The offload spec resolves the same rows through the model hook.
    vllm_config = NS(model_config=NS(architecture=ARCH))
    assert raiden_store.resolve_kernel_geometry(vllm_config, native) == geometry


def test_paired_main_kv_pool_rows_are_whole_carrier_pages(monkeypatch):
    """Paired main KV pages move whole, both layers' head slots included."""
    _, layout, native, _ = qwen_pool(monkeypatch, packed=True, paired=True)
    assert len(layout.main_owners) == 12
    main = [
        g.layer_names
        for g in native.kv_cache_groups
        if g.layer_names[0].endswith(".self_attn")
    ]
    # Six target pairs and the MTP draft's own pages.
    assert len(main) == 7 and [layout.main_owners[0][1]] in main
    assert offload_pool_geometry(None, native) == (
        layout.block_size,
        (layout.block_size, 1, 4, 256),
        torch.float8_e4m3fn,
        layout.block_size,
    )


def test_pool_geometry_rejects_mixed_pages(monkeypatch):
    _, layout, native, _ = qwen_pool(monkeypatch)
    native.kv_cache_groups[0] = NS(
        kv_cache_spec=NS(page_size_bytes=layout.page_bytes // 2)
    )
    with pytest.raises(ValueError, match="uniform pool page"):
        offload_pool_geometry(None, native)


def test_other_models_keep_their_geometry_path():
    from vllm_torchtpu.offload import raiden_store

    for arch in ("Qwen3NextForCausalLM", None):
        assert (
            raiden_store.resolve_model_owned_pool_geometry(
                NS(model_config=NS(architecture=arch)), None
            )
            is None
        )


# -- byte-exact page round trip ------------------------------------------------


def _fake_tpu_sync(captured):
    """Stand-in for tpu_sync: capture what each rank registers with Raiden."""

    class KVCacheManager:
        def __init__(self, *, kv_caches, **kwargs):
            captured.append((kv_caches, kwargs))

    import sys
    import types

    api = types.ModuleType("tpu_sync.api.torch")
    api.kv_cache_manager = types.SimpleNamespace(KVCacheManager=KVCacheManager)
    return {
        "tpu_sync": types.ModuleType("tpu_sync"),
        "tpu_sync.api": types.ModuleType("tpu_sync.api"),
        "tpu_sync.api.torch": api,
        "tpu_sync.api.torch.kv_cache_manager": api.kv_cache_manager,
    }, sys


def _fill(page_bytes: torch.Tensor, dtype, offset, numel, generator):
    """Write random values of ``dtype`` into a byte range of one page."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    region = page_bytes[offset : offset + numel * itemsize].view(dtype)
    if dtype.is_floating_point and dtype.itemsize >= 2:
        values = torch.randn(numel, generator=generator).to(dtype)
    else:
        values = torch.randint(
            0, 256, (numel * itemsize,), generator=generator, dtype=torch.uint8
        ).view(dtype)
    region.copy_(values)


@pytest.mark.parametrize("packed", [False, True])
def test_every_group_kind_round_trips_bit_exactly(monkeypatch, packed):
    """Store pages of each group kind, clobber the pool, load into new blocks.

    Raiden moves rows of the tensors a rank registers; registration goes
    through the real ``TPURaidenOffloadingConnector.register_kv_caches`` and
    ``RaidenStoreOffloadingWorker`` and the copy uses those row views, so a
    row is checked to be exactly one manager page of the pool for every
    group: typed contents written through each group's own dtype must come
    back bit for bit in the destination block.
    """
    from unittest.mock import patch

    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector
    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    cfg, layout, native, _ = qwen_pool(monkeypatch, packed)
    page = layout.page_bytes
    shape = (NUM_BLOCKS, layout.block_size, 1, 4, 256)
    pool = torch.zeros(shape, dtype=torch.float8_e4m3fn)
    pool_bytes = pool.view(torch.uint8).view(NUM_BLOCKS, page)

    # The model's live caches as the runner hands them to the connector.
    kv_caches = {}
    for group in native.kv_cache_groups:
        for name in group.layer_names:
            if isinstance(group.kv_cache_spec, MambaSpec):
                kv_caches[name] = [pool]
            else:
                kv_caches[name] = pool
    captured_canonical = []
    connector = NS(
        connector_worker=NS(
            kv_cache_config=native, _init_worker=captured_canonical.append
        )
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)
    (canonical,) = captured_canonical
    assert len(canonical.tensors) == 1
    assert canonical.tensors[0].page_size_bytes == page

    registered = []
    modules, sys = _fake_tpu_sync(registered)
    vllm_config = NS(model_config=NS(architecture=ARCH))
    with (
        patch.dict(sys.modules, modules),
        patch.object(RaidenStoreOffloadingWorker, "_controller_reachable") as up,
    ):
        up.return_value = True
        worker = RaidenStoreOffloadingWorker(
            canonical,
            vllm_config=vllm_config,
            kv_cache_config=native,
            host_blocks_to_allocate=8,
            controller_address="127.0.0.1:1",
            rank=0,
        )
        worker._register_thread.join(timeout=10)
    assert registered, "worker never registered"
    ((rows,),) = registered[0][0]
    assert rows.shape == shape and rows.dtype == torch.float8_e4m3fn
    assert rows.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()

    generator = torch.Generator().manual_seed(0)
    kinds = {}
    src_block = 1
    for gid, group in enumerate(native.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, CircularBufferSpec):
            continue  # per-request ring: never offloaded
        kind = type(spec).__name__ + (
            ".ple" if group.layer_names[0].endswith(".ple") else ""
        )
        if kind in kinds:
            continue
        page_view = pool_bytes[src_block]
        if isinstance(spec, MambaSpec) and kind.endswith(".ple"):
            _fill(page_view, torch.bfloat16, 0, layout.ple_state_bytes // 2, generator)
        elif isinstance(spec, MambaSpec):
            # GDN: SSM rows first, then the FP32 conv region.
            ssm_dtype = spec.dtypes[1]
            ssm_numel = (layout.gdn_state_bytes // 2) // ssm_dtype.itemsize
            _fill(page_view, ssm_dtype, 0, ssm_numel, generator)
            _fill(
                page_view,
                torch.float32,
                layout.gdn_state_bytes // 2,
                (layout.gdn_state_bytes // 2) // 4,
                generator,
            )
        elif isinstance(spec, MLAAttentionSpec):
            # Compressed keys: BF16 kernel pages, one slot per packed layer.
            per_slot = layout.main_kernel_pages * layout.compressed_kernel_bytes
            for slot in range(layout.compressed_slots):
                _fill(
                    page_view, torch.bfloat16, slot * per_slot, per_slot // 2, generator
                )
        else:
            assert isinstance(spec, FullAttentionSpec)
            # Main KV: FP8 head-packed rows across the whole page.
            _fill(page_view, torch.float8_e4m3fn, 0, page, generator)
        kinds[kind] = (gid, src_block)
        src_block += 1
    assert set(kinds) == {
        "MambaSpec",
        "MambaSpec.ple",
        "FullAttentionSpec",
        "MLAAttentionSpec",
    }
    originals = {kind: pool_bytes[b].clone() for kind, (_, b) in kinds.items()}
    assert all(original.any() for original in originals.values())

    # Save: one host slot per (group, block) key, copied from the row view.
    host = {kind: rows[b].clone() for kind, (_, b) in kinds.items()}
    # The blocks are recycled.
    pool_bytes.copy_(
        torch.randint(0, 256, pool_bytes.shape, generator=generator, dtype=torch.uint8)
    )
    # Load into different blocks, again through the row view.
    for i, (kind, slot) in enumerate(host.items()):
        rows[NUM_BLOCKS - 1 - i].copy_(slot)
        restored = pool_bytes[NUM_BLOCKS - 1 - i]
        assert torch.equal(restored, originals[kind]), kind


# -- scheduler: store and load plan against an in-memory Raiden store -----------


def _offloading_config(scheduler_kvc, world_size=8):
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingGroupConfig,
        OffloadingModelConfig,
        OffloadingParallelConfig,
    )

    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=scheduler_kvc.kv_cache_groups[
                gid
            ].kv_cache_spec.block_size,
            layer_names=tuple(scheduler_kvc.kv_cache_groups[gid].layer_names),
            group_id=gid,
        )
        for gid in scheduler_kvc.prefix_cacheable_group_ids
    )
    tensor = scheduler_kvc.kv_cache_tensors[0]
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=tensor.size // scheduler_kvc.num_blocks,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": 64 * GiB},
        engine_id="engine",
        model=OffloadingModelConfig(name="Qwen/Qwen3.8-Flash-Next-FP8", dtype="fp8"),
        cache=OffloadingCacheConfig(tokens_per_hash=512, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def _scheduler(monkeypatch, scheduler_kvc):
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    monkeypatch.setattr(raiden_store, "_all_ranks_kv_bytes_per_block", None)
    vllm_config = NS(
        model_config=NS(
            is_hybrid=True,
            architecture=ARCH,
            model="Qwen/Qwen3.8-Flash-Next-FP8",
            revision=None,
            quantization="fp8",
        ),
        cache_config=NS(
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256",
            prefix_cache_retention_interval=None,
        ),
        parallel_config=NS(
            world_size=8,
            decode_context_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=spec_config(3),
        kv_events_config=None,
    )
    spec = raiden_store.TPURaidenStoreOffloadingSpec(
        _offloading_config(scheduler_kvc), vllm_config, scheduler_kvc
    )
    # One Raiden key per scheduler block: 8 ranks x one 512 KiB page each.
    assert spec.device_block_size_factor == 1
    assert spec.offload_logical_blocks_capacity == 64 * GiB // (8 * 512 * 1024)
    store = FakeKVCacheStore()
    spec._manager = make_manager(
        store,
        device_block_size_factor=1,
        num_blocks=spec.offload_logical_blocks_capacity,
        world_size=8,
        key_namespace=spec.key_namespace,
    )
    return TPURaidenOffloadingScheduler(spec, vllm_config, scheduler_kvc), store


def _tokens(num_tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 200000, (num_tokens,), generator=generator).tolist()


def _request(req_id, tokens):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=get_request_block_hasher(512, sha256),
    )


def _group_of(key: bytes) -> int:
    return int.from_bytes(key[-4:], "big")


def _key_group_of_sub_hash(sub_hash: bytes, namespace: bytes) -> tuple[bytes, int]:
    key = sub_hash[len(namespace) : -4]
    return key, _group_of(key)


class _Blocks:
    """Block IDs handed out per group, like the scheduler's block pool."""

    def __init__(self):
        self.next_id = 1

    def take(self, count):
        ids = list(range(self.next_id, self.next_id + count))
        self.next_id += count
        return ids


def _is_mamba(kvc, gid):
    return isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec)


@pytest.mark.parametrize("packed", [False, True])
def test_host_hit_restores_every_group_at_the_mamba_boundary(monkeypatch, packed):
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.offload.raiden_connector import (
        RaidenOffloadingConnectorMetadata,
        RaidenOffloadingWorkerMetadata,
    )

    _, layout, _, kvc = qwen_pool(monkeypatch, packed)
    assert layout.block_size == 512
    sched, store = _scheduler(monkeypatch, kvc)
    namespace = sched._raiden_manager._key_namespace
    cacheable = set(kvc.prefix_cacheable_group_ids)
    ring_groups = {
        gid
        for gid, group in enumerate(kvc.kv_cache_groups)
        if isinstance(group.kv_cache_spec, CircularBufferSpec)
    }
    assert ring_groups and not ring_groups & cacheable
    mamba = [gid for gid in sorted(cacheable) if _is_mamba(kvc, gid)]
    attention = [gid for gid in sorted(cacheable) if not _is_mamba(kvc, gid)]
    assert len(mamba) == 37  # 36 GDN + 1 PLE
    assert len(attention) == (15 if packed else 26)  # 13 QSA layers incl. MTP

    # -- turn 1: a 3-block prompt prefills in one step and is offloaded.
    prompt = 3 * 512 + 5
    first = _request("r1", _tokens(prompt, 0))
    sched.on_new_request(first)
    assert sched.get_num_new_matched_tokens(first, 0) == (0, False)
    blocks = _Blocks()
    table = []
    checkpoints = {}
    for gid in range(len(kvc.kv_cache_groups)):
        if gid in ring_groups:
            table.append(blocks.take(1))
        elif _is_mamba(kvc, gid):
            # Align mode: nulls, the prompt checkpoint at block 2, then the
            # running state and its three MTP checkpoint blocks.
            ckpt, *rest = blocks.take(5)
            checkpoints[gid] = ckpt
            table.append([0, 0, ckpt, *rest])
        else:
            table.append(blocks.take(4))
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NS(req_id="r1", block_ids=tuple(table))]
    output.num_scheduled_tokens = {"r1": prompt}
    output.total_num_scheduled_tokens = prompt
    output.kv_connector_block_state = KVConnectorBlockState(
        req_ids={"r1"},
        resolve_block_ids=lambda req_id: tuple(table),
        boundary_state_offloads={
            "r1": [(gid, checkpoints[gid], 3 * 512) for gid in mamba]
        },
    )
    meta = sched.build_connector_meta(output)
    assert isinstance(meta, RaidenOffloadingConnectorMetadata)
    assert meta.fence_job_ids and not meta.load_jobs and not meta.store_jobs

    stored = {}
    for job_id in meta.fence_job_ids:
        parked = sched._fence_pending[job_id]
        stored.update(zip(parked.keys, parked.device_block_ids))
    expected = {}
    for gid in attention:
        for chunk in range(3):
            expected[(gid, chunk)] = table[gid][chunk]
    for gid in mamba:
        expected[(gid, 2)] = checkpoints[gid]
    chunk_of = {bytes(h): i for i, h in enumerate(first.block_hashes)}
    got = {
        (_group_of(key), chunk_of[bytes(key[:-4])]): block
        for key, block in stored.items()
    }
    assert got == expected
    assert not {_group_of(key) for key in stored} & ring_groups

    # Every rank acknowledges the fence; Raiden saves, then reports done.
    ack = RaidenOffloadingWorkerMetadata(
        completed_jobs={}, fenced_jobs={job: 8 for job in meta.fence_job_ids}
    )
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ack))
    saved = [h for call in store.calls_named("save") for h in call.args[0]]
    assert len(saved) == len(expected)
    store.complete_save(saved)
    done = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert sorted(done.finished_store_job_ids) == sorted(meta.fence_job_ids)
    sched.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                completed_jobs={job: 8 for job in done.finished_store_job_ids}
            )
        )
    )

    # -- turn 2 extends the prompt: everything up to the checkpoint is a host
    # hit, reported as external tokens beyond any local (HBM) hit.
    second = _request("r2", first.prompt_token_ids + _tokens(700, 1))
    assert second.block_hashes[:3] == first.block_hashes[:3]
    sched.on_new_request(second)
    assert sched.get_num_new_matched_tokens(second, 512) == (2 * 512, True)
    assert sched.get_num_new_matched_tokens(second, 0) == (3 * 512, True)

    # The load lands in fresh blocks of every group: attention chunks 0-2 and
    # the Mamba checkpoint at the hit boundary (chunk 2), nothing else.
    alloc = []
    for gid in range(len(kvc.kv_cache_groups)):
        if gid in ring_groups:
            ids = blocks.take(1)
        elif _is_mamba(kvc, gid):
            ids = [0, 0, *blocks.take(1)]
        else:
            ids = blocks.take(3)
        alloc.append([KVCacheBlock(i, is_null=i == 0) for i in ids])
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    sched.update_state_after_alloc(second, KVCacheBlocks(tuple(alloc)), 3 * 512)
    load = SchedulerOutput.make_empty()
    meta = sched.build_connector_meta(load)
    (load_call,) = store.calls_named("load")
    hashes, device_blocks = load_call.args
    loaded = {}
    for sub_hash, block in zip(hashes, device_blocks):
        key, gid = _key_group_of_sub_hash(sub_hash, namespace)
        loaded[(gid, chunk_of[bytes(key[:-4])])] = block
    expected_load = {}
    for gid in attention:
        for chunk in range(3):
            expected_load[(gid, chunk)] = alloc[gid][chunk].block_id
    for gid in mamba:
        expected_load[(gid, 2)] = alloc[gid][2].block_id
    assert loaded == expected_load

    # A host hit needs every Mamba group's checkpoint at the boundary: with
    # one evicted there is no earlier boundary to fall back to.
    store.complete_load(hashes)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    victim = next(
        h
        for h in list(store.entries)
        if _key_group_of_sub_hash(h, namespace)[1] == mamba[-1]
    )
    del store.entries[victim]
    third = _request("r3", first.prompt_token_ids + _tokens(300, 2))
    sched.on_new_request(third)
    assert sched.get_num_new_matched_tokens(third, 0) == (0, False)
