# SPDX-License-Identifier: Apache-2.0
"""A host-DRAM prefix tier on the served Qwen4Exp PD prefill (CPU).

The served prefill is a PD producer of eight TP1 pipeline stages on the gcd
group-size rule. Its host tier is the Raiden offload connector composed with
the model's PD connector in one TPUMultiConnector. These tests pin:

- admission: the composed producer (PP8, DP8), offload under PP alone, and
  every refused variant;
- geometry: a PP8 stage's pool is two raw regions; the offload geometry,
  the region guard and the host capacity follow it;
- bytes: every cache group kind of a two-region stage round-trips bit for
  bit through the real registration and Raiden row views, across regions;
- the scheduler plan on the served PP8 group structure (3,200-token pages):
  a turn-2 host hit loads every attention block and every GDN/PLE
  checkpoint at the boundary, exactly what an HBM hit at that boundary
  reads, and nothing when one checkpoint is gone;
- the multi connector's composition contracts the PD producer relies on.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

from tests.models.vllm.test_qwen4_exp_parallel import config as base_config
from tests.runner.test_qwen4_exp_offload import (
    GiB,
    _Blocks,
    _fake_tpu_sync,
    _fill,
    _group_of,
    _key_group_of_sub_hash,
    _tokens,
)
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig
from vllm_torchtpu.models.vllm.qwen4_exp.offload import (
    offload_pool_geometry,
    pool_regions,
)

ARCH = "Qwen4ExpForConditionalGeneration"
STAGES = 8
PAGE = 3200 * 1024  # served PP8 TP1 page: 3,200 tokens x 1,024 B fp8 main KV
BLOCK = 3200
NUM_BLOCKS = 40


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.delenv("QWEN4_EXP_PACK_COMPRESSED_K", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PACK_MAIN_KV", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR", raising=False)


def _pd_child(role="kv_producer"):
    return {
        "kv_connector": qconfig.PD_CONNECTOR,
        "kv_connector_module_path": qconfig.PD_CONNECTOR_MODULE,
        "kv_role": role,
        "kv_port": 14579,
    }


def _offload_child(cpu_bytes=600 * GiB, role="kv_both"):
    return {
        "kv_connector": qconfig.OFFLOAD_CONNECTOR,
        "kv_connector_module_path": qconfig.OFFLOAD_CONNECTOR_MODULE,
        "kv_role": role,
        "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
    }


def multi_transfer(children=None, role="kv_producer"):
    return NS(
        kv_connector=qconfig.MULTI_CONNECTOR,
        kv_connector_module_path=qconfig.MULTI_CONNECTOR_MODULE,
        kv_role=role,
        kv_connector_extra_config={
            "connectors": children if children is not None else [_pd_child(), _offload_child()]
        },
    )


def served_prefill(pp=8, dp=1):
    """The served PD prefill shape: PP8 TP1 (or DP8), FP8 KV, APC, align."""
    cfg = base_config(tp=1, dp=dp)
    cfg.parallel_config.pipeline_parallel_size = pp
    cfg.parallel_config.enable_expert_parallel = dp > 1
    cfg.cache_config.cache_dtype = "fp8"
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.mamba_cache_mode = "align"
    cfg.cache_config.mamba_ssm_cache_dtype = "float32"
    cfg.kv_transfer_config = multi_transfer()
    return cfg


# -- admission -------------------------------------------------------------------


@pytest.mark.parametrize("pp,dp", [(8, 1), (1, 8)])
def test_admits_pd_producer_with_host_tier(pp, dp):
    cfg = served_prefill(pp=pp, dp=dp)
    qconfig.validate_config(cfg)
    assert qconfig.pd_transfer_enabled(cfg)
    assert qconfig.host_offload_enabled(cfg)
    assert qconfig.pd_transfer_config(cfg).kv_role == "kv_producer"
    assert qconfig.host_offload_transfer_config(cfg).kv_role == "kv_both"
    # A PD engine keeps static pool sizing: both legs plan the pool alike.
    assert not qconfig.hbm_pool_sizing(cfg)


def test_plain_pd_and_plain_offload_keep_their_admission():
    cfg = served_prefill()
    cfg.kv_transfer_config = NS(**_pd_child())
    qconfig.validate_config(cfg)
    assert qconfig.pd_transfer_enabled(cfg) and not qconfig.host_offload_enabled(cfg)
    assert not qconfig.hbm_pool_sizing(cfg)
    # Offload alone is now admitted on pipeline stages too (no PD peer).
    cfg.kv_transfer_config = NS(
        **{k: v for k, v in _offload_child().items()}
    )
    qconfig.validate_config(cfg)
    assert qconfig.host_offload_enabled(cfg) and not qconfig.pd_transfer_enabled(cfg)
    assert qconfig.hbm_pool_sizing(cfg)


def _mutated(mutate):
    cfg = served_prefill()
    mutate(cfg)
    return cfg


@pytest.mark.parametrize(
    "mutate,error",
    [
        # decode side: not admitted here
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([_pd_child("kv_consumer"), _offload_child()],
                                          role="kv_consumer")), "PD producer"),
        # top-level role must be the PD child's
        (lambda c: setattr(c.kv_transfer_config, "kv_role", "kv_both"), "PD producer"),
        # exactly the two children
        (lambda c: setattr(c, "kv_transfer_config", multi_transfer([_pd_child()])), "PD producer"),
        (lambda c: setattr(c, "kv_transfer_config", multi_transfer([_offload_child()])), "PD producer"),
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([_pd_child(), _offload_child(), _offload_child()])),
         "PD producer"),
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([_pd_child(), {**_offload_child(),
                                                         "kv_connector": "OffloadingConnector"}])),
         "PD producer"),
        (lambda c: setattr(c.kv_transfer_config, "kv_connector_module_path", "x.y"), "PD producer"),
        # the offload child's own rules
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([_pd_child(), _offload_child(role="kv_producer")])),
         "kv_both"),
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([_pd_child(), _offload_child(cpu_bytes=0)])),
         "cpu_bytes_to_use"),
        (lambda c: setattr(c.cache_config, "mamba_cache_mode", "none"), "align"),
        # the PD child's own rules still apply
        (lambda c: setattr(c, "kv_transfer_config",
                           multi_transfer([{**_pd_child(), "kv_connector_module_path": "x.y"},
                                           _offload_child()])),
         "PD requires"),
        (lambda c: setattr(c.cache_config, "cache_dtype", "auto"), "PD requires"),
    ],
)
def test_rejects_unsupported_compositions(mutate, error):
    cfg = _mutated(mutate)
    with pytest.raises(ValueError, match=error):
        qconfig.validate_config(cfg)


def test_rejects_offload_with_stage_page_remap(monkeypatch):
    """The connector moves pool row = block ID; a stage page map breaks that."""
    monkeypatch.setenv("QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR", "2")
    with pytest.raises(ValueError, match="LOGICAL_BLOCKS_FACTOR"):
        qconfig.validate_config(served_prefill())


# -- the served PP8 pool: stages, regions, capacity -------------------------------


def _gcd_rule(monkeypatch):
    """Install the TPU gcd group-size rule (the served default) for one test."""
    import vllm.v1.core.kv_cache_utils as kcu

    monkeypatch.setenv("TPU_KV_CACHE_GROUP_SIZE_RULE", "gcd")
    monkeypatch.setattr(kcu, "_tpu_group_size_patch", False, raising=False)
    monkeypatch.setattr(
        kcu, "_get_kv_cache_groups_uniform_page_size",
        kcu._get_kv_cache_groups_uniform_page_size,
    )
    qconfig.ensure_kv_group_size_rule()


def pp8_pool(monkeypatch, num_blocks=NUM_BLOCKS):
    """Per-stage native configs and the scheduler config of the served PP8 pool."""
    from vllm.distributed.utils import get_pp_indices
    from vllm.v1.core.kv_cache_utils import (
        _get_kv_cache_bytes_per_block,
        _project_kv_cache_groups_to_worker,
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs
    from vllm_torchtpu.models.vllm.qwen4_exp.pp import owns_cache

    _gcd_rule(monkeypatch)
    cfg = pool_config(tp=1, kv="fp8_e4m3", ssm="float32")
    layout, model_specs = pooled_cache_specs(cfg)
    model_groups = get_kv_cache_groups(cfg, dict(model_specs))
    num_layers = cfg.model_config.hf_text_config.num_hidden_layers
    stages = []
    for stage in range(STAGES):
        start, end = get_pp_indices(num_layers, stage, STAGES)
        specs = {
            name: spec
            for name, spec in model_specs.items()
            if owns_cache(name, range(start, end), stage == STAGES - 1)
        }
        groups = _project_kv_cache_groups_to_worker(model_groups, specs)
        native = get_kv_cache_config_from_groups(
            cfg, groups, num_blocks * _get_kv_cache_bytes_per_block(groups)
        )
        assert native.num_blocks == num_blocks
        stages.append(native)
    return cfg, layout, stages, generate_scheduler_kv_cache_config(stages)


def test_served_stage_pool_is_two_regions_and_offload_geometry_accepts_it(monkeypatch):
    from vllm_torchtpu.offload import raiden_store

    _, layout, stages, scheduler = pp8_pool(monkeypatch)
    assert layout.block_size == BLOCK and layout.page_bytes == PAGE
    regions = [pool_regions(stage) for stage in stages]
    # The served log's "pools=2": the busiest stage backs two layer slots.
    assert max(regions) == 2 and min(regions) >= 1
    for stage in stages:
        assert offload_pool_geometry(None, stage) == (
            BLOCK, (BLOCK, 1, 4, 256), torch.float8_e4m3fn, BLOCK
        )
    assert offload_pool_geometry(None, scheduler)[0] == BLOCK
    # Host capacity: one key holds the block's row of every region on every
    # stage, so bytes per key = sum(regions) pages.
    total = raiden_store.record_all_ranks_kv_bytes_per_block(stages)
    assert total == sum(regions) * PAGE
    monkeypatch.setattr(raiden_store, "_all_ranks_kv_bytes_per_block", None)


def _stage_caches(native, regions):
    """The runner's named caches for one stage: each layer -> its region tensor."""
    from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

    index = layer_to_pool_index(native)
    caches = {}
    for group in native.kv_cache_groups:
        for name in group.layer_names:
            raw = regions[index[name]]
            caches[name] = [raw] if isinstance(group.kv_cache_spec, MambaSpec) else raw
    return caches, index


def _register(native, caches):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    captured = []
    connector = NS(
        _vllm_config=NS(model_config=NS(architecture=ARCH)),
        connector_worker=NS(kv_cache_config=native, _init_worker=captured.append),
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, caches)
    return captured


def test_registration_needs_every_region(monkeypatch):
    _, _, stages, _ = pp8_pool(monkeypatch, num_blocks=4)
    native = next(s for s in stages if pool_regions(s) == 2)
    shape = (4, BLOCK, 1, 4, 256)
    regions = [torch.zeros(shape, dtype=torch.float8_e4m3fn) for _ in range(2)]
    caches, _ = _stage_caches(native, regions)
    (canonical,) = _register(native, caches)
    assert len(canonical.tensors) == 2
    assert all(t.page_size_bytes == PAGE for t in canonical.tensors)
    # Every group references both regions: a key moves the block's row of each.
    assert all(len(refs) == 2 for refs in canonical.group_data_refs)
    # A registration that sees one storage for two regions would drop half
    # the layers of every block; it must fail loudly.
    collapsed, _ = _stage_caches(native, [regions[0], regions[0]])
    with pytest.raises(AssertionError, match="every pool region"):
        _register(native, collapsed)


def test_every_group_round_trips_across_regions_bit_exactly(monkeypatch):
    """Save a block's rows of both regions, recycle, load elsewhere: equal bytes."""
    from unittest.mock import patch

    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    nb = 12
    _, layout, stages, _ = pp8_pool(monkeypatch, num_blocks=nb)
    native = next(s for s in stages if pool_regions(s) == 2)
    shape = (nb, BLOCK, 1, 4, 256)
    regions = [torch.zeros(shape, dtype=torch.float8_e4m3fn) for _ in range(2)]
    region_bytes = [r.view(torch.uint8).view(nb, PAGE) for r in regions]
    caches, index = _stage_caches(native, regions)
    (canonical,) = _register(native, caches)

    registered = []
    modules, sys = _fake_tpu_sync(registered)
    with (
        patch.dict(sys.modules, modules),
        patch.object(RaidenStoreOffloadingWorker, "_controller_reachable") as up,
    ):
        up.return_value = True
        worker = RaidenStoreOffloadingWorker(
            canonical,
            vllm_config=NS(model_config=NS(architecture=ARCH)),
            kv_cache_config=native,
            host_blocks_to_allocate=8,
            controller_address="127.0.0.1:1",
            rank=0,
        )
        worker._register_thread.join(timeout=10)
    assert registered, "worker never registered"
    rows = [entry[0] for entry in registered[0][0]]
    assert len(rows) == 2
    for row, raw in zip(rows, regions):
        assert row.shape == shape
        assert row.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()

    # One block per group with a layer on this stage: fill each layer's page
    # in its own region with that group kind's dtype layout.
    generator = torch.Generator().manual_seed(1)
    owned = [
        (gid, g) for gid, g in enumerate(native.kv_cache_groups)
        if g.layer_names and not isinstance(g.kv_cache_spec, CircularBufferSpec)
    ]
    assert owned, "a served stage owns cacheable groups"
    block_of = {}
    for i, (gid, group) in enumerate(owned):
        block = 1 + i % (nb - 2)
        block_of[gid] = block
        for name in group.layer_names:
            page = region_bytes[index[name]][block]
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec) and name.endswith(".ple"):
                _fill(page, torch.bfloat16, 0, layout.ple_state_bytes // 2, generator)
            elif isinstance(spec, MambaSpec):
                _fill(page, spec.dtypes[1], 0,
                      (layout.gdn_state_bytes // 2) // spec.dtypes[1].itemsize, generator)
                _fill(page, torch.float32, layout.gdn_state_bytes // 2,
                      (layout.gdn_state_bytes // 2) // 4, generator)
            else:
                _fill(page, torch.uint8, 0, PAGE, generator)
    kinds = {type(g.kv_cache_spec).__name__ for _, g in owned}
    assert {"MambaSpec", "FullAttentionSpec"} <= kinds
    assert len(set(block_of.values())) == len(owned), "one block per group"
    snap = {
        (gid, n): region_bytes[index[n]][block_of[gid]].clone()
        for gid, g in owned for n in g.layer_names
    }
    assert all(v.any() for v in snap.values()), "every owned page was filled"
    dst = nb - 1
    for gid, group in owned:
        block = block_of[gid]
        for n in group.layer_names:
            region_bytes[index[n]][block].copy_(snap[(gid, n)])
        host = [row[block].clone() for row in rows]  # the key's host slot: every region
        for rb in region_bytes:  # the blocks are recycled
            rb.copy_(torch.randint(0, 256, rb.shape, generator=generator, dtype=torch.uint8))
        for row, slot in zip(rows, host):  # a host hit lands in a fresh block
            row[dst].copy_(slot)
        for n in group.layer_names:
            assert torch.equal(region_bytes[index[n]][dst], snap[(gid, n)]), (gid, n)


# -- the scheduler's store and load plan on the served PP8 groups ---------------


def _pp8_offloading_config(scheduler_kvc, stages, cpu_bytes):
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingGroupConfig,
        OffloadingModelConfig,
        OffloadingParallelConfig,
    )

    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=scheduler_kvc.kv_cache_groups[gid].kv_cache_spec.block_size,
            layer_names=tuple(scheduler_kvc.kv_cache_groups[gid].layer_names),
            group_id=gid,
        )
        for gid in scheduler_kvc.prefix_cacheable_group_ids
    )
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=scheduler_kvc.kv_cache_tensors[0].size
        // scheduler_kvc.num_blocks,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": cpu_bytes},
        engine_id="engine",
        model=OffloadingModelConfig(name="Qwen/Qwen3.8-Flash-Next-FP8", dtype="fp8"),
        cache=OffloadingCacheConfig(tokens_per_hash=BLOCK, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0, world_size=STAGES, tp_size=1, pp_size=STAGES, pcp_size=1, dcp_size=1,
            data_parallel_index=0, data_parallel_size=1, data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def _pp8_scheduler(monkeypatch, scheduler_kvc, stages, cpu_bytes=64 * GiB, store=None):
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    monkeypatch.setattr(raiden_store, "_all_ranks_kv_bytes_per_block", None)
    per_key = raiden_store.record_all_ranks_kv_bytes_per_block(stages)
    vllm_config = NS(
        model_config=NS(is_hybrid=True, architecture=ARCH,
                        model="Qwen/Qwen3.8-Flash-Next-FP8", revision=None,
                        quantization="fp8"),
        cache_config=NS(enable_prefix_caching=True, prefix_caching_hash_algo="sha256",
                        prefix_cache_retention_interval=None),
        parallel_config=NS(world_size=STAGES, decode_context_parallel_size=1,
                           data_parallel_rank=0, data_parallel_rank_local=0,
                           cp_kv_cache_interleave_size=1, pipeline_parallel_size=STAGES),
        speculative_config=None,
        kv_events_config=None,
    )
    spec = raiden_store.TPURaidenStoreOffloadingSpec(
        _pp8_offloading_config(scheduler_kvc, stages, cpu_bytes), vllm_config, scheduler_kvc
    )
    assert spec.device_block_size_factor == 1
    # Capacity from the sum over the stages' regions, not one stage x 8.
    assert spec.offload_logical_blocks_capacity == cpu_bytes // per_key
    store = FakeKVCacheStore() if store is None else store
    spec._manager = make_manager(
        store, device_block_size_factor=1, num_blocks=spec.offload_logical_blocks_capacity,
        world_size=STAGES, key_namespace=spec.key_namespace,
    )
    return TPURaidenOffloadingScheduler(spec, vllm_config, scheduler_kvc), store


def _request(req_id, tokens):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        req_id, prompt_token_ids=tokens, sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None, block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def test_pp8_host_hit_restores_every_group_at_the_mamba_boundary(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.offload.raiden_connector import (
        RaidenOffloadingConnectorMetadata,
        RaidenOffloadingWorkerMetadata,
    )

    _, layout, stages, kvc = pp8_pool(monkeypatch)
    sched, store = _pp8_scheduler(monkeypatch, kvc, stages)
    namespace = sched._raiden_manager._key_namespace
    cacheable = set(kvc.prefix_cacheable_group_ids)
    is_mamba = lambda gid: isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec)
    ring_groups = {
        gid for gid, g in enumerate(kvc.kv_cache_groups)
        if isinstance(g.kv_cache_spec, CircularBufferSpec)
    }
    assert ring_groups and not ring_groups & cacheable
    mamba = [gid for gid in sorted(cacheable) if is_mamba(gid)]
    attention = [gid for gid in sorted(cacheable) if not is_mamba(gid)]
    # gcd rule, 12-layer groups: GDN x3 + PLE; main KV + compressed keys.
    assert len(mamba) == 4 and len(attention) == 2

    # -- turn 1: a 3-block-plus prompt is prefilled and offloaded.
    prompt = 3 * BLOCK + 5
    first = _request("r1", _tokens(prompt, 0))
    sched.on_new_request(first)
    assert sched.get_num_new_matched_tokens(first, 0) == (0, False)
    blocks = _Blocks()
    table, checkpoints = [], {}
    for gid in range(len(kvc.kv_cache_groups)):
        if gid in ring_groups:
            table.append(blocks.take(1))
        elif is_mamba(gid):
            ckpt, running = blocks.take(2)  # align, no MTP: checkpoint + running
            checkpoints[gid] = ckpt
            table.append([0, 0, ckpt, running])
        else:
            table.append(blocks.take(4))
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NS(req_id="r1", block_ids=tuple(table))]
    output.num_scheduled_tokens = {"r1": prompt}
    output.total_num_scheduled_tokens = prompt
    output.kv_connector_block_state = KVConnectorBlockState(
        req_ids={"r1"},
        resolve_block_ids=lambda req_id: tuple(table),
        boundary_state_offloads={"r1": [(gid, checkpoints[gid], 3 * BLOCK) for gid in mamba]},
    )
    meta = sched.build_connector_meta(output)
    assert isinstance(meta, RaidenOffloadingConnectorMetadata)
    assert meta.fence_job_ids and not meta.load_jobs and not meta.store_jobs
    stored = {}
    for job_id in meta.fence_job_ids:
        parked = sched._fence_pending[job_id]
        stored.update(zip(parked.keys, parked.device_block_ids))
    expected = {(gid, chunk): table[gid][chunk] for gid in attention for chunk in range(3)}
    expected.update({(gid, 2): checkpoints[gid] for gid in mamba})
    chunk_of = {bytes(h): i for i, h in enumerate(first.block_hashes)}
    got = {(_group_of(k), chunk_of[bytes(k[:-4])]): b for k, b in stored.items()}
    assert got == expected

    # A store launches only after all eight stages acknowledge the fence.
    seven = RaidenOffloadingWorkerMetadata(
        completed_jobs={}, fenced_jobs={job: STAGES - 1 for job in meta.fence_job_ids})
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=seven))
    assert not store.calls_named("save")
    last = RaidenOffloadingWorkerMetadata(
        completed_jobs={}, fenced_jobs={job: 1 for job in meta.fence_job_ids})
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=last))
    saved = [h for call in store.calls_named("save") for h in call.args[0]]
    assert len(saved) == len(expected)
    assert sched.telemetry["stores"] == len(meta.fence_job_ids)
    assert sched.telemetry["store_keys"] == len(expected)
    store.complete_save(saved)
    done = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert sorted(done.finished_store_job_ids) == sorted(meta.fence_job_ids)
    assert sched.telemetry["stores_done"] == len(meta.fence_job_ids)
    sched.update_connector_output(KVConnectorOutput(
        kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
            completed_jobs={job: STAGES for job in done.finished_store_job_ids})))

    # -- turn 2 extends the prompt (the multi-turn pattern): a host hit up to
    # the checkpoint boundary, the same 3 blocks an HBM hit would cover.
    second = _request("r2", first.prompt_token_ids + _tokens(2000, 1))
    assert second.block_hashes[:3] == first.block_hashes[:3]
    sched.on_new_request(second)
    assert sched.get_num_new_matched_tokens(second, BLOCK) == (2 * BLOCK, True)
    assert sched.get_num_new_matched_tokens(second, 0) == (3 * BLOCK, True)
    alloc = []
    for gid in range(len(kvc.kv_cache_groups)):
        if gid in ring_groups:
            ids = blocks.take(1)
        elif is_mamba(gid):
            ids = [0, 0, *blocks.take(1)]
        else:
            ids = blocks.take(3)
        alloc.append([KVCacheBlock(i, is_null=i == 0) for i in ids])
    sched.update_state_after_alloc(second, KVCacheBlocks(tuple(alloc)), 3 * BLOCK)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    (load_call,) = store.calls_named("load")
    hashes, device_blocks = load_call.args
    loaded = {}
    for sub_hash, block in zip(hashes, device_blocks):
        key, gid = _key_group_of_sub_hash(sub_hash, namespace)
        loaded[(gid, chunk_of[bytes(key[:-4])])] = block
    expected_load = {(gid, c): alloc[gid][c].block_id for gid in attention for c in range(3)}
    expected_load.update({(gid, 2): alloc[gid][2].block_id for gid in mamba})
    assert loaded == expected_load
    assert sched.telemetry["loads"] == 1 and sched.telemetry["load_keys"] == len(expected_load)

    # No partial hit: with one checkpoint gone there is no boundary to load.
    store.complete_load(hashes)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    assert sched.telemetry["loads_done"] == 1 and sched.telemetry["failed_load_blocks"] == 0
    line = sched.telemetry_line()
    assert line.startswith("offload-tier telemetry: loads=1 ") and "mem_max=" in line
    victim = next(h for h in list(store.entries)
                  if _key_group_of_sub_hash(h, namespace)[1] == mamba[0])
    del store.entries[victim]
    third = _request("r3", first.prompt_token_ids + _tokens(300, 2))
    sched.on_new_request(third)
    assert sched.get_num_new_matched_tokens(third, 0) == (0, False)


def _auto_save_store():
    """FakeKVCacheStore whose saves land at once (a victim store is drained in the same step)."""
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore

    class AutoSave(FakeKVCacheStore):
        def save(self, block_hashes, dst_raiden_id=None):
            ok = super().save(block_hashes, dst_raiden_id)
            if ok:
                self.complete_save(list(block_hashes))
            return ok

    return AutoSave()


def test_victim_policy_stores_evicted_blocks_before_reuse_and_they_hit(monkeypatch):
    """QWEN4_EXP_HOST_TIER_POLICY=victim on the served PP8 groups: computing stores nothing; blocks evicted for
    reuse are stored under the keys a later lookup asks for, drained within the step, never echoed to the
    workers; a block freed too recently is not stored; and a turn-2 request host-hits the stored blocks."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput

    from vllm_torchtpu.offload import victim

    monkeypatch.setenv(victim.POLICY_ENV, "victim")
    victim.deactivate()
    _, layout, stages, kvc = pp8_pool(monkeypatch)
    store = _auto_save_store()
    sched, _ = _pp8_scheduler(monkeypatch, kvc, stages, store=store)
    try:
        tap = victim.active_tap()
        assert tap is sched._victim_tap and sched._victim_min_age == STAGES + 2
        cacheable = set(kvc.prefix_cacheable_group_ids)
        is_mamba = lambda gid: isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec)
        ring = {gid for gid, g in enumerate(kvc.kv_cache_groups)
                if isinstance(g.kv_cache_spec, CircularBufferSpec)}
        mamba = [g for g in sorted(cacheable) if is_mamba(g)]
        attention = [g for g in sorted(cacheable) if not is_mamba(g)]

        # turn 1, prefilled as in the write-through test: nothing is stored on compute
        prompt = 3 * BLOCK + 5
        first = _request("r1", _tokens(prompt, 0))
        sched.on_new_request(first)
        blocks = _Blocks()
        table, checkpoints = [], {}
        for gid in range(len(kvc.kv_cache_groups)):
            if gid in ring:
                table.append(blocks.take(1))
            elif is_mamba(gid):
                ckpt, running = blocks.take(2)
                checkpoints[gid] = ckpt
                table.append([0, 0, ckpt, running])
            else:
                table.append(blocks.take(4))
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [NS(req_id="r1", block_ids=tuple(table))]
        output.num_scheduled_tokens = {"r1": prompt}
        output.total_num_scheduled_tokens = prompt
        output.kv_connector_block_state = KVConnectorBlockState(
            req_ids={"r1"}, resolve_block_ids=lambda req_id: tuple(table),
            boundary_state_offloads={"r1": [(g, checkpoints[g], 3 * BLOCK) for g in mamba]},
        )
        meta = sched.build_connector_meta(output)
        assert not meta.fence_job_ids and not sched._fence_pending
        assert not store.calls_named("insert") and not store.calls_named("save")

        # HBM evicts turn 1's blocks for reuse (what BlockPool.get_new_blocks reports through the tap), freed
        # long ago; plus one block freed in this very step (young: a batch in flight may still write it).
        old = tap.step
        evicted = {(g, c): table[g][c] for g in attention for c in range(3)}
        evicted.update({(g, 2): checkpoints[g] for g in mamba})
        for (g, c), block_id in evicted.items():
            tap.on_freed([block_id])
        tap.step = old + sched._victim_min_age
        young = table[attention[0]][3]
        tap.on_freed([young])
        for (g, c), block_id in evicted.items():
            tap.pending.append(victim.Victim(block_id, make_block_hash_with_group_id(first.block_hashes[c], g),
                                             tap.freed_at[block_id]))
        young_hash = make_block_hash_with_group_id(_request("rx", _tokens(BLOCK, 9)).block_hashes[0],
                                                   attention[0])
        tap.pending.append(victim.Victim(young, young_hash, tap.freed_at[young]))
        done = sched.build_connector_meta(SchedulerOutput.make_empty())

        (save,) = store.calls_named("save")
        namespace = sched._raiden_manager._key_namespace
        chunk_of = {bytes(h): i for i, h in enumerate(first.block_hashes)}
        saved = {}
        for sub_hash in save.args[0]:
            key, g = _key_group_of_sub_hash(sub_hash, namespace)
            saved[(g, chunk_of[bytes(key[:-4])])] = store.entries[sub_hash].device_block_id
        assert saved == evicted                      # every victim, from its own block, under its lookup key
        assert young not in saved.values()
        assert not sched._raiden_manager._admissions  # drained before the metadata shipped
        assert not done.finished_store_job_ids        # never echoed to the workers
        tel = sched.telemetry
        assert tel["victim_stores"] == 1 and tel["victim_store_keys"] == len(evicted)
        assert tel["victim_young"] == 1 and tel["victim_seen"] == len(evicted) + 1

        # turn 2 extends turn 1: a host hit on the stored blocks, up to the checkpoint boundary
        second = _request("r2", first.prompt_token_ids + _tokens(2000, 1))
        sched.on_new_request(second)
        assert sched.get_num_new_matched_tokens(second, 0) == (3 * BLOCK, True)

        # evicted again later: already on the host, so not stored twice
        for (g, c), block_id in evicted.items():
            tap.pending.append(victim.Victim(block_id, make_block_hash_with_group_id(first.block_hashes[c], g),
                                             old))
        sched.build_connector_meta(SchedulerOutput.make_empty())
        assert len(store.calls_named("save")) == 1
        assert sched.telemetry["victim_resident"] == len(evicted)
    finally:
        victim.deactivate()


# -- the multi connector's contracts for a PD producer with a host tier ----------


class _Child:
    def __init__(self, count=None, matched=(0, False), finished=(False, None)):
        self.count, self.matched, self.finished = count, matched, finished
        self.allocs = []

    def get_finished_count(self):
        return self.count

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.matched

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.allocs.append(num_external_tokens)

    def request_finished_all_groups(self, request, block_ids):
        return self.finished


def _multi(children, world=STAGES):
    from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector

    m = object.__new__(TPUMultiConnector)
    m._connectors = children
    m._vllm_config = NS(parallel_config=NS(world_size=world))
    m._requests_to_connector = {}
    m._extra_async_saves = {}
    m._all_support_hma = True
    return m


def test_multi_finished_count_agrees_or_fails():
    # PP8 producer: the PD child counts one report per stage, the offload
    # child one echo per worker; both are the world size.
    assert _multi([_Child(STAGES), _Child(None)]).get_finished_count() is None
    assert _multi([_Child(0), _Child(None)]).get_finished_count() is None
    assert _multi([_Child(2), _Child(2)], world=8).get_finished_count() == 2
    with pytest.raises(ValueError, match="different numbers"):
        _multi([_Child(1), _Child(None)]).get_finished_count()


def test_multi_routes_host_loads_and_keeps_the_pd_hold():
    pd = _Child(count=STAGES, matched=(0, False), finished=(True, {"remote": "params"}))
    offload = _Child(matched=(3 * BLOCK, True), finished=(False, None))
    m = _multi([pd, offload])
    req = NS(request_id="r")
    # The producer never loads; the host tier's hit is the one reported.
    assert m.get_num_new_matched_tokens(req, BLOCK) == (3 * BLOCK, True)
    m.update_state_after_alloc(req, None, 3 * BLOCK)
    assert offload.allocs == [3 * BLOCK] and pd.allocs == [0]
    # The PD hold survives: blocks stay until the decode pulls.
    assert m.request_finished_all_groups(req, ([1],)) == (True, {"remote": "params"})


# -- shutdown against the real Raiden binding's eviction-callback signature ------


def test_shutdown_unregisters_eviction_callback_with_a_callable():
    """The image's KVCacheStore.set_eviction_callback(self, callback: object)
    rejects None with a TypeError (M: tpu-sync-torch 0.0.1.dev20261012084241,
    CPU probe in the served image; j-46084fb7 hit it at connector shutdown).
    A stub with that contract must see shutdown succeed, the manager's own
    callback detached and the job drain still reached."""
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager

    class BindingFaithfulStore(FakeKVCacheStore):
        def set_eviction_callback(self, callback) -> None:
            if callback is None or not callable(callback):
                raise TypeError(
                    "set_eviction_callback(): incompatible function arguments. The "
                    "following argument types are supported:\n    1. "
                    "set_eviction_callback(self, callback: object) -> None"
                )
            super().set_eviction_callback(callback)

    store = BindingFaithfulStore()
    manager = make_manager(store)
    assert store.eviction_callback == manager._on_eviction
    drained = []
    manager._admissions = {7: object()}
    manager.drain_jobs = lambda ids: drained.append(set(ids)) or []
    manager.shutdown()
    assert callable(store.eviction_callback)
    assert store.eviction_callback != manager._on_eviction
    store.simulate_evict([b"x" * 40])  # a late eviction is ignored, not queued
    assert manager.take_events() == []
    assert drained == [{7}]
