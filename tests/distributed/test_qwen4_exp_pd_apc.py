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
"""APC ownership through the real cache allocator and grouped PD descriptors."""

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_cache_hit_alignment_tokens,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request, RequestStatus

from tests.platforms.test_same_step_prefix_hits_patch import patched
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.distributed.kv_transfer import tpu_connector
from vllm_torchtpu.models.vllm.qwen4_exp.pd import TransferExtent, build_transfer_groups
from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
    Qwen4ExpRaidenConnector,
    QwenPagePlan,
)
from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import build_transfer_manifest
from vllm_torchtpu.models.vllm.qwen4_exp.pool_allocation import materialize_pool
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

from .test_qwen4_exp_pd_connector import qwen_worker
from .test_tpu_connector import _make_vllm_config


@pytest.fixture
def native_plan(monkeypatch):
    patched.__wrapped__(monkeypatch)
    for name, value in dict(
        TPU_KV_RESHARD_TRANSPORT="raiden",
        TPU_RAIDEN_PREFIX_AWARE_LOAD="1",
        TPU_RAIDEN_CONTROLLER_ADDRESS="source.test:27000",
        TPU_RAIDEN_TRANSFER_PARALLELISM="1",
        TPU_RAIDEN_ENGINE_ID="source-engine",
        TPU_RAIDEN_JOB_NAME="source",
        TPU_RAIDEN_QWEN35_ADMISSION="0",
        TPU_RAIDEN_KIMIK3_ADMISSION="0",
        TPU_RAIDEN_GLM_ADMISSION="0",
        TPU_RAIDEN_DSV4_ADMISSION="0",
        TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="1",
    ).items():
        monkeypatch.setenv(name, value)
    assert tpu_connector._prefix_aware_load_supported()
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ips", lambda: "127.0.0.1")
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ports", lambda: 9100)
    cfg = pool_config()
    cfg.model_config.hf_text_config.layer_types = [
        "linear_attention",
        "indexed_attention",
    ]
    cfg.model_config.hf_text_config.num_hidden_layers = 2
    cfg.model_config.hf_text_config.ple_layer_ids = [1]
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = cfg.cache_config.mamba_block_size = layout.block_size
    groups = get_kv_cache_groups(cfg, specs)
    pages = 1 + 2 * sum(
        group.kv_cache_spec.max_num_blocks_per_req(cfg, 3 * layout.block_size)
        for group in groups
    )
    plan = get_kv_cache_config_from_groups(
        cfg, groups, pages * layout.page_bytes * max(len(g.layer_names) for g in groups)
    )
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(plan, cfg)
    plan.hash_block_size = hash_block
    plan.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
        plan, cfg, scheduler_block, hash_block
    )
    return SimpleNamespace(
        cfg=cfg, layout=layout, plan=plan, block=scheduler_block, hash=hash_block
    )


def engine(native_plan, *, producer):
    cfg = _make_vllm_config(
        is_producer=producer, block_size=native_plan.block, enable_prefix_caching=True
    )
    cfg.cache_config = native_plan.cfg.cache_config
    cfg.kv_transfer_config.is_kv_consumer = not producer
    connector = Qwen4ExpRaidenConnector(
        cfg, KVConnectorRole.SCHEDULER, native_plan.plan
    )
    assert connector.connector_scheduler is not None
    return connector


def manager(native_plan):
    init_none_hash(sha256)
    return KVCacheManager(
        native_plan.plan,
        max_model_len=native_plan.cfg.model_config.max_model_len,
        scheduler_block_size=native_plan.block,
        hash_block_size=native_plan.hash,
        enable_caching=True,
    )


def request(native_plan, name, length):
    return Request(
        name,
        [i % 113 for i in range(length)],
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(native_plan.hash, sha256),
    )


def prime(native_plan, cache, length):
    req = request(native_plan, "prime", length + 1)
    scheduler = SimpleNamespace(
        cache_config=native_plan.cfg.cache_config,
        block_size=native_plan.block,
        hash_block_size=native_plan.hash,
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
    )
    assert Scheduler._mamba_block_aligned_split(scheduler, req, length + 1) == length
    assert cache.allocate_slots(req, length) is not None
    req.num_computed_tokens = length
    cache.cache_blocks(req, length)
    cache.new_step_starts()
    cache.free(req)


@pytest.mark.parametrize("warm", [False, True])
def test_real_allocator_suffix_targets_exclude_shared_pages_and_select_end_state(
    native_plan, monkeypatch, warm
):
    n = native_plan
    cache = manager(n)
    if warm:
        prime(n, cache, 2 * n.block)
    consumer = engine(n, producer=False)
    req = request(n, "destination", 3 * n.block + 1)
    hits, count, _ = cache.get_computed_blocks(req)
    assert count == (2 * n.block if warm else 0)
    req.kv_transfer_params = dict(
        req_id="source",
        uuid=71,
        num_tokens=3 * n.block,
        src_controller_address="source.test:27000",
        src_job_name="source",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        dst_tp_size=1,
        dst_dcp_size=1,
    )
    assert consumer.get_num_new_matched_tokens(req, count)[0] == 3 * n.block - count
    assert (
        cache.allocate_slots(
            req,
            0,
            count,
            hits,
            num_external_computed_tokens=3 * n.block - count,
            delay_cache_blocks=True,
        )
        is not None
    )
    owned = cache.get_blocks(req.request_id)
    before = tuple(tuple(row) for row in owned.get_block_ids())
    consumer.update_state_after_alloc(req, owned, 3 * n.block - count)
    load = consumer.connector_scheduler.reqs_to_load[req.request_id]
    assert load.skip_tokens == count and load.group_block_ids is not None
    groups = build_transfer_groups(n.plan.kv_cache_groups)
    expected_states = {}
    for group in groups:
        if group.extent is TransferExtent.REQUEST_STATE:
            column = (3 * n.block - 1) // group.block_size if group.checkpointed else 0
            expected_states[group.group_index] = before[group.group_index][column]
            assert load.group_block_ids[group.group_index] == [
                expected_states[group.group_index]
            ]
    materialized = materialize_pool(n.cfg, n.plan, n.layout, torch.device("cpu"))
    manifest = build_transfer_manifest(
        n.plan, materialized, index_tag_fields=n.layout.index_tag_fields
    )
    worker = qwen_worker(monkeypatch, producer=False)
    worker._qwen_page_plan = QwenPagePlan(groups, manifest)
    worker.vllm_config.model_config.max_model_len = 3 * n.block
    hints = worker._raiden_staging_blocks_per_pool(manifest, n.plan.kv_cache_groups)
    touched = {}
    for pool, group_index in worker._page_plan().pool_groups():
        touched.setdefault(pool.storage_index, set()).update(
            load.group_block_ids[group_index]
        )
    assert all(
        hint >= len(touched[pool.storage_index])
        for pool, hint in zip(manifest.pools, hints)
    )
    tags, ids, counts = worker._stage3_grouped_destination_pages(load, 3 * n.block)
    clips = worker._stage3_grouped_skip_bytes(load)
    offset = 0
    for (pool, gid), tag, size, clip in zip(
        worker._page_plan().pool_groups(), tags, counts, clips
    ):
        assert tag == pool.tag
        actual = ids[offset : offset + size]
        offset += size
        group = groups[gid]
        if group.extent is TransferExtent.TOKEN_PAGES:
            expected = list(before[gid][count // group.block_size :])
            assert actual == expected
            assert not set(actual).intersection(
                before[gid][: count // group.block_size]
            )
            assert clip == (count // group.block_size) * pool.block_stride_bytes
            assert all(
                block.ref_cnt == 1
                for block in owned.blocks[gid][count // group.block_size :]
            )
        else:
            assert actual == [expected_states[gid]] and clip == 0
    assert tuple(tuple(row) for row in owned.get_block_ids()) == before
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3PendingSubmit

    worker._stage3_source_facades["source.test:27000"] = (
        worker._raiden_controller_facade
    )
    monkeypatch.setattr(worker, "_stage3_source_work_units", lambda _: ["source"])
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: "HBM")
    pending = _Stage3PendingSubmit(
        req_meta=load,
        source_req_id="source",
        destination_req_id=req.request_id,
        uuid=71,
        num_tokens=3 * n.block,
        local_blocks=load.local_block_ids,
        dst_controller_address="destination.test:28000",
        dst_units=["destination"],
        defer_deadline=None,
        first_attempt_s=time.perf_counter(),
    )
    task = worker._stage_stage3_submit_task(pending, synchronous=True)
    assert task is not None
    assert task.call_kwargs["transfer_pool_tags"] == tags
    assert task.call_kwargs["dst_device_block_ids"] == ids
    assert task.call_kwargs["dst_block_counts"] == counts
    if warm:
        assert task.call_kwargs["dst_skip_bytes"] == clips
    else:
        assert "dst_skip_bytes" not in task.call_kwargs


def test_producer_committed_snapshot_uses_position_not_null_history_or_spare_tail(
    native_plan,
):
    n = native_plan
    cache = manager(n)
    req = request(n, "source", 3 * n.block)
    assert cache.allocate_slots(req, 3 * n.block) is not None
    req.num_computed_tokens = 3 * n.block
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}
    producer = engine(n, producer=True)
    tables = cache.get_blocks(req.request_id).get_block_ids()
    delayed, params = producer.request_finished_all_groups(req, tables)
    assert delayed and params["num_tokens"] == 3 * n.block
    send = producer.connector_scheduler.reqs_to_send[req.request_id]
    assert send.group_block_ids is not None
    for index, group in enumerate(n.plan.kv_cache_groups):
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert send.group_block_ids[index] == [tables[index][2]]
    producer.connector_scheduler.build_connector_meta()
    again, duplicate = producer.request_finished_all_groups(req, tables)
    assert again and duplicate == params
    assert not producer.connector_scheduler.build_connector_meta().reqs_to_send


def test_full_local_hit_releases_producer_generation_without_a_receive(native_plan):
    n = native_plan
    cache = manager(n)
    prime(n, cache, 2 * n.block)
    req = request(n, "full-hit", 2 * n.block + 1)
    blocks, count, _ = cache.get_computed_blocks(req)
    assert count == 2 * n.block
    req.kv_transfer_params = dict(
        req_id="source",
        uuid=93,
        num_tokens=count,
        src_controller_address="source.test:27000",
        src_job_name="source",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        dst_tp_size=1,
        dst_dcp_size=1,
    )
    consumer = engine(n, producer=False)
    assert consumer.get_num_new_matched_tokens(req, count) == (0, False)
    assert cache.allocate_slots(req, 1, count, blocks) is not None
    consumer.update_state_after_alloc(req, cache.get_blocks(req.request_id), 0)
    release = consumer.connector_scheduler.reqs_to_load[req.request_id]
    assert release.release_only and release.uuid == 93
    assert not release.local_block_ids and release.source_req_id == "source"


def test_native_fixture_reuses_only_scheduler_retained_producer_and_consumer_states(
    native_plan,
):
    n = native_plan
    producer = manager(n)
    # The proxy removes the final original prompt token on P.
    req = request(n, "producer-prime", 2 * n.block)
    scheduler = SimpleNamespace(
        cache_config=n.cfg.cache_config,
        block_size=n.block,
        hash_block_size=n.hash,
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
    )
    chunks = []
    while req.num_computed_tokens < req.num_prompt_tokens:
        scheduled = Scheduler._mamba_block_aligned_split(
            scheduler, req, req.num_prompt_tokens - req.num_computed_tokens
        )
        assert producer.allocate_slots(req, scheduled) is not None
        req.num_computed_tokens += scheduled
        producer.cache_blocks(req, req.num_computed_tokens)
        producer.new_step_starts()
        chunks.append(scheduled)
    producer.free(req)
    assert chunks == [n.block, n.block]
    repeated = request(n, "producer-repeat", 2 * n.block)
    assert producer.get_computed_blocks(repeated)[1] == n.block
    # An async D load commits the imported boundary before the final prompt
    # token runs. The next allocation must preserve that reusable snapshot.
    consumer = manager(n)
    req = request(n, "consumer-prime", 2 * n.block + 1)
    assert (
        consumer.allocate_slots(
            req, 0, num_external_computed_tokens=2 * n.block, delay_cache_blocks=True
        )
        is not None
    )
    req.num_computed_tokens = 2 * n.block
    consumer.cache_blocks(req, req.num_computed_tokens)
    consumer.new_step_starts()
    assert consumer.allocate_slots(req, 1) is not None
    req.num_computed_tokens += 1
    consumer.cache_blocks(req, req.num_computed_tokens)
    consumer.new_step_starts()
    consumer.free(req)
    repeated = request(n, "consumer-repeat", 2 * n.block + 1)
    assert consumer.get_computed_blocks(repeated)[1] == 2 * n.block


def test_apc_constructor_rejects_missing_clipping_or_unaligned_contract(
    native_plan, monkeypatch
):
    n = native_plan
    monkeypatch.setenv("TPU_RAIDEN_PREFIX_AWARE_LOAD", "0")
    with pytest.raises(ValueError, match="prefix-aware"):
        engine(n, producer=False)
    monkeypatch.setenv("TPU_RAIDEN_PREFIX_AWARE_LOAD", "1")
    n.plan.cache_hit_alignment_tokens = n.block - 1
    with pytest.raises(ValueError, match="token-page-aligned"):
        engine(n, producer=False)


def test_worker_dev_receipt_uses_initialized_plan_without_device_queries(
    native_plan, tmp_path
):
    n = native_plan
    groups = [
        dict(
            layer_names=list(group.layer_names),
            block_tokens=group.kv_cache_spec.block_size,
            prefix_cacheable=group.kv_cache_spec.prefix_cacheable,
        )
        for group in n.plan.kv_cache_groups
    ]
    payload = dict(
        hash_block_size=n.hash,
        cache_hit_alignment_tokens=n.block,
        num_blocks=n.plan.num_blocks,
        groups=groups,
    )
    target = tmp_path / "worker-plan.json"
    # Worker imports configure the runtime; isolate them from allocator-only tests.
    code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace as NS
from vllm_torchtpu.worker import tpu_worker
plan = json.loads(sys.argv[1])
plan['kv_cache_groups'] = [NS(layer_names=g['layer_names'],
    kv_cache_spec=NS(block_size=g['block_tokens'],
        prefix_cacheable=g['prefix_cacheable']))
    for g in plan['groups']]
del plan['groups']
tpu_worker.get_tensor_model_parallel_rank = lambda: 0
worker = NS(rank=5, parallel_config=NS(data_parallel_rank=5,
    data_parallel_size=8, tensor_parallel_size=1),
    cache_config=NS(enable_prefix_caching=True, mamba_cache_mode='align'),
    model_runner=NS(kv_cache_config=NS(**plan), max_model_len=int(sys.argv[3])))
Path(sys.argv[2]).write_text(json.dumps(tpu_worker.TPUWorker.get_kv_cache_plan(worker)))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(payload),
            str(target),
            str(n.cfg.model_config.max_model_len),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    receipt = json.loads(target.read_text())
    assert receipt["dp_rank"] == 5 and receipt["tp_rank"] == 0
    assert receipt["cache_hit_alignment_tokens"] == n.block
    assert receipt["prefix_caching"] and receipt["mamba_cache_mode"] == "align"
    assert receipt["groups"] == groups


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("warm", [False, True])
def test_loaded_state_reaches_actual_next_forward_without_overwriting_cached_state(
    native_plan, monkeypatch, inline, delta, warm
):
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    from tests.runner.test_mamba_state_seed_copies import _collect, _make_self, _sched

    n = native_plan
    cache = manager(n)
    # Keep a live reference to a real hashed prefix while receiving a new state.
    if warm:
        prime(n, cache, n.block)
    req = request(n, "next-forward", 2 * n.block + delta + 1)
    hits, count, _ = cache.get_computed_blocks(req)
    assert count == (n.block if warm else 0)
    published = 2 * n.block + delta
    monkeypatch.setattr(
        tpu_connector.dist_utils, "get_raiden_inline_load", lambda: inline
    )
    consumer = engine(n, producer=False)
    req.kv_transfer_params = dict(
        req_id="source",
        uuid=97,
        num_tokens=published,
        src_controller_address="source.test:27000",
        src_job_name="source",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        dst_tp_size=1,
        dst_dcp_size=1,
    )
    assert consumer.get_num_new_matched_tokens(req, count) == (
        published - count,
        not inline,
    )
    assert (
        cache.allocate_slots(
            req,
            int(inline),
            count,
            hits,
            num_external_computed_tokens=published - count,
            delay_cache_blocks=True,
        )
        is not None
    )
    consumer.update_state_after_alloc(
        req, cache.get_blocks(req.request_id), published - count
    )
    load = consumer.connector_scheduler.reqs_to_load[req.request_id]
    assert load.group_block_ids is not None
    state_groups = [
        (i, group.kv_cache_spec)
        for i, group in enumerate(n.plan.kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    source_words = torch.tensor([97, published, -71, 0x7FC00001], dtype=torch.int32)
    buffers = {}
    preserved = {}
    for gid, _ in state_groups:
        raw = torch.arange(n.plan.num_blocks * 4, dtype=torch.int32).reshape(-1, 4)
        cached = [block.block_id for block in hits.blocks[gid] if not block.is_null]
        preserved[gid] = [(bid, raw[bid].clone()) for bid in cached]
        destination = load.group_block_ids[gid][0]
        assert destination not in cached
        raw[destination] = source_words
        buffers[gid] = raw
    req.num_computed_tokens = published
    if not inline:
        assert cache.allocate_slots(req, 1) is not None
    tables = cache.get_blocks(req.request_id).get_block_ids()
    runner, _ = _make_self(
        [req.request_id],
        [published],
        [[tables[gid]] for gid, _ in state_groups],
        state_block_size=n.block,
        num_cols=max(len(tables[gid]) for gid, _ in state_groups),
    )
    runner._mamba_copy_plan = [
        (i + 1, [buffers[gid]]) for i, (gid, _) in enumerate(state_groups)
    ]
    _collect(runner, _sched({req.request_id: 1}))
    # These are the real runner's staged pairs, applied to CPU carriers here.
    for raws, src, dst in runner._pending_mamba_state_copies:
        for raw in raws:
            raw[dst.long()] = raw[src.long()]
    for gid, spec in state_groups:
        selected = mamba_get_block_table_tensor(
            torch.tensor([tables[gid]], dtype=torch.int32),
            torch.tensor([published + 1], dtype=torch.int32),
            spec,
            "align",
        )
        assert torch.equal(buffers[gid][selected.item()], source_words)
        for bid, old in preserved[gid]:
            assert torch.equal(buffers[gid][bid], old)


def _mtp_plan(monkeypatch, k: int):
    from vllm_torchtpu import _patch_vllm_qwen4_raiden_lookahead_hashes

    patched.__wrapped__(monkeypatch)
    _patch_vllm_qwen4_raiden_lookahead_hashes()
    for name, value in dict(
        TPU_KV_RESHARD_TRANSPORT="raiden",
        TPU_RAIDEN_PREFIX_AWARE_LOAD="1",
        TPU_RAIDEN_CONTROLLER_ADDRESS="source.test:27000",
        TPU_RAIDEN_TRANSFER_PARALLELISM="1",
        TPU_RAIDEN_ENGINE_ID="source-engine",
        TPU_RAIDEN_JOB_NAME="source",
        TPU_RAIDEN_QWEN35_ADMISSION="0",
        TPU_RAIDEN_KIMIK3_ADMISSION="0",
        TPU_RAIDEN_GLM_ADMISSION="0",
        TPU_RAIDEN_DSV4_ADMISSION="0",
        TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="1",
    ).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ips", lambda: "127.0.0.1")
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ports", lambda: 9100)
    cfg = pool_config()
    cfg.model_config.hf_text_config.layer_types = [
        "linear_attention",
        "indexed_attention",
    ]
    cfg.model_config.hf_text_config.num_hidden_layers = 2
    cfg.model_config.hf_text_config.ple_layer_ids = [1]
    cfg.speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=k,
        draft_tensor_parallel_size=1,
        use_eagle=lambda: True,
        use_multi_module_mtp=lambda: False,
        use_eagle_block_drop=lambda: True,
        disable_lookahead_block_hashes=False,
    )
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = cfg.cache_config.mamba_block_size = layout.block_size
    draft_names = {name for name, spec in specs.items() if spec.is_draft}
    groups = get_kv_cache_groups(cfg, specs)
    for group in groups:
        group.is_eagle_group = not draft_names.isdisjoint(group.layer_names)
    pages = 1 + 3 * sum(
        group.kv_cache_spec.max_num_blocks_per_req(cfg, 4 * layout.block_size)
        for group in groups
    )
    plan = get_kv_cache_config_from_groups(
        cfg, groups, pages * layout.page_bytes * max(len(g.layer_names) for g in groups)
    )
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(plan, cfg)
    plan.hash_block_size = hash_block
    plan.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
        plan, cfg, scheduler_block, hash_block
    )
    return SimpleNamespace(
        cfg=cfg,
        layout=layout,
        plan=plan,
        block=scheduler_block,
        hash=hash_block,
        k=k,
    )


def _mtp_engine(mtp_plan, *, producer: bool):
    cfg = _make_vllm_config(
        is_producer=producer, block_size=mtp_plan.block, enable_prefix_caching=True
    )
    cfg.cache_config = mtp_plan.cfg.cache_config
    cfg.kv_transfer_config.kv_connector = "Qwen4ExpRaidenConnector"
    cfg.kv_transfer_config.kv_role = "kv_producer" if producer else "kv_consumer"
    cfg.kv_transfer_config.is_kv_consumer = not producer
    cfg.speculative_config = mtp_plan.cfg.speculative_config
    cfg.num_prefill_lookahead_tokens = 1
    cfg.num_hash_lookahead_tokens = 1
    connector = Qwen4ExpRaidenConnector(cfg, KVConnectorRole.SCHEDULER, mtp_plan.plan)
    assert connector.connector_scheduler is not None
    return connector


def _mtp_manager(mtp_plan):
    init_none_hash(sha256)
    return KVCacheManager(
        mtp_plan.plan,
        max_model_len=mtp_plan.cfg.model_config.max_model_len,
        scheduler_block_size=mtp_plan.block,
        hash_block_size=mtp_plan.hash,
        enable_caching=True,
        use_eagle=False,
        num_hash_lookahead_tokens=1,
        num_prefill_lookahead=1,
    )


def _mtp_request(mtp_plan, name: str, length: int, tokens: list[int] | None = None):
    return Request(
        name,
        [i % 113 for i in range(length)] if tokens is None else list(tokens),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(
            mtp_plan.hash, sha256, num_hash_lookahead_tokens=1
        ),
    )


def _prime_mtp(mtp_plan, cache, length: int):
    req = _mtp_request(mtp_plan, "prime-mtp", length + 1)
    scheduler = SimpleNamespace(
        cache_config=mtp_plan.cfg.cache_config,
        block_size=mtp_plan.block,
        hash_block_size=mtp_plan.hash,
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=1,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
    )
    assert Scheduler._mamba_block_aligned_split(scheduler, req, length + 1) == length
    assert (
        cache.allocate_slots(req, length, num_lookahead_tokens=mtp_plan.k) is not None
    )
    req.num_computed_tokens = length
    cache.cache_blocks(req, length)
    cache.new_step_starts()
    cache.free(req)


@pytest.mark.parametrize("producer_k", [1, 3])
@pytest.mark.parametrize("warm", [False, True])
def test_mtp_apc_asymmetric_handoff_uses_lookahead_hashes_and_skip_bytes(
    monkeypatch, producer_k, warm
):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3PendingSubmit

    p_plan = _mtp_plan(monkeypatch, producer_k)
    d_plan = _mtp_plan(monkeypatch, 3)
    assert p_plan.block == d_plan.block
    p_groups = build_transfer_groups(p_plan.plan.kv_cache_groups)
    d_groups = build_transfer_groups(d_plan.plan.kv_cache_groups)
    assert [(g.extent, g.block_size, g.checkpointed) for g in p_groups] == [
        (g.extent, g.block_size, g.checkpointed) for g in d_groups
    ]

    # Producer side: multi-chunk aligned prefill retains draft_prompt_seeds and
    # boundary lookahead hashes until finish, ignoring throwaway sampled output.
    p_cache = _mtp_manager(p_plan)
    producer = _mtp_engine(p_plan, producer=True)
    src_req = _mtp_request(p_plan, "source-mtp", 3 * p_plan.block + 1)
    expected_tail = src_req.prompt_token_ids[-1]
    expected_p_hashes = list(src_req.block_hashes)
    assert len(expected_p_hashes) == 3
    producer.on_new_request(src_req)
    assert src_req.num_prompt_tokens == 3 * p_plan.block
    assert src_req.block_hashes == expected_p_hashes
    for _ in range(3):
        assert p_cache.allocate_slots(src_req, p_plan.block) is not None
        src_req.num_computed_tokens += p_plan.block
        sched_out = SchedulerOutput.make_empty()
        sched_out.num_scheduled_tokens = {src_req.request_id: p_plan.block}
        meta = producer.build_connector_meta(sched_out)
        assert meta.draft_prompt_seeds == {src_req.request_id: expected_tail}
    src_req.append_output_token_ids((expected_tail + 7) % 113)
    assert src_req.block_hashes == expected_p_hashes
    p_cache.cache_blocks(src_req, 3 * p_plan.block)
    src_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    p_tables = p_cache.get_block_ids_for_computed_tokens(
        src_req.request_id, 3 * p_plan.block
    )
    delay, xfer_params = producer.request_finished_all_groups(src_req, p_tables)
    assert delay and xfer_params is not None
    assert xfer_params["num_tokens"] == 3 * p_plan.block
    assert "qwen_mtp_prompt_digest" in xfer_params
    send = producer.connector_scheduler.reqs_to_send[src_req.request_id]
    assert send.group_block_ids is not None
    for gid, group in enumerate(p_plan.plan.kv_cache_groups):
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert len(p_tables[gid]) == 3 + producer_k
            assert send.group_block_ids[gid] == [p_tables[gid][2]]

    d_cache = _mtp_manager(d_plan)
    if warm:
        _prime_mtp(d_plan, d_cache, 2 * d_plan.block)
        # Exact repeat of 2*B + 1 hits 2*B locally on D (release_only, 0 external).
        full_hit_req = _mtp_request(d_plan, "full-hit-mtp", 2 * d_plan.block + 1)
        _, full_count, _ = d_cache.get_computed_blocks(full_hit_req)
        assert full_count == 2 * d_plan.block
        # Diverging at token 2*B misses block 1 since draft KV used token 2*B.
        div_tokens = [i % 113 for i in range(3 * d_plan.block + 1)]
        div_tokens[2 * d_plan.block] = (div_tokens[2 * d_plan.block] + 1) % 113
        divergent_req = _mtp_request(
            d_plan, "divergent-mtp", len(div_tokens), tokens=div_tokens
        )
        assert d_cache.get_computed_blocks(divergent_req)[1] == 0

    consumer = _mtp_engine(d_plan, producer=False)

    # Tail mismatch is rejected even when prefix blocks are warm in D's APC.
    tampered = _mtp_request(d_plan, "tampered-mtp", 3 * d_plan.block + 1)
    tampered.prompt_token_ids[-1] = (expected_tail + 1) % 113
    tampered.kv_transfer_params = dict(xfer_params)
    consumer.on_new_request(tampered)
    assert consumer.get_num_new_matched_tokens(tampered, 0) == (0, False)
    release = consumer.connector_scheduler.reqs_to_load[tampered.request_id]
    assert release.release_only and not release.report_completion
    consumer.build_connector_meta(SchedulerOutput.make_empty())

    # Continuation sharing [0 .. 2*B] reuses 2*B locally and pulls 1*B suffix.
    dst_req = _mtp_request(d_plan, "dest-mtp", 3 * d_plan.block + 1)
    dst_req.kv_transfer_params = dict(xfer_params)
    consumer.on_new_request(dst_req)
    hits, count, _ = d_cache.get_computed_blocks(dst_req)
    assert count == (2 * d_plan.block if warm else 0)
    external, async_load = consumer.get_num_new_matched_tokens(dst_req, count)
    assert (external, async_load) == (3 * d_plan.block - count, True)
    assert (
        d_cache.allocate_slots(
            dst_req,
            0,
            count,
            hits,
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
        )
        is not None
    )
    owned = d_cache.get_blocks(dst_req.request_id)
    before = tuple(tuple(row) for row in owned.get_block_ids())
    consumer.update_state_after_alloc(dst_req, owned, external)
    load = consumer.connector_scheduler.reqs_to_load[dst_req.request_id]
    assert load.skip_tokens == count and load.group_block_ids is not None
    for group in d_groups:
        if group.extent is TransferExtent.REQUEST_STATE:
            col = (
                (3 * d_plan.block - 1) // group.block_size if group.checkpointed else 0
            )
            if group.checkpointed:
                assert len(before[group.group_index]) == col + 1 + d_plan.k
            assert load.group_block_ids[group.group_index] == [
                before[group.group_index][col]
            ]

    materialized = materialize_pool(
        d_plan.cfg, d_plan.plan, d_plan.layout, torch.device("cpu")
    )
    manifest = build_transfer_manifest(
        d_plan.plan, materialized, index_tag_fields=d_plan.layout.index_tag_fields
    )
    worker = qwen_worker(monkeypatch, producer=False)
    worker._qwen_page_plan = QwenPagePlan(d_groups, manifest)
    worker.vllm_config.model_config.max_model_len = 4 * d_plan.block
    hints = worker._raiden_staging_blocks_per_pool(
        manifest, d_plan.plan.kv_cache_groups
    )
    touched = {}
    for pool, group_index in worker._page_plan().pool_groups():
        touched.setdefault(pool.storage_index, set()).update(
            load.group_block_ids[group_index]
        )
    assert all(
        hint >= len(touched[pool.storage_index])
        for pool, hint in zip(manifest.pools, hints)
    )
    tags, ids, counts = worker._stage3_grouped_destination_pages(load, 3 * d_plan.block)
    clips = worker._stage3_grouped_skip_bytes(load)
    token_groups_seen = 0
    offset = 0
    for (pool, gid), tag, size, clip in zip(
        worker._page_plan().pool_groups(), tags, counts, clips
    ):
        assert tag == pool.tag
        actual = ids[offset : offset + size]
        offset += size
        group = d_groups[gid]
        if group.extent is TransferExtent.TOKEN_PAGES:
            token_groups_seen += 1
            assert actual == list(before[gid][count // group.block_size :])
            assert not set(actual).intersection(
                before[gid][: count // group.block_size]
            )
            assert clip == (count // group.block_size) * pool.block_stride_bytes
        else:
            assert len(actual) == 1 and clip == 0
    # Both target self_attn and draft mtp.self_attn skip warm prefix bytes.
    assert token_groups_seen >= 2
    worker._stage3_source_facades["source.test:27000"] = (
        worker._raiden_controller_facade
    )
    monkeypatch.setattr(worker, "_stage3_source_work_units", lambda _: ["source"])
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: "HBM")
    pending = _Stage3PendingSubmit(
        req_meta=load,
        source_req_id="source-mtp",
        destination_req_id=dst_req.request_id,
        uuid=int(xfer_params["uuid"]),
        num_tokens=3 * d_plan.block,
        local_blocks=load.local_block_ids,
        dst_controller_address="destination.test:28000",
        dst_units=["destination"],
        defer_deadline=None,
        first_attempt_s=time.perf_counter(),
    )
    task = worker._stage_stage3_submit_task(pending, synchronous=True)
    assert task is not None
    if warm:
        assert task.call_kwargs["dst_skip_bytes"] == clips
    else:
        assert "dst_skip_bytes" not in task.call_kwargs


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("warm", [False, True])
def test_mtp_apc_loaded_state_seeds_spec_window_and_clears_recycled_read_offset(
    monkeypatch, inline, delta, warm
):
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    from tests.runner.test_mamba_state_seed_copies import _collect, _make_self, _sched

    n = _mtp_plan(monkeypatch, 3)
    cache = _mtp_manager(n)
    if warm:
        _prime_mtp(n, cache, n.block)
    req = _mtp_request(n, "mtp-next-forward", 2 * n.block + delta + 1)
    hits, count, _ = cache.get_computed_blocks(req)
    assert count == (n.block if warm else 0)
    published = 2 * n.block + delta
    monkeypatch.setattr(
        tpu_connector.dist_utils, "get_raiden_inline_load", lambda: inline
    )
    consumer = _mtp_engine(n, producer=False)
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import _prompt_digest

    req.kv_transfer_params = dict(
        req_id="source",
        uuid=197,
        num_tokens=published,
        src_controller_address="source.test:27000",
        src_job_name="source",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        dst_tp_size=1,
        dst_dcp_size=1,
        qwen_mtp_prompt_digest=_prompt_digest(req.prompt_token_ids),
    )
    consumer.on_new_request(req)
    assert consumer.get_num_new_matched_tokens(req, count) == (
        published - count,
        not inline,
    )
    assert (
        cache.allocate_slots(
            req,
            int(inline),
            count,
            hits,
            num_lookahead_tokens=n.k if inline else 0,
            num_external_computed_tokens=published - count,
            delay_cache_blocks=True,
        )
        is not None
    )
    consumer.update_state_after_alloc(
        req, cache.get_blocks(req.request_id), published - count
    )
    load = consumer.connector_scheduler.reqs_to_load[req.request_id]
    assert load.group_block_ids is not None
    state_groups = [
        (i, group.kv_cache_spec)
        for i, group in enumerate(n.plan.kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    source_words = torch.tensor([197, published, -71, 0x12345678], dtype=torch.int32)
    buffers = {}
    preserved = {}
    read_offsets = torch.zeros(n.plan.num_blocks + 8, dtype=torch.int32)
    for gid, _ in state_groups:
        raw = torch.arange((n.plan.num_blocks + 8) * 4, dtype=torch.int32).reshape(
            -1, 4
        )
        cached = [block.block_id for block in hits.blocks[gid] if not block.is_null]
        preserved[gid] = [(bid, raw[bid].clone()) for bid in cached]
        destination = load.group_block_ids[gid][0]
        assert destination not in cached
        raw[destination] = source_words
        # Simulate a recycled block retaining a non-zero MTP read offset.
        read_offsets[destination] = 2
        buffers[gid] = raw
    req.num_computed_tokens = published
    if not inline:
        assert cache.allocate_slots(req, 1, num_lookahead_tokens=n.k) is not None
    tables = cache.get_blocks(req.request_id).get_block_ids()
    runner, _ = _make_self(
        [req.request_id],
        [published],
        [[tables[gid]] for gid, _ in state_groups],
        state_block_size=n.block,
        num_cols=max(len(tables[gid]) for gid, _ in state_groups) + 2,
        ckpt_window=1 + n.k,
        read_offsets=read_offsets,
    )
    # First D forward: new owner must ignore stale recycled read_offset=2.
    runner._mamba_offset_seeded.clear()
    runner._mamba_copy_plan = [
        (i + 1, [buffers[gid]]) for i, (gid, _) in enumerate(state_groups)
    ]
    _collect(runner, _sched({req.request_id: 1}))
    for raws, src, dst in runner._pending_mamba_state_copies:
        for raw in raws:
            raw[dst.long()] = raw[src.long()]
    for gid, spec in state_groups:
        selected = mamba_get_block_table_tensor(
            torch.tensor([tables[gid]], dtype=torch.int32),
            torch.tensor([published + 1], dtype=torch.int32),
            spec,
            "align",
        )
        ckpt0 = int(selected[0, 0].item())
        assert torch.equal(buffers[gid][ckpt0], source_words)
        if delta == 0:
            assert int(read_offsets[ckpt0].item()) == 0
        for bid, old in preserved[gid]:
            assert torch.equal(buffers[gid][bid], old)
