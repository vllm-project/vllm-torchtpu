# SPDX-License-Identifier: Apache-2.0
"""Host tier in size classes (VLLM_TPU_OFFLOAD_TIER_CLASSES), on CPU.

Covers offload/tier_classes.py and its wiring in the connector.

- Routing: every key reaches only its class's store, for lookup, touch, prepare and the
  victim path.
- A split job: one child a class, terminal when every launched child is, a success only
  if all are.
- The pin cases of a split job: another request's pin on a finished child's key survives
  a sibling's failure; a prepare that fails part way gives back only its own pins; a
  refused launch leaves the parent waiting for the child in flight; a failed load
  reports every scheduler block ID and no finer device row.
- Failure mode 3: a class missing its partner gives a shorter hit or none, on the served
  PP8 groups.
- Failure mode 10: a drain that times out stops the engine before the step's metadata
  ships.
- The model's plan: classes, regions by stage and capacities at the served geometry;
  registration takes each class's regions from the plan and refuses tensors that do not
  follow it.
- Scope refusals, and the flag off: the same single store and the same store calls as
  before.
"""

import hashlib
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

from tests.offload.test_raiden_store_offload import (
    FAKE_STORE_TYPES,
    FakeBlockStatus,
    FakeKVCacheStore,
    _Entry,
)

GiB = 1024**3
SERVED_PARTITION = "6,6,6,6,6,6,7,5"
# 464 GiB, the tier reference 5 runs.
SERVED_BUDGET = 498_216_206_336
PAGE = 3200 * 1024
KEY_ROW = 409_600

# Groups of the served model (tests/runner fixtures): GDN A, B, C, PLE, main KV,
# compressed keys.
GDN_A, GDN_B, GDN_C, PLE, MAIN, KEYS = range(6)


def _ctx(req_id="req"):
    return ReqContext(req_id=req_id)


def _key(n, group):
    return make_offload_key(b"hash_%04d" % n, group)


def _sub(key):
    return bytes(key) + (0).to_bytes(4, "big")


def _resident(store, key):
    store.entries[_sub(key)] = _Entry(status=FakeBlockStatus.HOST)


# -- the composite manager over fake stores ------------------------------------------


def _tier(keys_rows_per_page=1, token_blocks=16, checkpoints=8):
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    from vllm_torchtpu.offload.tier_classes import (
        TierClass,
        TieredOffloadingManager,
        TierPlan,
    )

    one = ((0,),)
    classes = (
        TierClass("main", 0, "main", (MAIN,), one, 100),
        TierClass(
            "keys",
            1,
            "keys",
            (KEYS,),
            one,
            100 // keys_rows_per_page,
            keys_rows_per_page,
        ),
        TierClass("gdn-ab", 2, "gdn", (GDN_A, GDN_B), one, 100),
        TierClass("gdn-c", 3, "gdn", (GDN_C,), one, 100),
        TierClass("ple", 4, "ple", (PLE,), one, 100),
    )
    plan = TierPlan(
        classes, budget=0, ratio=2.0, token_blocks=token_blocks, checkpoints=checkpoints
    )
    stores = {c.name: FakeKVCacheStore() for c in classes}
    managers = [
        RaidenOffloadingManager(
            kernel_physical_blocks_capacity=plan.capacity(c),
            offload_logical_blocks_capacity=plan.capacity(c),
            device_block_size_factor=1,
            world_size=1,
            raiden_controller_port=0,
            raiden_job_name=f"test-{c.name}",
            device_row_stride=c.rows_per_page,
            track_held=True,
            store=stores[c.name],
            store_types=FAKE_STORE_TYPES,
        )
        for c in classes
    ]
    return TieredOffloadingManager(plan, managers), stores


def _hashes_by_store(stores, name):
    return {
        store_name: [h for call in store.calls_named(name) for h in call.args[0]]
        for store_name, store in stores.items()
    }


def test_routes_each_key_to_its_class_by_group():
    tier, stores = _tier()
    by_group = {g: _key(g, g) for g in (MAIN, KEYS, GDN_A, GDN_B, GDN_C, PLE)}
    owner = {
        MAIN: "main",
        KEYS: "keys",
        GDN_A: "gdn-ab",
        GDN_B: "gdn-ab",
        GDN_C: "gdn-c",
        PLE: "ple",
    }
    for group, key in by_group.items():
        _resident(stores[owner[group]], key)

    for group, key in by_group.items():
        before = {n: len(s.calls_named("lookup")) for n, s in stores.items()}
        assert tier.lookup(key, _ctx()).name == "HIT"
        after = {n: len(s.calls_named("lookup")) for n, s in stores.items()}
        assert {n for n in stores if after[n] != before[n]} == {owner[group]}

    # touch: lookup + release in the owning store only, with only its keys.
    for store in stores.values():
        store.calls.clear()
    tier.touch(list(by_group.values()), _ctx())
    released = _hashes_by_store(stores, "release")
    assert released == {
        "main": [_sub(by_group[MAIN])],
        "keys": [_sub(by_group[KEYS])],
        "gdn-ab": [_sub(by_group[GDN_A]), _sub(by_group[GDN_B])],
        "gdn-c": [_sub(by_group[GDN_C])],
        "ple": [_sub(by_group[PLE])],
    }

    # host_resident_prefix walks the classes in key order and stops at the first miss.
    missing = _key(99, KEYS)
    run = [_key(10, MAIN), _key(11, MAIN), missing, _key(12, MAIN)]
    for k in (run[0], run[1], run[3]):
        _resident(stores["main"], k)
    assert tier.host_resident_prefix(run) == 2

    # prepare_store keeps the job's order and marks each key in flight in its class.
    fresh = [_key(20, KEYS), _key(20, MAIN), _key(20, GDN_C), _key(21, MAIN)]
    out = tier.prepare_store(fresh, _ctx())
    assert out.keys_to_store == fresh and out.store_spec.keys == fresh
    assert tier.managers[0]._inflight_store_keys == {fresh[1], fresh[3]}
    assert tier.managers[1]._inflight_store_keys == {fresh[0]}
    assert tier.managers[3]._inflight_store_keys == {fresh[2]}
    tier.on_store_job_cancelled(fresh)
    assert not any(m._inflight_store_keys for m in tier.managers)

    # the victim path: resident keys are not stored again, the rest is, in order.
    tier.set_victim_mode(True)
    assert tier.prepare_store(fresh, _ctx()).keys_to_store == []
    victims = [by_group[KEYS], _key(30, KEYS), _key(30, PLE), by_group[MAIN]]
    assert tier.prepare_victim_store(victims) == [victims[1], victims[2]]

    # evictions come back through each class's callback.
    stores["keys"].simulate_evict([_sub(by_group[KEYS])])
    (event,) = tier.take_events()
    assert event.keys == [by_group[KEYS]]


def test_a_store_job_ends_when_every_child_does():
    tier, stores = _tier()
    keys = [_key(1, MAIN), _key(1, GDN_A), _key(1, GDN_B), _key(2, MAIN), _key(1, PLE)]
    ids = [10, 11, 12, 13, 14]
    tier.submit_store_job(5, keys, ids)
    inserted = {
        n: [
            (h, sl.device_block_id)
            for call in s.calls_named("insert")
            for h, sl in zip(call.args[0], call.args[1])
        ]
        for n, s in stores.items()
    }
    assert inserted == {
        "main": [(_sub(keys[0]), 10), (_sub(keys[3]), 13)],
        "keys": [],
        "gdn-ab": [(_sub(keys[1]), 11), (_sub(keys[2]), 12)],
        "gdn-c": [],
        "ple": [(_sub(keys[4]), 14)],
    }
    stores["main"].complete_save([_sub(keys[0]), _sub(keys[3])])
    stores["gdn-ab"].complete_save([_sub(keys[1]), _sub(keys[2])])
    assert tier.poll_finished_jobs() == [] and tier.is_job_launched(5)
    stores["ple"].complete_save([_sub(keys[4])])
    (done,) = tier.poll_finished_jobs()
    assert (done.job_id, done.op.value, done.success) == (5, "store", True)
    assert not tier.is_job_launched(5) and not tier.has_pending_work()
    # every pin the children took was spent by their saves
    assert not any(s.pinned_hashes() for s in stores.values())
    held = {c.name: m.held_stats()[0] for c, m in zip(tier.classes, tier.managers)}
    assert held == {"main": 2, "keys": 0, "gdn-ab": 2, "gdn-c": 0, "ple": 1}
    line = tier.telemetry_line(now=1000.0)
    assert (
        "tier[main] held=2/16 stored=2 evicted=0" in line
        and "tier[ple] held=1/8" in line
    )
    stores["main"].simulate_evict([_sub(keys[0])])
    line = tier.telemetry_line(now=1060.0)
    assert (
        "tier[main] held=1/16 stored=2 evicted=1 stores_per_min=0 evictions_per_min=1"
        in line
    )


def test_a_load_job_ends_when_every_child_does():
    tier, stores = _tier()
    keys = [_key(1, MAIN), _key(1, KEYS), _key(1, GDN_C)]
    for k, n in zip(keys, ("main", "keys", "gdn-c")):
        _resident(stores[n], k)
    spec = tier.prepare_load(keys, _ctx("r1"))
    assert spec.pinned and spec.keys == keys
    tier.submit_load_job(6, spec.keys, [21, 22, 23], "r1", pinned=spec.pinned)
    loads = {
        n: [c.args for c in s.calls_named("load")]
        for n, s in stores.items()
        if s.calls_named("load")
    }
    assert loads == {
        "main": [([_sub(keys[0])], [21])],
        "keys": [([_sub(keys[1])], [22])],
        "gdn-c": [([_sub(keys[2])], [23])],
    }
    stores["main"].complete_load([_sub(keys[0])])
    stores["gdn-c"].complete_load([_sub(keys[2])])
    assert tier.poll_finished_jobs() == []
    stores["keys"].complete_load([_sub(keys[1])])
    (done,) = tier.poll_finished_jobs()
    assert (done.job_id, done.success, done.req_id, done.failed_device_block_ids) == (
        6,
        True,
        "r1",
        [],
    )
    assert not any(s.pinned_hashes() for s in stores.values())


# -- the four pin cases of a split job
# --------------------------------------------------


def test_sibling_failure_keeps_another_requests_pin_on_a_finished_childs_key():
    # A load: the main child lands, a second request pins its key, the keys child fails.
    tier, stores = _tier()
    a, b = _key(1, MAIN), _key(1, KEYS)
    _resident(stores["main"], a)
    _resident(stores["keys"], b)
    spec = tier.prepare_load([a, b], _ctx("r1"))
    tier.submit_load_job(7, spec.keys, [11, 12], "r1")
    stores["main"].complete_load([_sub(a)])
    assert tier.poll_finished_jobs() == []
    assert tier.prepare_load([a], _ctx("r2")).pinned
    assert stores["main"].entries[_sub(a)].pin_count == 1
    releases = len(stores["main"].calls_named("release"))
    stores["keys"].complete_load([_sub(b)], success=False)
    (done,) = tier.poll_finished_jobs()
    assert not done.success and done.failed_device_block_ids == [11, 12]
    assert stores["main"].entries[_sub(a)].pin_count == 1  # r2's pin is still held
    assert len(stores["main"].calls_named("release")) == releases
    assert (
        stores["keys"].pinned_hashes() == {}
    )  # the failed child gave back its own pin

    # A store: the main save lands, a second request pins it, the keys save fails twice.
    a, b = _key(2, MAIN), _key(2, KEYS)
    tier.submit_store_job(8, [a, b], [21, 22])
    stores["main"].complete_save([_sub(a)])
    assert tier.poll_finished_jobs() == []
    assert tier.prepare_load([a], _ctx("r3")).pinned
    stores["keys"].complete_save([_sub(b)], success=False)
    assert tier.poll_finished_jobs() == []  # the keys child retries its save
    stores["keys"].complete_save([_sub(b)], success=False)
    (done,) = tier.poll_finished_jobs()
    assert (done.job_id, done.success) == (8, False)
    assert stores["main"].entries[_sub(a)].pin_count == 1
    # a host entry the successful child wrote stays
    assert stores["main"].entries[_sub(a)].status == FakeBlockStatus.HOST_AND_HBM


def test_prepare_that_fails_part_way_gives_back_only_its_own_pins():
    tier, stores = _tier()
    a, b = _key(3, MAIN), _key(3, KEYS)
    _resident(stores["main"], a)  # the keys class lost its entry after the lookup
    assert tier.prepare_load([a], _ctx("r0")).pinned  # another request's pin
    spec = tier.prepare_load([a, b], _ctx("r1"))
    assert not spec.pinned and spec.keys == [a, b]
    assert stores["main"].entries[_sub(a)].pin_count == 1
    assert stores["keys"].pinned_hashes() == {}
    tier.submit_load_job(9, spec.keys, [31, 32], "r1", pinned=spec.pinned)
    (done,) = tier.poll_finished_jobs()
    assert not done.success and done.failed_device_block_ids == [31, 32]
    assert not any(s.calls_named("load") for s in stores.values())
    assert not tier.is_job_launched(9)


def test_a_refused_launch_leaves_the_parent_waiting_for_the_child_in_flight():
    tier, stores = _tier()
    a, b = _key(4, MAIN), _key(4, KEYS)
    _resident(stores["main"], a)
    _resident(stores["keys"], b)
    stores["keys"].fail_load = True
    spec = tier.prepare_load([a, b], _ctx("r1"))
    tier.submit_load_job(10, spec.keys, [41, 42], "r1")
    assert tier.poll_finished_jobs() == [] and tier.is_job_launched(10)
    assert stores["keys"].pinned_hashes() == {}  # the refused child finalized itself
    assert (
        stores["main"].entries[_sub(a)].pin_count == 1
    )  # the running child still holds its pin
    stores["main"].complete_load([_sub(a)])
    (done,) = tier.poll_finished_jobs()
    assert not done.success and done.failed_device_block_ids == [41, 42]
    assert not any(s.pinned_hashes() for s in stores.values())

    # The same for a store whose keys child is refused at insert.
    c, d = _key(5, MAIN), _key(5, KEYS)
    stores["keys"].fail_insert = True
    tier.submit_store_job(11, [c, d], [51, 52])
    assert tier.poll_finished_jobs() == [] and tier.is_job_launched(11)
    stores["main"].complete_save([_sub(c)])
    (done,) = tier.poll_finished_jobs()
    assert (done.job_id, done.success) == (11, False)


def test_a_failed_load_reports_every_scheduler_block_id_and_no_device_row():
    tier, stores = _tier(keys_rows_per_page=8)
    keys = [_key(1, MAIN), _key(1, KEYS), _key(2, MAIN), _key(2, KEYS)]
    for k in keys:
        _resident(stores["keys" if k[-1] == KEYS else "main"], k)
    spec = tier.prepare_load(keys, _ctx("r1"))
    tier.submit_load_job(12, spec.keys, [3, 7, 5, 9], "r1")
    # Only the keys class's call into Raiden sees the device rows (8b).
    (keys_load,) = stores["keys"].calls_named("load")
    assert keys_load.args == ([_sub(keys[1]), _sub(keys[3])], [56, 72])
    (main_load,) = stores["main"].calls_named("load")
    assert main_load.args == ([_sub(keys[0]), _sub(keys[2])], [3, 5])
    stores["keys"].complete_load([_sub(keys[1]), _sub(keys[3])])
    stores["main"].complete_load([_sub(keys[0])], success=False)
    stores["main"].complete_load([_sub(keys[2])])
    (done,) = tier.poll_finished_jobs()
    assert not done.success and done.failed_device_block_ids == [3, 7, 5, 9]

    # A store in the keys class inserts and saves device row 8b too, and succeeds.
    k = _key(3, KEYS)
    tier.submit_store_job(13, [k], [6])
    (insert,) = stores["keys"].calls_named("insert")
    assert insert.args[1][0].device_block_id == 48
    stores["keys"].complete_save([_sub(k)])
    (done,) = tier.poll_finished_jobs()
    assert done.success


def test_a_child_that_never_completes_times_out_the_drain(monkeypatch):
    from vllm_torchtpu.offload import raiden_store

    monkeypatch.setattr(raiden_store, "_DRAIN_TIMEOUT_S", 0.02)
    tier, stores = _tier()
    a, b = _key(6, MAIN), _key(6, KEYS)
    tier.submit_store_job(14, [a, b], [61, 62])
    stores["main"].complete_save([_sub(a)])
    assert tier.drain_jobs({14}) == []
    assert tier.is_job_launched(14)
    assert tier.managers[1]._admissions[14].poisoned
    # once the stuck transfer resolves, the parent ends as a failure
    stores["keys"].complete_save([_sub(b)])
    (done,) = tier.poll_finished_jobs()
    assert (done.job_id, done.success) == (14, False)


# -- the served PP8 groups
# --------------------------------------------------------------


def _served(monkeypatch, classes=True, key_rows=False):
    from tests.runner.test_qwen4_exp_host_tier import pp8_pool

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", SERVED_PARTITION)
    if classes:
        monkeypatch.setenv("VLLM_TPU_OFFLOAD_TIER_CLASSES", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_CLASSES", raising=False)
    if key_rows:
        monkeypatch.setenv("VLLM_TPU_OFFLOAD_TIER_KEY_ROWS", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_KEY_ROWS", raising=False)
    return pp8_pool(monkeypatch)


def _vllm_config(cfg):
    """The pool config the fixtures plan with, plus what the offloading spec reads."""
    cfg.model_config.is_hybrid = True
    cfg.model_config.architecture = "Qwen4ExpForConditionalGeneration"
    cfg.model_config.model = "Qwen/Qwen3.8-Flash-Next-FP8"
    cfg.model_config.revision = None
    cfg.model_config.quantization = "fp8"
    cfg.cache_config.prefix_caching_hash_algo = "sha256"
    cfg.cache_config.prefix_cache_retention_interval = None
    for name, value in dict(
        world_size=8,
        decode_context_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        cp_kv_cache_interleave_size=1,
        pipeline_parallel_size=8,
    ).items():
        setattr(cfg.parallel_config, name, value)
    cfg.kv_events_config = None
    return cfg


def _spec(monkeypatch, cfg, stages, kvc, cpu_bytes=SERVED_BUDGET, **extra):
    from tests.runner.test_qwen4_exp_host_tier import _pp8_offloading_config
    from vllm_torchtpu.offload import raiden_store

    monkeypatch.setattr(raiden_store, "_all_ranks_kv_bytes_per_block", None)
    monkeypatch.setattr(raiden_store, "_worker_kv_cache_configs", None, raising=False)
    raiden_store.record_all_ranks_kv_bytes_per_block(stages)
    config = _pp8_offloading_config(kvc, stages, cpu_bytes)
    config.extra_config.update(extra)
    return raiden_store.TPURaidenStoreOffloadingSpec(config, _vllm_config(cfg), kvc)


def _tier_scheduler(
    monkeypatch, stores=None, cpu_bytes=64 * GiB, victim_policy=False, single_store=None
):
    """The connector scheduler on the served groups, tier classes or one store."""
    from vllm_torchtpu.offload import victim
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    if victim_policy:
        monkeypatch.setenv(victim.POLICY_ENV, "victim")
        victim.deactivate()
    cfg, layout, stages, kvc = _served(monkeypatch, classes=single_store is None)
    spec = _spec(monkeypatch, cfg, stages, kvc, cpu_bytes=cpu_bytes)
    if single_store is None:
        stores = stores or [FakeKVCacheStore() for _ in range(5)]
        spec._manager = spec.build_tier_manager(stores, FAKE_STORE_TYPES)
    else:
        from tests.offload.test_raiden_store_offload import make_manager

        spec._manager = make_manager(
            single_store,
            device_block_size_factor=1,
            num_blocks=spec.offload_logical_blocks_capacity,
            world_size=8,
            key_namespace=spec.key_namespace,
        )
    sched = TPURaidenOffloadingScheduler(spec, spec.vllm_config, kvc)
    return sched, kvc, stores


def _turn_one(sched, kvc, boundaries=(3,)):
    """Prefills a 3-block-plus prompt; its blocks and checkpoints go to the store."""
    from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput

    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _Blocks, _tokens

    prompt = 3 * BLOCK + 5
    first = request("r1", _tokens(prompt, 0))
    sched.on_new_request(first)
    blocks = _Blocks()
    table, offloads = [], []
    for gid, group in enumerate(kvc.kv_cache_groups):
        if isinstance(group.kv_cache_spec, CircularBufferSpec):
            table.append(blocks.take(1))
        elif isinstance(group.kv_cache_spec, MambaSpec):
            ids = [0, 0, 0, 0]
            for boundary in boundaries:
                ids[boundary - 1] = blocks.take(1)[0]
                offloads.append((gid, ids[boundary - 1], boundary * BLOCK))
            table.append(ids)
        else:
            table.append(blocks.take(4))
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NS(req_id="r1", block_ids=tuple(table))]
    output.num_scheduled_tokens = {"r1": prompt}
    output.total_num_scheduled_tokens = prompt
    output.kv_connector_block_state = KVConnectorBlockState(
        req_ids={"r1"},
        resolve_block_ids=lambda req_id: tuple(table),
        boundary_state_offloads={"r1": offloads},
    )
    return first, table, sched.build_connector_meta(output)


def _ack_fences(sched, meta):
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.offload.raiden_connector import RaidenOffloadingWorkerMetadata

    acks = RaidenOffloadingWorkerMetadata(
        completed_jobs={}, fenced_jobs={j: 8 for j in meta.fence_job_ids}
    )
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=acks))


def test_a_class_missing_its_partner_gives_a_shorter_hit_or_none(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _tokens

    sched, kvc, stores = _tier_scheduler(monkeypatch)
    first, table, meta = _turn_one(sched, kvc, boundaries=(2, 3))
    _ack_fences(sched, meta)
    for store in stores:
        store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    sched.build_connector_meta(SchedulerOutput.make_empty())
    assert not sched._raiden_manager.has_pending_work()
    # every class holds its part: main and keys 3 blocks, GDN A+B 2 x 2 checkpoints, GDN
    # C and PLE 2
    assert [len(s.entries) for s in stores] == [3, 3, 4, 2, 2]

    prefix = first.prompt_token_ids
    full = request("r2", prefix + _tokens(2000, 1))
    sched.on_new_request(full)
    assert sched.get_num_new_matched_tokens(full, 0) == (3 * BLOCK, True)

    # The keys class lost block 2's entry: the attention run is two blocks, and the hit
    # stops at the checkpoint there. Block 2's main KV is never loaded without its keys.
    namespace = sched._raiden_manager._key_namespace
    lost = _sub_of(first.block_hashes[2], KEYS, namespace)
    del stores[1].entries[lost]
    shorter = request("r3", prefix + _tokens(300, 2))
    sched.on_new_request(shorter)
    assert sched.get_num_new_matched_tokens(shorter, 0) == (2 * BLOCK, True)
    assert sched.host_resident_tokens(shorter, 0) == 2 * BLOCK

    # Block 1's keys gone as well: no checkpoint at boundary 1, so no hit.
    del stores[1].entries[_sub_of(first.block_hashes[1], KEYS, namespace)]
    none = request("r4", prefix + _tokens(300, 3))
    sched.on_new_request(none)
    assert sched.get_num_new_matched_tokens(none, 0) == (0, False)


def _sub_of(block_hash, group, namespace):
    return (
        namespace
        + bytes(make_offload_key(bytes(block_hash), group))
        + (0).to_bytes(4, "big")
    )


class _AutoSave(FakeKVCacheStore):
    def save(self, block_hashes, dst_raiden_id=None):
        ok = super().save(block_hashes, dst_raiden_id)
        if ok:
            self.complete_save(list(block_hashes))
        return ok


def _victims_of_turn_one(sched, first, table, kvc):
    """Report turn 1's blocks as evicted for reuse, freed long ago."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    from vllm_torchtpu.offload import victim

    tap = sched._victim_tap
    old = tap.step
    evicted = {}
    for gid in kvc.prefix_cacheable_group_ids:
        if isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec):
            evicted[(gid, 2)] = table[gid][2]
        else:
            evicted.update({(gid, c): table[gid][c] for c in range(3)})
    for block_id in evicted.values():
        tap.on_freed([block_id])
    tap.step = old + sched._victim_min_age
    for (g, c), block_id in evicted.items():
        tap.pending.append(
            victim.Victim(
                block_id,
                make_block_hash_with_group_id(first.block_hashes[c], g),
                tap.freed_at[block_id],
            )
        )
    return evicted


def test_a_drain_that_times_out_stops_the_engine_before_the_metadata_ships(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.offload import raiden_store, victim

    monkeypatch.setattr(raiden_store, "_DRAIN_TIMEOUT_S", 0.05)
    # Every class lands its saves at once except the keys class, which never completes.
    stores = [_AutoSave(), FakeKVCacheStore(), _AutoSave(), _AutoSave(), _AutoSave()]
    sched, kvc, _ = _tier_scheduler(monkeypatch, stores=stores, victim_policy=True)
    try:
        first, table, _ = _turn_one(sched, kvc)
        _victims_of_turn_one(sched, first, table, kvc)
        built = []
        with pytest.raises(
            RuntimeError, match="did not finish within the drain timeout"
        ):
            built.append(sched.build_connector_meta(SchedulerOutput.make_empty()))
        assert not built  # no metadata left the scheduler
        assert stores[1].calls_named("save") and stores[1]._saving
    finally:
        victim.deactivate()


def test_flag_off_a_drain_timeout_poisons_and_the_step_goes_on(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.offload import raiden_store, victim

    monkeypatch.setattr(raiden_store, "_DRAIN_TIMEOUT_S", 0.05)
    store = FakeKVCacheStore()  # never completes a save
    sched, kvc, _ = _tier_scheduler(monkeypatch, victim_policy=True, single_store=store)
    try:
        first, table, _ = _turn_one(sched, kvc)
        _victims_of_turn_one(sched, first, table, kvc)
        meta = sched.build_connector_meta(SchedulerOutput.make_empty())
        assert meta is not None and sched.telemetry["victim_store_failed"] == 1
        (admission,) = sched._raiden_manager._admissions.values()
        assert admission.poisoned
    finally:
        victim.deactivate()


def test_a_flush_drain_that_times_out_stops_the_engine(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.offload import raiden_store

    monkeypatch.setattr(raiden_store, "_DRAIN_TIMEOUT_S", 0.05)
    stores = [_AutoSave(), FakeKVCacheStore(), _AutoSave(), _AutoSave(), _AutoSave()]
    sched, kvc, _ = _tier_scheduler(monkeypatch, stores=stores)
    first, table, meta = _turn_one(sched, kvc)
    _ack_fences(sched, meta)
    assert stores[1]._saving
    output = SchedulerOutput.make_empty()
    output.preempted_req_ids = {"r1"}
    with pytest.raises(RuntimeError, match="did not finish within the drain timeout"):
        sched.build_connector_meta(output)


def test_victim_stores_split_by_class_and_hit_again(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _tokens
    from vllm_torchtpu.offload import victim

    stores = [_AutoSave() for _ in range(5)]
    sched, kvc, _ = _tier_scheduler(monkeypatch, stores=stores, victim_policy=True)
    try:
        first, table, _ = _turn_one(sched, kvc)
        evicted = _victims_of_turn_one(sched, first, table, kvc)
        meta = sched.build_connector_meta(SchedulerOutput.make_empty())
        assert (
            not meta.finished_store_job_ids
            and not sched._raiden_manager.has_pending_work()
        )
        namespace = sched._raiden_manager._key_namespace
        saved = {}
        for name, store in zip(("main", "keys", "gdn-ab", "gdn-c", "ple"), stores):
            for sub_hash in [h for c in store.calls_named("save") for h in c.args[0]]:
                group = int.from_bytes(sub_hash[-8:-4], "big")
                saved[group] = saved.get(group, set()) | {name}
                assert store.entries[sub_hash].device_block_id in evicted.values()
        assert saved == {
            MAIN: {"main"},
            KEYS: {"keys"},
            GDN_A: {"gdn-ab"},
            GDN_B: {"gdn-ab"},
            GDN_C: {"gdn-c"},
            PLE: {"ple"},
        }
        assert sched.telemetry["victim_stores"] == 1
        assert sched.telemetry["victim_store_keys"] == len(evicted)
        line = sched.telemetry_line()
        assert (
            "tier[main] held=3/" in line
            and "tier[ple] held=1/" in line
            and "oldest_age_s=" in line
        )
        second = request("r2", first.prompt_token_ids + _tokens(2000, 1))
        sched.on_new_request(second)
        assert sched.get_num_new_matched_tokens(second, 0) == (3 * BLOCK, True)
        assert namespace == sched._raiden_manager.managers[0]._key_namespace
    finally:
        victim.deactivate()


# -- the model's plan, capacities and registration
# -----------------------------------------


def test_classes_follow_the_models_region_plan(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import tier_group_kind
    from vllm_torchtpu.offload.tier_classes import build_tier_plan, check_served_plan

    _, _, stages, kvc = _served(monkeypatch)
    plan = build_tier_plan(
        stages, kvc.prefix_cacheable_group_ids, tier_group_kind, SERVED_BUDGET, 4.7
    )
    # The design's table, stages 0 to 7.
    assert {c.name: [len(r) for r in c.worker_regions] for c in plan.classes} == {
        "main": [1, 2, 1, 2, 1, 2, 1, 2],
        "keys": [1, 2, 1, 2, 1, 2, 1, 2],
        "gdn-ab": [2, 1, 2, 1, 2, 1, 2, 1],
        "gdn-c": [1, 2, 1, 2, 1, 2, 2, 1],
        "ple": [1, 0, 0, 0, 0, 0, 0, 0],
    }
    assert {c.name: c.groups for c in plan.classes} == {
        "main": (MAIN,),
        "keys": (KEYS,),
        "gdn-ab": (GDN_A, GDN_B),
        "gdn-c": (GDN_C,),
        "ple": (PLE,),
    }
    assert [c.index for c in plan.classes] == [0, 1, 2, 3, 4]
    check_served_plan(plan)
    # A plan that is not the served one is refused.
    partial = build_tier_plan(
        stages, (GDN_A, GDN_B, GDN_C, MAIN, KEYS), tier_group_kind, SERVED_BUDGET, 4.7
    )
    with pytest.raises(ValueError, match="not the served prefill's"):
        check_served_plan(partial)


def test_capacities_at_the_served_geometry(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import tier_group_kind
    from vllm_torchtpu.offload.tier_classes import build_tier_plan

    _, _, stages, kvc = _served(monkeypatch)
    groups = kvc.prefix_cacheable_group_ids
    for key_row, entry_bytes, token_bytes, (m, c) in (
        (
            None,
            {
                "main": 12 * PAGE,
                "keys": 12 * PAGE,
                "gdn-ab": 12 * PAGE,
                "gdn-c": 12 * PAGE,
                "ple": PAGE,
            },
            78_643_200,
            (4765, 1014),
        ),
        (
            KEY_ROW,
            {
                "main": 12 * PAGE,
                "keys": 12 * KEY_ROW,
                "gdn-ab": 12 * PAGE,
                "gdn-c": 12 * PAGE,
                "ple": PAGE,
            },
            44_236_800,
            (7111, 1513),
        ),
    ):
        plan = build_tier_plan(
            stages, groups, tier_group_kind, SERVED_BUDGET, 4.7, key_row_bytes=key_row
        )
        assert {x.name: x.bytes_per_entry for x in plan.classes} == entry_bytes
        checkpoint_bytes = 2 * 12 * PAGE + 12 * PAGE + PAGE
        assert checkpoint_bytes == 121_241_600
        assert (plan.token_blocks, plan.checkpoints) == (m, c)
        assert {x.name: plan.capacity(x) for x in plan.classes} == {
            "main": m,
            "keys": m,
            "gdn-ab": 2 * c,
            "gdn-c": c,
            "ple": c,
        }
        used = sum(plan.capacity(x) * x.bytes_per_entry for x in plan.classes)
        assert used == m * token_bytes + c * checkpoint_bytes
        # what flooring C, then M = int(4.7 C), can leave unused
        assert (
            SERVED_BUDGET - (4.7 * token_bytes + checkpoint_bytes) - token_bytes
            < used
            <= SERVED_BUDGET
        )
    # The ratio is a setting: 6 token-blocks a checkpoint.
    plan = build_tier_plan(
        stages, groups, tier_group_kind, SERVED_BUDGET, 6.0, key_row_bytes=KEY_ROW
    )
    assert plan.token_blocks == int(6.0 * plan.checkpoints)


def _register(stage_config, caches, vllm_config):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    captured = []
    connector = NS(
        _vllm_config=vllm_config,
        connector_worker=NS(kv_cache_config=stage_config, _init_worker=captured.append),
    )
    TPURaidenOffloadingConnector.register_kv_caches(connector, caches)
    return captured


@pytest.mark.parametrize("key_rows", [False, True])
def test_each_stage_registers_its_classes_regions_from_the_plan(monkeypatch, key_rows):
    from unittest.mock import patch

    from vllm.distributed import parallel_state

    from tests.runner.test_qwen4_exp_host_tier import _stage_caches
    from tests.runner.test_qwen4_exp_offload import _fake_tpu_sync
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    cfg, _, stages, kvc = _served(monkeypatch, key_rows=key_rows)
    _spec(monkeypatch, cfg, stages, kvc)  # records the per-worker configs
    nb = stages[0].num_blocks
    shape = (nb, 3200, 1, 4, 256)
    seen_ple = []
    for rank, stage in enumerate(stages):
        spec = raiden_store.TPURaidenStoreOffloadingSpec(
            _spec_config(kvc, stages), _vllm_config(cfg), stage
        )
        plan = spec.tier_plan
        regions = [
            torch.zeros(shape, dtype=torch.float8_e4m3fn)
            for _ in range(pool_regions(stage))
        ]
        caches, index = _stage_caches(stage, regions)
        (canonical,) = _register(stage, caches, spec.vllm_config)
        # each group refers to the regions its own layers occupy
        for gid, group in enumerate(stage.kv_cache_groups):
            assert sorted(
                r.tensor_idx for r in canonical.group_data_refs[gid]
            ) == sorted({index[n] for n in group.layer_names})
        registered = []
        modules, sys = _fake_tpu_sync(registered)
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                RaidenStoreOffloadingWorker, "_controller_reachable", return_value=True
            ),
            patch.object(
                parallel_state, "get_world_group", return_value=NS(rank_in_group=rank)
            ),
        ):
            worker = spec.get_worker(canonical)
            for class_worker in worker.workers.values():
                class_worker._register_thread.join(timeout=10)
        assert {c.name for c in plan.classes if c.worker_regions[rank]} == set(
            worker.workers
        )
        by_port = {
            kw["raiden_controller_address"]: (tensors, kw) for tensors, kw in registered
        }
        for c in plan.classes:
            address = f"127.0.0.1:{31515 + c.index}"
            if not c.worker_regions[rank]:
                assert address not in by_port
                continue
            tensors, kw = by_port[address]
            assert kw["host_blocks_to_allocate"] == plan.capacity(c)
            ptrs = [t[0].untyped_storage().data_ptr() for t in tensors]
            assert ptrs == [
                regions[r].untyped_storage().data_ptr() for r in c.worker_regions[rank]
            ]
            for (t,) in tensors:
                if c.name == "keys" and key_rows:
                    # 8 rows of the keys' leading 409,600 bytes a page, over the whole
                    # storage at offset 0
                    assert t.shape == (nb * 8, KEY_ROW) and t.storage_offset() == 0
                else:
                    assert t.shape == shape
            if c.name == "ple":
                seen_ple.append(rank)
        worker.shutdown()
    assert seen_ple == [0]


def _spec_config(kvc, stages):
    from tests.runner.test_qwen4_exp_host_tier import _pp8_offloading_config

    return _pp8_offloading_config(kvc, stages, SERVED_BUDGET)


def test_registration_refuses_tensors_that_do_not_follow_the_plan(monkeypatch):
    from tests.runner.test_qwen4_exp_host_tier import _stage_caches
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions

    cfg, _, stages, kvc = _served(monkeypatch)
    stage = next(s for s in stages if pool_regions(s) == 2)
    shape = (4, 3200, 1, 4, 256)
    regions = [torch.zeros(shape, dtype=torch.float8_e4m3fn) for _ in range(2)]
    vllm_config = _vllm_config(cfg)
    # Two regions collapsed into one storage.
    collapsed, _ = _stage_caches(stage, [regions[0], regions[0]])
    with pytest.raises(
        AssertionError, match="planned in pool region|share one storage"
    ):
        _register(stage, collapsed, vllm_config)
    # One layer given the other region's storage.
    caches, index = _stage_caches(stage, regions)
    name = next(n for n, r in index.items() if r == 1)
    caches[name] = [regions[0]] if isinstance(caches[name], list) else regions[0]
    with pytest.raises(AssertionError, match="planned in pool region"):
        _register(stage, caches, vllm_config)


# -- scope and the flag off ---------------------------------------------------------


@pytest.mark.parametrize(
    "extra, env, error",
    [
        (
            {
                "global_registry_address": "registry:50051",
                "store_server_ip": "10.0.0.1",
            },
            {},
            "global registry",
        ),
        ({"kv_pool_group": "pool-a"}, {}, "peer store group"),
        (
            {
                "raiden_storage_path": "STORAGE",
                "experimental_raiden_storage_skip_parallelism_check": True,
            },
            {"PYTHONHASHSEED": "0"},
            "storage tier",
        ),
        ({}, {"RAIDEN_SHM_KEY": "segment"}, "RAIDEN_SHM_KEY is not supported"),
    ],
)
def test_the_flag_is_refused_outside_the_local_dram_tier(
    monkeypatch, tmp_path, extra, env, error
):
    cfg, _, stages, kvc = _served(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    extra = {k: (str(tmp_path) if v == "STORAGE" else v) for k, v in extra.items()}
    with pytest.raises(ValueError, match=error):
        _spec(monkeypatch, cfg, stages, kvc, **extra)


@pytest.mark.parametrize(
    "flag", ["VLLM_TPU_OFFLOAD_TIER_KEY_ROWS", "VLLM_TPU_OFFLOAD_TIER_CHECK"]
)
def test_key_rows_and_check_need_the_classes(monkeypatch, flag):
    cfg, _, stages, kvc = _served(monkeypatch, classes=False)
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="need VLLM_TPU_OFFLOAD_TIER_CLASSES=1"):
        _spec(monkeypatch, cfg, stages, kvc)


@pytest.mark.parametrize("value", [None, "0"])
def test_flag_off_builds_the_single_store_as_before(monkeypatch, value):
    from unittest.mock import MagicMock

    from vllm_torchtpu.offload import raiden_store

    cfg, _, stages, kvc = _served(monkeypatch, classes=False)
    if value is not None:
        monkeypatch.setenv("VLLM_TPU_OFFLOAD_TIER_CLASSES", value)
    spec = _spec(monkeypatch, cfg, stages, kvc)
    assert spec.tier_plan is None
    built = MagicMock()
    monkeypatch.setattr(raiden_store, "RaidenOffloadingManager", built)
    spec.get_manager()
    (call,) = built.call_args_list
    assert set(call.kwargs) == {
        "kernel_physical_blocks_capacity",
        "offload_logical_blocks_capacity",
        "device_block_size_factor",
        "world_size",
        "raiden_controller_port",
        "raiden_job_name",
        "global_registry_address",
        "store_server_ip",
        "kv_pool_group",
        "key_namespace",
        "secondary_backend_configs",
    }
    assert (
        call.kwargs["raiden_controller_port"] == 31515
        and call.kwargs["world_size"] == 8
    )
    assert call.kwargs["offload_logical_blocks_capacity"] == SERVED_BUDGET // (
        16 * PAGE
    )


def _flag_off_call_log(monkeypatch):
    """The fake store's calls for write-through and victim traffic, flag off."""
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput

    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _Blocks, _tokens
    from vllm_torchtpu.offload import victim

    store = FakeKVCacheStore()
    sched, kvc, _ = _tier_scheduler(monkeypatch, single_store=store)
    first, table, meta = _turn_one(sched, kvc)
    _ack_fences(sched, meta)
    store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    sched.build_connector_meta(SchedulerOutput.make_empty())
    second = request("r2", first.prompt_token_ids + _tokens(2000, 1))
    sched.on_new_request(second)
    sched.get_num_new_matched_tokens(second, 0)
    blocks = _Blocks()
    blocks.next_id = 100
    alloc = []
    for group in kvc.kv_cache_groups:
        if isinstance(group.kv_cache_spec, CircularBufferSpec):
            ids = blocks.take(1)
        elif isinstance(group.kv_cache_spec, MambaSpec):
            ids = [0, 0, *blocks.take(1)]
        else:
            ids = blocks.take(3)
        alloc.append([KVCacheBlock(i, is_null=i == 0) for i in ids])
    sched.update_state_after_alloc(second, KVCacheBlocks(tuple(alloc)), 3 * BLOCK)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    (load,) = store.calls_named("load")
    store.complete_load(load.args[0])
    sched.build_connector_meta(SchedulerOutput.make_empty())

    # the victim path on a fresh scheduler
    victim_store = _AutoSave()
    vsched, vkvc, _ = _tier_scheduler(
        monkeypatch, victim_policy=True, single_store=victim_store
    )
    try:
        vfirst, vtable, _ = _turn_one(vsched, vkvc)
        _victims_of_turn_one(vsched, vfirst, vtable, vkvc)
        vsched.build_connector_meta(SchedulerOutput.make_empty())
    finally:
        victim.deactivate()

    def shown(value):
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (list, tuple)):
            return [shown(v) for v in value]
        if hasattr(value, "device_block_id"):
            return (
                "slice",
                value.device_block_id,
                getattr(value.status, "name", value.status),
            )
        if callable(value) or value is None:
            return None
        return value

    return [
        (c.name, shown(list(c.args))) for s in (store, victim_store) for c in s.calls
    ]


# Captured with this test module on f29e285dd3, the tree before the tier classes.
FLAG_OFF_CALLS_SHA256 = (
    "3a308365cb47119ea10d080dc9d334103998b3ed8b16fd0dfa303eb8c0ea8d9c"
)


def test_flag_off_issues_the_same_store_and_load_calls(monkeypatch):
    calls = _flag_off_call_log(monkeypatch)
    assert {name for name, _ in calls} >= {"insert", "save", "lookup", "load"}
    digest = hashlib.sha256(repr(calls).encode()).hexdigest()
    assert digest == FLAG_OFF_CALLS_SHA256, digest


def test_lookup_line_shows_each_class_and_restore_line_counts_class_bytes(monkeypatch):
    import json
    from unittest.mock import patch

    from vllm.v1.core.sched.output import SchedulerOutput

    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _tokens
    from vllm_torchtpu.offload import lookup_log

    monkeypatch.setenv("VLLM_TPU_PREFIX_LOOKUP_LOG", "1")
    sched, kvc, stores = _tier_scheduler(monkeypatch)
    first, table, meta = _turn_one(sched, kvc, boundaries=(2, 3))
    _ack_fences(sched, meta)
    for store in stores:
        store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    sched.build_connector_meta(SchedulerOutput.make_empty())
    namespace = sched._raiden_manager._key_namespace
    del stores[1].entries[_sub_of(first.block_hashes[2], KEYS, namespace)]
    del stores[3].entries[_sub_of(first.block_hashes[1], GDN_C, namespace)]
    lines = []
    with patch.object(
        lookup_log.logger, "info", side_effect=lambda fmt, *a: lines.append(fmt % a)
    ):
        second = request("r2", first.prompt_token_ids + _tokens(300, 2))
        sched.on_new_request(second)
        sched.get_num_new_matched_tokens(second, 0)
    (line,) = [
        json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("prefix-lookup ")
    ]
    assert line["tier_class"] == {
        "main": 3,
        "keys": 2,
        "gdn-ab": [2, 3],
        "gdn-c": [3],
        "ple": [2, 3],
    }
    assert line["tier_attn"] == 2

    # tier-restore: a job's bytes are its keys at their classes' entry sizes
    tier = sched._raiden_manager
    log = lookup_log.RestoreLog(tier, tier.bytes_of_keys)
    keys = [
        make_offload_key(bytes(first.block_hashes[0]), g) for g in (MAIN, KEYS, PLE)
    ]
    sizes = {c.name: c.bytes_per_entry for c in tier.classes}
    assert sizes["main"] == 12 * PAGE and sizes["ple"] == PAGE
    tier.submit_load_job(99, keys, [1, 2, 3], "r9", pinned=False)
    assert log.jobs[99][2] == sizes["main"] + sizes["keys"] + sizes["ple"]


def test_a_raiden_that_refuses_the_key_row_view_stops_registering(monkeypatch):
    from unittest.mock import patch

    from vllm.distributed import parallel_state

    from tests.runner.test_qwen4_exp_host_tier import _stage_caches
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    cfg, _, stages, kvc = _served(monkeypatch, key_rows=True)
    _spec(monkeypatch, cfg, stages, kvc)
    stage = stages[1]
    spec = raiden_store.TPURaidenStoreOffloadingSpec(
        _spec_config(kvc, stages), _vllm_config(cfg), stage
    )
    shape = (stage.num_blocks, 3200, 1, 4, 256)
    regions = [
        torch.zeros(shape, dtype=torch.float8_e4m3fn)
        for _ in range(pool_regions(stage))
    ]
    caches, _ = _stage_caches(stage, regions)
    (canonical,) = _register(stage, caches, spec.vllm_config)
    attempts = []

    class KVCacheManager:
        def __init__(self, *, kv_caches, **kwargs):
            attempts.append(kwargs["raiden_controller_address"])
            if kv_caches[0][0].shape[0] != shape[0]:
                raise ValueError(
                    "Tensor's leading (block) dimension does not match its base "
                    "storage buffer"
                )

    import sys
    import types

    api = types.ModuleType("tpu_sync.api.torch")
    api.kv_cache_manager = types.SimpleNamespace(KVCacheManager=KVCacheManager)
    modules = {
        "tpu_sync": types.ModuleType("tpu_sync"),
        "tpu_sync.api": types.ModuleType("tpu_sync.api"),
        "tpu_sync.api.torch": api,
        "tpu_sync.api.torch.kv_cache_manager": api.kv_cache_manager,
    }
    with (
        patch.dict(sys.modules, modules),
        patch.object(
            RaidenStoreOffloadingWorker, "_controller_reachable", return_value=True
        ),
        patch.object(
            parallel_state, "get_world_group", return_value=NS(rank_in_group=1)
        ),
    ):
        worker = spec.get_worker(canonical)
        for class_worker in worker.workers.values():
            class_worker._register_thread.join(timeout=10)
    keys_address = f"127.0.0.1:{31515 + 1}"
    # one attempt for the refused view, no retry; the other classes registered
    assert attempts.count(keys_address) == 1
    assert not worker.workers["keys"]._register_thread.is_alive()
    assert worker.workers["keys"]._mgr is None
    assert all(w._mgr is not None for n, w in worker.workers.items() if n != "keys")
    worker.shutdown()


# -- check mode (VLLM_TPU_OFFLOAD_TIER_CHECK) -----------------------------------------


def _cpu_rows(region, first, count):
    rows = region.view(torch.uint8).reshape(-1, region.shape[-2] * region.shape[-1])
    return rows[first : first + count].numpy()


def test_check_digest_covers_only_the_bytes_a_class_moves():
    from vllm_torchtpu.offload.tier_check import TierCheck

    gen = torch.Generator().manual_seed(0)
    shape = (6, 8, 1, 4, 256)  # 6 blocks of 8 rows of 1,024 B
    regions = [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen).view(
            torch.float8_e4m3fn
        )
        for _ in range(2)
    ]
    reads = []

    def reader(region, first, count):
        reads.append((first, count))
        return _cpu_rows(region, first, count)

    check = TierCheck(
        {"main": regions, "keys": regions}, {"main": 8192, "keys": 2048}, 0, reader
    )
    page = lambda r, b: regions[r].view(torch.uint8)[b].reshape(-1)  # noqa: E731
    check.run([("main", b"m", 1), ("keys", b"k", 2)], [])
    # a whole-page restore of block 1 into block 4: equal
    for r in range(2):
        page(r, 4).copy_(page(r, 1))
    # the keys class moves the leading 2,048 B only: restore those into block 5
    for r in range(2):
        page(r, 5)[:2048].copy_(page(r, 2)[:2048])
    check.run([], [("main", b"m", 4), ("keys", b"k", 5)])
    assert check.take_counts() == {"main": [1, 0, 0, 0], "keys": [1, 0, 0, 0]}
    assert (16, 2) in reads and (40, 2) in reads  # row 8b, 2 rows, for the keys class
    # one changed byte in what was moved, or the keys landing one row off
    page(1, 4)[5000] ^= 0xFF
    for r in range(2):
        page(r, 3)[1024:3072].copy_(page(r, 2)[:2048])
    check.run([], [("main", b"m", 4), ("keys", b"k", 3), ("keys", b"never", 0)])
    assert check.take_counts() == {"main": [1, 1, 0, 0], "keys": [1, 1, 1, 0]}


def _check_world(monkeypatch, rank):
    """A stage's caches registered with check mode on, rows read on the CPU."""
    from unittest.mock import patch

    from vllm.distributed import parallel_state

    from tests.runner.test_qwen4_exp_host_tier import _stage_caches
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    cfg, _, stages, kvc = _served(monkeypatch)
    _spec(monkeypatch, cfg, stages, kvc)
    stage = stages[rank]
    spec = raiden_store.TPURaidenStoreOffloadingSpec(
        _spec_config(kvc, stages), _vllm_config(cfg), stage
    )
    gen = torch.Generator().manual_seed(rank)
    shape = (stage.num_blocks, 3200, 1, 4, 256)
    regions = [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen).view(
            torch.float8_e4m3fn
        )
        for _ in range(pool_regions(stage))
    ]
    caches, _ = _stage_caches(stage, regions)
    connector = NS(
        _vllm_config=spec.vllm_config,
        connector_worker=NS(
            kv_cache_config=stage, _init_worker=lambda c: None, spec=spec
        ),
    )
    with patch.object(
        parallel_state, "get_world_group", return_value=NS(rank_in_group=rank)
    ):
        TPURaidenOffloadingConnector.register_kv_caches(connector, caches)
    return connector, regions, spec.tier_plan


def test_check_mode_counts_a_load_that_returns_other_bytes(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _Blocks, _tokens
    from vllm_torchtpu.offload.raiden_connector import (
        RaidenOffloadingWorkerMetadata,
        TPURaidenOffloadingConnector,
    )

    monkeypatch.setenv("VLLM_TPU_OFFLOAD_TIER_CHECK", "1")
    sched, kvc, stores = _tier_scheduler(monkeypatch)
    first, table, meta = _turn_one(sched, kvc)
    assert not meta.tier_check_stores  # nothing saved yet
    _ack_fences(sched, meta)
    for store in stores:
        store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    saved = sched.build_connector_meta(SchedulerOutput.make_empty())
    # every saved entry with its class and source block
    source = {}
    for gid in kvc.prefix_cacheable_group_ids:
        if isinstance(kvc.kv_cache_groups[gid].kv_cache_spec, MambaSpec):
            source[bytes(make_offload_key(bytes(first.block_hashes[2]), gid))] = table[
                gid
            ][2]
        else:
            for c in range(3):
                key = make_offload_key(bytes(first.block_hashes[c]), gid)
                source[bytes(key)] = table[gid][c]
    assert {k: b for _, k, b in saved.tier_check_stores} == source
    names = {k: n for n, k, _ in saved.tier_check_stores}
    assert names[bytes(make_offload_key(bytes(first.block_hashes[0]), KEYS))] == "keys"

    second = request("r2", first.prompt_token_ids + _tokens(2000, 1))
    sched.on_new_request(second)
    assert sched.get_num_new_matched_tokens(second, 0) == (3 * BLOCK, True)
    blocks = _Blocks()
    blocks.next_id = 30
    alloc = []
    for group in kvc.kv_cache_groups:
        if isinstance(group.kv_cache_spec, CircularBufferSpec):
            ids = blocks.take(1)
        elif isinstance(group.kv_cache_spec, MambaSpec):
            ids = [0, 0, *blocks.take(1)]
        else:
            ids = blocks.take(3)
        alloc.append([KVCacheBlock(i, is_null=i == 0) for i in ids])
    sched.update_state_after_alloc(second, KVCacheBlocks(tuple(alloc)), 3 * BLOCK)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    for store in stores:
        for call in store.calls_named("load"):
            store.complete_load(call.args[0])
    landed = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert len(landed.tier_check_loads) == len(source)

    # Stage 1's worker hashes what was saved, the bytes land (one load gets
    # other bytes), and it compares.
    rank = 1
    connector, regions, plan = _check_world(monkeypatch, rank)
    check = connector._tier_check
    check._read = _cpu_rows
    connector._get_connector_metadata = lambda: saved
    TPURaidenOffloadingConnector._run_tier_check(connector)
    by_name = {c.name: c for c in plan.classes}
    wrong = None
    for name, key, dst in landed.tier_check_loads:
        own = by_name[name].worker_regions[rank]
        for r in own:
            regions[r].view(torch.uint8)[dst].copy_(
                regions[r].view(torch.uint8)[source[key]]
            )
        if wrong is None and name == "gdn-c" and own:
            wrong = (own[0], dst)
    regions[wrong[0]].view(torch.uint8)[wrong[1], 7] ^= (
        1  # a load that returns other bytes
    )
    connector._get_connector_metadata = lambda: landed
    TPURaidenOffloadingConnector._run_tier_check(connector)
    counts = check.take_counts()
    assert counts["gdn-c"][:3] == [1, 1, 0]
    assert all(v[1] == 0 for n, v in counts.items() if n != "gdn-c")
    assert counts["main"][0] == 3  # one comparison a key, over the class's regions
    assert "ple" not in counts  # stage 1 has no PLE region

    # The counts reach the scheduler's telemetry line.
    report = RaidenOffloadingWorkerMetadata(completed_jobs={}, tier_check=counts)
    merged = report.aggregate(
        RaidenOffloadingWorkerMetadata(completed_jobs={}, tier_check=counts)
    )
    assert merged.tier_check["gdn-c"][:2] == [2, 2]
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=merged))
    line = sched.telemetry_line()
    assert "tier_check[gdn-c] compared=2 mismatched=2 missing=0 errors=0" in line


def test_check_mode_off_issues_no_checksum(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.offload import tier_check

    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_CHECK", raising=False)
    built = []
    monkeypatch.setattr(
        tier_check.TierCheck, "__init__", lambda *a, **k: built.append(a)
    )
    sched, kvc, stores = _tier_scheduler(monkeypatch)
    first, table, meta = _turn_one(sched, kvc)
    _ack_fences(sched, meta)
    for store in stores:
        store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    done = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert done.tier_check_stores == [] and done.tier_check_loads == []
    assert "tier_check[" not in sched.telemetry_line()
    connector, _, _ = _check_world(monkeypatch, 0)
    assert "_tier_check" not in vars(connector) and not built


def test_class_ports_stay_out_of_the_ephemeral_range(monkeypatch):
    from vllm_torchtpu.offload import raiden_store

    cfg, _, stages, kvc = _served(monkeypatch)
    monkeypatch.setattr(raiden_store, "ephemeral_port_range", lambda: (32768, 60999))
    spec = _spec(monkeypatch, cfg, stages, kvc)
    assert [spec.tier_controller_port(c) for c in spec.tier_plan.classes] == list(
        range(31515, 31520)
    )
    # A base inside the range is refused: a port-0 listener could hold it.
    with pytest.raises(ValueError, match="ephemeral range 32768-60999"):
        _spec(monkeypatch, cfg, stages, kvc, raiden_controller_port=51515)


def test_single_store_port_stays_out_of_the_ephemeral_range(monkeypatch):
    from vllm_torchtpu.offload import raiden_store

    cfg, _, stages, kvc = _served(monkeypatch, classes=False)
    monkeypatch.setattr(raiden_store, "ephemeral_port_range", lambda: (32768, 60999))
    from unittest.mock import MagicMock

    monkeypatch.setattr(raiden_store, "RaidenOffloadingManager", MagicMock())
    spec = _spec(monkeypatch, cfg, stages, kvc)
    # The store binds this port and every worker dials it: one value.
    assert spec.tier_plan is None and spec.raiden_controller_port == 31515
    assert spec.raiden_controller_port == raiden_store._DEFAULT_CONTROLLER_PORT
    spec.get_manager()
    _spec(monkeypatch, cfg, stages, kvc, raiden_controller_port=27800).get_manager()
    inside = _spec(monkeypatch, cfg, stages, kvc, raiden_controller_port=51515)
    with pytest.raises(ValueError, match=r"controller ports \[51515\]"):
        inside.get_manager()
    with pytest.raises(ValueError, match=r"controller ports \[51515\]"):
        inside.get_worker(None)


def test_check_reads_rows_byte_for_byte_through_the_dma_gather():
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.offload import tier_check

    gen = np.random.default_rng(0)
    region = gen.integers(
        0, 256, (4, 8, 1, 4, 256), dtype=np.uint8
    )  # 4 pages of 8 rows
    read = tier_check.read_rows_fn(interpret=True)
    pool = jnp.asarray(region).view(jnp.float8_e4m3fn)
    idx = jnp.arange(16, 18, dtype=jnp.int32)  # page 2, its first two rows
    got = np.asarray(read(pool, idx))
    assert got.dtype == np.uint8 and got.shape == (2, 4, 256)
    assert np.array_equal(got, region.reshape(-1, 4, 256)[16:18])  # NaN bytes included


def test_check_read_op_builds():
    pytest.importorskip("torch_tpu._internal.pallas")
    from vllm_torchtpu.offload import tier_check

    op = tier_check.read_rows_op()
    assert callable(op)


def test_check_forgets_a_saved_block_that_a_same_step_load_overwrites():
    """j-3177bdaa: a victim's source block is also a load destination of the same
    step; the load can land before the workers read the block for the victim's
    digest. Such entries go out as block -1, the workers drop the key's digest, and
    the key's later restore counts as missing, not mismatched."""
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler
    from vllm_torchtpu.offload.tier_check import TierCheck

    gen = torch.Generator().manual_seed(3)
    shape = (6, 8, 1, 4, 256)
    region = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen).view(
        torch.float8_e4m3fn
    )
    page = lambda b: region.view(torch.uint8)[b].reshape(-1)  # noqa: E731
    check = TierCheck({"main": [region]}, {"main": 8192}, 0, _cpu_rows)
    saved_y = page(2).clone()  # Raiden saved victim Y from block 2 ...
    page(2).copy_(page(4))  # ... then this step's load of X landed in block 2
    sched = NS()
    stores = TPURaidenOffloadingScheduler._unhashable_as_forget(
        sched, [("main", b"y", 2), ("main", b"z", 3)], {2, 5}
    )
    assert stores == [("main", b"y", -1), ("main", b"z", 3)]
    assert sched._tier_check_unhashed == {"main": 1}
    check.run(stores, [])
    page(1).copy_(saved_y)  # Y restored, byte for byte, into block 1
    page(0).copy_(page(3))  # Z restored into block 0
    check.run([], [("main", b"y", 1), ("main", b"z", 0)])
    assert check.take_counts() == {"main": [1, 0, 1, 0]}  # Z compared, Y missing

    # Without the filter Y's digest would be of X's bytes: a false mismatch.
    unfiltered = TierCheck({"main": [region]}, {"main": 8192}, 0, _cpu_rows)
    page(2).copy_(page(4))
    unfiltered.run([("main", b"y", 2)], [])
    unfiltered.run([], [("main", b"y", 1)])
    assert unfiltered.take_counts() == {"main": [1, 1, 0, 0]}


@pytest.mark.parametrize("classes", [True, False])
def test_flushed_stores_drain_before_same_step_loads_with_the_classes(
    monkeypatch, classes
):
    """With the classes, a launched store whose source block this step reallocates
    finishes before any load (possibly into that block) is submitted. Flag off keeps
    today's order: loads first, then the flush drain."""
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
        TransferJob,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_offload.base import GPULoadStoreSpec

    from vllm_torchtpu.offload.raiden_store import (
        AdmissionOp,
        FinishedJob,
        RaidenLoadStoreSpec,
    )

    if classes:
        sched, kvc, _ = _tier_scheduler(monkeypatch)
    else:
        sched, kvc, _ = _tier_scheduler(monkeypatch, single_store=FakeKVCacheStore())
    manager = sched._raiden_manager
    order = []
    key = make_offload_key(b"hash_0001", MAIN)
    crafted = OffloadingConnectorMetadata(
        load_jobs={
            7: TransferJob(
                req_id="r2",
                src_spec=RaidenLoadStoreSpec([key]),
                dst_spec=GPULoadStoreSpec([12], group_sizes=[1], block_indices=[0]),
            )
        },
        store_jobs={},
        jobs_to_flush={5},
    )
    with (
        patch.object(
            OffloadingConnectorScheduler, "build_connector_meta", return_value=crafted
        ),
        patch.object(manager, "is_job_launched", side_effect=lambda j: j == 5),
        patch.object(
            manager,
            "drain_jobs",
            side_effect=lambda ids: order.append(("drain", sorted(ids)))
            or [FinishedJob(5, AdmissionOp.STORE, True)],
        ),
        patch.object(
            manager,
            "submit_load_job",
            side_effect=lambda *a, **k: order.append(("load", a[0])),
        ),
        patch.object(manager, "poll_finished_jobs", return_value=[]),
    ):
        meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    expected = (
        [("drain", [5]), ("load", 7)] if classes else [("load", 7), ("drain", [5])]
    )
    assert order == expected
    assert meta.finished_store_job_ids == [5]
