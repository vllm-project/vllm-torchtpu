# SPDX-License-Identifier: Apache-2.0
"""Host tier classes with merged compressed keys (QWEN4_EXP_MERGE_COMPRESSED_K), on CPU.

With merged keys there is no compressed-key group: each QSA layer's keys for block ``b`` sit
at row ``cols * N + b`` (rows of the keys' live bytes) of its main-KV region. A ``keys``
mirror class over the main class's groups and regions carries them: every main-KV key is
stored, loaded and looked up in both, and the mirror's device rows are offset past the
scheduler pages.
"""

from types import SimpleNamespace as NS

import pytest

from tests.offload.test_raiden_store_offload import FAKE_STORE_TYPES, FakeKVCacheStore
from tests.offload.test_tier_classes import (
    GDN_A,
    GDN_B,
    GDN_C,
    KEY_ROW,
    MAIN,
    PAGE,
    PLE,
    _ctx,
    _key,
    _resident,
    _sub,
)

N = 9830  # scheduler pages of the served prefill pool
OFFSET = (PAGE // KEY_ROW) * N


def _classes():
    from vllm_torchtpu.offload.tier_classes import TierClass

    regions = ((0,), (0, 1))
    return [
        TierClass("main", 0, "main", (MAIN,), regions, PAGE),
        TierClass("gdn-ab", 1, "gdn", (GDN_A, GDN_B), regions, PAGE),
        TierClass("gdn-c", 2, "gdn", (GDN_C,), regions, PAGE),
        TierClass("ple", 3, "ple", (PLE,), ((0,), ()), PAGE),
    ]


def test_mirror_follows_the_main_class():
    from vllm_torchtpu.offload.tier_classes import _with_merged_key_mirror

    out = _with_merged_key_mirror(_classes(), PAGE, KEY_ROW, [NS(num_blocks=N)] * 2)
    assert [c.name for c in out] == ["main", "keys", "gdn-ab", "gdn-c", "ple"]
    assert [c.index for c in out] == list(range(5))
    keys = out[1]
    assert keys.mirror and keys.kind == "keys" and keys.groups == (MAIN,)
    assert keys.worker_regions == out[0].worker_regions
    assert (keys.row_bytes, keys.rows_per_page, keys.row_offset) == (KEY_ROW, 1, 8 * N)
    assert keys.bytes_per_entry == 3 * KEY_ROW


def test_mirror_refusals():
    from vllm_torchtpu.offload.tier_classes import TierClass, _with_merged_key_mirror

    with pytest.raises(ValueError, match="one block count"):
        _with_merged_key_mirror(_classes(), PAGE, KEY_ROW, [NS(num_blocks=N), NS(num_blocks=N + 1)])
    with pytest.raises(ValueError, match="do not tile"):
        _with_merged_key_mirror(_classes(), PAGE, KEY_ROW + 1, [NS(num_blocks=N)])
    keyed = _classes() + [TierClass("keys", 4, "keys", (5,), ((0,),), KEY_ROW, 8)]
    with pytest.raises(ValueError, match="compressed-key group"):
        _with_merged_key_mirror(keyed, PAGE, KEY_ROW, [NS(num_blocks=N)])


def test_served_table_holds_with_the_mirror():
    from vllm_torchtpu.offload.tier_classes import (
        TierClass,
        TierPlan,
        _with_merged_key_mirror,
        check_served_plan,
    )

    # The served prefill: main KV in regions [1, 2, 1, 2, ...] over 8 workers.
    main_regions = tuple((0,) if w % 2 == 0 else (0, 1) for w in range(8))
    classes = [
        TierClass("main", 0, "main", (MAIN,), main_regions, PAGE),
        TierClass("gdn-ab", 1, "gdn", (GDN_A, GDN_B), main_regions, PAGE),
        TierClass("gdn-c", 2, "gdn", (GDN_C,), main_regions, PAGE),
        TierClass("ple", 3, "ple", (PLE,), ((0,),) + ((),) * 7, PAGE),
    ]
    out = _with_merged_key_mirror(classes, PAGE, KEY_ROW, [NS(num_blocks=N)] * 8)
    check_served_plan(TierPlan(tuple(out), 0, 4.7, 10, 2))


def _tier(token_blocks=16, checkpoints=8):
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    from vllm_torchtpu.offload.tier_classes import (
        TieredOffloadingManager,
        TierPlan,
        _with_merged_key_mirror,
    )

    classes = _with_merged_key_mirror(_classes(), PAGE, KEY_ROW, [NS(num_blocks=N)])
    plan = TierPlan(
        tuple(classes), budget=0, ratio=2.0, token_blocks=token_blocks, checkpoints=checkpoints
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
            device_row_offset=c.row_offset,
            track_held=True,
            store=stores[c.name],
            store_types=FAKE_STORE_TYPES,
        )
        for c in classes
    ]
    return TieredOffloadingManager(plan, managers), stores


def test_a_main_key_hits_only_where_both_copies_are():
    tier, stores = _tier()
    k = _key(1, MAIN)
    _resident(stores["main"], k)
    assert tier.lookup(k, _ctx()).name == "MISS"
    assert tier.host_resident_prefix([k]) == 0
    _resident(stores["keys"], k)
    assert tier.lookup(k, _ctx()).name == "HIT"
    assert tier.host_resident_prefix([k, _key(2, MAIN)]) == 1
    # A GDN key is not mirrored.
    g = _key(3, GDN_C)
    _resident(stores["gdn-c"], g)
    assert tier.lookup(g, _ctx()).name == "HIT"
    assert not stores["keys"].calls_named("lookup") or all(
        _sub(g) not in call.args[0] for call in stores["keys"].calls_named("lookup")
    )


def test_stores_and_loads_go_to_both_copies_at_their_own_rows():
    tier, stores = _tier()
    keys = [_key(1, MAIN), _key(1, GDN_A), _key(2, MAIN)]
    tier.submit_store_job(5, keys, [10, 11, 12])
    inserted = {
        n: [
            (h, sl.device_block_id)
            for call in s.calls_named("insert")
            for h, sl in zip(call.args[0], call.args[1])
        ]
        for n, s in stores.items()
    }
    assert inserted["main"] == [(_sub(keys[0]), 10), (_sub(keys[2]), 12)]
    assert inserted["keys"] == [(_sub(keys[0]), OFFSET + 10), (_sub(keys[2]), OFFSET + 12)]
    assert inserted["gdn-ab"] == [(_sub(keys[1]), 11)]
    stores["main"].complete_save([_sub(keys[0]), _sub(keys[2])])
    stores["gdn-ab"].complete_save([_sub(keys[1])])
    assert tier.poll_finished_jobs() == []  # the mirror child is still running
    stores["keys"].complete_save([_sub(keys[0]), _sub(keys[2])])
    (done,) = tier.poll_finished_jobs()
    assert done.success and not tier.has_pending_work()
    assert tier.bytes_of_keys([keys[0]]) == 3 * PAGE + 3 * KEY_ROW

    # Device rows: whole pages for main, the key region (offset) for the mirror.
    assert tier.managers[0]._expand_device_blocks([10, 12]) == [10, 12]
    assert tier.managers[1]._expand_device_blocks([10, 12]) == [OFFSET + 10, OFFSET + 12]

    spec = tier.prepare_load([keys[0]], _ctx("r"))
    assert spec.pinned
    tier.submit_load_job(6, spec.keys, [40], "r", pinned=True)
    loads = {n: [c.args for c in s.calls_named("load")] for n, s in stores.items() if s.calls_named("load")}
    assert set(loads) == {"main", "keys"}
    stores["main"].complete_load([_sub(keys[0])])
    assert tier.poll_finished_jobs() == []
    stores["keys"].complete_load([_sub(keys[0])])
    (done,) = tier.poll_finished_jobs()
    assert done.success and done.req_id == "r"


def test_victim_store_takes_a_key_only_if_every_copy_needs_it():
    tier, stores = _tier()
    tier.set_victim_mode(True)
    both, main_only, fresh, gdn = _key(1, MAIN), _key(2, MAIN), _key(3, MAIN), _key(4, GDN_C)
    for s in ("main", "keys"):
        _resident(stores[s], both)
    _resident(stores["main"], main_only)
    picked = tier.prepare_victim_store([both, main_only, fresh, gdn])
    assert picked == [fresh, gdn]
    # The mirror's claim on main_only was given back: nothing of it stays in flight.
    assert main_only not in tier.managers[1]._inflight_store_keys
    assert tier.managers[0]._inflight_store_keys == {fresh}
    assert tier.managers[1]._inflight_store_keys == {fresh}
