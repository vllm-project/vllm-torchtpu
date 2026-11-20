# SPDX-License-Identifier: Apache-2.0
"""q38 HOLE-DIAG: divergent local hybrid hits on the PD prefill with the host tier (VLLM_TPU_OFFLOAD_DIVERGENT_HITS), CPU.

The prefix-cache "hole" on the B1 runs (reports/hole_diag_1120.md): a conversation's next turn finds its shared head
(the Marconi junction, 4-13 blocks) in HBM with no recurrent-state checkpoint there, and its own tail plus the previous
turn's checkpoint in the host tier. The convergent local hit is 0, the tier lookup starts at block 0, the head is not in
the tier (HBM keeps it hot, so the victim policy never stores it, and exclusive restore discards the copy a load brings
back), and the whole prompt is recomputed. With the flag, the local hit is the attention groups' cached run (the head),
no state block is adopted, and the tier's load supplies the attention blocks past the head and every state group's
checkpoint at the boundary it restores.

Rows (tag in each assertion message):
- FLAG     the key is registered once, off by default, and compile-ignored.
- RULE     a tier restores state with every load only when it offloads every prefix-cacheable state group.
- CONN     the offload connector reports divergent hits only with the flag and a scheduler whose loads restore state.
- MULTI    the multi connector leaves a pure producer out of the vote with the flag; without it, upstream's all().
- HOLE     flag off, on the served PP8 prefill groups with a real KVCacheManager: the split chain hits nothing.
- SPLIT    flag on: the same request adopts the HBM head and loads only the tier's tail and the state checkpoints.
- NODUP    flag on with the head also in the tier: the head is adopted from HBM, not loaded again.
- FALLBACK flag on, one state checkpoint missing from the tier: no load, and the convergent re-lookup hits nothing.
- LOG      the lookup line still records the HBM side on the divergent lookup; tier_runs shows a run past block 0.
"""

from types import SimpleNamespace as NS

import pytest
from vllm.v1.kv_cache_interface import MambaSpec

from tests.runner.test_qwen4_exp_host_tier import (
    BLOCK,
    _multi,
    _pp8_scheduler,
    _request,
    pp8_pool,
)
from tests.runner.test_qwen4_exp_offload import _key_group_of_sub_hash, _tokens

FLAG = "VLLM_TPU_OFFLOAD_DIVERGENT_HITS"
J, K = 3, 6  # shared head blocks (the junction), and the previous turn's checkpoint boundary


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    for name in ("QWEN4_EXP_PACK_COMPRESSED_K", "QWEN4_EXP_PACK_MAIN_KV", "QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR",
                 FLAG, "VLLM_TPU_PREFIX_LOOKUP_LOG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine_patches():
    """The prefill engine-core patches this path runs under (= test_qwen4_exp_decode_prefix_cache.engine_patches,
    credited); vLLM restored afterwards."""
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    from vllm_torchtpu.core.divergent_hybrid_hits import patch_divergent_hybrid_hits
    from vllm_torchtpu.core.mamba_checkpoint_retention import patch_mamba_checkpoint_retention

    saved = [
        (cls, name, cls.__dict__[name])
        for cls, names in (
            (MambaManager, ("cache_blocks", "_remove_blocks_in_range", "remove_skipped_blocks", "pop_blocks_for_free")),
            (KVCacheCoordinator, ("free", "pop_blocks_for_free")),
            (KVCacheManager, ("get_computed_blocks", "get_computed_blocks_for_connector")),
            (Scheduler, ("add_request",)),
        )
        for name in names
    ]
    patch_mamba_checkpoint_retention()
    patch_divergent_hybrid_hits()
    yield
    for cls, name, value in saved:
        setattr(cls, name, value)


# -- FLAG / RULE / CONN / MULTI ----------------------------------------------------


def test_flag_is_registered_off_by_default_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG in envs.environment_variables, "FLAG not registered"
    assert envs.VLLM_TPU_OFFLOAD_DIVERGENT_HITS is False, "FLAG default is not off"
    monkeypatch.setenv(FLAG, "1")
    assert envs.VLLM_TPU_OFFLOAD_DIVERGENT_HITS is True, "FLAG does not read 1"
    assert FLAG in tpu_compiler._TPU_COMPILE_ENV_IGNORED, "FLAG is not compile-ignored"


def _groups(kvc):
    cacheable = sorted(kvc.prefix_cacheable_group_ids)
    mamba = [g for g in cacheable if isinstance(kvc.kv_cache_groups[g].kv_cache_spec, MambaSpec)]
    return cacheable, mamba, [g for g in cacheable if g not in mamba]


def test_a_tier_restores_state_only_when_it_offloads_every_state_group(monkeypatch):
    from vllm_torchtpu.offload.raiden_connector import divergent_hits_restore_state

    _, _, stages, kvc = pp8_pool(monkeypatch)
    cacheable, mamba, attention = _groups(kvc)
    assert len(mamba) == 4 and len(attention) == 2
    assert divergent_hits_restore_state(kvc, cacheable), "RULE every group offloaded"
    assert not divergent_hits_restore_state(kvc, [g for g in cacheable if g != mamba[-1]]), "RULE a state group left out"
    assert not divergent_hits_restore_state(kvc, attention), "RULE attention only"
    no_state = NS(kv_cache_groups=[g for g in kvc.kv_cache_groups if not isinstance(g.kv_cache_spec, MambaSpec)])
    assert not divergent_hits_restore_state(no_state, range(len(no_state.kv_cache_groups))), "RULE no state groups"
    sched, _ = _pp8_scheduler(monkeypatch, kvc, stages)
    assert sched.restores_state_with_every_load, "RULE the served scheduler's verdict"


def test_offload_connector_reports_divergent_hits_only_with_the_flag(monkeypatch):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    def conn(restores):
        c = object.__new__(TPURaidenOffloadingConnector)
        c.connector_scheduler = None if restores is None else NS(restores_state_with_every_load=restores)
        return c

    assert not conn(True).supports_divergent_local_hybrid_hits, "CONN flag off reports True"
    monkeypatch.setenv(FLAG, "1")
    assert conn(True).supports_divergent_local_hybrid_hits, "CONN flag on, restoring tier"
    assert not conn(False).supports_divergent_local_hybrid_hits, "CONN a tier that cannot restore state"
    assert not conn(None).supports_divergent_local_hybrid_hits, "CONN worker role (no scheduler)"


def _child(role, supports):
    return NS(_kv_transfer_config=NS(kv_role=role), supports_divergent_local_hybrid_hits=supports)


def test_multi_connector_leaves_a_pure_producer_out_of_the_vote(monkeypatch):
    producer, offload = _child("kv_producer", False), _child("kv_both", True)
    assert not _multi([producer, offload]).supports_divergent_local_hybrid_hits, "MULTI flag off: upstream all()"
    assert _multi([_child("kv_both", True)]).supports_divergent_local_hybrid_hits, "MULTI flag off, one supporter"
    monkeypatch.setenv(FLAG, "1")
    assert _multi([producer, offload]).supports_divergent_local_hybrid_hits, "MULTI the PD prefill's children"
    assert not _multi([producer, _child("kv_consumer", False), offload]).supports_divergent_local_hybrid_hits, (
        "MULTI a loading child that does not support it vetoes")
    assert not _multi([producer, _child("kv_both", False)]).supports_divergent_local_hybrid_hits, (
        "MULTI the offload child cannot restore state")
    assert not _multi([producer]).supports_divergent_local_hybrid_hits, "MULTI no loading child"


# -- the split chain on the served PP8 prefill groups --------------------------------


def _split_chain(monkeypatch, *, head_in_tier=False, drop_state=False):
    """HBM: the shared head's attention blocks [0, J), cached by another conversation, with no state checkpoint at
    any boundary (the measured hole: hbm_attn == junction, no HBM checkpoint). Tier: the turn's attention blocks
    [J, K) (and [0, J) with head_in_tier) and every state group's checkpoint at K (one left out with drop_state)."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_offload.base import make_offload_key

    from tests.offload.test_raiden_store_offload import FakeBlockStatus, _Entry
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    _, _, stages, kvc = pp8_pool(monkeypatch, num_blocks=256)
    sched, store = _pp8_scheduler(monkeypatch, kvc, stages)
    _, mamba, attention = _groups(kvc)
    kvm = KVCacheManager(kvc, max_model_len=262144, scheduler_block_size=BLOCK, hash_block_size=BLOCK,
                         enable_caching=True, use_eagle=False)
    head = _tokens(J * BLOCK, 7)
    other = _request("other", head + _tokens(BLOCK + 9, 8))
    blocks, hit, _ = kvm.get_computed_blocks(other)
    assert hit == 0
    assert kvm.allocate_slots(other, other.num_tokens, new_computed_blocks=blocks) is not None
    other.num_computed_tokens = other.num_tokens
    head_ids = {g: [b.block_id for b in kvm.coordinator.single_type_managers[g].req_to_blocks["other"][:J]]
                for g in attention}
    state = [b for g in mamba for b in kvm.coordinator.single_type_managers[g].req_to_blocks["other"]]
    kvm.free(other)
    for b in state:  # HBM's LRU took the junction state; the head's attention blocks stay cached
        if b.block_hash is not None:
            kvm.block_pool._maybe_evict_cached_block(b)

    turn = _request("turn", head + _tokens((K - J) * BLOCK, 9) + _tokens(2000, 10))
    ns = sched._raiden_manager._key_namespace

    def put(gid, chunk):
        key = bytes(make_offload_key(turn.block_hashes[chunk], gid))
        store.entries[ns + key + (0).to_bytes(4, "big")] = _Entry(status=FakeBlockStatus.HOST)

    for gid in attention:
        for chunk in range(0 if head_in_tier else J, K):
            put(gid, chunk)
    for gid in mamba[: len(mamba) - int(drop_state)]:
        put(gid, K - 1)
    offload = object.__new__(TPURaidenOffloadingConnector)
    offload.connector_scheduler = sched
    offload._kv_transfer_config = NS(kv_role="kv_both")
    multi = _multi([_child("kv_producer", False), offload])
    return NS(kvc=kvc, kvm=kvm, sched=sched, store=store, turn=turn, attention=attention, mamba=mamba,
              head_ids=head_ids, multi=multi, ns=ns)


def _admit(s):
    """The scheduler's admission of ``s.turn`` (vLLM ``Scheduler._get_local_prefix_cache_hit`` itself, then the
    connector, the divergent fallback and the allocation as ``Scheduler.schedule`` runs them)."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    req = s.turn
    s.sched.on_new_request(req)
    blocks, local, junction, diverged = Scheduler._get_local_prefix_cache_hit(
        NS(connector=s.multi, kv_cache_manager=s.kvm), req)
    ext, load_async = s.sched.get_num_new_matched_tokens(req, local)
    first = (local, junction, diverged, ext)
    if diverged and ext == 0:
        blocks, local, junction = s.kvm.get_computed_blocks(req)
    loads = []
    if ext:
        assert load_async
        assert s.kvm.allocate_slots(req, 0, num_new_computed_tokens=local, new_computed_blocks=blocks,
                                    num_external_computed_tokens=ext, delay_cache_blocks=True) is not None
        s.sched.update_state_after_alloc(req, s.kvm.get_blocks(req.request_id), ext)
        s.sched.build_connector_meta(SchedulerOutput.make_empty())
        chunk_of = {bytes(h): i for i, h in enumerate(req.block_hashes)}
        for call in s.store.calls_named("load"):
            for sub_hash, block in zip(*call.args):
                key, gid = _key_group_of_sub_hash(sub_hash, s.ns)
                loads.append((gid, chunk_of[bytes(key[:-4])], block))
    return NS(first=first, local=local, ext=ext, loads=loads)


@pytest.mark.usefixtures("engine_patches")
def test_flag_off_the_split_chain_hits_nothing(monkeypatch):
    s = _split_chain(monkeypatch)
    from vllm_torchtpu.offload import lookup_log

    attn, ckpt = lookup_log.hbm_presence(s.kvm.block_pool, s.turn.block_hashes, s.attention, s.mamba, K + 1)
    assert attn == [True] * J + [False] * (K + 1 - J) and not any(ckpt), "HOLE the measured HBM state"
    a = _admit(s)
    local, junction, diverged, ext = a.first
    assert (local, diverged, ext) == (0, False, 0), f"HOLE local 0, convergent, no tier hit: {a.first}"
    assert junction == J * BLOCK, f"HOLE the junction is the head (the measured junction == hbm_attn): {a.first}"
    assert a.loads == [] and not s.store.calls_named("load"), "HOLE nothing loaded"


@pytest.mark.usefixtures("engine_patches")
def test_flag_on_adopts_the_hbm_head_and_loads_only_the_tail_and_the_state(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_torchtpu.offload import lookup_log

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_TPU_PREFIX_LOOKUP_LOG", "1")
    monkeypatch.setattr(lookup_log, "logger", NS(info=lambda *a: None, exception=lambda *a: None))
    lookup_log.patch_prefix_lookup_log()
    assert getattr(KVCacheManager.get_computed_blocks_for_connector, lookup_log._PATCH_FLAG, False), "LOG not hooked"
    assert getattr(Scheduler.add_request, lookup_log._PATCH_FLAG, False)
    s = _split_chain(monkeypatch)
    assert s.multi.supports_divergent_local_hybrid_hits, "SPLIT the prefill composition reports divergent hits"
    a = _admit(s)
    assert a.first == (J * BLOCK, 0, True, (K - J) * BLOCK), f"SPLIT local = head, diverged, tier = tail: {a.first}"
    want = {(g, c) for g in s.attention for c in range(J, K)} | {(g, K - 1) for g in s.mamba}
    assert {(g, c) for g, c, _ in a.loads} == want and len(a.loads) == len(want), (
        f"SPLIT loads exactly the tail and each state checkpoint: {sorted((g, c) for g, c, _ in a.loads)}")
    table = s.kvm.get_block_ids("turn")
    for g in s.attention:
        assert list(table[g][:J]) == s.head_ids[g], "SPLIT the head is HBM's cached blocks, adopted"
        assert not set(table[g][:J]) & {b for gg, _, b in a.loads if gg == g}, "SPLIT no load writes an adopted block"
    # The lookup line's HBM side is recorded on the divergent lookup.
    attn, ckpt = getattr(s.turn, lookup_log._HBM_ATTR, None) or (None, None)
    assert attn is not None and attn[: J + 1] == [True] * J + [False] and not any(ckpt), "LOG HBM side on the divergent lookup"


@pytest.mark.usefixtures("engine_patches")
def test_flag_on_with_the_head_also_in_the_tier_does_not_load_it_again(monkeypatch):
    s_off = _split_chain(monkeypatch, head_in_tier=True)
    off = _admit(s_off)
    assert off.first[:2] == (0, J * BLOCK) and off.ext == K * BLOCK, f"NODUP flag off loads from block 0: {off.first}"
    assert {c for g, c, _ in off.loads if g in s_off.attention} == set(range(K)), "NODUP flag off reloads the head"
    monkeypatch.setenv(FLAG, "1")
    s_on = _split_chain(monkeypatch, head_in_tier=True)
    on = _admit(s_on)
    assert on.first == (J * BLOCK, 0, True, (K - J) * BLOCK), f"NODUP flag on: {on.first}"
    assert {c for g, c, _ in on.loads if g in s_on.attention} == set(range(J, K)), "NODUP the head is not loaded again"


@pytest.mark.usefixtures("engine_patches")
def test_flag_on_without_every_state_checkpoint_falls_back_and_loads_nothing(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    s = _split_chain(monkeypatch, drop_state=True)
    a = _admit(s)
    assert a.first == (J * BLOCK, 0, True, 0), f"FALLBACK diverged, no tier hit: {a.first}"
    assert a.local == 0 and a.loads == [] and not s.store.calls_named("load"), "FALLBACK convergent 0, nothing loaded"


# -- LOG: tier_runs ---------------------------------------------------------------------


def test_tier_runs_report_blocks_past_a_missing_block_zero():
    from vllm_torchtpu.offload import lookup_log
    from vllm_torchtpu.offload.tier_classes import TierClass

    def cls(name, index, kind, groups):
        return TierClass(name=name, index=index, kind=kind, groups=groups, worker_regions=((0,),), row_bytes=1)

    held = {"m2", "m3", "m4", "m5", "k2", "k3", "k5", "s5"}
    store = NS(host_resident_prefix=lambda keys: next((i for i, k in enumerate(keys) if k not in held), len(keys)))
    manager = NS(classes=(cls("main", 0, "main", (4,)), cls("keys", 1, "keys", (5,)), cls("gdn-ab", 2, "gdn", (0,))),
                 managers=[store, store, store])
    keys_of = {4: [f"m{i}" for i in range(6)], 5: [f"k{i}" for i in range(6)], 0: [f"s{i}" for i in range(6)]}
    assert lookup_log.class_runs(manager, keys_of) == {"main": [[2, 6]], "keys": [[2, 4], [5, 6]]}, "LOG tier_runs"
    # tier_class keeps its contiguous-from-the-hit meaning: 0 here.
    assert lookup_log.class_presence(manager, keys_of, 0)["main"] == 0
