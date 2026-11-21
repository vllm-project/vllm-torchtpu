# SPDX-License-Identifier: Apache-2.0
"""q38 ARRIVAL-TOUCH: pin a waiting request's HBM prefix and refresh its host-tier entries at arrival
(VLLM_TPU_ARRIVAL_TOUCH), CPU.

HUNTER-19 (reports/hunt_iter19_b1_cliff_1121.md): on the B1 prefill at c227 the queue wait (median 45.8 s) outlived the
HBM pool's retention (about 22 s), so returners that arrived with their prefix in HBM lost it while they waited. With
the flag the prefix is pinned when the request enters the waiting queue and released when it is scheduled or ends, and
the host-tier entries a load would read are moved to the store's most recently used end.

Rows (tag in each assertion message):
- FLAG     the five keys are registered once, the master switch is off by default, and all are compile-ignored.
- IDENT    flag off: the patch installs nothing and a scripted run on a real vLLM scheduler is identical to the pristine
           one (scheduled requests, hits, block tables, free-queue order); flag on with the HBM pins and the tier
           refresh off: identical as well (the arrival lookup reads the cache only); flag on: the same script differs.
- PIN      a returner's cached prefix survives other work taking every free block while it waits (flag off: lost).
- RELS     the pin is released when the request is scheduled: every adopted block has one owner, nothing stays pinned.
- RELF     the pin is released when a waiting request is aborted: the blocks are free again, hashes kept.
- BOUND    a pin past the HBM_FRAC budget of distinct blocks is refused, and that prefix is lost as with the flag off;
           of two requests the second is refused when the two would pass the budget.
- SHARED   two waiting requests on one prefix: its blocks count once in the budget, and releasing one keeps the other.
- FLOOR    a pin that would take the pool below MIN_FREE_FRAC is refused; below the floor, the newest pins go first.
- TTL      a pin older than TTL_S is released before the next schedule pass.
- HYBRID   served PP8 groups (real KVCacheManager, engine patches): the pin is exactly the convergent hit plus, with
           divergent hits, the attention run; every block the scheduler then adopts was pinned; a full drain of the free
           queue evicts none of them.
- TIER     served tier classes: the refresh asks each class for the keys a load would read past the local hit, in one
           pinned lookup and one release each, loads nothing and leaves no pin.
- HOOK     the scheduler hands the tier member the block-aligned local hit; TIER=0 skips it; a failing tier stops only
           the refresh.
"""

from types import SimpleNamespace as NS

import pytest

FLAG = "VLLM_TPU_ARRIVAL_TOUCH"
KNOBS = ("VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC", "VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC", "VLLM_TPU_ARRIVAL_TOUCH_TTL_S",
         "VLLM_TPU_ARRIVAL_TOUCH_TIER")
BS = 16  # real-scheduler block size (tokens)
NB = 128  # real-scheduler pool (blocks, the null block included)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    for name in (FLAG, *KNOBS, "QWEN4_EXP_PACK_COMPRESSED_K", "QWEN4_EXP_PACK_MAIN_KV",
                 "QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR", "VLLM_TPU_OFFLOAD_DIVERGENT_HITS", "VLLM_TPU_PREFIX_LOOKUP_LOG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def restore_scheduler():
    """vLLM's Scheduler methods restored after a test installs the patch."""
    from vllm.v1.core.sched.scheduler import Scheduler

    saved = {name: Scheduler.__dict__[name] for name in ("add_request", "schedule", "finish_requests")}
    yield
    for name, value in saved.items():
        setattr(Scheduler, name, value)


def _install(monkeypatch, on=True, **knobs):
    from vllm_torchtpu.core import arrival_touch

    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    for key, value in knobs.items():
        monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH_" + key.upper(), str(value))
    arrival_touch.patch_arrival_touch()
    return arrival_touch


# -- FLAG -------------------------------------------------------------------------------


def test_flag_keys_registered_defaults_and_compile_ignored(monkeypatch):
    import re

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    src = open(envs.__file__).read()
    for key in (FLAG, *KNOBS):
        assert key in envs.environment_variables, f"FLAG {key} not registered"
        assert len(re.findall(r'^    "%s": ' % key, src, re.M)) == 1, f"FLAG {key} registered more than once"
        assert key in tpu_compiler._TPU_COMPILE_ENV_IGNORED, f"FLAG {key} is not compile-ignored"
    assert envs.VLLM_TPU_ARRIVAL_TOUCH is False, "FLAG master switch is not off by default"
    assert envs.VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC == 0.25 and envs.VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC == 0.10
    assert envs.VLLM_TPU_ARRIVAL_TOUCH_TTL_S == 120.0 and envs.VLLM_TPU_ARRIVAL_TOUCH_TIER is True
    monkeypatch.setenv(FLAG, "1")
    assert envs.VLLM_TPU_ARRIVAL_TOUCH is True, "FLAG does not read 1"


# -- a real vLLM scheduler on CPU ---------------------------------------------------------


def _scheduler(monkeypatch, num_blocks=NB):
    """A real vLLM scheduler on CPU with prefix caching and no connector (= test_prefix_lookup_log._scheduler,
    credited; pool size a parameter)."""
    import torch
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    config = VllmConfig(
        scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=2048, max_model_len=2048,
                                         enable_chunked_prefill=True, is_encoder_decoder=False),
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048),
        cache_config=CacheConfig(block_size=BS, gpu_memory_utilization=0.9, cache_dtype="auto",
                                 enable_prefix_caching=True),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=64,
                                                      dtype=torch.float16))])
    init_none_hash(sha256)
    return Scheduler(vllm_config=config, kv_cache_config=kv_cache_config,
                     structured_output_manager=StructuredOutputManager(config), block_size=BS, hash_block_size=BS,
                     log_stats=False)


def _req(req_id, tokens):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request

    return Request(req_id, list(tokens), SamplingParams(max_tokens=4), None, arrival_time=0.0,
                   block_hasher=get_request_block_hasher(BS, sha256))


def _toks(n, seed):
    return [1000 * (seed + 1) + t for t in range(n)]


def _abort(sched, *req_ids):
    from vllm.v1.request import RequestStatus

    sched.finish_requests(list(req_ids), RequestStatus.FINISHED_ABORTED)


def _cached_turn(sched, req_id, tokens):
    """A finished turn whose prefix sits in the free queue as cached blocks: allocated (hashed) and freed through the
    scheduler's own KV cache manager, as a turn that prefilled and finished seconds ago leaves it. (Not through
    schedule + abort: a request aborted before its step's output is processed does not leave a usable prefix here,
    dev run j-0f4adcd0.) The same-step patch's unwritten set is cleared as the next pass would."""
    kvm = sched.kv_cache_manager
    req = _req(req_id, tokens)
    blocks, hit, _ = kvm.get_computed_blocks(req)
    assert kvm.allocate_slots(req, req.num_tokens - hit, num_new_computed_tokens=hit,
                              new_computed_blocks=blocks) is not None
    req.num_computed_tokens = req.num_tokens
    kvm.free(req)
    kvm.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
    probe = _req(req_id + "-probe", tokens + [7])
    _, hit, _ = kvm.coordinator.find_longest_cache_hit(probe.block_hashes, probe.num_tokens - 1)
    assert hit == len(tokens) // BS * BS, f"precondition: the finished turn is cached ({hit})"


def _churn(sched, keep_free=0):
    """Other work takes every free block but ``keep_free`` (evicting what it takes) and gives them back hashless."""
    pool = sched.kv_cache_manager.block_pool
    taken = pool.get_new_blocks(pool.get_num_free_blocks() - keep_free)
    pool.free_blocks(taken)
    return len(taken)


def _admit(sched, req_id):
    """One schedule pass; returns the local hit (tokens) the request was admitted with."""
    out = sched.schedule()
    new = {r.req_id: r for r in out.scheduled_new_reqs}
    assert req_id in new, f"{req_id} not admitted: {list(new)}"
    return new[req_id].num_computed_tokens


def _pins(sched):
    from vllm_torchtpu.core import arrival_touch

    return sched.__dict__.get(arrival_touch._STATE_ATTR)


A1 = _toks(30 * BS, 1)            # a finished turn: 30 cached blocks
A2 = A1 + _toks(2 * BS, 2)        # its returner: 32 blocks, 30 of them cached


def _returner_run(monkeypatch, on, **knobs):
    """A1 cached and freed; A2 arrives; other work takes every free block; A2 is scheduled."""
    if on is not None:
        _install(monkeypatch, on=on, **knobs)
    sched = _scheduler(monkeypatch)
    _cached_turn(sched, "A1", A1)
    sched.add_request(_req("A2", A2))
    _churn(sched)
    return sched, _admit(sched, "A2")


# -- IDENT ------------------------------------------------------------------------------------


def _script(monkeypatch):
    """Adds, schedule passes, aborts and churn on a real scheduler; returns what the scheduler did."""
    from vllm_torchtpu.core import arrival_touch

    t = iter(range(1000))
    monkeypatch.setattr(arrival_touch, "_clock", lambda: float(next(t)))
    sched = _scheduler(monkeypatch)
    kvm = sched.kv_cache_manager
    trace = []

    def step():
        out = sched.schedule()
        trace.append((
            sorted((r.req_id, r.num_computed_tokens, tuple(map(tuple, r.block_ids))) for r in out.scheduled_new_reqs),
            sorted(out.num_scheduled_tokens.items()),
            kvm.block_pool.get_num_free_blocks(),
            tuple(b.block_id for b in kvm.block_pool.free_block_queue.get_all_free_blocks()),
            len(kvm.block_pool.cached_block_hash_to_block),
        ))

    x1 = _toks(20 * BS, 11)
    sched.add_request(_req("X1", x1))
    sched.add_request(_req("Y1", _toks(15 * BS + 3, 12)))
    step()
    _abort(sched, "X1")
    step()
    sched.add_request(_req("X2", x1 + _toks(2 * BS, 13)))
    sched.add_request(_req("Z1", _toks(25 * BS, 14)))
    _churn(sched, keep_free=5)
    step()
    _abort(sched, "Y1", "Z1")
    step()
    sched.add_request(_req("X3", x1 + _toks(3 * BS + 5, 15)))
    sched.add_request(_req("W1", _toks(10 * BS, 16)))
    _abort(sched, "W1")
    _churn(sched, keep_free=5)
    step()
    _abort(sched, "X2", "X3")
    step()
    return trace


def test_flag_off_installs_nothing_and_runs_identically(monkeypatch, restore_scheduler):
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_torchtpu.core import arrival_touch

    pristine = _script(monkeypatch)
    _install(monkeypatch, on=False)
    assert not getattr(Scheduler.add_request, arrival_touch._PATCH_FLAG, False), "IDENT flag off wrapped add_request"
    assert not getattr(Scheduler.schedule, arrival_touch._PATCH_FLAG, False), "IDENT flag off wrapped schedule"
    assert _script(monkeypatch) == pristine, "IDENT flag off differs from pristine"
    _install(monkeypatch, on=True, hbm_frac=0, tier=0)
    assert getattr(Scheduler.add_request, arrival_touch._PATCH_FLAG, False)
    assert _script(monkeypatch) == pristine, "IDENT the arrival lookup changed the scheduler (pins and tier off)"
    monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC", "0.5")
    assert _script(monkeypatch) != pristine, "IDENT flag on: the script cannot see the pins"


# -- PIN / RELS / RELF ------------------------------------------------------------------------


def test_pinned_prefix_survives_the_wait(monkeypatch, restore_scheduler):
    _, hit_off = _returner_run(monkeypatch, on=None)
    assert hit_off == 0, f"PIN flag off: the churn evicts the waiting returner's prefix ({hit_off})"
    sched, hit_on = _returner_run(monkeypatch, on=True)
    assert hit_on == 30 * BS, f"PIN flag on: the pinned prefix survives the churn ({hit_on})"
    s = _pins(sched).stats
    assert (s["arrivals"], s["pinned"], s["pinned_blocks"]) == (1, 1, 30), f"PIN counters {s}"


def test_pin_is_released_when_the_request_is_scheduled(monkeypatch, restore_scheduler):
    sched, hit = _returner_run(monkeypatch, on=True)
    assert hit == 30 * BS
    pins = _pins(sched)
    assert not pins.pins and not pins.refs and sum(pins.unique.values()) == 0, "RELS pins left after schedule"
    blocks = sched.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks["A2"]
    assert all(b.ref_cnt == 1 for b in blocks[:30]), f"RELS adopted blocks: {[b.ref_cnt for b in blocks[:30]]}"
    assert pins.stats["rel_sched"] == 1 and pins.stats["adopted_blocks"] == 30, f"RELS counters {pins.stats}"


def test_pin_is_released_when_a_waiting_request_is_aborted(monkeypatch, restore_scheduler):
    _install(monkeypatch)
    sched = _scheduler(monkeypatch)
    pool = sched.kv_cache_manager.block_pool
    _cached_turn(sched, "A1", A1)
    free0 = pool.get_num_free_blocks()
    sched.add_request(_req("A2", A2))
    pins = _pins(sched)
    held = pins.pins["A2"].groups[0]
    assert len(held) == 30 and pool.get_num_free_blocks() == free0 - 30, "RELF pinned 30 blocks out of the free queue"
    _abort(sched, "A2")
    assert "A2" not in pins.pins and not pins.refs, "RELF the abort left the pin"
    assert pool.get_num_free_blocks() == free0 and all(b.ref_cnt == 0 for b in held), "RELF blocks not free again"
    assert all(b.block_hash is not None for b in held), "RELF a released block lost its hash"
    tail = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()[-30:]]
    assert sorted(tail) == sorted(b.block_id for b in held), "RELF released blocks are not at the MRU end"
    assert pins.stats["rel_finish"] == 1


# -- BOUND / SHARED / FLOOR / TTL -------------------------------------------------------------


def test_budget_bounds_the_pinned_blocks(monkeypatch, restore_scheduler):
    # 0.2 of 128 blocks = 25.6 < 30: refused, and the churn takes the prefix as with the flag off.
    sched, hit = _returner_run(monkeypatch, on=True, hbm_frac=0.2)
    s = _pins(sched).stats
    assert (s["pinned"], s["denied_budget"]) == (0, 1), f"BOUND a 30-block pin past a 25-block budget: {s}"
    assert hit == 0, f"BOUND the refused prefix is lost ({hit})"
    # 0.4 of 128 = 51.2: A2's 30 blocks fit, B2's 25 more do not.
    monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC", "0.4")
    sched = _scheduler(monkeypatch)
    b1 = _toks(25 * BS, 5)
    _cached_turn(sched, "A1", A1)
    _cached_turn(sched, "B1", b1)
    sched.add_request(_req("A2", A2))
    sched.add_request(_req("B2", b1 + _toks(BS, 6)))
    pins = _pins(sched)
    assert list(pins.pins) == ["A2"] and pins.stats["denied_budget"] == 1, f"BOUND second pin {pins.stats}"
    assert sum(pins.unique.values()) == 30 <= 0.4 * NB


def test_a_shared_prefix_counts_once_and_survives_one_release(monkeypatch, restore_scheduler):
    _install(monkeypatch, hbm_frac=0.3)  # 38 blocks: two 30-block pins of one prefix fit only if counted once
    sched = _scheduler(monkeypatch)
    _cached_turn(sched, "A1", A1)
    sched.add_request(_req("A2", A2))
    sched.add_request(_req("A2b", A1 + _toks(3 * BS, 7)))
    pins = _pins(sched)
    assert sorted(pins.pins) == ["A2", "A2b"], f"SHARED both pinned {pins.stats}"
    assert sum(pins.unique.values()) == 30, f"SHARED distinct blocks counted once: {pins.unique}"
    held = pins.pins["A2"].groups[0]
    assert all(b.ref_cnt == 2 for b in held)
    _abort(sched, "A2")
    assert all(b.ref_cnt == 1 for b in held) and sum(pins.unique.values()) == 30, "SHARED the other pin kept"
    _churn(sched)
    assert _admit(sched, "A2b") == 30 * BS, "SHARED the remaining pin still protects the prefix"


def test_free_floor_refuses_and_releases_newest_first(monkeypatch, restore_scheduler):
    from vllm.v1.core.sched.interface import PauseState

    from vllm_torchtpu.core import arrival_touch

    ticks = iter(range(10, 10_000))
    monkeypatch.setattr(arrival_touch, "_clock", lambda: float(next(ticks)))

    # Refused: pinning 30 of 127 free blocks would leave 97 < 0.9 x 128.
    sched, hit = _returner_run(monkeypatch, on=True, min_free_frac=0.9)
    s = _pins(sched).stats
    assert (s["pinned"], s["denied_floor"]) == (0, 1) and hit == 0, f"FLOOR refusal {s}"
    # Released: two pins, then other work holds all but 5 free blocks (< 0.1 x 128).
    monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC", "0.1")
    monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC", "1")
    sched = _scheduler(monkeypatch)
    pool = sched.kv_cache_manager.block_pool
    b1 = _toks(25 * BS, 5)
    _cached_turn(sched, "A1", A1)
    _cached_turn(sched, "B1", b1)
    sched.add_request(_req("A2", A2))
    sched.add_request(_req("B2", b1 + _toks(BS, 6)))
    pins = _pins(sched)
    assert sorted(pins.pins) == ["A2", "B2"] and pins.pins["B2"].t > pins.pins["A2"].t  # B2 is the newer pin
    held = pool.get_new_blocks(pool.get_num_free_blocks() - 5)
    sched.set_pause_state(PauseState.PAUSED_NEW)
    sched.schedule()
    assert list(pins.pins) == ["A2"] and pins.stats["rel_pressure"] == 1, f"FLOOR newest first {pins.stats}"
    assert pool.get_num_free_blocks() == 30 >= 0.1 * NB
    pool.free_blocks(held)


def test_ttl_releases_old_pins(monkeypatch, restore_scheduler):
    from vllm.v1.core.sched.interface import PauseState

    from vllm_torchtpu.core import arrival_touch

    now = [100.0]
    monkeypatch.setattr(arrival_touch, "_clock", lambda: now[0])
    _install(monkeypatch, ttl_s=5)
    sched = _scheduler(monkeypatch)
    _cached_turn(sched, "A1", A1)
    sched.add_request(_req("A2", A2))
    sched.set_pause_state(PauseState.PAUSED_NEW)
    now[0] = 104.0
    sched.schedule()
    assert "A2" in _pins(sched).pins, "TTL released early"
    now[0] = 106.0
    sched.schedule()
    assert "A2" not in _pins(sched).pins and _pins(sched).stats["rel_ttl"] == 1, "TTL pin kept past ttl"


# -- HYBRID: the served PP8 groups --------------------------------------------------------------


@pytest.fixture
def engine_patches():
    """The prefill engine-core patches (= test_q38_holefix_divhit.engine_patches, credited); vLLM restored after."""
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    from vllm_torchtpu.core.divergent_hybrid_hits import patch_divergent_hybrid_hits
    from vllm_torchtpu.core.mamba_checkpoint_retention import patch_mamba_checkpoint_retention

    saved = [
        (cls, name, cls.__dict__[name])
        for cls, names in (
            (MambaManager, ("cache_blocks", "_remove_blocks_in_range", "remove_skipped_blocks", "pop_blocks_for_free")),
            (KVCacheCoordinator, ("free", "pop_blocks_for_free")),
            (KVCacheManager, ("get_computed_blocks", "get_computed_blocks_for_connector")),
        )
        for name in names
    ]
    patch_mamba_checkpoint_retention()
    patch_divergent_hybrid_hits()
    yield
    for cls, name, value in saved:
        setattr(cls, name, value)


def _pp8(monkeypatch, evict_state):
    """Served PP8 prefill groups: another conversation computed a shared head (J blocks) and one block more. Split
    chain (``evict_state``, the measured hole): HBM's LRU took its state checkpoints and the turn shares only the head.
    Convergent: the checkpoints stay and the turn continues the other conversation's whole prompt."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import MambaSpec

    from tests.runner.test_qwen4_exp_host_tier import BLOCK, _request, pp8_pool
    from tests.runner.test_qwen4_exp_offload import _tokens

    _, _, _, kvc = pp8_pool(monkeypatch, num_blocks=256)
    kvm = KVCacheManager(kvc, max_model_len=262144, scheduler_block_size=BLOCK, hash_block_size=BLOCK,
                         enable_caching=True, use_eagle=False)
    mamba = [g for g, grp in enumerate(kvc.kv_cache_groups) if isinstance(grp.kv_cache_spec, MambaSpec)]
    J = 3
    head = _tokens(J * BLOCK, 7)
    other = _request("other", head + _tokens(BLOCK + 9, 8))
    blocks, hit, _ = kvm.get_computed_blocks(other)
    assert kvm.allocate_slots(other, other.num_tokens, new_computed_blocks=blocks) is not None
    other.num_computed_tokens = other.num_tokens
    state = [b for g in mamba for b in kvm.coordinator.single_type_managers[g].req_to_blocks["other"]]
    kvm.free(other)
    if evict_state:
        for b in state:
            if b.block_hash is not None:
                kvm.block_pool._maybe_evict_cached_block(b)
    # The next schedule pass would clear the same-step patch's unwritten set (no-op when the patch is not active).
    kvm.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
    if evict_state:
        turn = _request("turn", head + _tokens(3 * BLOCK, 9) + _tokens(2000, 10))
    else:
        turn = _request("turn", list(other.prompt_token_ids) + _tokens(2000, 10))
    return kvm, turn, J, BLOCK


def _state_groups(kvm, groups):
    from vllm.v1.kv_cache_interface import MambaSpec

    specs = [g.kv_cache_spec for g in kvm.kv_cache_config.kv_cache_groups]
    return [[b for b in g if not b.is_null] for g, spec in zip(groups, specs) if isinstance(spec, MambaSpec)]


def _ids(groups):
    return [sorted(b.block_id for b in g if not b.is_null) for g in groups]


@pytest.mark.usefixtures("engine_patches")
@pytest.mark.parametrize("evict_state", [True, False], ids=["split", "convergent"])
def test_served_groups_pin_the_attention_run_and_the_state_checkpoint(monkeypatch, evict_state):
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from vllm_torchtpu.core import arrival_touch
    from vllm_torchtpu.core.divergent_hybrid_hits import divergent_local_hit

    kvm, turn, J, BLOCK = _pp8(monkeypatch, evict_state)
    conn = NS(supports_divergent_local_hybrid_hits=True)
    groups, local = arrival_touch.arrival_blocks(kvm, conn, turn)
    conv_blocks, conv_hit, _ = KVCacheManager.get_computed_blocks(kvm, turn)
    div_blocks, div_hit, _, _ = divergent_local_hit(kvm, turn)
    want = [sorted(set(a) | set(b)) for a, b in zip(_ids(conv_blocks.blocks), _ids(div_blocks.blocks))]
    assert _ids(groups) == want, "HYBRID the pin is the convergent hit plus the attention run"
    assert local == max(conv_hit, div_hit), f"HYBRID local {local} conv {conv_hit} div {div_hit}"
    if evict_state:
        assert conv_hit == 0 and div_hit == J * BLOCK, "HYBRID the split chain: only the divergent lookup hits"
        assert not any(b for g in _state_groups(kvm, groups) for b in g), "HYBRID split chain: no state block to pin"
    else:
        assert conv_hit > 0, f"HYBRID convergent: a checkpoint backs the hit ({conv_hit})"
        assert all(_state_groups(kvm, groups)), "HYBRID convergent: every state group's checkpoint is pinned"
    # Without divergent hits the pin is the convergent hit alone.
    g_conv, l_conv = arrival_touch.arrival_blocks(kvm, NS(supports_divergent_local_hybrid_hits=False), turn)
    assert _ids(g_conv) == _ids(conv_blocks.blocks) and l_conv == conv_hit, "HYBRID convergent-only pin"
    pins = arrival_touch.ArrivalPins(hbm_frac=1.0, min_free_frac=0.0, ttl_s=1e9, tier=False)
    assert pins.pin(kvm, "turn", groups, 0.0)
    pinned = {id(b) for g in groups for b in g if not b.is_null}
    pool = kvm.block_pool
    drained = pool.get_new_blocks(pool.get_num_free_blocks())
    assert not {id(b) for b in drained} & pinned, "HYBRID a drain took a pinned block"
    assert all(b.block_hash is not None for g in groups for b in g if not b.is_null), "HYBRID a pinned block lost its hash"
    pool.free_blocks(drained)
    # The scheduler's own choice (divergent, falling back to convergent when the tier has nothing) adopts pinned blocks only.
    if div_hit > conv_hit:  # divergent: the head is adopted and the tier's load supplies the rest
        blocks, hit = div_blocks, div_hit
        assert kvm.allocate_slots(turn, 0, num_new_computed_tokens=hit, new_computed_blocks=blocks,
                                  num_external_computed_tokens=BLOCK, delay_cache_blocks=True) is not None
    else:  # convergent: a local hit, the rest computed
        blocks, hit = conv_blocks, conv_hit
        assert kvm.allocate_slots(turn, turn.num_tokens - hit, num_new_computed_tokens=hit,
                                  new_computed_blocks=blocks) is not None
    adopted = {id(b) for g in blocks.blocks for b in g if not b.is_null}
    assert adopted <= pinned, "HYBRID the scheduler adopted a block the pin did not hold"
    pins.release(kvm, "turn", "rel_sched")
    assert all(b.ref_cnt == 1 for g in blocks.blocks for b in g if not b.is_null), "HYBRID one owner after release"


# -- TIER / HOOK ----------------------------------------------------------------------------------


def _tier_turn(monkeypatch):
    """Served tier classes holding one finished 3-block turn (= test_tier_classes, credited)."""
    from vllm.v1.core.sched.output import SchedulerOutput

    from tests.offload.test_tier_classes import _ack_fences, _tier_scheduler, _turn_one
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _tokens
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    sched, kvc, stores = _tier_scheduler(monkeypatch)
    first, _, meta = _turn_one(sched, kvc, boundaries=(2, 3))
    _ack_fences(sched, meta)
    for store in stores:
        store.complete_save([h for c in store.calls_named("save") for h in c.args[0]])
    sched.build_connector_meta(SchedulerOutput.make_empty())
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn.connector_scheduler = sched
    conn._bounding_group_ids = ()
    turn = request("r2", first.prompt_token_ids + _tokens(2000, 1))
    return sched, conn, stores, turn


def _expected_keys(sched, turn, start, end):
    from vllm.v1.kv_offload.base import make_offload_key

    keys = []
    for group in sched.config.kv_group_configs:
        tpc, step = group.tokens_per_chunk, group.hashes_per_chunk
        stop = end // tpc
        first = start // tpc if group.sliding_window_size_in_chunks is None else max(
            stop - group.sliding_window_size_in_chunks, 0)
        keys += [make_offload_key(turn.block_hashes[(c + 1) * step - 1], group.group_idx) for c in range(first, stop)]
    return keys


@pytest.mark.parametrize("local_blocks", [0, 1])
def test_tier_refresh_asks_each_class_once_and_loads_nothing(monkeypatch, local_blocks):
    from tests.runner.test_qwen4_exp_host_tier import BLOCK

    sched, conn, stores, turn = _tier_turn(monkeypatch)
    start = local_blocks * BLOCK
    hit = conn.host_tier_hit_tokens(turn, start)
    assert hit == 3 * BLOCK - start, f"TIER the tier holds the turn's 3 blocks ({hit})"
    marks = [len(s.calls) for s in stores]
    n = conn.arrival_touch(turn, start)
    want = _expected_keys(sched, turn, start, start + hit)
    assert n == len(want) > 0, f"TIER keys asked {n} want {len(want)}"
    manager = sched._raiden_manager
    want_by_class = {}
    for key in want:
        for c in manager.classes_of(key):
            want_by_class.setdefault(c, []).extend(manager.managers[c]._expand_keys([key]))
    for c, store in enumerate(stores):
        new = store.calls[marks[c]:]
        names = [call.name for call in new]
        if c not in want_by_class:
            assert names == [], f"TIER class {c} asked without keys: {names}"
            continue
        # host_tier_hit_tokens probes first (pin_found=False, LRU untouched); then one pinned lookup and one release.
        probes = [call for call in new if call.name == "lookup" and not call.args[2]]
        pinned = [call for call in new if call.name == "lookup" and call.args[2]]
        released = [call for call in new if call.name == "release"]
        assert len(pinned) == 1 and len(released) == 1 and len(probes) + 2 == len(new), (
            f"TIER class {c}: probes, then one pinned lookup and one release, got {names}")
        assert new[-2] is pinned[0] and new[-1] is released[0], f"TIER class {c} order {names}"
        hashes, enable_global, _ = pinned[0].args
        assert hashes == want_by_class[c] and not enable_global, f"TIER class {c} lookup keys / local only"
        assert released[0].args[0] == want_by_class[c], f"TIER class {c} releases what it pinned"
        assert all(e.pin_count == 0 for e in store.entries.values()), f"TIER class {c} left a pin"
    moved = [c.name for i, st in enumerate(stores) for c in st.calls[marks[i]:] if c.name not in ("lookup", "release")]
    assert moved == [], f"TIER the refresh called {moved}"


def test_tiered_touch_run_routes_each_key_to_its_class():
    from vllm_torchtpu.offload.tier_classes import TieredOffloadingManager

    seen = {0: [], 1: []}
    manager = object.__new__(TieredOffloadingManager)
    manager._managers = [NS(touch_run=lambda ks, c=c: seen[c].extend(ks) or len(ks)) for c in (0, 1)]
    manager._classes_of_group = {4: (0,), 0: (1,)}
    from vllm.v1.kv_offload.base import make_offload_key

    keys = [make_offload_key(bytes([i]) * 32, g) for i, g in ((1, 4), (2, 0), (3, 4))]
    assert manager.touch_run(keys) == 3
    assert seen == {0: [keys[0], keys[2]], 1: [keys[1]]}, "TIER routing"


def test_scheduler_hands_the_tier_member_the_block_aligned_local_hit(monkeypatch):
    from vllm_torchtpu.core import arrival_touch

    calls = []
    member = NS(arrival_touch=lambda request, local: calls.append(local) or 7)
    multi = NS(sub_connectors=[NS(), member], supports_divergent_local_hybrid_hits=False)
    hits = {"r": 37}
    kvm = NS(enable_caching=True, prefix_cache_lookup_enabled=lambda r: True,
             coordinator=NS(find_longest_cache_hit=lambda h, n: ([[]], hits["r"], 0), single_type_managers=[]))
    sched = NS(kv_cache_manager=kvm, connector=multi, block_size=16)
    pins = arrival_touch.ArrivalPins(hbm_frac=0.0, min_free_frac=0.1, ttl_s=10, tier=True)
    arrival_touch.on_arrival(sched, NS(request_id="r", block_hashes=[], num_tokens=100), pins)
    assert calls == [32] and pins.stats["tier_requests"] == 1 and pins.stats["tier_keys"] == 7, f"HOOK {calls}"
    pins.tier = False
    arrival_touch.on_arrival(sched, NS(request_id="r", block_hashes=[], num_tokens=100), pins)
    assert calls == [32], "HOOK TIER=0 still refreshed"

    def boom(request, local):
        raise RuntimeError("store down")

    pins = arrival_touch.ArrivalPins(hbm_frac=0.0, min_free_frac=0.1, ttl_s=10, tier=True)
    sched.connector = NS(sub_connectors=[NS(arrival_touch=boom)], supports_divergent_local_hybrid_hits=False)
    arrival_touch.on_arrival(sched, NS(request_id="r", block_hashes=[], num_tokens=100), pins)
    assert pins.tier_off and pins.stats["tier_errors"] == 1 and not pins.off, "HOOK a tier failure stops the refresh only"
    line = pins.line()
    assert line.startswith("arrival-touch: arrivals=1 ") and "tier_errors=1" in line
