# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-VASYNC: VLLM_TPU_OFFLOAD_VICTIM_ASYNC (asynchronous victim stores behind an evict-ahead hold).

The scheduler connector is the served one (tests/offload/test_tier_classes.py ``_tier_scheduler``: PP8 groups, host tier
classes, the victim policy); the block pool is vLLM's own ``BlockPool`` with the victim hooks. A small world model checks
integrity from outside the connector:

* D1 no block enters the free queue while a store reading it is in flight (the fake stores' ``_saving``);
* D2 when the pass's metadata ships, no block a request holds is the source of a store in flight (the new owner's forward
  writes it next);
* D3 every store that lands reads, from its device block, the bytes of the key it is published under;
* D4 a parked block is in neither the free queue nor the prefix cache and no request holds it; and every block is in
  exactly one place (free queue, parked, a request, the null block).

Each test prints one ``Q38_VASYNC <NAME>`` line.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time

import pytest

from tests.offload.test_raiden_store_offload import FakeKVCacheStore

BS = 16
ENV = "VLLM_TPU_OFFLOAD_VICTIM_ASYNC"
MAIN, KEYS = 4, 5  # the served model's main-KV and compressed-key groups (two host tier classes)


@pytest.fixture(autouse=True)
def _clean():
    from vllm_torchtpu.offload import raiden_connector as rc
    from vllm_torchtpu.offload import victim

    victim.deactivate()
    rc._VASYNC = None
    yield
    victim.deactivate()
    rc._VASYNC = None


def _mark(name, **kv):
    print(f"Q38_VASYNC {name} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


# -- the world model ---------------------------------------------------------------------------------------------


class _World:
    """Device memory (block id -> bytes, one token per key), every store's saves in flight, and the violations."""

    def __init__(self, rng=None, latency=(1, 4), fail_p=0.0):
        self.rng = rng or random.Random(0)
        self.latency = latency
        self.fail_p = fail_p
        self.hbm: dict[int, object] = {}
        self.stores: list[FakeKVCacheStore] = []
        self.ns = b""
        self.violations: list[str] = []
        self.landed = 0

    def key_bytes(self, key) -> tuple:
        return ("K", bytes(key))

    def inflight_blocks(self) -> set[int]:
        return {s.entries[h].device_block_id for s in self.stores for h in s._saving}

    def land(self, store, sub_hash) -> None:
        block_id = store.entries[sub_hash].device_block_id
        key = sub_hash[len(self.ns):-4]
        want, got = self.key_bytes(key), self.hbm.get(block_id)
        self.landed += 1
        if got != want:
            self.violations.append(f"D3 store of key {key[:6].hex()} landed from block {block_id} holding {got!r}")

    def guard_queue(self, pool) -> None:
        """D1 on every way into the free queue."""
        queue = pool.free_block_queue
        for name in ("append", "append_n", "prepend_n"):
            orig = getattr(queue, name)

            def wrapped(arg, _orig=orig, _many=name != "append", _name=name):
                blocks = list(arg) if _many else [arg]
                live = self.inflight_blocks()
                for b in blocks:
                    if b.block_id in live:
                        self.violations.append(f"D1 block {b.block_id} entered the free queue ({_name}) with its store in flight")
                return _orig(blocks) if _many else _orig(arg)

            setattr(queue, name, wrapped)

    def check_ship(self, held_blocks, t) -> None:
        live = self.inflight_blocks()
        for b in held_blocks:
            if b.block_id in live:
                self.violations.append(f"D2 pass {t}: block {b.block_id} is held by a request while its store is in flight")

    def check_places(self, sched, pool, held_blocks, t) -> None:
        vasync = sched.__dict__.get("_vasync")
        parked = [b for rec in vasync.parked.values() for b in rec.blocks] if vasync is not None else []
        queued = []
        b = pool.free_block_queue.fake_free_list_head.next_free_block
        while b is not None and b is not pool.free_block_queue.fake_free_list_tail:
            queued.append(b)
            if len(queued) > pool.num_gpu_blocks:
                self.violations.append(f"D4 pass {t}: the free queue does not end within the pool (a cycle)")
                return
            b = b.next_free_block
        qids = {x.block_id for x in queued}
        for p in parked:
            if p.block_id in qids or p.prev_free_block is not None or p.next_free_block is not None:
                self.violations.append(f"D4 pass {t}: parked block {p.block_id} is in the free queue")
            if p.block_hash is not None:
                self.violations.append(f"D4 pass {t}: parked block {p.block_id} is still in the prefix cache")
            if p.ref_cnt != 0:
                self.violations.append(f"D4 pass {t}: parked block {p.block_id} has ref_cnt {p.ref_cnt}")
        held_ids = sorted({x.block_id for x in held_blocks})
        ids = [x.block_id for x in queued] + [x.block_id for x in parked] + held_ids + [0]
        if len(queued) != pool.free_block_queue.num_free_blocks:
            self.violations.append(f"D4 pass {t}: free queue walks {len(queued)} blocks, counts {pool.free_block_queue.num_free_blocks}")
        if sorted(ids) != list(range(pool.num_gpu_blocks)):
            self.violations.append(f"D4 pass {t}: blocks not each in exactly one place ({len(ids)} vs {pool.num_gpu_blocks})")
        if vasync is not None and sum(vasync.count.values()) != len(parked):
            self.violations.append(f"D4 pass {t}: parked count {sum(vasync.count.values())} vs {len(parked)} parked blocks")


def _check_pins(world, t) -> None:
    """D5: every save in flight still holds its admission's pin (FakeKVCacheStore.save needs one, and so does a retry)."""
    for s in world.stores:
        for h in s._saving:
            if s.entries[h].pin_count < 1:
                world.violations.append(f"D5 pass {t}: an in-flight save lost its admission pin (block {s.entries[h].device_block_id})")


def _check_targets(world, sched, blocks, t, what) -> bool:
    """D6: no arrival-touch target (pinned on arrival or adopted when scheduled) is a parked block."""
    vasync = sched.__dict__.get("_vasync")
    parked = {b.block_id for rec in vasync.parked.values() for b in rec.blocks} if vasync is not None else set()
    bad = [b.block_id for b in blocks if b.block_id in parked]
    if bad:
        world.violations.append(f"D6 pass {t}: {what} targets parked blocks {bad}")
    return not bad


def _check_host_hits(world, keys, t) -> None:
    """D7: a key the host tier counts as resident is on the host, not a save in flight."""
    from tests.offload.test_raiden_store_offload import FakeBlockStatus

    host = (FakeBlockStatus.HOST, FakeBlockStatus.HOST_AND_HBM)
    for key in keys:
        h = world.ns + bytes(key) + (0).to_bytes(4, "big")
        homes = [s for s in world.stores if h in s.entries]
        if not any(s.entries[h].status in host and h not in s._saving for s in homes):
            world.violations.append(f"D7 pass {t}: key {bytes(key)[:6].hex()} counted as a host hit while not on the host")


class _Store(FakeKVCacheStore):
    """Saves land after a random number of polls (out of order across jobs), some fail once (the manager retries), and
    a landing reads its device block (D3)."""

    def __init__(self, world, manual=False):
        super().__init__()
        self.world = world
        self.manual = manual
        self.left: dict[bytes, int] = {}
        world.stores.append(self)

    def save(self, block_hashes, dst_raiden_id=None):
        ok = super().save(block_hashes, dst_raiden_id)
        if ok:
            for h in block_hashes:
                self.left[h] = self.world.rng.randint(*self.world.latency)
        return ok

    def land_now(self, block_hashes) -> None:
        for h in block_hashes:
            self.left.pop(h, None)
            self.world.land(self, h)
        self.complete_save(list(block_hashes))

    def poll_save_status(self):
        if not self.manual:
            for h in list(self.left):
                self.left[h] -= 1
                if self.left[h] <= 0:
                    del self.left[h]
                    fail = self.world.rng.random() < self.world.fail_p
                    if not fail:
                        self.world.land(self, h)
                    self.complete_save([h], success=not fail)
        return super().poll_save_status()


class _TimedStore(_Store):
    """Saves land ``latency_s`` of wall time after launch (the backlog leg)."""

    def __init__(self, world, latency_s):
        super().__init__(world)
        self.latency_s = latency_s
        self.due: dict[bytes, float] = {}

    def save(self, block_hashes, dst_raiden_id=None):
        ok = FakeKVCacheStore.save(self, block_hashes, dst_raiden_id)
        if ok:
            now = time.perf_counter()
            for h in block_hashes:
                self.due[h] = now + self.latency_s
        return ok

    def poll_save_status(self):
        now = time.perf_counter()
        ready = [h for h, at in self.due.items() if at <= now]
        for h in ready:
            del self.due[h]
            self.world.land(self, h)
        if ready:
            self.complete_save(ready)
        return FakeKVCacheStore.poll_save_status(self)


# -- the served scheduler, vLLM's block pool, a workload ---------------------------------------------------------


def _sched(monkeypatch, world, reserve, store_cls=_Store, single=False, **store_kw):
    from tests.offload.test_tier_classes import _tier_scheduler

    if reserve:
        monkeypatch.setenv(ENV, str(reserve))
    else:
        monkeypatch.delenv(ENV, raising=False)
    if single:
        store = store_cls(world, **store_kw)
        sched, kvc, _ = _tier_scheduler(monkeypatch, victim_policy=True, single_store=store)
    else:
        stores = [store_cls(world, **store_kw) for _ in range(5)]
        sched, kvc, _ = _tier_scheduler(monkeypatch, stores=stores, victim_policy=True)
    world.ns = sched._raiden_manager._key_namespace
    return sched


def _pool(world, n):
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=n, enable_caching=True, hash_block_size=BS)
    world.guard_queue(pool)
    return pool


def _request(req_id, n_tokens, seed):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        req_id, prompt_token_ids=[seed * 100_000 + i for i in range(n_tokens)],
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _echo(sched, meta):
    """What the workers send back for this metadata (update_connector_output): vLLM looks every echoed job id up in
    its own job table."""
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.offload.raiden_connector import RaidenOffloadingWorkerMetadata

    done = {j: 8 for j in meta.finished_store_job_ids}
    done.update({j: 8 for j in meta.finished_load_jobs})
    acks = RaidenOffloadingWorkerMetadata(completed_jobs=done, fenced_jobs={j: 8 for j in meta.fence_job_ids})
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=acks))


class _Run:
    """A schedule of requests: each pass frees the finished ones, admits new ones (all random draws are made before the
    free-block check, so two arms of one seed see one schedule), builds the metadata, checks D2 / D4, then runs the
    forward (writes the new blocks and caches their full ones)."""

    def __init__(self, sched, pool, world, seed, groups=(MAIN, KEYS), sizes=(2, 6), life=(1, 3), admit=(0, 3),
                 max_running=8, dup_p=0.1, partial_p=0.5, forward_s=0.0, timed=False):
        self.sched, self.pool, self.world = sched, pool, world
        self.rng = random.Random(seed)
        self.groups, self.sizes, self.life, self.admit = groups, sizes, life, admit
        self.max_running, self.dup_p, self.partial_p = max_running, dup_p, partial_p
        self.forward_s, self.timed = forward_s, timed
        self.running: list = []
        self.seq = 0
        self.t = 0
        self.starved = 0
        self.free_seq: list[int] = []
        self.sched_s = 0.0

    def held(self):
        return [b for r in self.running for b in r[1]]

    def step(self, admit=True):
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.kv_offload.base import make_offload_key

        t, rng, pool = self.t, self.rng, self.pool
        keep = []
        for r in self.running:
            if r[0] <= t:
                pool.free_blocks(list(reversed(r[1])))
            else:
                keep.append(r)
        self.running = keep
        draws = [
            (rng.randint(*self.sizes), rng.random() < self.partial_p, rng.random(), rng.randint(*self.life))
            for _ in range(rng.randint(*self.admit) if admit else 0)
        ]
        new = []
        t0 = time.perf_counter()
        for n_full, partial, dup, life in draws:
            if len(self.running) + len(new) >= self.max_running:
                break
            need = n_full + int(partial)
            free = pool.get_num_free_blocks()
            self.free_seq.append(free)
            if need > free:  # KVCacheManager.allocate_slots refuses (kv_cache_manager.py)
                self.starved += 1
                continue
            blocks = pool.get_new_blocks(need)
            self.seq += 1
            seed = self.seq if (dup >= self.dup_p or self.seq < 3) else 1 + int(dup * 1000) % (self.seq - 1)
            new.append([t + life, blocks, n_full, seed, self.groups[self.seq % len(self.groups)]])
        meta = self.sched.build_connector_meta(SchedulerOutput.make_empty())
        self.sched_s += time.perf_counter() - t0
        _echo(self.sched, meta)
        held = self.held() + [b for r in new for b in r[1]]
        self.world.check_ship(held, t)
        self.world.check_places(self.sched, pool, held, t)
        for fin, blocks, n_full, seed, group in new:
            req = _request(f"r{seed}", n_full * BS, seed)
            for i, b in enumerate(blocks[:n_full]):
                self.world.hbm[b.block_id] = self.world.key_bytes(make_offload_key(req.block_hashes[i], group))
            for b in blocks[n_full:]:
                self.world.hbm[b.block_id] = ("partial", t, b.block_id)
            pool.cache_full_blocks(req, blocks, 0, n_full, BS, group)
            self.running.append((fin, blocks))
        if self.forward_s:
            time.sleep(self.forward_s)
        self.t += 1
        return meta

    def settle(self, limit=200):
        """Passes with no admission until nothing is parked or in flight."""
        for _ in range(limit):
            vasync = self.sched.__dict__.get("_vasync")
            if (vasync is None or not vasync.parked) and not self.world.inflight_blocks():
                return True
            self.step(admit=False)
        return False


def _tel(sched, name):
    return sched.telemetry.get(name, 0)


# -- integrity of record: random workloads on the served (tier classes) and the single-store scheduler -------------


@pytest.mark.parametrize("seed,min_age,pool_blocks,reserve,fail_p", [
    (1, 3, 72, 12, 0.0),
    (2, 3, 72, 24, 0.05),
    (3, 10, 160, 16, 0.05),
    (4, 3, 48, 32, 0.05),
])
def test_integrity_sim_tier_classes(monkeypatch, seed, min_age, pool_blocks, reserve, fail_p):
    """Key on, served form: out-of-order landings, failed saves retried, duplicates. No D1-D4 violation; the async path
    engages (jobs parked and released) and every landed store read the bytes of its key."""
    world = _World(random.Random(1000 + seed), latency=(1, 5), fail_p=fail_p)
    sched = _sched(monkeypatch, world, reserve)
    sched._victim_min_age = min_age
    pool = _pool(world, pool_blocks)
    run = _Run(sched, pool, world, seed)
    for _ in range(160):
        run.step()
    assert run.settle(), "parked jobs never drained"
    tel = sched.telemetry
    _mark("SIM_TIER", seed=seed, min_age=min_age, pool=pool_blocks, reserve=reserve, fail_p=fail_p,
          async_jobs=_tel(sched, "victim_async_stores"), async_keys=_tel(sched, "victim_async_store_keys"),
          released=_tel(sched, "victim_async_released_blocks"), failed=_tel(sched, "victim_async_failed"),
          sync_jobs=_tel(sched, "victim_stores"), fallbacks=_tel(sched, "victim_async_fallbacks"),
          landed=world.landed, starved=run.starved, violations=len(world.violations))
    assert world.violations == [], world.violations[:5]
    assert tel["victim_async_stores"] > 0 and tel["victim_async_released_blocks"] > 0
    assert tel["victim_async_done"] + tel.get("victim_async_failed", 0) == tel["victim_async_stores"]
    assert world.landed > 0


def test_integrity_sim_single_store(monkeypatch):
    """Key on with one store (drain timeouts poison instead of stopping the engine): no violation."""
    world = _World(random.Random(77), latency=(1, 6), fail_p=0.05)
    sched = _sched(monkeypatch, world, 16, single=True)
    sched._victim_min_age = 3
    pool = _pool(world, 64)
    run = _Run(sched, pool, world, 7)
    for _ in range(160):
        run.step()
    assert run.settle()
    _mark("SIM_SINGLE", async_jobs=_tel(sched, "victim_async_stores"), sync_jobs=_tel(sched, "victim_stores"),
          fallbacks=_tel(sched, "victim_async_fallbacks"), violations=len(world.violations))
    assert world.violations == [], world.violations[:5]
    assert _tel(sched, "victim_async_stores") > 0 and not sched._vasync.fatal


# -- (a) release only after the store job is terminal -------------------------------------------------------------


def _parked_setup(monkeypatch, reserve=4, n=12, free=True, manual=True, latency=(1, 4)):
    """A pool whose free queue holds 2 old cached requests of 2 blocks each (MAIN); by default stores that land only on
    demand (manual=False: after ``latency`` polls). With free=False the two requests still hold their blocks
    (_free_aged frees them)."""
    world = _World(latency=latency)
    sched = _sched(monkeypatch, world, reserve, manual=manual)
    pool = _pool(world, n)
    from vllm.v1.kv_offload.base import make_offload_key

    reqs = []
    for seed in (1, 2):
        blocks = pool.get_new_blocks(2)
        req = _request(f"r{seed}", 2 * BS, seed)
        for i, b in enumerate(blocks):
            world.hbm[b.block_id] = world.key_bytes(make_offload_key(req.block_hashes[i], MAIN))
        pool.cache_full_blocks(req, blocks, 0, 2, BS, MAIN)
        reqs.append((req, blocks))
    # The pool takes part once one of its allocations evicted a block the policy stores: as if that happened.
    sched._vasync.pools[id(pool)] = pool
    # The never-used (hashless) blocks go to a request, so the cached ones are the free queue.
    rest = pool.get_new_blocks(pool.free_block_queue.num_free_blocks)
    if free:
        for _, blocks in reqs:
            _free_aged(sched, pool, blocks)
    return world, sched, pool, reqs, rest


def _free_aged(sched, pool, blocks):
    """Frees blocks and ages them past the young-block guard."""
    pool.free_blocks(list(reversed(blocks)))
    sched._victim_tap.step += sched._victim_min_age


def _queue_ids(pool):
    out, b = [], pool.free_block_queue.fake_free_list_head.next_free_block
    while b is not None and b is not pool.free_block_queue.fake_free_list_tail:
        out.append(b.block_id)
        assert len(out) <= pool.num_gpu_blocks, "the free queue does not end within the pool (a cycle)"
        b = b.next_free_block
    return out


def test_blocks_return_only_after_their_store_job_is_terminal(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, _ = _parked_setup(monkeypatch)
    vasync = sched._vasync
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    parked = {b.block_id for rec in vasync.parked.values() for b in rec.blocks}
    assert parked == {b.block_id for _, bl in reqs for b in bl}
    assert not parked & set(_queue_ids(pool))
    assert pool.get_num_free_blocks() == 4  # parked blocks still count as free
    for _ in range(3):  # passes with the stores still in flight: nothing returns
        meta = sched.build_connector_meta(SchedulerOutput.make_empty())
        _echo(sched, meta)
        assert not parked & set(_queue_ids(pool)) and len(vasync.parked) == 1
    store = world.stores[0]  # the main class
    store.land_now(list(store._saving))
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    _echo(sched, meta)
    assert not vasync.parked and set(_queue_ids(pool)) == parked
    assert all(pool.blocks[i].block_hash is None for i in parked)
    _mark("RELEASE_AFTER_TERMINAL", parked=len(parked), violations=len(world.violations))
    assert world.violations == []


def test_parked_blocks_are_out_of_the_queue_and_the_prefix_cache(monkeypatch):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, rest = _parked_setup(monkeypatch)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    for req, blocks in reqs:
        for i, b in enumerate(blocks):
            assert b.prev_free_block is None and b.next_free_block is None and b.ref_cnt == 0
            assert b.block_hash is None
            assert pool.get_cached_block(req.block_hashes[i], [MAIN]) is None
            assert not pool.cached_block_hash_to_block.contain(make_block_hash_with_group_id(req.block_hashes[i], MAIN), b.block_id)
    world.check_places(sched, pool, rest, 0)
    _mark("PARKED_UNREACHABLE", violations=len(world.violations))
    assert world.violations == []


# -- (c) ordering: each job releases only its own blocks, on its own terminal record -------------------------------


def test_out_of_order_completion_releases_only_the_finished_job(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, _ = _parked_setup(monkeypatch, reserve=4, free=False)
    vasync = sched._vasync
    _free_aged(sched, pool, reqs[0][1])
    sched.build_connector_meta(SchedulerOutput.make_empty())  # parks request 1's blocks (job J1)
    (j1,) = vasync.parked
    _free_aged(sched, pool, reqs[1][1])
    sched.build_connector_meta(SchedulerOutput.make_empty())  # parks request 2's blocks (job J2)
    (j2,) = [j for j in vasync.parked if j != j1]
    assert j1 < j2
    first = {b.block_id for b in vasync.parked[j1].blocks}
    second = {b.block_id for b in vasync.parked[j2].blocks}
    assert first == {b.block_id for b in reqs[0][1]} and second == {b.block_id for b in reqs[1][1]}
    store = world.stores[0]
    store.land_now([h for h in store._saving if store.entries[h].device_block_id in second])  # J2 lands first
    _echo(sched, sched.build_connector_meta(SchedulerOutput.make_empty()))
    assert list(vasync.parked) == [j1]
    assert second <= set(_queue_ids(pool)) and not first & set(_queue_ids(pool))
    store.land_now(list(store._saving))
    _echo(sched, sched.build_connector_meta(SchedulerOutput.make_empty()))
    assert not vasync.parked and first | second <= set(_queue_ids(pool))
    _mark("OUT_OF_ORDER", violations=len(world.violations))
    assert world.violations == []


def test_a_job_submitted_this_pass_is_not_read_as_done(monkeypatch):
    """Stale status: completion comes from the job's own terminal record, never from a status read taken before the
    pass submitted it."""
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, _ = _parked_setup(monkeypatch)
    vasync = sched._vasync
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    (job,) = vasync.parked
    assert sched._raiden_manager.is_job_launched(job)
    assert job not in meta.finished_store_job_ids
    assert not {b.block_id for b in vasync.parked[job].blocks} & set(_queue_ids(pool))
    _mark("STALE_STATUS", violations=len(world.violations))
    assert world.violations == []


def test_victim_jobs_never_reach_the_workers_echo(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, _ = _parked_setup(monkeypatch)
    vasync = sched._vasync
    sched.build_connector_meta(SchedulerOutput.make_empty())
    (job,) = vasync.parked
    store = world.stores[0]
    store.land_now(list(store._saving))
    meta = sched.build_connector_meta(SchedulerOutput.make_empty())
    assert job not in meta.finished_store_job_ids and not vasync.parked
    _echo(sched, meta)  # vLLM's update_connector_output raises KeyError on a job id it does not know
    _mark("ECHO_FILTER", job=job)


# -- (b) the free queue running dry: fall back to a synchronous drain, never starve, never hand out a held block ----


def _dry_arm(monkeypatch, reserve, seed):
    world = _World(random.Random(500 + seed), latency=(2, 8), fail_p=0.0)
    sched = _sched(monkeypatch, world, reserve)
    sched._victim_min_age = 3
    pool = _pool(world, 64)
    run = _Run(sched, pool, world, seed, sizes=(3, 7), admit=(0, 3), max_running=5, life=(1, 3))
    for _ in range(120):
        run.step()
    settled = run.settle()
    return sched, run, world, settled


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_free_queue_dry_same_decisions_as_key_off(monkeypatch, seed):
    """The reserve (48) exceeds what the free queue usually holds. One schedule, key off and key on: identical free-block
    counts at every allocation decision (parked blocks count as free), hence the same starvation; with the key on, every
    allocation the free queue cannot hold drains parked jobs first, and no block is handed out under a live store."""
    from vllm_torchtpu.offload import raiden_connector as rc
    from vllm_torchtpu.offload import victim

    off_sched, off, off_world, off_ok = _dry_arm(monkeypatch, 0, seed)
    off_sched.shutdown()
    victim.deactivate()
    rc._VASYNC = None
    on_sched, on, on_world, on_ok = _dry_arm(monkeypatch, 48, seed)
    diverge = next((i for i, (a, b) in enumerate(zip(off.free_seq, on.free_seq)) if a != b), None)
    _mark("DRY_SAME", seed=seed, starved_off=off.starved, starved_on=on.starved, decisions=len(on.free_seq),
          diverge_at=diverge, fallbacks=_tel(on_sched, "victim_async_fallbacks"),
          fallback_jobs=_tel(on_sched, "victim_async_fallback_jobs"), async_jobs=_tel(on_sched, "victim_async_stores"),
          violations=len(on_world.violations) + len(off_world.violations), settled=int(off_ok and on_ok))
    assert off_world.violations == [] and on_world.violations == [], (off_world.violations + on_world.violations)[:5]
    assert off_ok and on_ok
    assert on.free_seq == off.free_seq and on.starved == off.starved, diverge
    assert _tel(on_sched, "victim_async_stores") > 0


def test_free_queue_dry_falls_back_to_a_synchronous_drain(monkeypatch):
    """All 4 cached blocks are parked and the free queue is empty; the pool still counts 4 free. An allocation of 2
    drains the oldest parked job (its store lands during the drain), takes 2 of its blocks, and nothing is handed out
    or returned under a live store."""
    world, sched, pool, reqs, rest = _parked_setup(monkeypatch, manual=False, latency=(6, 6))
    from vllm.v1.core.sched.output import SchedulerOutput

    vasync = sched._vasync
    sched.build_connector_meta(SchedulerOutput.make_empty())
    parked = {b.block_id for rec in vasync.parked.values() for b in rec.blocks}
    assert len(parked) == 4 and pool.free_block_queue.num_free_blocks == 0 and pool.get_num_free_blocks() == 4
    assert world.inflight_blocks() == parked
    got = pool.get_new_blocks(2)
    assert {b.block_id for b in got} <= parked and not world.inflight_blocks() and not vasync.parked
    assert pool.free_block_queue.num_free_blocks == 2 and pool.get_num_free_blocks() == 2
    world.check_ship(got, 0)
    world.check_places(sched, pool, rest + got, 0)
    _mark("DRY_FALLBACK", fallbacks=_tel(sched, "victim_async_fallbacks"), fallback_jobs=_tel(sched, "victim_async_fallback_jobs"),
          landed=world.landed, violations=len(world.violations))
    assert world.violations == [], world.violations[:5]
    assert _tel(sched, "victim_async_fallback_jobs") == 1 and world.landed == 4


def test_a_fallback_drain_that_times_out_stops_the_engine(monkeypatch):
    """Tier classes: a parked job that does not finish within the drain timeout stops the engine before its blocks
    return to the free queue."""
    from vllm_torchtpu.offload import raiden_store

    monkeypatch.setattr(raiden_store, "_DRAIN_TIMEOUT_S", 0.05)
    from vllm.v1.core.sched.output import SchedulerOutput

    world, sched, pool, reqs, rest = _parked_setup(monkeypatch)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    parked = {b.block_id for rec in sched._vasync.parked.values() for b in rec.blocks}
    assert len(parked) == 4 and pool.free_block_queue.num_free_blocks == 0 and pool.get_num_free_blocks() == 4
    with pytest.raises(RuntimeError, match="did not finish within the drain timeout"):
        pool.get_new_blocks(2)
    assert not parked & set(_queue_ids(pool))
    _mark("FALLBACK_TIMEOUT_FATAL", parked=len(parked), violations=len(world.violations))
    assert world.violations == []


# -- the young-block guard and what evict-ahead leaves alone ---------------------------------------------------------


def test_evict_ahead_takes_only_what_an_allocation_would_store(monkeypatch):
    """Freed too recently (a batch in flight may still write it), never stamped, a group the tier does not hold, or
    readable until reused (unpin_blocks): left in the queue with its hash. Old, stamped, offloaded: parked."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_offload.base import make_offload_key

    world = _World()
    sched = _sched(monkeypatch, world, 8, manual=True)
    pool = _pool(world, 12)
    tap = sched._victim_tap
    sched._vasync.pools[id(pool)] = pool

    def cached(seed, group):
        blocks = pool.get_new_blocks(1)
        req = _request(f"r{seed}", BS, seed)
        world.hbm[blocks[0].block_id] = world.key_bytes(make_offload_key(req.block_hashes[0], group))
        pool.cache_full_blocks(req, blocks, 0, 1, BS, group)
        return blocks[0]

    pool.get_new_blocks(pool.free_block_queue.num_free_blocks - 5)  # 5 blocks left for the cached ones
    old = cached(1, MAIN)
    foreign = cached(2, 9)  # not an offloaded group
    unstamped = cached(3, MAIN)
    watched = cached(4, MAIN)
    pool.free_blocks([old, foreign, unstamped])
    tap.freed_at.pop(unstamped.block_id)
    if "unpin_blocks" in type(pool).__dict__:
        pool.unpin_blocks([watched], on_reuse=lambda b: None)
    else:
        pool.free_blocks([watched])
        pool._reuse_watchers = {watched.block_id: None}
    tap.step += sched._victim_min_age
    young = cached(5, MAIN)
    pool.free_blocks([young])  # freed this pass
    assert _queue_ids(pool) == [old.block_id, foreign.block_id, unstamped.block_id, watched.block_id, young.block_id]
    sched.build_connector_meta(SchedulerOutput.make_empty())
    parked = {b.block_id for rec in sched._vasync.parked.values() for b in rec.blocks}
    assert parked == {old.block_id}
    for b in (foreign, unstamped, watched, young):
        assert b.block_hash is not None and b.block_id in _queue_ids(pool)
    _mark("GUARD", parked=sorted(parked))


# -- key-off: nothing installed, nothing changed ---------------------------------------------------------------------

_KEY_OFF = r'''
import json, os, sys
os.environ.pop("VLLM_TPU_OFFLOAD_VICTIM_ASYNC", None)
from _pytest.monkeypatch import MonkeyPatch
from tests.offload.test_raiden_store_offload import FakeKVCacheStore
from tests.offload.test_tier_classes import _tier_scheduler
from vllm.v1.core.block_pool import BlockPool
from vllm_torchtpu.offload import raiden_connector as rc, victim
mp = MonkeyPatch()
sched, kvc, _ = _tier_scheduler(mp, stores=[FakeKVCacheStore() for _ in range(5)], victim_policy=True)
m = sched._raiden_manager
print("Q38_KEYOFF " + json.dumps({
    "vasync_attr": "_vasync" in sched.__dict__,
    "module_active": rc._VASYNC is not None,
    "block_pool_patched": bool(BlockPool.__dict__.get("_q38_vasync_patch", False)),
    "poll_wrapped": "poll_finished_jobs" in m.__dict__ and getattr(m.__dict__["poll_finished_jobs"], "__self__", None) is not None
                    and type(m.__dict__["poll_finished_jobs"].__self__).__name__ == "_VictimAsync",
    "telemetry_async": "victim_async" in sched.telemetry_line(),
    "victim": bool(sched.__dict__.get("_victim", False)),
}))
victim.deactivate()
'''


def test_key_off_installs_nothing():
    """A fresh process with the key unset and the victim policy on: no scheduler state, no BlockPool patch, the
    manager's poll unwrapped, the telemetry line unchanged."""
    import json

    env = dict(os.environ)
    env.pop(ENV, None)
    r = subprocess.run([sys.executable, "-c", _KEY_OFF], capture_output=True, text=True, env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith("Q38_KEYOFF ")), None)
    assert line is not None, r.stderr[-3000:]
    got = json.loads(line[len("Q38_KEYOFF "):])
    _mark("KEY_OFF", **{k: int(v) for k, v in got.items()})
    assert got == {"vasync_attr": False, "module_active": False, "block_pool_patched": False,
                   "poll_wrapped": False, "telemetry_async": False, "victim": True}


def test_key_needs_the_victim_policy(monkeypatch):
    from tests.offload.test_tier_classes import _tier_scheduler

    monkeypatch.setenv(ENV, "8")
    monkeypatch.delenv("QWEN4_EXP_HOST_TIER_POLICY", raising=False)
    with pytest.raises(ValueError, match="needs QWEN4_EXP_HOST_TIER_POLICY=victim"):
        _tier_scheduler(monkeypatch, stores=[FakeKVCacheStore() for _ in range(5)])
    _mark("NEEDS_VICTIM")


def test_env_key_registered_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED, _tpu_compile_env_factors

    monkeypatch.delenv(ENV, raising=False)
    assert envs.VLLM_TPU_OFFLOAD_VICTIM_ASYNC == 0
    off = _tpu_compile_env_factors()
    monkeypatch.setenv(ENV, "256")
    assert envs.VLLM_TPU_OFFLOAD_VICTIM_ASYNC == 256
    assert ENV in _TPU_COMPILE_ENV_IGNORED and ENV not in off and _tpu_compile_env_factors() == off
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValueError):
        envs.VLLM_TPU_OFFLOAD_VICTIM_ASYNC  # noqa: B018
    _mark("ENV_KEY", compile_ignored=1)


# -- the backlog leg: the scheduler-thread time the hold removes -----------------------------------------------------


def _backlog_arm(monkeypatch, reserve, latency_s, forward_s, warm, measure, seed=21):
    world = _World(random.Random(3))
    sched = _sched(monkeypatch, world, reserve, store_cls=_TimedStore, latency_s=latency_s)
    sched._victim_min_age = 3
    pool = _pool(world, 96)
    run = _Run(sched, pool, world, seed, sizes=(3, 6), admit=(1, 3), max_running=6, life=(1, 2), dup_p=0.0,
               forward_s=forward_s)
    for _ in range(warm):
        run.step()
    run.sched_s = 0.0
    drain0, sync0 = _tel(sched, "victim_drain_us"), _tel(sched, "victim_stores")
    async0 = _tel(sched, "victim_async_stores")
    for _ in range(measure):
        run.step()
    out = dict(sched_ms=run.sched_s * 1e3, drain_ms=(_tel(sched, "victim_drain_us") - drain0) / 1e3,
               sync_jobs=_tel(sched, "victim_stores") - sync0, async_jobs=_tel(sched, "victim_async_stores") - async0,
               fallback_ms=_tel(sched, "victim_async_fallback_us") / 1e3, topup_ms=_tel(sched, "victim_async_topup_us") / 1e3,
               starved=run.starved, violations=len(world.violations))
    run.forward_s = 0.0
    assert run.settle()
    out["violations"] = len(world.violations)
    sched.shutdown()
    return out, world


def test_backlog_scheduler_thread_saving(monkeypatch):
    """Stores take 20 ms of wall time, a forward 30 ms. Key off, every pass with stored victims waits for its drain on
    the scheduler thread; key on, the stores run while the forward does. Pass: key-on scheduler-thread time in the
    measured window <= 25 % of key-off's, and no violation in either arm."""
    from vllm_torchtpu.offload import raiden_connector as rc
    from vllm_torchtpu.offload import victim

    L, F, warm, measure = 0.020, 0.030, 30, 40
    off, off_world = _backlog_arm(monkeypatch, 0, L, F, warm, measure)
    victim.deactivate()
    rc._VASYNC = None
    on, on_world = _backlog_arm(monkeypatch, 24, L, F, warm, measure)
    saved = off["sched_ms"] - on["sched_ms"]
    _mark("BACKLOG", L_ms=int(L * 1e3), F_ms=int(F * 1e3), passes=measure,
          off_sched_ms=f"{off['sched_ms']:.1f}", on_sched_ms=f"{on['sched_ms']:.1f}", saved_ms=f"{saved:.1f}",
          saved_pct=f"{100 * saved / max(off['sched_ms'], 1e-9):.1f}", off_drain_ms=f"{off['drain_ms']:.1f}",
          on_drain_ms=f"{on['drain_ms']:.1f}", off_sync_jobs=off["sync_jobs"], on_sync_jobs=on["sync_jobs"],
          on_async_jobs=on["async_jobs"], on_fallback_ms=f"{on['fallback_ms']:.1f}", on_topup_ms=f"{on['topup_ms']:.1f}",
          violations=off["violations"] + on["violations"])
    assert off_world.violations == [] and on_world.violations == [], (off_world.violations + on_world.violations)[:5]
    assert off["sync_jobs"] > 0 and on["async_jobs"] > 0
    assert on["sched_ms"] <= 0.25 * off["sched_ms"], (on, off)


# -- the served combination: VLLM_TPU_OFFLOAD_VICTIM_ASYNC with VLLM_TPU_ARRIVAL_TOUCH=1 ------------------------------
#
# The Car D / E / F / G forms set VLLM_TPU_ARRIVAL_TOUCH=1 (sub-keys at their defaults; no TIER_X2 / READ_ROOM) in the
# same scheduler. Arrival touch reaches HBM blocks only through prefix-cache lookups (core/arrival_touch.py
# arrival_blocks -> find_longest_cache_hit, then BlockPool.touch) and the host tier only through
# host_resident_prefix + touch_run (TPURaidenOffloadingConnector.arrival_touch).


def _kvm(pool):
    from types import SimpleNamespace as NS

    return NS(coordinator=NS(single_type_managers=[NS(block_pool=pool, req_to_blocks={}) for _ in range(6)]))


def _arrival_pins(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.core.arrival_touch import ArrivalPins

    monkeypatch.setenv("VLLM_TPU_ARRIVAL_TOUCH", "1")
    for k in ("VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC", "VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC", "VLLM_TPU_ARRIVAL_TOUCH_TTL_S",
              "VLLM_TPU_ARRIVAL_TOUCH_TIER"):
        monkeypatch.delenv(k, raising=False)
    return ArrivalPins(hbm_frac=float(envs.VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC),
                       min_free_frac=float(envs.VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC),
                       ttl_s=float(envs.VLLM_TPU_ARRIVAL_TOUCH_TTL_S), tier=bool(envs.VLLM_TPU_ARRIVAL_TOUCH_TIER))


def _hits(pool, req, group, n_tokens):
    """The run a prefix lookup adopts for one full-attention group: get_cached_block per block until the first miss,
    over at most (num_tokens - 1) tokens (find_longest_cache_hit's max length)."""
    out = []
    for i in range((n_tokens - 1) // BS):
        got = pool.get_cached_block(req.block_hashes[i], [group])
        if not got:
            break
        out.append(got[0])
    return out


class _ArrivalRun(_Run):
    """_Run with arrival touch: a new request (40 % of them return an earlier prompt) arrives in the waiting queue,
    where the tree's ArrivalPins pins the blocks its prefix lookup finds and the host tier refreshes the keys past its
    local hit; it waits 0-15 passes (a pass is 10 s of the arrival clock against the 120 s TTL); before each pass
    ArrivalPins.before_schedule releases expired and pressure pins; when scheduled it adopts its current hit
    (BlockPool.touch, as allocate_slots does), allocates the rest and its pin is released (rel_sched). Checks D1-D7."""

    def __init__(self, sched, pool, world, seed, pins, **kw):
        super().__init__(sched, pool, world, seed, **kw)
        self.pins, self.kvm = pins, _kvm(pool)
        self.waiting: list[dict] = []
        self.refreshed = 0
        self.adopted = 0
        self.prompts: list[tuple[int, int]] = []

    def pinned(self):
        return [b for pin in self.pins.pins.values() for keep in pin.groups for b in keep]

    def step(self, admit=True):
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.kv_offload.base import make_offload_key

        t, rng, pool, world, pins = self.t, self.rng, self.pool, self.world, self.pins
        now = 10.0 * t
        manager = self.sched._raiden_manager
        keep = []
        for r in self.running:
            if r[0] <= t:
                pool.free_blocks(list(reversed(r[1])))
            else:
                keep.append(r)
        self.running = keep
        draws = [
            (rng.randint(*self.sizes), rng.random() < self.partial_p, rng.random(), rng.randint(*self.life),
             rng.randint(0, 15), rng.random())
            for _ in range(rng.randint(*self.admit) if admit else 0)
        ]
        for n_full, partial, dup, life, wait, pick in draws:
            self.seq += 1
            if dup < 0.4 and self.prompts:
                seed, group = self.prompts[int(pick * len(self.prompts))]
                n_full = max(n_full, 2)
            else:
                seed, group = self.seq, self.groups[self.seq % len(self.groups)]
                self.prompts.append((seed, group))
            req = _request(f"r{seed}", n_full * BS + int(partial), seed)
            rid = f"q{self.seq}"
            hits = _hits(pool, req, group, req.num_tokens)
            if _check_targets(world, self.sched, hits, t, "an arrival pin") and hits and pins.hbm_frac > 0:
                groups = [[] for _ in range(6)]
                groups[group] = hits
                pins.pin(self.kvm, rid, groups, now)
            keys = [make_offload_key(req.block_hashes[i], group) for i in range(len(hits), n_full)]
            run = manager.host_resident_prefix(keys)
            _check_host_hits(world, keys[:run], t)
            if run:
                self.refreshed += manager.touch_run(keys[:run])
            _check_pins(world, t)
            self.waiting.append(dict(rid=rid, req=req, group=group, n_full=n_full, partial=int(partial), life=life,
                                     ready=t + wait))
        pins.before_schedule(self.kvm, now)
        new = []
        t0 = time.perf_counter()
        for w in list(self.waiting):
            if w["ready"] > t:
                continue
            if len(self.running) + len(new) >= self.max_running:
                break
            hits = _hits(pool, w["req"], w["group"], w["req"].num_tokens)
            if not _check_targets(world, self.sched, hits, t, "an adoption"):
                hits = []
            need = w["n_full"] - len(hits) + w["partial"]
            free = pool.get_num_free_blocks()
            self.free_seq.append(free)
            if need + sum(1 for b in hits if b.ref_cnt == 0) > free:
                self.starved += 1
                continue
            pool.touch(hits)
            blocks = hits + (pool.get_new_blocks(need) if need else [])
            pins.release(self.kvm, w["rid"], "rel_sched")
            self.adopted += len(hits)
            self.waiting.remove(w)
            new.append([t + w["life"], blocks, w])
        meta = self.sched.build_connector_meta(SchedulerOutput.make_empty())
        self.sched_s += time.perf_counter() - t0
        _echo(self.sched, meta)
        held = self.held() + [b for r in new for b in r[1]] + self.pinned()
        world.check_ship(held, t)
        world.check_places(self.sched, pool, held, t)
        _check_pins(world, t)
        for fin, blocks, w in new:
            req, group, n_full = w["req"], w["group"], w["n_full"]
            h = sum(1 for b in blocks[:n_full] if b.block_hash is not None)
            for i, b in enumerate(blocks[h:n_full], start=h):
                world.hbm[b.block_id] = world.key_bytes(make_offload_key(req.block_hashes[i], group))
            for b in blocks[n_full:]:
                world.hbm[b.block_id] = ("partial", t, b.block_id)
            pool.cache_full_blocks(req, blocks, h, n_full, BS, group)
            self.running.append((fin, blocks))
        self.t += 1
        return meta

    def settle(self, limit=200):
        self.pins.release_all(self.kvm, "rel_finish")
        self.waiting = []
        return super().settle(limit)


@pytest.mark.parametrize("seed,pool_blocks,min_age", [(21, 96, 3), (22, 384, 3), (23, 384, 10)])
def test_integrity_sim_with_arrival_touch(monkeypatch, seed, pool_blocks, min_age):
    """The served combination: VLLM_TPU_OFFLOAD_VICTIM_ASYNC=256 with VLLM_TPU_ARRIVAL_TOUCH=1 on the tier-class
    scheduler (out-of-order landings, failed saves retried, returners). No D1-D7 violation; arrival pins, adoptions and
    host refreshes all happen while victim stores are parked."""
    world = _World(random.Random(2000 + seed), latency=(1, 6), fail_p=0.05)
    sched = _sched(monkeypatch, world, 256)
    sched._victim_min_age = min_age
    pool = _pool(world, pool_blocks)
    pins = _arrival_pins(monkeypatch)
    run = _ArrivalRun(sched, pool, world, seed, pins, sizes=(2, 6), admit=(0, 4), max_running=8, life=(1, 3))
    for _ in range(160):
        run.step()
    assert run.settle(), "parked jobs never drained"
    st = pins.stats
    _mark("SIM_ARRIVAL", seed=seed, pool=pool_blocks, reserve=256, min_age=min_age,
          async_jobs=_tel(sched, "victim_async_stores"), released=_tel(sched, "victim_async_released_blocks"),
          sync_jobs=_tel(sched, "victim_stores"), fallbacks=_tel(sched, "victim_async_fallbacks"),
          pinned=st["pinned"], pinned_blocks=st["pinned_blocks"], denied=st["denied_budget"] + st["denied_floor"],
          rel_ttl=st["rel_ttl"], rel_pressure=st["rel_pressure"], rel_sched=st["rel_sched"], adopted=run.adopted,
          refreshed=run.refreshed, landed=world.landed, violations=len(world.violations))
    assert world.violations == [], world.violations[:5]
    assert _tel(sched, "victim_async_stores") > 0 and _tel(sched, "victim_async_released_blocks") > 0
    assert st["pinned"] > 0 and run.adopted > 0 and run.refreshed > 0


def test_arrival_touch_never_reaches_a_parked_block(monkeypatch):
    """A returner whose prefix is parked finds no HBM hit (nothing to pin), the host tier counts none of its in-flight
    keys as resident, and the refresh leaves the saves' pins as they were; once the stores land the keys are host hits
    and the refresh moves them. A pinned cached block is never parked (it is not in the free queue); released, it is
    young; aged, it is parked and invisible to the next arrival."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_offload.base import make_offload_key

    world, sched, pool, reqs, rest = _parked_setup(monkeypatch)
    pins, kvm = _arrival_pins(monkeypatch), _kvm(pool)
    vasync, manager, store = sched._vasync, sched._raiden_manager, world.stores[0]
    sched.build_connector_meta(SchedulerOutput.make_empty())
    parked = {b.block_id for rec in vasync.parked.values() for b in rec.blocks}
    assert len(parked) == 4
    req1 = _request("r1", 2 * BS + 1, 1)
    hits = _hits(pool, req1, MAIN, req1.num_tokens)
    _check_targets(world, sched, hits, 0, "an arrival pin")
    assert hits == []
    keys = [make_offload_key(req1.block_hashes[i], MAIN) for i in range(2)]
    pins_before = {h: store.entries[h].pin_count for h in store._saving}
    assert manager.host_resident_prefix(keys) == 0
    manager.touch_run(keys)
    assert {h: store.entries[h].pin_count for h in store._saving} == pins_before and all(pins_before.values())
    store.land_now(list(store._saving))
    _echo(sched, sched.build_connector_meta(SchedulerOutput.make_empty()))
    assert not vasync.parked
    assert manager.host_resident_prefix(keys) == 2 and manager.touch_run(keys) == 2
    assert all(store.entries[world.ns + bytes(k) + (0).to_bytes(4, "big")].pin_count == 0 for k in keys)
    # A cached block pinned on arrival stays out of the free queue, so evict-ahead cannot take it.
    blocks = pool.get_new_blocks(2)
    req3 = _request("r3", 2 * BS + 1, 3)
    for i, b in enumerate(blocks):
        world.hbm[b.block_id] = world.key_bytes(make_offload_key(req3.block_hashes[i], MAIN))
    pool.cache_full_blocks(req3, blocks, 0, 2, BS, MAIN)
    _free_aged(sched, pool, blocks)
    hits = _hits(pool, req3, MAIN, req3.num_tokens)
    assert [b.block_id for b in hits] == [b.block_id for b in blocks]
    groups = [[] for _ in range(6)]
    groups[MAIN] = hits
    assert pins.pin(kvm, "q3", groups, 0.0)
    sched.build_connector_meta(SchedulerOutput.make_empty())
    assert not vasync.parked and all(b.ref_cnt == 1 for b in blocks)
    pins.release(kvm, "q3", "rel_finish")
    sched.build_connector_meta(SchedulerOutput.make_empty())
    assert not vasync.parked  # just freed: young
    sched._victim_tap.step += sched._victim_min_age
    sched.build_connector_meta(SchedulerOutput.make_empty())
    assert {b.block_id for rec in vasync.parked.values() for b in rec.blocks} == {b.block_id for b in blocks}
    late = _hits(pool, req3, MAIN, req3.num_tokens)
    _check_targets(world, sched, late, 1, "an arrival pin")
    assert late == []
    held = rest + [b for rec in vasync.parked.values() for b in rec.blocks]
    world.check_places(sched, pool, rest, 1)
    _check_pins(world, 1)
    _mark("ARRIVAL_PARKED", parked=len(held) - len(rest), violations=len(world.violations))
    assert world.violations == [], world.violations[:5]
