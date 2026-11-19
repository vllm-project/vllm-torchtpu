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
"""Cross-step memo: a find_longest_cache_hit result reused across scheduler
steps must equal a fresh lookup exactly.

Drives the pinned vLLM's hybrid KVCacheManager (full attention + 2 Mamba
"align" groups, fine-grained hash block 2 < block 4, eagle drop on) through
random admissions, chunked prefills, decodes, frees, evictions, unwritten-block
clears and prefix-cache resets, and compares every memoized lookup with a fresh
one."""

import random

import pytest

torch = pytest.importorskip("torch")

from vllm import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request  # noqa: E402

import vllm_torchtpu  # noqa: E402
from vllm_torchtpu.core import prefix_change_log  # noqa: E402
from vllm_torchtpu.core.prefix_aware_scheduler import (  # noqa: E402
    PrefixAwareScheduler,
    _hit_signature,
)

HASH_BS = 2
BS = 4


def _begin_step(pool_owner_mgr):
    """Step prologue before ``new_step_starts``: nothing to do on main, where
    the same-step patch clears the unwritten set in ``new_step_starts`` (the
    simulation calls it right after this)."""


@pytest.fixture
def patched_pool():
    names = (
        "_insert_block_hash",
        "_remove_cached_block_hashes",
        "reset_prefix_cache",
        "get_cached_block",
    )
    saved_pool = {k: BlockPool.__dict__[k] for k in names}
    saved_new_step = KVCacheManager.__dict__["new_step_starts"]
    # The registry's order: the same-step patch, the epoch, the change log.
    vllm_torchtpu._patch_vllm_same_step_prefix_hits()
    prefix_change_log.install_epoch()
    assert prefix_change_log.install() is None
    assert prefix_change_log.installed()
    init_none_hash(sha256)
    try:
        yield _begin_step
    finally:
        for k, v in saved_pool.items():
            setattr(BlockPool, k, v)
        KVCacheManager.new_step_starts = saved_new_step


def _manager(num_blocks):
    mamba = MambaSpec(
        block_size=BS,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(["m0"], mamba),
            KVCacheGroupSpec(["m1"], mamba),
        ],
    )
    return KVCacheManager(
        cfg,
        max_model_len=4096,
        scheduler_block_size=BS,
        hash_block_size=HASH_BS,
        enable_caching=True,
        use_eagle=True,
    )


_HASHER = None


def _req(rid, ids):
    global _HASHER
    if _HASHER is None:
        _HASHER = get_request_block_hasher(HASH_BS, sha256)
    sp = SamplingParams(max_tokens=64)
    sp.update_from_generation_config({}, eos_token_id=100_000)
    return Request(
        request_id=rid,
        prompt_token_ids=list(ids),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=_HASHER,
    )


def _paw(mgr, verify):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.kv_cache_manager = mgr
    s._paw_scoring_broken = False
    s._enable_memo_xstep()
    assert s._memo_xstep
    s._memo_verify_left = verify
    return s


def _lookup(s, req):
    epoch = s._get_prefix_cache_epoch()
    max_len = req.num_tokens - 1
    c = s._memo_get(req, epoch, max_len)
    if c is None:
        c = s._memo_compute(req, epoch, max_len)
    return c


def _simulate(begin_step, seed, steps, verify=0, num_blocks=160, on_check=None):
    rng = random.Random(seed)
    mgr = _manager(num_blocks)
    s = _paw(mgr, verify)
    convs = [[rng.randrange(1000) for _ in range(rng.randint(4, 30))] for _ in range(6)]
    running: dict[str, Request] = {}
    waiting: list[Request] = []
    n = 0
    checks = nontrivial = resets = 0

    def new_waiting():
        nonlocal n
        c = rng.randrange(len(convs))
        if rng.random() < 0.6:
            convs[c] = convs[c] + [
                rng.randrange(1000) for _ in range(rng.randint(1, 12))
            ]
        base = convs[c][: rng.randint(1, len(convs[c]))]
        n += 1
        return _req(
            f"w{n}", base + [rng.randrange(1000) for _ in range(rng.randint(1, 9))]
        )

    for _ in range(8):
        waiting.append(new_waiting())

    def check_all():
        nonlocal checks, nontrivial
        for w in waiting:
            memo = _hit_signature(_lookup(s, w)[2])
            fresh = _hit_signature(
                mgr.coordinator.find_longest_cache_hit(w.block_hashes, w.num_tokens - 1)
            )
            assert memo == fresh, (w.request_id, memo[:2], fresh[:2])
            if on_check is not None:
                on_check(s, w)
            checks += 1
            nontrivial += memo[0] > 0

    for _ in range(steps):
        begin_step(mgr)
        mgr.new_step_starts()
        check_all()
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            if op < 0.30 and waiting:
                # admit a waiting request: chunked prefill (first chunk now)
                req = waiting.pop(rng.randrange(len(waiting)))
                blocks, num_hit, _ = mgr.get_computed_blocks(req)
                remaining = req.num_tokens - num_hit
                chunk = (
                    remaining
                    if rng.random() < 0.5
                    else max(1, min(remaining, BS * rng.randint(1, 3)))
                )
                if mgr.allocate_slots(req, chunk, num_hit, blocks) is not None:
                    req.num_computed_tokens = num_hit + chunk
                    running[req.request_id] = req
                waiting.append(new_waiting())
            elif op < 0.65 and running:
                req = running[rng.choice(list(running))]
                if req.num_computed_tokens < req.num_tokens:
                    chunk = req.num_tokens - req.num_computed_tokens
                    if mgr.allocate_slots(req, chunk) is not None:
                        req.num_computed_tokens += chunk
                else:
                    req.append_output_token_ids(rng.randrange(1000))
                    if mgr.allocate_slots(req, 1) is not None:
                        req.num_computed_tokens += 1
            elif op < 0.85 and running:
                rid = rng.choice(list(running))
                mgr.free(running.pop(rid))
            elif op < 0.87 and not running:
                resets += bool(mgr.reset_prefix_cache())
            else:
                waiting.append(new_waiting())
                if len(waiting) > 14:
                    waiting.pop(0)
        check_all()  # mid-step: freshly inserted blocks are still unwritten
    print(f"seed={seed} resets={resets}")
    return s, checks, nontrivial


def test_xstep_memo_equals_fresh_lookup(patched_pool):
    tot_hits = tot_checks = tot_nontrivial = tot_computes = 0
    for seed in range(12):
        s, checks, nontrivial = _simulate(patched_pool, seed, steps=150)
        tot_hits += s._memo_hits
        tot_checks += checks
        tot_nontrivial += nontrivial
        tot_computes += s._memo_computes
        assert s._memo_xstep  # never disabled
    # Cross-step reuse really happens, on non-trivial hits.
    assert tot_checks > 10_000
    assert tot_hits > tot_checks // 4
    assert tot_nontrivial > tot_checks // 6
    print(
        f"checks={tot_checks} xstep_hits={tot_hits} computes={tot_computes} "
        f"nontrivial={tot_nontrivial}"
    )


def _local_hit_view(mgr, hit):
    blocks, num_local, boundary, diverged = hit
    ids = tuple(tuple(b.block_id for b in group) for group in blocks.blocks)
    return blocks is mgr.empty_kv_cache_blocks, ids, num_local, boundary, diverged


def test_memo_local_hit_equals_the_upstream_lookup(patched_pool):
    """The memo branch of _get_local_prefix_cache_hit returns exactly what
    vLLM's Scheduler._get_local_prefix_cache_hit returns, across steps."""
    from vllm.v1.core.sched.scheduler import Scheduler

    counts = {"checks": 0, "from_memo": 0, "hits": 0}

    def compare(s, w):
        s.connector = None
        before = s._memo_computes
        got = s._get_local_prefix_cache_hit(w)
        counts["from_memo"] += s._memo_computes == before
        want = Scheduler._get_local_prefix_cache_hit(s, w)
        mgr = s.kv_cache_manager
        assert _local_hit_view(mgr, got) == _local_hit_view(mgr, want), w.request_id
        # The caller may extend the returned lists; the memo must not change.
        if got[0] is not mgr.empty_kv_cache_blocks:
            for group in got[0].blocks:
                group.append(None)
        again = s._get_local_prefix_cache_hit(w)
        assert _local_hit_view(mgr, again) == _local_hit_view(mgr, want)
        counts["checks"] += 1
        counts["hits"] += want[1] > 0

    for seed in range(4):
        _simulate(patched_pool, seed, steps=60, on_check=compare)
    assert counts["from_memo"] == counts["checks"] > 1000
    assert counts["hits"] > counts["checks"] // 6
    print(counts)


def test_runtime_verifier_agrees(patched_pool):
    s, checks, _ = _simulate(patched_pool, 99, steps=150, verify=10_000)
    assert s._memo_xstep and s._memo_verified > 100 and s._memo_hits == 0


def test_memo_mismatch_turns_the_change_log_off(patched_pool):
    """Once a verified reuse disagrees, the cross-step memo is off and nothing
    trims the change log, so the pool must stop keeping it."""
    mgr = _manager(64)
    d = mgr.block_pool.__dict__
    s = _paw(mgr, verify=10)
    req = _req("m1", list(range(13)))
    max_len = req.num_tokens - 1
    cached = s._memo_compute(req, s._get_prefix_cache_epoch(), max_len)
    # Unrelated inserts move the epoch without touching req's probed hashes.
    other = _req("m2", list(range(500, 513)))
    assert mgr.allocate_slots(other, other.num_tokens) is not None
    mgr.new_step_starts()
    epoch = s._get_prefix_cache_epoch()
    assert epoch != cached[0] and s._memo_still_exact(cached, epoch)
    wrong = (cached[2][0], cached[2][1] + 4, cached[2][2])
    req._paw_hit_cache = cached[:2] + (wrong,) + cached[3:]
    assert s._memo_get(req, epoch, max_len)[2][1] == cached[2][1]
    assert not s._memo_xstep
    assert prefix_change_log.LOG not in d
    late = _req("m3", list(range(900, 913)))
    assert mgr.allocate_slots(late, late.num_tokens) is not None
    mgr.new_step_starts()
    assert prefix_change_log.LOG not in d


def test_change_log_install_is_idempotent_over_the_same_step_patch(patched_pool):
    assert prefix_change_log.install() is None
    wrapped = BlockPool._insert_block_hash
    assert prefix_change_log.install() is None
    assert BlockPool._insert_block_hash is wrapped
    # The plugin patch must not re-apply on top (it keys on get_cached_block).
    vllm_torchtpu._patch_vllm_same_step_prefix_hits()
    assert BlockPool._insert_block_hash is wrapped


def _churn(mgr, rid):
    """Inserts, clears the unwritten set, evicts and resets the prefix cache."""
    req = _req(rid, list(range(13)))
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, req.num_tokens, num_hit, blocks) is not None
    mgr.new_step_starts()
    mgr.free(req)
    assert mgr.reset_prefix_cache()


def test_wrappers_leave_a_pool_alone_until_it_is_opted_in(patched_pool):
    mgr = _manager(32)
    d = mgr.block_pool.__dict__
    _churn(mgr, "a1")
    for name in (
        prefix_change_log.EPOCH,
        prefix_change_log.LOG,
        prefix_change_log.BASE,
    ):
        assert name not in d
    # The change log needs a counting epoch.
    assert not prefix_change_log.enable(mgr.block_pool)
    assert prefix_change_log.LOG not in d
    assert prefix_change_log.enable_epoch(mgr.block_pool)
    assert prefix_change_log.enable(mgr.block_pool)
    _churn(mgr, "a2")
    assert d[prefix_change_log.EPOCH] > 0
    assert d[prefix_change_log.BASE] > 0  # the reset raised the log base


def test_disable_drops_the_change_log(patched_pool):
    mgr = _manager(32)
    d = mgr.block_pool.__dict__
    assert prefix_change_log.enable_epoch(mgr.block_pool)
    assert prefix_change_log.enable(mgr.block_pool)
    req = _req("d1", list(range(13)))
    assert mgr.allocate_slots(req, req.num_tokens) is not None
    assert d[prefix_change_log.LOG]
    prefix_change_log.disable(mgr.block_pool)
    req2 = _req("d2", list(range(100, 113)))
    assert mgr.allocate_slots(req2, req2.num_tokens) is not None
    mgr.new_step_starts()
    assert prefix_change_log.LOG not in d and prefix_change_log.BASE not in d


def test_epoch_moves_on_every_prefix_cache_mutation(patched_pool):
    mgr = _manager(32)
    d = mgr.block_pool.__dict__
    assert prefix_change_log.enable_epoch(mgr.block_pool)
    req = _req("e1", list(range(13)))
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    e0 = d.get("_tpu_prefix_cache_epoch", 0)
    assert mgr.allocate_slots(req, req.num_tokens, num_hit, blocks) is not None
    e1 = d["_tpu_prefix_cache_epoch"]
    assert e1 > e0  # hashes inserted
    assert d.get("_tpu_unwritten_block_ids")
    mgr.new_step_starts()
    assert d["_tpu_prefix_cache_epoch"] == e1 + 1  # the unwritten set cleared
    assert not d.get("_tpu_unwritten_block_ids")
    mgr.new_step_starts()
    assert d["_tpu_prefix_cache_epoch"] == e1 + 1  # nothing to clear
    mgr.free(req)
    assert mgr.reset_prefix_cache()
    assert d["_tpu_prefix_cache_epoch"] > e1 + 1  # hashes removed, then reset


@pytest.mark.parametrize("case", ["pool_not_opted_in", "no_epoch_counters"])
def test_without_a_counting_epoch_there_is_no_memo(patched_pool, monkeypatch, case):
    mgr = _manager(32)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.kv_cache_manager = mgr
    s._paw_scoring_broken = False
    if case == "no_epoch_counters":
        assert prefix_change_log.enable_epoch(mgr.block_pool)
        monkeypatch.setattr(prefix_change_log, "epoch_installed", lambda: False)
    assert s._get_prefix_cache_epoch() is None
    req = _req("n1", list(range(13)))
    s._prefix_warmth(req)
    s._prefix_warmth(req)
    assert s._memo_computes == 2  # nothing was reused
    assert getattr(req, "_paw_hit_cache", None) is None
