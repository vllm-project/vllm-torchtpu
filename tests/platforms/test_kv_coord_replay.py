# SPDX-License-Identifier: Apache-2.0
"""The hybrid KV coordinator speedups are exact.

See `vllm_torchtpu.core.kv_coord_replay`. A randomized scheduler-like driver
(admissions with full and partial prefix hits, chunked prefill, speculative
decode steps with random acceptance, finishes, preemptions and allocation
failures) runs twice on the same seed: once with upstream vLLM, once with
``VLLM_TPU_KV_COORD_TAIL_SCAN`` and ``VLLM_TPU_KV_COORD_GROUP_REPLAY``. Every
allocation result, every prefix hit, and after every step every manager's
per-request state, every block's refcount and hash, the free queue order and
the pool's hash map must be identical.

``test_microbench`` prints CPU timings at the Qwen3.8 layout (37 Mamba
"align" groups + one attention group, block 512, hash 128); run with ``-s``.
"""

import random
import time
from collections import deque

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.core import kv_coord_replay

_TOUCHED = (
    (HybridKVCacheCoordinator, "remove_skipped_blocks"),
    (HybridKVCacheCoordinator, "get_num_blocks_to_allocate"),
    (HybridKVCacheCoordinator, "allocate_new_blocks"),
    (HybridKVCacheCoordinator, "cache_blocks"),
    (HybridKVCacheCoordinator, "free"),
    (HybridKVCacheCoordinator, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
    (MambaManager, "_get_num_evictable_blocks"),
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (MambaManager, "_needs_internal_checkpoint"),
    (MambaManager, "get_num_blocks_to_allocate"),
    (MambaManager, "new_step_starts"),
    (BlockPool, "get_cached_block"),
    (BlockPool, "_insert_block_hash"),
)
_ABSENT = object()


@pytest.fixture
def restore():
    """Put every class attribute the patches touch back as it was."""
    saved = [(o, n, o.__dict__.get(n, _ABSENT)) for o, n in _TOUCHED]
    # Start from upstream: earlier tests in this process may have patched.
    import inspect

    for owner, name, raw in saved:
        if raw is not _ABSENT and inspect.isfunction(raw):
            setattr(owner, name, inspect.unwrap(raw))
    yield
    for owner, name, raw in saved:
        if raw is _ABSENT:
            if name in owner.__dict__:
                delattr(owner, name)
        else:
            setattr(owner, name, raw)


def _install_tpu_manager_patches(monkeypatch, retention: bool, nobound: bool):
    from vllm_torchtpu.core import mamba_checkpoint_retention, mamba_tail_fold

    if retention:
        monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
        monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
        mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    if nobound:
        mamba_tail_fold._patch_nobound_manager()


def _install_kv_coord(monkeypatch, tail: bool, replay: bool):
    monkeypatch.setenv("VLLM_TPU_KV_COORD_TAIL_SCAN", "1" if tail else "0")
    monkeypatch.setenv("VLLM_TPU_KV_COORD_GROUP_REPLAY", "1" if replay else "0")
    kv_coord_replay.patch_kv_coord()
    # The patch only installs over the upstream vLLM source it was written
    # against (source-hash check); on another vLLM it stays off by design, so
    # there is nothing to compare.
    import vllm

    missing = []
    if tail and not getattr(
        MambaManager._get_num_evictable_blocks, kv_coord_replay._PATCH_FLAG, False
    ):
        missing.append("tail scan")
    if replay and not getattr(
        HybridKVCacheCoordinator.remove_skipped_blocks,
        kv_coord_replay._PATCH_FLAG,
        False,
    ):
        missing.append("group replay")
    if missing:
        pytest.skip(
            f"{' and '.join(missing)} not applied: this vLLM "
            f"({getattr(vllm, '__version__', '?')}) does not match the upstream "
            "source hashes the patch checks"
        )


def _manager(
    *,
    num_mamba: int,
    block: int,
    hash_block: int,
    num_blocks: int,
    spec_blocks: int = 0,
    checkpoint: bool = False,
    attention_at: int = -1,
    two_specs: bool = False,
    retention_interval=None,
) -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=block, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    groups = []
    for g in range(num_mamba):
        shapes = ((1, 64),) if not (two_specs and g % 3 == 2) else ((2, 32),)
        state = MambaSpec(
            block_size=block,
            shapes=shapes,
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=attention.page_size_bytes,
            num_speculative_blocks=spec_blocks,
            num_prefill_checkpoint_blocks=1 if checkpoint else 0,
            prefill_checkpoint_alignment=hash_block if checkpoint else None,
        )
        groups.append(KVCacheGroupSpec([f"state{g}"], state))
    position = len(groups) if attention_at < 0 else attention_at
    groups.insert(position, KVCacheGroupSpec(["attention"], attention))
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=block,
        hash_block_size=hash_block,
        enable_caching=True,
    )


def _ids(value):
    if isinstance(value, KVCacheBlock):
        return ("B", value.block_id)
    if isinstance(value, dict):
        return sorted(((repr(k), _ids(v)) for k, v in value.items()), key=repr)
    if isinstance(value, set):
        return sorted(repr(_ids(v)) for v in value)
    if isinstance(value, (list, tuple, deque)):
        return [_ids(v) for v in value]
    if isinstance(value, (int, float, str, bool, type(None), bytes)):
        return value
    return repr(type(value))


def _fingerprint(manager: KVCacheManager):
    coordinator = manager.coordinator
    pool = coordinator.block_pool
    managers = []
    for m in coordinator.single_type_managers:
        state = {}
        for name, value in sorted(m.__dict__.items()):
            if name in ("block_pool", "kv_cache_spec", "_null_block"):
                continue
            state[name] = _ids(value)
        managers.append(state)
    blocks = [
        (b.block_id, b.ref_cnt, repr(b.block_hash), b.block_hash_num_tokens)
        for b in pool.blocks
    ]
    free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    cached = _ids(pool.cached_block_hash_to_block._cache)
    return managers, blocks, free, cached, pool.get_num_free_blocks()


def _scenario(seed: int, config: dict, steps: int, *, check_every: int = 1):
    """Drive the manager like the scheduler; return the full trace."""
    rng = random.Random(seed)
    m = _manager(**config)
    B = config["block"]
    H = config["hash_block"]
    K = config.get("spec_blocks", 0)
    hasher = get_request_block_hasher(H, sha256)
    prefixes = [
        [
            rng.randrange(5000)
            for _ in range(rng.choice((2, 3, 5)) * B + rng.randrange(B))
        ]
        for _ in range(4)
    ]
    trace = []
    waiting: deque = deque()
    running: dict = {}
    next_id = 0

    def chunk_end(req, start):
        n = req.num_tokens
        end = min(n, start + rng.choice((H, 2 * H, B, 2 * B, 4 * B)))
        if end < n:
            aligned = end // B * B
            end = aligned if aligned > start else min(n, (start // B + 1) * B)
        return end

    for step in range(steps):
        m.new_step_starts()
        if rng.random() < 0.6 and len(waiting) + len(running) < 10:
            p = rng.choice(prefixes)
            cut = rng.randrange(len(p) // 2, len(p) + 1)
            tail = [rng.randrange(5000) for _ in range(rng.randrange(0, 2 * B))]
            toks = p[:cut] + tail
            while len(toks) < 2:
                toks.append(7)
            waiting.append(
                Request(
                    f"r{next_id}",
                    toks,
                    SamplingParams(max_tokens=1 << 20),
                    None,
                    block_hasher=hasher,
                )
            )
            next_id += 1

        for rid in list(running):
            req = running[rid]
            u = rng.random()
            if u < 0.03:
                m.free(req)
                trace.append(("finish", rid))
                del running[rid]
                continue
            if u < 0.06:
                m.free(req)
                req.status = RequestStatus.PREEMPTED
                req.num_computed_tokens = 0
                waiting.appendleft(req)
                trace.append(("preempt", rid))
                del running[rid]
                continue
            c = req.num_computed_tokens
            if c < req.num_tokens - 1 or (c < req.num_tokens and c == 0):
                end = chunk_end(req, c)
                out = m.allocate_slots(req, end - c, num_lookahead_tokens=K)
                trace.append(("chunk", rid, c, end, out and out.get_block_ids()))
                if out is None:
                    m.free(req)
                    req.status = RequestStatus.PREEMPTED
                    req.num_computed_tokens = 0
                    waiting.appendleft(req)
                    del running[rid]
                    continue
                req.num_computed_tokens = end
                if end == req.num_tokens:
                    req.append_output_token_ids([rng.randrange(5000)])
                continue
            # Decode: one verified token plus K drafts.
            drafts = K
            out = m.allocate_slots(req, 1 + drafts, num_lookahead_tokens=K)
            trace.append(("decode", rid, c, out and out.get_block_ids()))
            if out is None:
                m.free(req)
                req.status = RequestStatus.PREEMPTED
                req.num_computed_tokens = 0
                waiting.appendleft(req)
                del running[rid]
                continue
            accepted = rng.randint(0, drafts)
            req.num_computed_tokens = c + 1 + accepted
            req.append_output_token_ids(
                [rng.randrange(5000) for _ in range(1 + accepted)]
            )

        budget = 2
        while waiting and budget:
            budget -= 1
            req = waiting[0]
            blocks, hit, junction = m.get_computed_blocks(req)
            if hasattr(req, "shared_prefix_boundary"):
                req.shared_prefix_boundary = junction
            start = hit
            end = chunk_end(req, start) if start < req.num_tokens else req.num_tokens
            if end <= start:
                end = min(req.num_tokens, start + 1)
            trace.append(("hit", req.request_id, hit, junction, blocks.get_block_ids()))
            out = m.allocate_slots(
                req,
                end - start,
                num_new_computed_tokens=hit,
                new_computed_blocks=blocks,
                num_lookahead_tokens=K,
            )
            trace.append(
                ("admit", req.request_id, start, end, out and out.get_block_ids())
            )
            if out is None:
                break
            waiting.popleft()
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens = end
            if end == req.num_tokens:
                req.append_output_token_ids([rng.randrange(5000)])
            running[req.request_id] = req

        if step % check_every == 0:
            trace.append(("state", step, _fingerprint(m)))
    for req in list(running.values()):
        m.free(req)
    trace.append(("end", _fingerprint(m)))
    return trace, m


_CONFIGS = {
    # Qwen4Exp order: state groups first; fine-grained (partial) hits.
    "mamba_first": dict(num_mamba=9, block=64, hash_block=16, num_blocks=420),
    # MTP scratch blocks and an internal prefill checkpoint (tail fold).
    "spec_ckpt": dict(
        num_mamba=9,
        block=64,
        hash_block=16,
        num_blocks=420,
        spec_blocks=2,
        checkpoint=True,
    ),
    # Attention between the state groups: followers are not contiguous.
    "interleaved": dict(
        num_mamba=8,
        block=64,
        hash_block=16,
        num_blocks=380,
        attention_at=3,
        spec_blocks=1,
    ),
    # Two same-spec sets; tight pool, so allocations fail and preempt.
    "two_specs_tight": dict(
        num_mamba=9,
        block=64,
        hash_block=16,
        num_blocks=170,
        two_specs=True,
    ),
    # Block-aligned hits only (hash == block).
    "coarse": dict(num_mamba=6, block=32, hash_block=32, num_blocks=300),
}


def _tpu_variant(name):
    # (retention patch, tail-fold no-boundary manager patch, retention interval)
    return {
        "upstream": (False, False, None),
        "retention": (True, False, 0),
        "retention_nobound": (True, True, 0),
    }[name]


@pytest.mark.parametrize("variant", ["upstream", "retention", "retention_nobound"])
@pytest.mark.parametrize("config_name", sorted(_CONFIGS))
def test_flags_on_equal_flags_off(restore, monkeypatch, config_name, variant):
    retention, nobound, interval = _tpu_variant(variant)
    config = dict(_CONFIGS[config_name], retention_interval=interval)
    if nobound and not config.get("spec_blocks"):
        pytest.skip("the no-boundary fold needs speculative scratch blocks")
    _install_tpu_manager_patches(monkeypatch, retention, nobound)
    seeds = (1, 2, 3)
    steps = 160
    off = [_scenario(s, config, steps)[0] for s in seeds]
    _install_kv_coord(monkeypatch, tail=True, replay=True)
    skips = runs = 0
    for s, expected in zip(seeds, off):
        got, m = _scenario(s, config, steps)
        plan = m.coordinator.__dict__[kv_coord_replay._PLAN_ATTR]
        assert plan is not None
        skips += plan.skips
        runs += plan.runs
        assert len(got) == len(expected)
        for i, (a, b) in enumerate(zip(got, expected)):
            assert a == b, f"seed {s}: first difference at trace entry {i}: {b[:2]}"
    # The replay path really ran: most leader calls skipped their followers.
    assert skips > runs > 0, (skips, runs)


def test_flags_off_installs_nothing(restore, monkeypatch):
    before = {(o, n): o.__dict__.get(n, _ABSENT) for o, n in _TOUCHED}
    _install_kv_coord(monkeypatch, tail=False, replay=False)
    assert before == {(o, n): o.__dict__.get(n, _ABSENT) for o, n in _TOUCHED}


def test_tail_scan_matches_full_count(restore, monkeypatch):
    _install_kv_coord(monkeypatch, tail=True, replay=False)
    base = SingleTypeKVCacheManager._get_num_evictable_blocks.__func__
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
    null = pool.null_block
    rng = random.Random(0)
    for _ in range(3000):
        n = rng.randrange(0, 12)
        blocks = []
        for _ in range(n):
            u = rng.random()
            if u < 0.5:
                blocks.append(null)
            else:
                b = KVCacheBlock(rng.randrange(1, 15), ref_cnt=rng.choice((0, 0, 1, 2)))
                if u > 0.95:
                    b.is_null = True  # a null-flagged block that is not the null
                blocks.append(b)
        if rng.random() < 0.5:
            # The Mamba hit shape: nulls then one state block.
            k = rng.randrange(0, 10)
            blocks = [null] * k + [KVCacheBlock(3, ref_cnt=rng.choice((0, 1)))]
        for seq in (blocks, tuple(blocks)):
            assert MambaManager._get_num_evictable_blocks(seq) == base(
                MambaManager, seq
            )


def test_upstream_drift_disables(restore, monkeypatch):
    monkeypatch.setattr(
        kv_coord_replay, "_UPSTREAM_SOURCES", {"cache_blocks": "0" * 16}
    )
    monkeypatch.setenv("VLLM_TPU_KV_COORD_GROUP_REPLAY", "1")
    monkeypatch.setenv("VLLM_TPU_KV_COORD_TAIL_SCAN", "0")
    kv_coord_replay.patch_kv_coord()
    assert not getattr(
        HybridKVCacheCoordinator.remove_skipped_blocks,
        kv_coord_replay._PATCH_FLAG,
        False,
    )


# ---------------------------------------------------------------------------
# CPU microbench at the Qwen3.8 layout.


def _bench_layout(seed_prompt: int):
    m = _manager(num_mamba=37, block=512, hash_block=128, num_blocks=20000)
    hasher = get_request_block_hasher(128, sha256)
    rng = random.Random(seed_prompt)
    prompt = 114_000
    base = [rng.randrange(150000) for _ in range(prompt)]

    def mk(rid, toks):
        return Request(
            rid, toks, SamplingParams(max_tokens=1 << 20), None, block_hasher=hasher
        )

    # Turn 0 computes the prompt so its blocks are cached.
    r0 = mk("r0", base)
    blocks, nc, _ = m.get_computed_blocks(r0)
    done = 0
    while done < prompt:
        n = min(8192, prompt - done)
        out = m.allocate_slots(
            r0,
            n,
            num_new_computed_tokens=0 if done else nc,
            new_computed_blocks=blocks if not done else None,
            num_lookahead_tokens=3,
        )
        assert out is not None
        r0.num_computed_tokens += n
        done += n
        m.new_step_starts()
    m.free(r0)
    m.new_step_starts()
    return m, base, mk, rng


def _bench(label):
    m, base, mk, rng = _bench_layout(0)
    admit = []
    for i in range(23):
        toks = base + [rng.randrange(150000) for _ in range(2000)]
        r = mk(f"a{i}", toks)
        b, nc, _ = m.get_computed_blocks(r)
        t0 = time.perf_counter()
        out = m.allocate_slots(
            r,
            len(toks) - nc,
            num_new_computed_tokens=nc,
            new_computed_blocks=b,
            num_lookahead_tokens=3,
        )
        admit.append((time.perf_counter() - t0) * 1e3)
        assert out is not None
        m.free(r)
        m.new_step_starts()
    admit = sorted(admit[3:])
    # Ten running requests, as at c32 (HOSTADMIT reqs p50 10), decoding.
    running = []
    for i in range(10):
        toks = base[: 60_000 + 5_000 * i] + [rng.randrange(150000) for _ in range(500)]
        r = mk(f"run{i}", toks)
        b, nc, _ = m.get_computed_blocks(r)
        out = m.allocate_slots(
            r,
            len(toks) - nc,
            num_new_computed_tokens=nc,
            new_computed_blocks=b,
            num_lookahead_tokens=3,
        )
        assert out is not None
        r.num_computed_tokens = len(toks)
        r.append_output_token_ids([1])
        running.append(r)
    m.new_step_starts()
    per = []
    for _ in range(300):
        for r in running:
            t0 = time.perf_counter()
            out = m.allocate_slots(r, 4, num_lookahead_tokens=3)
            per.append((time.perf_counter() - t0) * 1e6)
            assert out is not None
            r.num_computed_tokens += 4
            r.append_output_token_ids([1, 2, 3, 4])
        m.new_step_starts()
    per.sort()
    line = (
        f"[kv_coord microbench] {label:<14} admit allocate_slots ms p50 "
        f"{admit[len(admit) // 2]:.3f} | running allocate_slots us p50 "
        f"{per[len(per) // 2]:.1f} p90 {per[int(len(per) * 0.9)]:.1f}"
    )
    print(line)
    return line


def test_microbench(restore, monkeypatch):
    lines = [_bench("off")]
    _install_kv_coord(monkeypatch, tail=True, replay=False)
    lines.append(_bench("tail_scan"))
    _install_kv_coord(monkeypatch, tail=True, replay=True)
    lines.append(_bench("tail+replay"))
    print("\n".join(lines))


def test_equivalence_check_catches_a_wrong_skip(restore, monkeypatch):
    """Negative control: replaying the leader with no checks at all (always
    skip the followers) must be caught by the comparison."""
    config = dict(_CONFIGS["mamba_first"], retention_interval=None)
    expected = _scenario(1, config, 80)[0]
    _install_kv_coord(monkeypatch, tail=True, replay=True)
    monkeypatch.setattr(
        kv_coord_replay._Plan, "decide", lambda self, *args, **kwargs: True
    )
    got = _scenario(1, config, 80)[0]
    assert got != expected


def test_engine_core_stage_installs_with_flags():
    """The registry installs both patches in a fresh engine-core process."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        JAX_PLATFORMS="cpu",
        VLLM_TPU_KV_COORD_TAIL_SCAN="1",
        VLLM_TPU_KV_COORD_GROUP_REPLAY="1",
    )
    code = """
from vllm_torchtpu import patch_registry
patch_registry.apply("engine_core")
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator as H
from vllm.v1.core.single_type_kv_cache_manager import MambaManager as M
flag = "_tpu_kv_coord_patch"
assert getattr(H.allocate_new_blocks, flag, False)
assert getattr(M._get_num_evictable_blocks, flag, False)
print("installed")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "installed" in out.stdout
