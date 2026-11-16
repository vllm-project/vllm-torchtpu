# SPDX-License-Identifier: Apache-2.0
"""Per-request prefix lookup line (VLLM_TPU_PREFIX_LOOKUP_LOG): the block fields from per-level presence, the HBM
presence read from a real BlockPool, one line per request from the connector scheduler's lookup, one tier-restore
line per load job with the store calls unchanged, and one request-arrival line per request a real scheduler adds."""

import json
from types import SimpleNamespace as NS

import pytest

from vllm_torchtpu.offload import lookup_log

BS = 16


def test_split_chain_shows_a_union_hit_that_neither_level_has_alone():
    # Head in HBM with no checkpoint there; tail and the newest checkpoint in the tier.
    T, F = True, False
    out = lookup_log.summarize(hbm_attn=[T, T, T, T, F, F], hbm_ckpt=[F] * 6,
                               tier_attn=[F, F, F, F, T, T], tier_ckpt=[F, F, F, F, F, T], local_blocks=0, cap=6)
    assert out == {"tier_attn": 0, "tier_attn_any": 2, "tier_ckpts": [6], "hbm_attn": 4, "hbm_ckpts": [],
                   "union_attn": 6, "union_ckpt": 6}


def test_lost_tail_leaves_no_checkpoint_inside_the_rest():
    T, F = True, False
    out = lookup_log.summarize(hbm_attn=[F] * 6, hbm_ckpt=[F] * 6, tier_attn=[T, T, T, F, F, F],
                               tier_ckpt=[F, F, F, F, F, T], local_blocks=0, cap=6)
    assert (out["tier_attn"], out["union_attn"], out["union_ckpt"], out["tier_ckpts"]) == (3, 3, 0, [6])


def test_tier_run_starts_at_the_hbm_hit_and_fields_stop_at_cap():
    T, F = True, False
    out = lookup_log.summarize(hbm_attn=[T, T, F, F, F], hbm_ckpt=[F, T, F, F, F], tier_attn=[F, F, T, T, T],
                               tier_ckpt=[F, F, F, T, T], local_blocks=2, cap=4)
    assert (out["tier_attn"], out["tier_ckpts"], out["hbm_ckpts"], out["union_attn"], out["union_ckpt"]) == (4, [4], [2], 4, 4)
    assert lookup_log.summarize(None, None, [T], [T], 0, 1)["hbm_attn"] is None


def test_hbm_presence_reads_the_block_pool_per_group():
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    req = Request("r", prompt_token_ids=list(range(4 * BS)), sampling_params=SamplingParams(max_tokens=1),
                  pooling_params=None, block_hasher=get_request_block_hasher(BS, sha256))
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=BS)
    pool.cache_full_blocks(req, pool.get_new_blocks(3), 0, 3, BS, 0)   # attention group 0: blocks 0-2
    pool.cache_full_blocks(req, pool.get_new_blocks(3), 0, 3, BS, 1)   # attention group 1: blocks 0-2
    state = pool.get_new_blocks(3)
    pool.cache_full_blocks(req, state, 0, 3, BS, 2)                     # state group: then drop all but block 1
    for b in (state[0], state[2]):
        pool._maybe_evict_cached_block(b)
    attn, ckpt = lookup_log.hbm_presence(pool, req.block_hashes, [0, 1], [2], cap=3)
    assert attn == [True, True, True] and ckpt == [False, True, False]


def test_log_lookup_writes_one_line_per_request(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    lines = []
    monkeypatch.setattr(lookup_log, "logger", NS(info=lambda fmt, *a: lines.append(fmt % a)))
    attn_spec, state_spec = object.__new__(FullAttentionSpec), object.__new__(MambaSpec)
    configs = [NS(tokens_per_chunk=BS, kv_cache_spec=state_spec), NS(tokens_per_chunk=BS, kv_cache_spec=attn_spec)]
    keys = {"s": [f"s{k}" for k in range(6)], "a": [f"a{k}" for k in range(6)]}
    in_tier = {"a4", "a5", "s5"}   # split chain: tail and newest checkpoint in the tier
    manager = NS(_sub_hashes=lambda k: [k], _block_status=NS(HOST="H", HOST_AND_HBM="HH"),
                 _probe=lambda hs: [(h, NS(status="H")) for h in hs if h in in_tier])
    sched = NS(config=NS(kv_group_configs=configs), _raiden_manager=manager,
               _req_status={"r": NS(group_states=(NS(offload_keys=keys["s"]), NS(offload_keys=keys["a"])))})
    T, F = True, False
    req = NS(request_id="r", num_tokens=6 * BS + 1, shared_prefix_boundary=4 * BS,
             _tpu_hbm_lookup=([T, T, T, T, F, F], [F] * 6))
    lookup_log.log_lookup(sched, req, 0, 0)
    lookup_log.log_lookup(sched, req, 0, 0)
    assert len(lines) == 1 and lines[0].startswith("prefix-lookup ")
    line = json.loads(lines[0].split(" ", 1)[1])
    assert line["cap"] == 6 and line["junction_tokens"] == 4 * BS
    assert (line["hbm_attn"], line["tier_attn"], line["tier_ckpts"], line["union_attn"], line["union_ckpt"]) == (4, 0, [6], 6, 6)


def test_flag_is_off_by_default_and_host_only(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.delenv("VLLM_TPU_PREFIX_LOOKUP_LOG", raising=False)
    assert envs.VLLM_TPU_PREFIX_LOOKUP_LOG is False
    assert "VLLM_TPU_PREFIX_LOOKUP_LOG" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def _restore_run(monkeypatch, flag: bool):
    """Two load jobs through the connector scheduler (one lands, one fails); returns the store calls, what the
    scheduler echoed each step, and the tier-restore lines."""
    from test_raiden_store_offload import (FakeKVCacheStore, TestSchedulerFenceFlow, _ctx, _key, make_manager,
                                           seed_host_entries, sub_hashes)

    if flag:
        monkeypatch.setenv("VLLM_TPU_PREFIX_LOOKUP_LOG", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_PREFIX_LOOKUP_LOG", raising=False)
    ticks = iter(range(1, 100))
    monkeypatch.setattr(lookup_log, "_clock", lambda: float(next(ticks)))
    lines = []
    monkeypatch.setattr(lookup_log, "logger", NS(info=lambda fmt, *a: lines.append(fmt % a)))
    store = FakeKVCacheStore()
    manager = make_manager(store)
    for k in range(3):
        seed_host_entries(store, _key(k))
    log = lookup_log.install_restore_log(manager, 1000)
    wrapped = {"prepare_load", "submit_load_job", "poll_finished_jobs", "drain_jobs"} & set(vars(manager))
    assert (log is not None, bool(wrapped)) == (flag, flag)

    h = TestSchedulerFenceFlow()
    sched = h._make_scheduler(manager)
    manager.prepare_load([_key(0), _key(1)], _ctx("r1"))
    manager.prepare_load([_key(2)], _ctx("r2"))
    jobs = {3: h._load_job([_key(0), _key(1)], [10, 11], "r1"), 4: h._load_job([_key(2)], [12], "r2")}
    echoes = []
    for land in (None, (sub_hashes(_key(0)) + sub_hashes(_key(1)), True), (sub_hashes(_key(2)), False)):
        if land:
            store.complete_load(land[0], success=land[1])
        meta = h._build_meta(sched, h._stock_meta(load_jobs=jobs if land is None else None))
        echoes.append((meta.finished_load_jobs, meta.failed_load_device_block_ids))
    calls = [(c.name, c.args) for c in store.calls if c.name != "set_eviction_callback"]
    return calls, echoes, [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("tier-restore ")]


def test_tier_restore_line_per_load_job_and_same_store_calls(monkeypatch):
    calls_off, echoes_off, lines_off = _restore_run(monkeypatch, flag=False)
    calls_on, echoes_on, lines_on = _restore_run(monkeypatch, flag=True)
    assert calls_on == calls_off and any(name == "load" for name, _ in calls_on)
    assert echoes_on == echoes_off == [({}, []), ({3: "r1"}, []), ({4: "r2"}, [12])]
    assert lines_off == []
    # Clock: prepare r1 = 1, prepare r2 = 2, launches 3 and 4, polls 5, 6, 7.
    assert lines_on == [
        {"req_id": "r1", "job": 3, "keys": 2, "bytes": 2000, "submit": 1.0, "start": 3.0, "end_lo": 5.0, "end": 6.0,
         "ok": True},
        {"req_id": "r2", "job": 4, "keys": 1, "bytes": 1000, "submit": 2.0, "start": 4.0, "end_lo": 6.0, "end": 7.0,
         "ok": False},
    ]


def _scheduler(monkeypatch):
    """A real vLLM scheduler on CPU with prefix caching and no connector."""
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
    config.cache_config.num_gpu_blocks = 64
    kv_cache_config = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=64,
                                                      dtype=torch.float16))])
    init_none_hash(sha256)
    return Scheduler(vllm_config=config, kv_cache_config=kv_cache_config,
                     structured_output_manager=StructuredOutputManager(config), block_size=BS, hash_block_size=BS,
                     log_stats=False)


def _arrival_run(monkeypatch, flag: bool):
    """Three requests through a real scheduler (two added, a schedule pass, a third added, then a duplicate id);
    returns what the scheduler did and the request-arrival lines."""
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request

    if flag:
        monkeypatch.setenv("VLLM_TPU_PREFIX_LOOKUP_LOG", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_PREFIX_LOOKUP_LOG", raising=False)
    walls, monos = iter(range(100, 200)), iter(range(5, 100))
    monkeypatch.setattr(lookup_log, "_clock", lambda: float(next(walls)))
    monkeypatch.setattr(lookup_log, "_monotonic", lambda: float(next(monos)))
    lines = []
    monkeypatch.setattr(lookup_log, "logger", NS(info=lambda fmt, *a: lines.append(fmt % a)))
    lookup_log.patch_prefix_lookup_log()
    sched = _scheduler(monkeypatch)

    def request(i, n):
        return Request(f"r{i}", [1000 * (i + 1) + t for t in range(n)], SamplingParams(max_tokens=4), None,
                       arrival_time=50.0 + i, block_hasher=get_request_block_hasher(BS, sha256))

    sched.add_request(request(0, 40))
    sched.add_request(request(1, 24))
    admitted = [r.req_id for r in sched.schedule().scheduled_new_reqs]
    sched.add_request(request(2, 33))
    with pytest.raises(AssertionError):   # a second request with a held id: no line, the error passes through
        sched.add_request(request(2, 33))
    state = (admitted, [r.request_id for r in sched.waiting], [r.request_id for r in sched.running])
    return state, [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("request-arrival ")]


def test_request_arrival_line_per_added_request(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    for cls, name in ((Scheduler, "add_request"), (KVCacheManager, "get_computed_blocks")):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    state_off, lines_off = _arrival_run(monkeypatch, flag=False)
    assert lines_off == [] and not getattr(Scheduler.add_request, lookup_log._PATCH_FLAG, False)
    state_on, lines_on = _arrival_run(monkeypatch, flag=True)
    assert getattr(Scheduler.add_request, lookup_log._PATCH_FLAG, False)
    assert state_on == state_off == (["r0", "r1"], ["r2"], ["r0", "r1"])
    # Clocks: one wall and one monotonic read per new request.
    assert lines_on == [
        {"req_id": "r0", "wall": 100.0, "mono": 5.0, "api_arrival": 50.0, "prompt_tokens": 40,
         "waiting": 0, "running": 0},
        {"req_id": "r1", "wall": 101.0, "mono": 6.0, "api_arrival": 51.0, "prompt_tokens": 24,
         "waiting": 1, "running": 0},
        {"req_id": "r2", "wall": 102.0, "mono": 7.0, "api_arrival": 52.0, "prompt_tokens": 33,
         "waiting": 0, "running": 2},
    ]
