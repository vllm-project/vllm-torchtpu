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
"""VLLM_TPU_TAIL_HOLD_STEPS on a real hybrid (attention + Mamba align)
PrefixAwareScheduler, through schedule()."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.core.tail_hold import TailHold, is_mamba_tail, tail_geometry

ATTN_BLOCK = 64
MAMBA_BLOCK = 64
# "a": one Mamba block, then a 36-token tail that the align split runs alone.
PROMPT_A = 100
B_LAST = PROMPT_A // MAMBA_BLOCK * MAMBA_BLOCK


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048
    )


def _scheduler(
    model_config, monkeypatch, *, steps, cls=PrefixAwareScheduler, max_num_seqs=2
):
    monkeypatch.setenv("VLLM_TPU_TAIL_HOLD_STEPS", str(steps))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=512,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=ATTN_BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    attention = FullAttentionSpec(
        block_size=ATTN_BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    mamba = MambaSpec(
        block_size=MAMBA_BLOCK,
        shapes=((1, 256),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        page_size_padded=attention.page_size_bytes,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], attention),
            KVCacheGroupSpec(["state"], mamba),
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler_block, hash_block = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    scheduler = cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=scheduler_block,
        hash_block_size=hash_block,
    )
    assert scheduler.need_mamba_block_aligned_split
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(hash_block, sha256)


def _request(rid, n, hasher, seed):
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 10_000 + i for i in range(n)]
    return Request(rid, prompt, params, None, block_hasher=hasher)


def _start(scheduler, hasher, waiting=True):
    """Step 1: `d` (a 32-token prompt, then decoding) and `a` (first chunk
    [0, 64)) take both seats; `w` waits for a seat when ``waiting``."""
    reqs = {
        "d": _request("d", 32, hasher, 1),
        "a": _request("a", PROMPT_A, hasher, 2),
    }
    for r in reqs.values():
        scheduler.add_request(r)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"d": 32, "a": B_LAST}
    if waiting:
        reqs["w"] = _request("w", 200, hasher, 3)
        scheduler.add_request(reqs["w"])
    return reqs, [out.num_scheduled_tokens]


def _steps(scheduler, n):
    return [dict(scheduler.schedule().num_scheduled_tokens) for _ in range(n)]


def test_tail_geometry_and_predicate():
    sched = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        block_size=512,
        use_eagle_block_drop=False,
    )
    assert tail_geometry(sched) == (128, 512)
    sched.use_eagle_block_drop = True
    assert tail_geometry(sched) == (128, 1024)

    def req(computed, prompt, out=0):
        return SimpleNamespace(
            num_computed_tokens=computed,
            num_prompt_tokens=prompt,
            num_output_tokens=out,
        )

    assert is_mamba_tail(req(512, 1000), 512, 512)
    assert not is_mamba_tail(req(0, 300), 512, 512)  # first chunk
    assert not is_mamba_tail(req(512, 1100), 512, 512)  # more than a block
    assert not is_mamba_tail(req(500, 900), 512, 512)  # mid-block
    assert not is_mamba_tail(req(1000, 1000), 512, 512)  # prefill done
    assert not is_mamba_tail(req(512, 1000, out=3), 512, 512)  # resumed


@pytest.mark.parametrize("waiting", [True, False])
def test_k0_matches_upstream(model_config, monkeypatch, waiting):
    runs = []
    for cls in (AsyncScheduler, PrefixAwareScheduler):
        scheduler, hasher = _scheduler(model_config, monkeypatch, steps=0, cls=cls)
        if cls is PrefixAwareScheduler:
            assert scheduler._tail_hold is None
        _, outs = _start(scheduler, hasher, waiting)
        outs += _steps(scheduler, 2)
        scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
        outs += _steps(scheduler, 3)
        runs.append(outs)
    assert runs[0] == runs[1]
    # Upstream runs the tail alone right after the chunk.
    assert runs[0][1]["a"] == PROMPT_A - B_LAST


def test_k2_tail_merges_into_the_next_prefill_step(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, monkeypatch, steps=2)
    reqs, _ = _start(scheduler, hasher)
    # No seat for `w` and nothing else prefills: `a`'s tail is held.
    (s2,) = _steps(scheduler, 1)
    assert "a" not in s2 and "d" in s2
    assert reqs["a"].num_computed_tokens == B_LAST
    # A seat frees up: `w` is admitted and `a`'s tail rides in its step.
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    (s3,) = _steps(scheduler, 1)
    assert s3["a"] == PROMPT_A - B_LAST
    assert s3["w"] > 0
    assert scheduler._tail_hold.n_merged == 1
    assert scheduler._tail_hold.n_expired == 0


def test_k2_tail_runs_alone_after_two_steps(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, monkeypatch, steps=2)
    reqs, _ = _start(scheduler, hasher)
    s2, s3, s4 = _steps(scheduler, 3)
    assert "a" not in s2 and "a" not in s3
    assert s4["a"] == PROMPT_A - B_LAST
    assert "w" not in s4  # still no seat: the tail ran with decodes only
    assert reqs["a"].num_computed_tokens == PROMPT_A
    assert scheduler._tail_hold.n_expired == 1
    assert scheduler._tail_hold.n_hold_steps == 2


def test_no_hold_when_nothing_waits(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, monkeypatch, steps=2)
    _start(scheduler, hasher, waiting=False)
    (s2,) = _steps(scheduler, 1)
    assert s2["a"] == PROMPT_A - B_LAST
    assert scheduler._tail_hold.n_hold_steps == 0


def test_no_hold_when_the_step_would_be_empty(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, monkeypatch, steps=2, max_num_seqs=1)
    a = _request("a", PROMPT_A, hasher, 2)
    scheduler.add_request(a)
    assert scheduler.schedule().num_scheduled_tokens == {"a": B_LAST}
    scheduler.add_request(_request("w", 200, hasher, 3))
    # `w` waits for the only seat, but `a` is all that runs: no hold.
    assert scheduler.schedule().num_scheduled_tokens == {"a": PROMPT_A - B_LAST}


def test_no_hold_while_another_prefill_runs(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, monkeypatch, steps=2, max_num_seqs=3)
    scheduler.add_request(_request("a", PROMPT_A, hasher, 2))
    # `long` needs several 512-budget-limited chunks.
    scheduler.add_request(_request("long", 1500, hasher, 4))
    out1 = scheduler.schedule().num_scheduled_tokens
    assert out1["a"] == B_LAST and out1["long"] > 0
    scheduler.add_request(_request("w", 200, hasher, 3))
    out2 = scheduler.schedule().num_scheduled_tokens
    assert out2["a"] == PROMPT_A - B_LAST and "long" in out2


def _hit_after_a(scheduler, hasher):
    """Prefix hit of a new request with `a`'s prompt."""
    probe = _request("probe", PROMPT_A, hasher, 2)
    _, hit, *_ = scheduler.kv_cache_manager.get_computed_blocks(probe)
    return hit


def test_split_and_prefix_boundaries_unchanged(model_config, monkeypatch):
    """`a` runs the same chunks with and without the hold, and its prompt is
    reusable from the same Mamba checkpoint."""
    chunks, hits = {}, {}
    for steps in (0, 2):
        scheduler, hasher = _scheduler(model_config, monkeypatch, steps=steps)
        reqs, outs = _start(scheduler, hasher)
        outs += _steps(scheduler, 3)
        prefill, done = [], 0
        for n in (o["a"] for o in outs if "a" in o):
            if done >= PROMPT_A:
                break  # decode steps follow
            prefill.append(n)
            done += n
        chunks[steps] = prefill
        assert reqs["a"].num_computed_tokens >= PROMPT_A
        hits[steps] = _hit_after_a(scheduler, hasher)
    assert chunks[0] == chunks[2] == [B_LAST, PROMPT_A - B_LAST]
    assert hits[0] == hits[2] == B_LAST


def test_plan_is_inert_without_mamba_align():
    hold = TailHold(2)
    sched = SimpleNamespace(need_mamba_block_aligned_split=False)
    assert hold.plan(sched, seat_free=False) == frozenset()
