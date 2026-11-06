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
"""Scheduler-side prefix hits for MiniMax M3 under DCP, on CPU.

The deployment's KV cache is one group: the two DCP-sharded MSA caches and
the replicated DSpark draft cache (`kv_cache_manager.dcp_replicated_page_tokens`).
A real `Scheduler` over that group, as EngineCore builds it, shows:

- a shared prefix hits as deep at DCP 2 and 8 as at DCP 1 once
  ``prefix_match_unit`` is the sparse block, which the model hook sets
  under DCP; without it hits stop at the ``128 * dcp``-token manager block;
- in a multi-session replay (prompt = previous prompt + output + new text),
  DCP's larger cache keeps more sessions resident: the hit rate rises with
  DCP's capacity rather than falling.
"""

import random

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
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK = 128
MAX_LEN = 65536
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _tpu_scheduler_patches(monkeypatch):
    """The engine-core patches that shape block sizes and prefix hits."""
    import inspect

    from vllm.v1.core import kv_cache_coordinator

    from vllm_torchtpu import (
        _patch_vllm_dcp_fine_prefix_hits,
        _patch_vllm_hybrid_pcp_block_sizes,
    )

    cls = kv_cache_coordinator.UnitaryKVCacheCoordinator
    monkeypatch.setattr(cls, "__init__", inspect.unwrap(cls.__init__))
    monkeypatch.setattr(
        cls, "find_longest_cache_hit", inspect.unwrap(cls.find_longest_cache_hit)
    )
    monkeypatch.setattr(cls, "_tpu_dcp_fine_prefix_patch", False, raising=False)
    monkeypatch.setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        inspect.unwrap(kv_cache_utils.resolve_kv_cache_block_sizes),
    )
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_dcp_fine_prefix_hits()


def _scheduler(dcp: int, capacity_tokens: int, unit: int | None, max_seqs: int = 8):
    specs = {
        "model.layers.0.self_attn.attn": MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=256, dtype=FP8
        ),
        "model.layers.0.self_attn.attn.index_cache": MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=128, dtype=FP8
        ),
        # The replicated draft: same block, dcp times the page bytes.
        "draft_model.model.layers.0.self_attn.attn": FullAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=128,
            dtype=FP8,
            page_size_padded=dcp * 64 * 1024,
            sliding_window=4096,
        ),
    }
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=8192,
            max_model_len=MAX_LEN,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=MAX_LEN
        ),
        cache_config=CacheConfig(
            block_size=BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_config.parallel_config.decode_context_parallel_size = dcp
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    vllm_config.cache_config.prefix_match_unit = unit
    groups = get_kv_cache_groups(vllm_config, dict(specs))
    assert len(groups) == 1  # MSA caches and the draft share one group
    # EngineCore hands the scheduler one leaf of a uniform-type group.
    for group in groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    num_blocks = capacity_tokens // (BLOCK * dcp)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(hash_block_size, sha256)


def _request(request_id, prompt, hasher, max_tokens=8):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    return Request(request_id, prompt, params, None, block_hasher=hasher)


@pytest.mark.parametrize(
    "dcp,unit,expected",
    [
        (1, None, 12288),
        (2, BLOCK, 12288),  # the hook's default under DCP: as deep as DCP 1
        (8, BLOCK, 12288),
        (2, None, 12288 // 256 * 256),
        (8, None, 12288 // 1024 * 1024),
    ],
)
def test_shared_prefix_hit_depth(dcp, unit, expected):
    scheduler, hasher = _scheduler(dcp, 400_000, unit)
    prompt = list(range(20_000))
    first = _request("a", prompt, hasher)
    scheduler.add_request(first)
    for _ in range(3):
        scheduler.schedule()
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    second = _request("b", prompt[:12_300] + [10**6 + i for i in range(100)], hasher)
    scheduler.add_request(second)
    (new,) = scheduler.schedule().scheduled_new_reqs
    assert new.num_computed_tokens == expected


def _replay(dcp: int, capacity_tokens: int, unit: int | None, seed: int = 0) -> float:
    """Token hit rate of a multi-session replay: each finished turn submits
    the session's next prompt (previous prompt + output + fresh text)."""
    rng = random.Random(seed)
    scheduler, hasher = _scheduler(dcp, capacity_tokens, unit)
    sessions, turns = 8, 4
    context = {
        s: [10**7 * (s + 1) + i for i in range(rng.randint(6000, 10000))]
        for s in range(sessions)
    }
    remaining = dict.fromkeys(range(sessions), turns)
    live: dict[str, tuple[int, Request]] = {}
    counter = [0]

    def submit(s):
        counter[0] += 1
        request = _request(
            f"s{s}-{counter[0]}", list(context[s]), hasher, rng.randint(20, 40)
        )
        scheduler.add_request(request)
        live[request.request_id] = (s, request)

    for s in range(sessions):
        submit(s)
    prompt_tokens = hit_tokens = 0
    for _ in range(20_000):
        if not live:
            break
        out = scheduler.schedule()
        for new in out.scheduled_new_reqs:
            prompt_tokens += live[new.req_id][1].num_prompt_tokens
            hit_tokens += new.num_computed_tokens
        req_ids = list(out.num_scheduled_tokens)
        sampled = []
        for req_id in req_ids:
            request = live[req_id][1]
            done = (
                request.num_computed_tokens + out.num_scheduled_tokens[req_id]
                >= request.num_prompt_tokens
            )
            sampled.append([rng.randint(0, 50_000)] if done else [])
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        for req_id, (s, request) in list(live.items()):
            if request.is_finished():
                fresh = [rng.randint(0, 50_000) + 7 * 10**7 for _ in range(500)]
                context[s] = (
                    list(request.prompt_token_ids)
                    + list(request.output_token_ids)
                    + fresh
                )
                del live[req_id]
                remaining[s] -= 1
                if remaining[s]:
                    submit(s)
    assert not live
    return hit_tokens / prompt_tokens


def test_multi_session_hits_rise_with_dcp_capacity():
    # 8 sessions of 6-12K tokens: a 60K-token cache thrashes (LRU over a
    # cyclic working set hits almost nothing), DCP's 1.82x / 2.3x more tokens
    # in the same HBM hold them all.
    base = _replay(1, 60_000, None)
    dcp2 = _replay(2, int(60_000 * 1.82), BLOCK)
    dcp8 = _replay(8, int(60_000 * 2.3), BLOCK)
    assert base < 0.3
    assert dcp2 > 0.6 and dcp8 > 0.6
    # Same capacity: DCP hits as much as DCP 1 (the manager block is coarser,
    # the hit unit is not).
    roomy = _replay(1, 200_000, None)
    assert _replay(2, 200_000, BLOCK) >= roomy - 0.01
    assert _replay(8, 200_000, BLOCK) >= roomy - 0.01
