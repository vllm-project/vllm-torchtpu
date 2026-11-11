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
"""Unit tests for PrefillFirstScheduler solo and regular step invariants."""

import json
import random
from collections import deque
from types import SimpleNamespace
from typing import cast

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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    DEFAULT_SOLO_DECODE_STEPS,
    DEFAULT_SOLO_MAX_CONSECUTIVE,
    DEFAULT_SOLO_OVER_TOKENS,
    SCHEDULER_CLS,
    PrefillFirstScheduler,
)
from vllm_torchtpu.platforms.tpu_platform import (
    _validate_prefill_first_scheduler_config,
)

BLOCK_SIZE = 16


@pytest.fixture(scope="module")
def shared_model_config(tmp_path_factory):
    cfg_dir = tmp_path_factory.mktemp("tiny_opt")
    (cfg_dir / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["OPTForCausalLM"],
                model_type="opt",
                hidden_size=64,
                num_attention_heads=2,
                num_hidden_layers=1,
                ffn_dim=128,
                vocab_size=1024,
                max_position_embeddings=32768,
            )
        ),
        encoding="utf-8",
    )
    return ModelConfig(
        model=str(cfg_dir),
        tokenizer=str(cfg_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        seed=42,
        max_model_len=32768,
    )


def _build(
    model_config: ModelConfig,
    *,
    over_tokens: int | None = 32,
    max_consecutive: int | None = None,
    decode_steps: int | None = None,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 128,
    max_model_len: int = 2048,
    long_prefill_token_threshold: int = 0,
    enable_prefix_caching: bool = False,
    data_parallel_size: int = 1,
    scheduler_reserve_full_isl: bool = True,
    sliding_window: int | None = None,
    async_scheduling: bool = True,
):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=scheduler_reserve_full_isl,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=async_scheduling,
            policy="fcfs",
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=enable_prefix_caching,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    if data_parallel_size != 1:
        vllm_config.parallel_config.data_parallel_size = data_parallel_size
        vllm_config.parallel_config.data_parallel_size_local = data_parallel_size
    spec_kw = dict(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    spec = (
        FullAttentionSpec(**spec_kw)
        if sliding_window is None
        else SlidingWindowSpec(sliding_window=sliding_window, **spec_kw)
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    kw = {
        k: v
        for k, v in (
            ("solo_over_tokens", over_tokens),
            ("solo_max_consecutive", max_consecutive),
            ("solo_decode_steps", decode_steps),
        )
        if v is not None
    }
    scheduler = PrefillFirstScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        **kw,
    )
    init_none_hash(sha256)
    hasher = (
        get_request_block_hasher(BLOCK_SIZE, sha256) if enable_prefix_caching else None
    )
    return scheduler, hasher


def _request(
    request_id: str,
    num_tokens: int,
    hasher,
    seed: int = 0,
    prefix: list[int] | None = None,
) -> Request:
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = list(prefix or []) + [
        seed * 100_000 + i for i in range(num_tokens - len(prefix or []))
    ]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _add(scheduler, hasher, *specs: tuple[str, int, int]) -> None:
    for request_id, num_tokens, seed in specs:
        scheduler.add_request(_request(request_id, num_tokens, hasher, seed))


def _runner_output(scheduler, out, token_id: int = 7) -> ModelRunnerOutput:
    req_ids = list(out.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[
            [] if scheduler.requests[r].is_prefill_chunk else [token_id]
            for r in req_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_default_thresholds_without_env_vars(shared_model_config):
    scheduler, _ = _build(
        shared_model_config,
        over_tokens=None,
        max_num_batched_tokens=16384,
        max_model_len=16384,
    )
    assert scheduler.solo_over_tokens == DEFAULT_SOLO_OVER_TOKENS == 8192
    assert scheduler.solo_decode_steps == DEFAULT_SOLO_DECODE_STEPS == 1
    assert scheduler.solo_max_consecutive == DEFAULT_SOLO_MAX_CONSECUTIVE == 0


def test_long_prompt_runs_whole_and_regular_steps_honor_cap(shared_model_config):
    scheduler, hasher = _build(shared_model_config, over_tokens=32)
    _add(scheduler, hasher, ("a", 16, 1), ("b", 16, 2))
    assert scheduler.schedule().num_scheduled_tokens == {"a": 16, "b": 16}
    _add(scheduler, hasher, ("long", 100, 3))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 100}
    assert [r.request_id for r in scheduler.running] == ["a", "b", "long"]
    assert scheduler.schedule().num_scheduled_tokens == {"a": 1, "b": 1, "long": 1}

    scheduler2, hasher2 = _build(shared_model_config, over_tokens=32)
    _add(scheduler2, hasher2, ("s1", 24, 1), ("s2", 24, 2), ("s3", 24, 3))
    totals = [scheduler2.schedule().total_num_scheduled_tokens for _ in range(4)]
    assert totals[0] == 32 and max(totals) <= 32


@pytest.mark.parametrize(
    ("max_consecutive", "decode_steps", "expected"),
    [
        (0, 1, ["L1", "L2", "L3", "regular"]),
        (1, 1, ["L1", "regular", "L2", "regular"]),
        (2, 2, ["L1", "L2", "regular", "regular", "L3", "regular"]),
    ],
)
def test_solo_steps_interleave_with_decode_steps(
    shared_model_config, max_consecutive, decode_steps, expected
):
    scheduler, hasher = _build(
        shared_model_config,
        over_tokens=32,
        max_consecutive=max_consecutive,
        decode_steps=decode_steps,
    )
    _add(scheduler, hasher, ("a", 16, 1))
    scheduler.schedule()
    _add(scheduler, hasher, ("L1", 100, 2), ("L2", 100, 3), ("L3", 100, 4))
    seen = []
    for _ in expected:
        scheduled = scheduler.schedule().num_scheduled_tokens
        solo = [r for r in scheduled if r.startswith("L") and scheduled[r] == 100]
        if solo:
            assert len(scheduled) == 1
            seen.append(solo[0])
        else:
            assert all(scheduled[r] == 1 for r in scheduled)
            seen.append("regular")
    assert seen == expected


def test_fcfs_order_and_fit_gate(shared_model_config):
    scheduler, hasher = _build(shared_model_config, over_tokens=32, max_consecutive=1)
    _add(scheduler, hasher, ("s1", 16, 1))
    scheduler.schedule()
    _add(
        scheduler, hasher, ("L1", 100, 2), ("s2", 16, 3), ("L2", 100, 4), ("s3", 16, 5)
    )
    assert scheduler.schedule().num_scheduled_tokens == {"L1": 100}
    assert scheduler.schedule().num_scheduled_tokens == {"s1": 1, "L1": 1, "s2": 16}
    assert [r.request_id for r in scheduler.waiting] == ["L2", "s3"]
    assert scheduler.schedule().num_scheduled_tokens == {"L2": 100}

    tight, tight_hasher = _build(
        shared_model_config, over_tokens=32, num_blocks=11, max_num_batched_tokens=256
    )
    _add(tight, tight_hasher, ("a", 16, 1))
    tight.schedule()
    _add(tight, tight_hasher, ("long", 150, 2))
    assert tight.schedule().num_scheduled_tokens == {"a": 1}
    tight.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert tight.schedule().num_scheduled_tokens == {"long": 150}


def test_oversized_prompt_falls_through_without_hol_deadlock(shared_model_config):
    scheduler, hasher = _build(
        shared_model_config,
        over_tokens=32,
        num_blocks=11,
        max_num_batched_tokens=256,
        scheduler_reserve_full_isl=False,
    )
    _add(scheduler, hasher, ("oversized", 180, 1), ("short", 16, 2))
    assert scheduler.schedule().num_scheduled_tokens == {"oversized": 32}
    scheduler.finish_requests("oversized", RequestStatus.FINISHED_ABORTED)
    assert scheduler.schedule().num_scheduled_tokens == {"short": 16}


@pytest.mark.parametrize("manual_reset", [True, False])
def test_prefix_cache_hit_falls_through_to_regular_steps(
    shared_model_config, manual_reset
):
    scheduler, hasher = _build(
        shared_model_config,
        over_tokens=32,
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    prefix = [(900_000 if manual_reset else 910_000) + i for i in range(32)]
    scheduler.add_request(_request("cold", 96, hasher, seed=1, prefix=prefix))
    out_cold = scheduler.schedule()
    assert out_cold.num_scheduled_tokens == {"cold": 96}
    scheduler.update_from_output(out_cold, _runner_output(scheduler, out_cold))
    scheduler.finish_requests("cold", RequestStatus.FINISHED_ABORTED)

    warm = _request("warm", 96, hasher, seed=2, prefix=prefix)
    if manual_reset:
        scheduler.kv_cache_manager.new_step_starts()
        assert scheduler.kv_cache_manager.get_computed_blocks(warm)[1] == 32
    scheduler.add_request(warm)
    assert scheduler.schedule().num_scheduled_tokens == {"warm": 32}


def test_async_resumed_request_never_leaks_prompt_tokens(shared_model_config):
    scheduler, hasher = _build(shared_model_config, over_tokens=32)
    _add(scheduler, hasher, ("a", 16, 1))
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens == {"a": 16}

    _add(scheduler, hasher, ("long1", 100, 2))
    out_solo1 = scheduler.schedule()
    assert out_solo1.num_scheduled_tokens == {"long1": 100}
    assert scheduler.schedule().scheduled_cached_reqs.all_token_ids["a"] == [-1]

    scheduler.update_from_output(out1, _runner_output(scheduler, out1, token_id=42))
    scheduler.update_from_output(
        out_solo1, _runner_output(scheduler, out_solo1, token_id=9)
    )
    _add(scheduler, hasher, ("long2", 100, 3))
    assert scheduler.schedule().num_scheduled_tokens == {"long2": 100}
    assert scheduler.schedule().scheduled_cached_reqs.all_token_ids["a"] == [42, -1]


def test_config_validation_rejects_unsupported_modes(shared_model_config, monkeypatch):
    for kw, match in (
        (dict(over_tokens=32, data_parallel_size=2), "data parallelism"),
        (dict(over_tokens=128, max_num_batched_tokens=128), "step budget"),
        (
            dict(over_tokens=32, long_prefill_token_threshold=64),
            "long_prefill_token_threshold",
        ),
        (dict(over_tokens=32, max_consecutive=-1), "non-negative"),
        (dict(over_tokens=32, max_consecutive=1, decode_steps=0), "solo_decode_steps"),
    ):
        with pytest.raises(ValueError, match=match):
            _build(shared_model_config, **kw)

    scheduler, _ = _build(shared_model_config, over_tokens=32)
    monkeypatch.setattr(scheduler, "need_mamba_block_aligned_split", True)
    with pytest.raises(ValueError, match="Mamba"):
        scheduler._check_solo_config()

    def _plat_cfg(cls_ref, dp=1, pp=1, pcp=1, dcp=1, mnbt=16384, lptt=0, **extra):
        return cast(
            VllmConfig,
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_size=dp,
                    pipeline_parallel_size=pp,
                    prefill_context_parallel_size=pcp,
                    decode_context_parallel_size=dcp,
                ),
                scheduler_config=SimpleNamespace(
                    scheduler_cls=cls_ref,
                    max_num_batched_tokens=mnbt,
                    long_prefill_token_threshold=lptt,
                ),
                **extra,
            ),
        )

    for cls_ref in (SCHEDULER_CLS, PrefillFirstScheduler):
        for cfg_obj, match in (
            (_plat_cfg(cls_ref, dp=2), "data parallelism"),
            (_plat_cfg(cls_ref, pp=2), "pipeline parallelism"),
            (_plat_cfg(cls_ref, pcp=2), "PCP or DCP"),
            (_plat_cfg(cls_ref, dcp=2), "PCP or DCP"),
            (_plat_cfg(cls_ref, lora_config=object()), "LoRA"),
            (_plat_cfg(cls_ref, speculative_config=object()), "speculative"),
            (_plat_cfg(cls_ref, kv_transfer_config=object()), "KV transfer"),
            (_plat_cfg(cls_ref, mnbt=8192), "max_num_batched_tokens"),
            (_plat_cfg(cls_ref, lptt=2048), "long_prefill_token_threshold"),
        ):
            with pytest.raises(ValueError, match=match):
                _validate_prefill_first_scheduler_config(cfg_obj)
        _validate_prefill_first_scheduler_config(_plat_cfg(cls_ref))


def test_idle_solo_rejection_falls_through_without_zero_token_livelock(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build(
        shared_model_config,
        over_tokens=32,
        num_blocks=8,
        max_num_batched_tokens=256,
        sliding_window=32,
        scheduler_reserve_full_isl=False,
        async_scheduling=False,
    )
    _add(scheduler, hasher, ("swa_long", 128, 1), ("short", 16, 2))
    steps = 0
    while scheduler.get_num_unfinished_requests() > 0:
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens > 0 and steps < 40
        scheduler.update_from_output(out, _runner_output(scheduler, out))
        steps += 1

    scheduler2, hasher2 = _build(
        shared_model_config,
        over_tokens=32,
        num_blocks=16,
        max_num_batched_tokens=256,
        async_scheduling=False,
    )
    orig_alloc = scheduler2.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        scheduler2.kv_cache_manager,
        "allocate_slots",
        lambda req, n, *a, **kw: None if n > 32 else orig_alloc(req, n, *a, **kw),
    )
    _add(scheduler2, hasher2, ("solo_rejected", 128, 1), ("tail", 16, 2))
    steps = 0
    while scheduler2.get_num_unfinished_requests() > 0:
        out = scheduler2.schedule()
        assert (
            out.total_num_scheduled_tokens > 0
            and not scheduler2._skip_solo
            and steps < 40
        )
        scheduler2.update_from_output(out, _runner_output(scheduler2, out))
        steps += 1


def test_random_async_workload_finishes_cleanly(shared_model_config):
    rng, over_tokens = random.Random(0), 32
    scheduler, hasher = _build(
        shared_model_config,
        over_tokens=over_tokens,
        max_consecutive=1,
        num_blocks=48,
        max_num_seqs=6,
        max_num_batched_tokens=256,
    )
    in_flight: deque = deque()
    num_added = 0
    for step in range(240):
        if step < 120 and rng.random() < 0.2:
            req = _request(
                f"r{num_added}", rng.choice([8, 24, 48, 96]), hasher, num_added
            )
            req.max_tokens = rng.randint(1, 8)
            scheduler.add_request(req)
            num_added += 1
        fresh = {r.request_id for r in scheduler.waiting if not r.num_computed_tokens}
        out = scheduler.schedule()
        scheduled = out.num_scheduled_tokens
        solo = [r for r, n in scheduled.items() if n > over_tokens]
        if solo:
            assert (
                len(scheduled) == 1
                and solo[0] in fresh
                and scheduled[solo[0]] == scheduler.requests[solo[0]].num_tokens
            )
        else:
            assert out.total_num_scheduled_tokens <= over_tokens
        in_flight.append((out, _runner_output(scheduler, out)))
        if len(in_flight) == 2 or not scheduled:
            scheduler.update_from_output(*in_flight.popleft())
    while in_flight:
        scheduler.update_from_output(*in_flight.popleft())
    assert num_added >= 15 and scheduler.get_num_unfinished_requests() == 0
