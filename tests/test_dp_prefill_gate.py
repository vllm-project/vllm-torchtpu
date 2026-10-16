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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs
from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
from vllm_torchtpu.core.experimental import dp_prefill_gate
from vllm_torchtpu.core.experimental.dp_prefill_gate import DpPrefillGateScheduler
from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import SrptPrefillScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"
GATE_ENVS = (
    "VLLM_TPU_PREFILL_GATE_MIN_CADENCES",
    "VLLM_TPU_PREFILL_GATE_LOG_STEPS",
)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


@pytest.fixture
def make_scheduler(model_config, monkeypatch):
    """Build a real DpPrefillGateScheduler the way EngineCore does."""

    def make(
        *,
        min_cadences: int = 0,
        log_steps: int = 1000,
        max_num_seqs: int = 8,
        max_num_active_seqs: int | None = None,
        max_num_batched_tokens: int = 64,
        long_prefill_token_threshold: int = 0,
        parallel_drafting: bool | None = None,
    ) -> DpPrefillGateScheduler:
        monkeypatch.setenv("VLLM_TPU_PREFILL_GATE_MIN_CADENCES", str(min_cadences))
        monkeypatch.setenv("VLLM_TPU_PREFILL_GATE_LOG_STEPS", str(log_steps))
        parallel_config = ParallelConfig()
        speculative_config = None
        if parallel_drafting is not None:
            # K=2 draft model: 1 drafting slot per request, or K with PARD.
            speculative_config = SpeculativeConfig(
                target_model_config=model_config,
                target_parallel_config=parallel_config,
                method="draft_model",
                model=MODEL,
                num_speculative_tokens=2,
                parallel_drafting=parallel_drafting,
            )
        vllm_config = VllmConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_active_seqs=max_num_active_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=2048,
                enable_chunked_prefill=True,
                long_prefill_token_threshold=long_prefill_token_threshold,
                is_encoder_decoder=False,
                async_scheduling=True,
                prefill_schedule_interval=4,
            ),
            model_config=model_config,
            cache_config=CacheConfig(
                block_size=BLOCK_SIZE,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
                enable_prefix_caching=True,
            ),
            parallel_config=parallel_config,
            speculative_config=speculative_config,
            device_config=DeviceConfig(device="cpu"),
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer0"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=64,
                        dtype=torch.float16,
                    ),
                )
            ],
        )
        vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        register_all_kvcache_specs(vllm_config)
        return DpPrefillGateScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK_SIZE,
        )

    return make


def _request(
    request_id: str, num_tokens: int, seed: int, prefix: tuple[int, ...] = ()
) -> Request:
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = list(prefix) + [
        seed * 100_000 + i for i in range(num_tokens - len(prefix))
    ]
    return Request(
        request_id,
        prompt,
        params,
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _running(request_id: str, prefilling: bool, seed: int) -> Request:
    request = _request(request_id, 16, seed)
    request.status = RequestStatus.RUNNING
    request.is_prefill_chunk = prefilling
    return request


def _fake_step(
    scheduler: DpPrefillGateScheduler,
    throttle: bool,
    tokens: dict[str, int],
    upstream_bound: bool | None = None,
):
    """Run schedule() with upstream replaced by a fake that schedules ``tokens``
    and, like upstream on a step it did not hold back, may rewrite
    prefill_capacity_bound."""
    output = SimpleNamespace(
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
    )

    def fake(*args, **kwargs):
        if upstream_bound is not None:
            scheduler.prefill_capacity_bound = upstream_bound
        return output

    with patch.object(SrptPrefillScheduler, "schedule", side_effect=fake) as upstream:
        assert scheduler.schedule(throttle_prefills=throttle) is output
    upstream.assert_called_once_with(throttle)


def _starved(scheduler: DpPrefillGateScheduler) -> dict[str, int]:
    return {"p0": scheduler.max_num_scheduled_tokens}


def test_inherits_prefix_aware_warmest_first_admission(make_scheduler):
    scheduler = make_scheduler(max_num_seqs=2)
    assert isinstance(scheduler, SrptPrefillScheduler)
    assert isinstance(scheduler, PrefixAwareScheduler)
    warm_prefix = tuple(range(48))
    scheduler.add_request(_request("a", 64, seed=1, prefix=warm_prefix))
    scheduler.schedule()
    # "cold" arrives first, but "warm" reuses a's cached 48-token prefix.
    scheduler.add_request(_request("cold", 64, seed=2))
    scheduler.add_request(_request("warm", 64, seed=3, prefix=warm_prefix))

    output = scheduler.schedule()

    assert "warm" in output.num_scheduled_tokens
    assert "cold" not in output.num_scheduled_tokens


def test_srpt_hold_composes_with_dp_hold_back(make_scheduler):
    """SRPT never holds on a throttled step, and a chunk it holds on a cadence
    step is restored and not counted as prefill."""
    scheduler = make_scheduler(max_num_seqs=4, max_num_batched_tokens=16)
    long = _request("long", 64, seed=1)
    scheduler.add_request(long)
    assert scheduler.schedule(throttle_prefills=False).num_scheduled_tokens == {
        "long": 16
    }
    scheduler.add_request(_request("short", 16, seed=2))

    # Throttled: no decoder, so upstream lets the chunk run and SRPT does not
    # hold it; the budget is gone before "short" is reached.
    held_back = scheduler.schedule(throttle_prefills=True)
    assert held_back.num_scheduled_tokens == {"long": 16}
    # Cadence: SRPT holds "long" (32 left) to admit "short" (16).
    cadence = scheduler.schedule(throttle_prefills=False)
    assert cadence.num_scheduled_tokens == {"short": 16}

    assert long in scheduler.running
    assert scheduler.max_num_running_reqs == 4
    stats = scheduler._gate_stats
    assert stats["throttled_prefill_no_decoder"] == 1
    assert stats["throttled_prefill_tokens"] == 16
    assert stats["cadence_prefill_tokens"] == 32


def test_gate_off_passes_through(make_scheduler):
    scheduler = make_scheduler(min_cadences=0, log_steps=0)
    scheduler.add_request(_request("w", 16, seed=1))

    _fake_step(scheduler, throttle=False, tokens={}, upstream_bound=True)

    assert scheduler.prefill_capacity_bound is True
    assert not scheduler._gate_stats


def test_stats_only_keeps_upstream_flag(make_scheduler):
    scheduler = make_scheduler(min_cadences=0)
    scheduler.running = [_running("d0", prefilling=False, seed=1)]
    scheduler.add_request(_request("w", 16, seed=2))

    _fake_step(scheduler, throttle=False, tokens={"d0": 1}, upstream_bound=True)

    assert scheduler.prefill_capacity_bound is True
    assert scheduler._gate_stats["cadence_waiting_other"] == 1


def test_throttled_prefill_attributed_to_escape_reason(make_scheduler):
    scheduler = make_scheduler()
    decoder = _running("d0", prefilling=False, seed=1)
    chunk = _running("p0", prefilling=True, seed=2)

    scheduler.running = [decoder, chunk]
    scheduler.prefill_capacity_bound = True
    _fake_step(scheduler, throttle=True, tokens={"p0": 32, "d0": 1})
    scheduler.running = [chunk]
    scheduler.prefill_capacity_bound = False
    _fake_step(scheduler, throttle=True, tokens={"p0": 32})
    # Held back: only the decode ran.
    scheduler.running = [decoder, chunk]
    _fake_step(scheduler, throttle=True, tokens={"d0": 1})

    stats = scheduler._gate_stats
    assert stats["throttled"] == 3
    assert stats["throttled_prefill_bound"] == 1
    assert stats["throttled_prefill_no_decoder"] == 1
    assert stats["throttled_prefill_other"] == 0
    assert stats["throttled_prefill_tokens"] == 64


def test_prefill_tokens_exclude_spec_verify_and_padded_admission():
    output = SimpleNamespace(
        num_scheduled_tokens={"verify": 7, "chunk": 32, "padded": 7, "new": 20},
        scheduled_spec_decode_tokens={"verify": [-1] * 6, "padded": [-1] * 6},
    )
    # "padded" and "new" were admitted this step from a waiting queue.
    prefilling_before = {"verify": False, "chunk": True}

    assert DpPrefillGateScheduler._prefill_tokens(output, prefilling_before) == 52


def test_bound_needs_consecutive_starved_cadences(make_scheduler):
    scheduler = make_scheduler(min_cadences=2)
    scheduler.add_request(_request("w", 16, seed=1))

    _fake_step(
        scheduler, throttle=False, tokens=_starved(scheduler), upstream_bound=True
    )
    assert scheduler.prefill_capacity_bound is False
    # A held-back step in between neither advances nor resets the streak.
    _fake_step(scheduler, throttle=True, tokens={})
    assert scheduler.prefill_capacity_bound is False
    _fake_step(
        scheduler, throttle=False, tokens=_starved(scheduler), upstream_bound=True
    )
    assert scheduler.prefill_capacity_bound is True
    # A cadence step with budget to spare resets the streak.
    _fake_step(scheduler, throttle=False, tokens={"p0": 1}, upstream_bound=True)
    assert scheduler.prefill_capacity_bound is False
    assert scheduler._gate_stats["cadence_waiting_budget"] == 2


def test_escape_stops_once_budget_no_longer_exhausted(make_scheduler):
    scheduler = make_scheduler(min_cadences=1)
    scheduler.add_request(_request("w", 16, seed=1))
    _fake_step(
        scheduler, throttle=False, tokens=_starved(scheduler), upstream_bound=True
    )
    assert scheduler.prefill_capacity_bound is True

    _fake_step(
        scheduler, throttle=True, tokens=_starved(scheduler), upstream_bound=True
    )
    assert scheduler.prefill_capacity_bound is True

    scheduler.finish_requests("w", RequestStatus.FINISHED_ABORTED)
    _fake_step(scheduler, throttle=True, tokens={"p0": 8}, upstream_bound=False)
    assert scheduler.prefill_capacity_bound is False


def test_logs_per_rank_counts_and_resets(make_scheduler, monkeypatch):
    scheduler = make_scheduler(log_steps=2)
    scheduler.running = [_running("d0", prefilling=False, seed=1)]
    logger = MagicMock()
    monkeypatch.setattr(dp_prefill_gate, "logger", logger)

    _fake_step(scheduler, throttle=True, tokens={"d0": 1})
    logger.info.assert_not_called()
    _fake_step(scheduler, throttle=False, tokens={"d0": 1})

    fmt, dp_rank, min_cadences, steps, throttled = logger.info.call_args.args[:5]
    assert fmt.startswith("DP prefill gate: dp_rank=%d")
    assert (dp_rank, min_cadences, steps, throttled) == (0, 0, 2, 1)
    assert not scheduler._gate_stats


def test_admissible_waiting_skips_blocked_and_counts_kv_holding(make_scheduler):
    scheduler = make_scheduler()
    remote = _request("remote", 32, seed=1)
    remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.requests["remote"] = remote
    scheduler._enqueue_waiting_request(remote)
    grammar = _request("grammar", 32, seed=2)
    grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.waiting.add_request(grammar)
    assert len(scheduler.kv_holding_waiting) == 1
    assert not scheduler._has_admissible_waiting()

    preempted = _request("preempted", 32, seed=3)
    preempted.status = RequestStatus.PREEMPTED
    preempted.num_computed_tokens = 16
    scheduler._enqueue_waiting_request(preempted)

    assert len(scheduler.kv_holding_waiting) == 2
    assert scheduler._has_admissible_waiting()


@pytest.mark.parametrize("parallel_drafting", [False, True])
def test_real_spec_verify_steps_are_not_prefill(make_scheduler, parallel_drafting):
    scheduler = make_scheduler(parallel_drafting=parallel_drafting)
    for i in range(2):
        scheduler.add_request(_request(f"d{i}", 16, seed=i))

    scheduler.schedule(throttle_prefills=False)
    for _ in range(2):
        output = scheduler.schedule(throttle_prefills=True)
        assert output.num_scheduled_tokens == {"d0": 3, "d1": 3}

    stats = scheduler._gate_stats
    assert stats["throttled"] == 2
    assert stats["cadence_prefill_tokens"] == 32
    for reason in ("bound", "no_decoder", "other"):
        assert stats[f"throttled_prefill_{reason}"] == 0


def test_real_budget_exhausted_reserves_draft_slots(make_scheduler):
    scheduler = make_scheduler(min_cadences=1, parallel_drafting=True)
    for i in range(2):
        scheduler.add_request(_request(f"long{i}", 200, seed=i))

    output = scheduler.schedule(throttle_prefills=False)

    # 64 batched tokens minus 2 drafting slots for the one admitted request.
    assert output.total_num_scheduled_tokens == 62
    assert output.total_num_scheduled_tokens < scheduler.max_num_scheduled_tokens
    assert scheduler.prefill_capacity_bound is True
    assert scheduler._gate_stats["cadence_waiting_budget"] == 1


@pytest.mark.parametrize("min_cadences, escapes", [(0, 1), (1, 0)])
def test_real_queue_held_by_max_active_seqs_does_not_escape(
    make_scheduler, min_cadences, escapes
):
    scheduler = make_scheduler(
        min_cadences=min_cadences,
        max_num_active_seqs=2,
        max_num_batched_tokens=128,
        long_prefill_token_threshold=32,
    )
    for request_id, num_tokens in (("dec", 16), ("long", 200), ("w", 16)):
        scheduler.add_request(_request(request_id, num_tokens, seed=len(request_id)))

    first = scheduler.schedule(throttle_prefills=False)
    assert first.num_scheduled_tokens == {"dec": 16, "long": 32}
    second = scheduler.schedule(throttle_prefills=True)

    assert ("long" in second.num_scheduled_tokens) == bool(escapes)
    stats = scheduler._gate_stats
    assert stats["cadence_waiting_seqs"] == 1
    assert stats["throttled_prefill_bound"] == escapes
    assert stats["throttled_prefill_tokens"] == 32 * escapes


def test_env_vars_reach_ray_actors_and_skip_compile_cache_key(monkeypatch):
    for name in GATE_ENVS:
        monkeypatch.setenv(name, "3")
        assert name in get_env_vars_to_copy()
        assert name in _TPU_COMPILE_ENV_IGNORED
    assert envs.VLLM_TPU_PREFILL_GATE_MIN_CADENCES == 3

    monkeypatch.setenv("VLLM_TPU_PREFILL_GATE_MIN_CADENCES", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = envs.VLLM_TPU_PREFILL_GATE_MIN_CADENCES
