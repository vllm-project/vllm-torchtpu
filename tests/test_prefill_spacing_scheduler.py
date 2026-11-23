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
"""PrefillSpacingScheduler prefill-token caps, on a real scheduler."""

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
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.prefill_spacing_scheduler import (
    _DECODE,
    _LONG,
    PrefillSpacingScheduler,
)

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build_real_scheduler(
    model_config: ModelConfig,
    *,
    scheduler_cls=PrefillSpacingScheduler,
    num_blocks: int = 32,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    long_prefill_token_threshold: int = 0,
    scheduler_reserve_full_isl: bool = False,
    policy: str = "fcfs",
):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=scheduler_reserve_full_isl,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
            policy=policy,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
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
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return scheduler, hasher


def _make_real_request(
    request_id: str,
    num_tokens: int,
    hasher,
    seed: int = 0,
    shared_prefix: list[int] | None = None,
) -> Request:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prefix = list(shared_prefix) if shared_prefix is not None else []
    tail_len = num_tokens - len(prefix)
    prompt = prefix + [seed * 100_000 + i for i in range(tail_len)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _decoding_scheduler(model_config, monkeypatch, **env):
    """A scheduler with one decoding request `dec` and measured step times."""
    monkeypatch.setenv("VLLM_TPU_PREFILL_SPACING_MIN_TOKENS", "32")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    scheduler, hasher = _build_real_scheduler(
        model_config,
        scheduler_cls=PrefillSpacingScheduler,
        num_blocks=64,
        max_num_batched_tokens=64,
    )
    scheduler.add_request(_make_real_request("dec", 16, hasher, seed=1))
    assert set(scheduler.schedule().num_scheduled_tokens) == {"dec"}
    assert not scheduler.requests["dec"].is_prefill_chunk
    scheduler._step_s = {_DECODE: 0.01, _LONG: 0.11}
    scheduler._long_tokens = 50.0  # 2 ms per prefill token
    scheduler.add_request(_make_real_request("short", 16, hasher, seed=2))
    scheduler.add_request(_make_real_request("big", 48, hasher, seed=3))
    return scheduler


@pytest.mark.parametrize(
    ("short_tokens", "expected"),
    [
        # Default: the debt defers all prefill.
        (0, {"dec"}),
        # `short` (16 left) fits whole; `big` (48 left) waits.
        (16, {"dec", "short"}),
        (40, {"dec", "short"}),
    ],
)
def test_debt_runs_whole_short_prompts(
    shared_model_config, monkeypatch, short_tokens, expected
):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0.5,
        VLLM_TPU_PREFILL_SPACING_SHORT_TOKENS=short_tokens,
    )
    scheduler._debt_s = 0.05
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == expected
    if "short" in expected:
        assert out.num_scheduled_tokens["short"] == 16
        # The capped step pays the debt like a decode-only step.
        assert scheduler._debt_s == pytest.approx(0.04)


def test_no_debt_runs_prefill_freely(shared_model_config, monkeypatch):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0.5,
        VLLM_TPU_PREFILL_SPACING_SHORT_TOKENS=16,
    )
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"dec", "short", "big"}
    # `big` has >= 32 tokens left: a long-chunk step, charged to the debt.
    assert scheduler._debt_s == pytest.approx(0.5 * 0.11)


@pytest.mark.parametrize(
    ("priced", "slack_s", "expected"),
    [
        # Unpriced: any slack below a long step's extra time defers prefill.
        (False, 0.04, {"dec": 8}),
        # 0.04 s / 2 ms = 20 tokens: `short` whole, 4 left for `big` is under
        # one block, so `big` waits.
        (True, 0.04, {"dec": 8, "short": 16}),
        # 0.07 s = 35 tokens: `short` whole and a 19-token chunk of `big`.
        (True, 0.07, {"dec": 8, "short": 16, "big": 19}),
        # Less than one block fits: defer.
        (True, 0.02, {"dec": 8}),
    ],
)
def test_priced_tpot_cuts_prefill_to_slack(
    shared_model_config, monkeypatch, priced, slack_s, expected
):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0,
        VLLM_TPU_PREFILL_TPOT_TARGET_MS=20,
        VLLM_TPU_PREFILL_TPOT_PRICED=int(priced),
    )
    monkeypatch.setattr(scheduler, "_short_of_slack", lambda decoders, now: slack_s)
    out = scheduler.schedule()
    got = {k: v for k, v in out.num_scheduled_tokens.items()}
    assert set(got) == set(expected)
    for k in ("short", "big"):
        if k in expected:
            assert got[k] == expected[k]
    # A cut chunk of a long prompt is not a long-chunk step.
    assert scheduler._debt_s == 0.0


def test_priced_tpot_without_slack_runs_prefill(shared_model_config, monkeypatch):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0,
        VLLM_TPU_PREFILL_TPOT_TARGET_MS=20,
        VLLM_TPU_PREFILL_TPOT_PRICED=1,
    )
    monkeypatch.setattr(scheduler, "_short_of_slack", lambda decoders, now: None)
    assert set(scheduler.schedule().num_scheduled_tokens) == {"dec", "short", "big"}


@pytest.mark.parametrize(
    ("short_tokens", "slack_frac", "slack_s", "debt_s", "expected"),
    [
        # Off: the TPOT hold defers all prefill.
        (40, 0.0, 0.08, 0.0, {"dec": 8}),
        # 0.5 x 0.08 s / 2 ms = 20 tokens: `short` whole; `big` never cut.
        (40, 0.5, 0.08, 0.0, {"dec": 8, "short": 16}),
        # Only 10 tokens fit: `short` (16) does not run.
        (40, 0.5, 0.04, 0.0, {"dec": 8}),
        # Short-prompt cap below `short`: nothing runs.
        (8, 1.0, 0.08, 0.0, {"dec": 8}),
        # 1.0 x 0.2 s = 100 tokens, cap 64: still whole prompts only, so
        # `short` and `big` (48 left) both fit in 64 tokens.
        (64, 1.0, 0.2, 0.0, {"dec": 8, "short": 16, "big": 48}),
        # With the debt too: the debt's cap (40) also applies, `big` waits.
        (40, 1.0, 0.2, 0.05, {"dec": 8, "short": 16}),
    ],
)
def test_short_prompts_bypass_tpot_hold(
    shared_model_config,
    monkeypatch,
    short_tokens,
    slack_frac,
    slack_s,
    debt_s,
    expected,
):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0.5,
        VLLM_TPU_PREFILL_TPOT_TARGET_MS=20,
        VLLM_TPU_PREFILL_SPACING_SHORT_TOKENS=short_tokens,
        VLLM_TPU_PREFILL_SPACING_SHORT_TPOT_SLACK=slack_frac,
    )
    scheduler._debt_s = debt_s
    monkeypatch.setattr(scheduler, "_short_of_slack", lambda decoders, now: slack_s)
    out = scheduler.schedule()
    assert dict(out.num_scheduled_tokens) == expected
