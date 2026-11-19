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
"""PrefillSpacingScheduler DP-aligned spacing (VLLM_TPU_PREFILL_SPACING_DP_ALIGN)."""

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
    scheduler.add_request(_make_real_request("short", 16, hasher, seed=2))
    scheduler.add_request(_make_real_request("big", 48, hasher, seed=3))
    return scheduler


def test_dp_align_off_without_dp(shared_model_config, monkeypatch):
    scheduler = _decoding_scheduler(
        shared_model_config, monkeypatch, VLLM_TPU_PREFILL_SPACING_DP_ALIGN=1
    )
    assert scheduler.dp_align is False


@pytest.mark.parametrize(
    ("throttle", "debt_s", "expected"),
    [
        # A throttled (off-cadence) step is decode-only on every engine.
        (True, 0.0, {"dec"}),
        # An on-cadence step prefills even with this engine's own debt.
        (False, 0.05, {"dec", "short", "big"}),
    ],
)
def test_dp_align_follows_cadence_not_debt(
    shared_model_config, monkeypatch, throttle, debt_s, expected
):
    scheduler = _decoding_scheduler(
        shared_model_config,
        monkeypatch,
        VLLM_TPU_PREFILL_SPACING_FRAC=0.5,
        VLLM_TPU_PREFILL_TPOT_TARGET_MS=0,
    )
    scheduler.dp_align = True  # as with DATA_PARALLEL_SIZE > 1 and the flag
    scheduler._step_s = {_DECODE: 0.01, _LONG: 0.11}
    scheduler._debt_s = debt_s
    out = scheduler.schedule(throttle)
    assert set(out.num_scheduled_tokens) == expected
