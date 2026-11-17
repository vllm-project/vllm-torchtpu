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
"""PrefillSpacingScheduler: selection by additional_config, its knobs, and
its ordering / spacing on a real scheduler."""

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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import prefill_spacing_scheduler as pss
from vllm_torchtpu.core.prefill_spacing_scheduler import PrefillSpacingScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.platforms.tpu_platform import _configure_prefill_spacing_scheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"
PAW = "vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler"


def _config(additional, scheduler_cls=None):
    return SimpleNamespace(
        additional_config=additional or {},
        scheduler_config=SimpleNamespace(scheduler_cls=scheduler_cls),
    )


@pytest.mark.parametrize("additional", [None, {}, {"prefill_spacing_frac": 0.5}])
@pytest.mark.parametrize("scheduler_cls", [None, PAW, "x.Other"])
def test_key_off_keeps_the_configured_scheduler(additional, scheduler_cls):
    config = _config(additional, scheduler_cls)
    _configure_prefill_spacing_scheduler(config)
    assert config.scheduler_config.scheduler_cls == scheduler_cls


@pytest.mark.parametrize("scheduler_cls", [None, PAW, pss.SCHEDULER_CLS])
def test_key_on_selects_the_scheduler(scheduler_cls):
    config = _config({"prefill_spacing_scheduler": True}, scheduler_cls)
    _configure_prefill_spacing_scheduler(config)
    assert config.scheduler_config.scheduler_cls == pss.SCHEDULER_CLS


def test_key_on_refuses_another_scheduler():
    config = _config(
        {"prefill_spacing_scheduler": True},
        "vllm_torchtpu.core.tpu_scheduler.TpuDpScheduler",
    )
    with pytest.raises(ValueError, match="prefill_spacing_scheduler"):
        _configure_prefill_spacing_scheduler(config)


def test_options_and_their_r10_defaults():
    assert pss.spacing_options(None) == (0.0, 8192, 0.0, 10.0, False)
    # The r10 c48 setting (spacing off, 22 ms target, priced holds).
    got = pss.spacing_options(
        {
            "prefill_spacing_frac": 0,
            "prefill_tpot_target_ms": 22,
            "prefill_tpot_priced": True,
        }
    )
    assert got == (0.0, 8192, 0.022, 10.0, True)
    got = pss.spacing_options(
        {"prefill_spacing_frac": 0.7, "prefill_spacing_min_tokens": 4096}
    )
    assert got == (0.7, 4096, 0.0, 10.0, False)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _scheduler(model_config, additional, *, max_num_batched_tokens, threshold=0):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=threshold,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
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
        additional_config=additional or {},
    )
    num_blocks = 64
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
    scheduler = PrefillSpacingScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(BLOCK_SIZE, sha256)


def _request(request_id, num_tokens, hasher, seed, max_tokens=64):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def test_reads_its_knobs_from_additional_config(model_config):
    scheduler, _ = _scheduler(
        model_config,
        {"prefill_spacing_frac": 0.7, "prefill_tpot_target_ms": 6.5},
        max_num_batched_tokens=32,
    )
    assert isinstance(scheduler, PrefixAwareScheduler)
    assert scheduler.spacing_frac == 0.7
    assert scheduler.long_prompt_tokens == 8192
    assert scheduler.tpot_target_s == pytest.approx(0.0065)
    assert scheduler.max_hold_s == 10.0
    assert scheduler.priced is False


def test_running_prefill_chunks_run_fewest_remaining_first(model_config):
    scheduler, hasher = _scheduler(
        model_config, None, max_num_batched_tokens=24, threshold=16
    )
    scheduler.add_request(_request("long", 64, hasher, seed=1))
    scheduler.add_request(_request("short", 48, hasher, seed=2))
    # The waiting window is sorted by uncached tokens: `short` admits first.
    assert scheduler.schedule().num_scheduled_tokens == {"short": 16, "long": 8}
    # RUNNING chunks too: `short` (32 left) ahead of `long` (56 left).
    assert scheduler.schedule().num_scheduled_tokens == {"short": 16, "long": 8}


def test_long_chunk_owes_decode_only_steps(model_config):
    scheduler, hasher = _scheduler(
        model_config,
        {"prefill_spacing_frac": 0.5, "prefill_spacing_min_tokens": 32},
        max_num_batched_tokens=32,
        threshold=16,
    )
    scheduler.add_request(_request("dec", 16, hasher, seed=1))
    assert scheduler.schedule().num_scheduled_tokens == {"dec": 16}
    # Measured step times (normally from `update_from_output`'s clock).
    scheduler._step_s = {"long": 1.0, "decode": 0.25}
    scheduler.add_request(_request("long", 128, hasher, seed=2))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"dec": 1, "long": 16}
    assert scheduler._debt_s == pytest.approx(0.5)
    # 0.5 s owed at 0.25 s per decode-only step: two steps without prefill.
    for _ in range(2):
        assert scheduler.schedule().num_scheduled_tokens == {"dec": 1}
    assert scheduler._debt_s == 0.0
    assert scheduler.schedule().num_scheduled_tokens == {"dec": 1, "long": 16}


def test_no_spacing_by_default(model_config):
    scheduler, hasher = _scheduler(
        model_config,
        {"prefill_spacing_min_tokens": 32},
        max_num_batched_tokens=32,
        threshold=16,
    )
    scheduler.add_request(_request("dec", 16, hasher, seed=1))
    scheduler.schedule()
    scheduler._step_s = {"long": 1.0, "decode": 0.25}
    scheduler.add_request(_request("long", 128, hasher, seed=2))
    for _ in range(3):
        assert scheduler.schedule().num_scheduled_tokens == {"dec": 1, "long": 16}


def test_step_times_come_from_the_output_clock(model_config, monkeypatch):
    scheduler, hasher = _scheduler(model_config, None, max_num_batched_tokens=32)
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    monkeypatch.setattr(AsyncScheduler, "update_from_output", lambda *args: {})
    clock = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr(pss, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    scheduler.add_request(_request("dec", 16, hasher, seed=1))
    first = scheduler.schedule()
    second = scheduler.schedule()
    third = scheduler.schedule()
    scheduler.update_from_output(first, None)
    # Timed only from an output that had the next step queued.
    assert scheduler._step_s == {}
    scheduler.update_from_output(second, None)
    assert scheduler._step_s == {"decode": pytest.approx(0.5)}
    scheduler.update_from_output(third, None)
    assert scheduler._step_s["decode"] == pytest.approx(0.5)
