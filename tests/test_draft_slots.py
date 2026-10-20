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
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import draft_slots

K = 6
BLOCK_SIZE = 16


def spec_config(method: str, parallel_drafting: bool) -> SpeculativeConfig:
    """A real SpeculativeConfig for ``method`` without loading a drafter."""
    spec = SpeculativeConfig(
        model="ngram",
        num_speculative_tokens=K,
        prompt_lookup_max=3,
        prompt_lookup_min=1,
    )
    spec.method = method
    spec.parallel_drafting = parallel_drafting
    return spec


@pytest.fixture
def install_patch(monkeypatch):
    # Snapshot upstream's property so teardown restores it.
    monkeypatch.setattr(
        SpeculativeConfig,
        "max_num_new_slots_for_drafting",
        SpeculativeConfig.__dict__["max_num_new_slots_for_drafting"],
    )
    return draft_slots.patch_drafting_slot_budget


@pytest.mark.parametrize(
    ("method", "parallel_drafting", "upstream_slots", "tpu_slots"),
    [
        ("dflash", True, K, 0),
        ("dspark", True, K - 1, 0),
        # Methods without a TPU block drafter keep upstream's charge.
        ("eagle3", True, K - 1, K - 1),
        ("draft_model", True, K, K),
        ("draft_model", False, 1, 1),
        ("ngram", False, 0, 0),
    ],
)
def test_only_tpu_block_drafters_stop_charging_slots(
    install_patch, method, parallel_drafting, upstream_slots, tpu_slots
):
    spec = spec_config(method, parallel_drafting)
    assert spec.max_num_new_slots_for_drafting == upstream_slots
    install_patch()
    assert spec.max_num_new_slots_for_drafting == tpu_slots


def build_scheduler(max_num_batched_tokens: int) -> Scheduler:
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    num_blocks = 32
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
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
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )


@pytest.mark.parametrize(
    ("method", "expected_prefill"),
    [
        # TPU block drafter: its K - 1 drafting slots go to prefill.
        ("dspark", 40),
        # PARD has no TPU block drafter and keeps its K-slot charge.
        ("draft_model", 40 - K),
    ],
)
def test_tpu_block_drafter_slots_go_to_prefill(install_patch, method, expected_prefill):
    install_patch()
    scheduler = build_scheduler(max_num_batched_tokens=40)
    # Scheduler.schedule() reads the drafting slots from the config every step.
    scheduler.vllm_config.speculative_config = spec_config(method, True)
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    scheduler.add_request(Request("prefill", list(range(96)), params, None))

    assert scheduler.schedule().num_scheduled_tokens == {"prefill": expected_prefill}


def test_patch_installs_once(install_patch):
    install_patch()
    installed = SpeculativeConfig.__dict__["max_num_new_slots_for_drafting"]
    install_patch()
    assert SpeculativeConfig.__dict__["max_num_new_slots_for_drafting"] is installed
