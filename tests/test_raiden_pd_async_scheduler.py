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
"""Raiden P send completion installs on vLLM's official scheduler entrypoints."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.core.raiden_pd import (
    _ProducerCompletion,
    patch_scheduler_init,
)
from vllm_torchtpu.distributed import utils as dist_utils

BLOCK_SIZE = 16


@pytest.fixture(scope="module")
def shared_model_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("opt_init_config")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 1000,
            }
        )
    )
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        seed=42,
        max_model_len=2048,
    )


@pytest.mark.parametrize(
    ("async_scheduling", "expected"),
    [
        (False, Scheduler),
        (True, AsyncScheduler),
    ],
)
def test_default_entrypoint_selects_the_official_scheduler(async_scheduling, expected):
    config = SchedulerConfig(
        async_scheduling=async_scheduling, max_model_len=32, is_encoder_decoder=False
    )
    assert config.scheduler_cls is None
    assert config.get_scheduler_cls() is expected


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
@pytest.mark.parametrize("send_completion", [False, True])
def test_init_patch_installs_producer_completion_on_real_schedulers(
    shared_model_config, scheduler_cls, send_completion, monkeypatch
):
    tpu_envs.__dict__.pop("TPU_KV_RESHARD_TRANSPORT", None)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    patch_scheduler_init()

    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=scheduler_cls is AsyncScheduler,
        ),
        model_config=shared_model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPURaidenConnector",
            kv_role="kv_producer",
        ),
        additional_config={"raiden_pd_send_completion": send_completion},
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = 8
    register_all_kvcache_specs(vllm_config)

    with patch(
        "vllm.v1.core.sched.scheduler.KVConnectorFactory.create_connector",
        return_value=SimpleNamespace(
            requires_kv_delivery=False, bind_kv_cache_manager=lambda _mgr: None
        ),
    ):
        scheduler = scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK_SIZE,
        )

    assert type(scheduler) is scheduler_cls
    if send_completion:
        assert isinstance(scheduler.update_from_output.__self__, _ProducerCompletion)
    else:
        assert scheduler.update_from_output.__self__ is scheduler
    if scheduler_cls is AsyncScheduler:
        assert (
            scheduler._update_after_schedule.__func__
            is AsyncScheduler._update_after_schedule
        )
        assert (
            scheduler._update_request_with_output.__func__
            is AsyncScheduler._update_request_with_output
        )


def test_decode_inline_load_with_fail_policy_initializes_without_error(
    shared_model_config, monkeypatch
):
    tpu_envs.__dict__.pop("TPU_KV_RESHARD_TRANSPORT", None)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(dist_utils, "get_raiden_inline_load", lambda: True)
    patch_scheduler_init()

    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=shared_model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPURaidenConnector",
            kv_role="kv_consumer",
            kv_load_failure_policy="fail",
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = 8
    register_all_kvcache_specs(vllm_config)

    with patch(
        "vllm.v1.core.sched.scheduler.KVConnectorFactory.create_connector",
        return_value=SimpleNamespace(
            requires_kv_delivery=False, bind_kv_cache_manager=lambda _mgr: None
        ),
    ):
        scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK_SIZE,
        )
    assert scheduler.recompute_kv_load_failures is False
    assert not isinstance(scheduler.update_from_output.__self__, _ProducerCompletion)
