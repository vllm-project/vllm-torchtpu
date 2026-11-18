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
"""TPU test for Omni Stage-Output Prefix Caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheStep,
)
from vllm_omni.core.prefix_cache.group_view import stage_prefix_cache_config
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.core.prefix_cache.runner_mixin import PrefixCacheRunnerMixin
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import (
    _resolve_worker_cls,
    validate_worker_omni_connector,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_HIDDEN_DIM = 8
_NUM_BLOCKS = 8


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="TPUWorker has no omni model runner (vllm-torchtpu#2065)",
)
def test_omni_stage_output_prefix_caching(block_size: int) -> None:
    """Omni Stage-Output Prefix Caching

    - TPU path: resolve ``current_omni_platform.get_omni_ar_worker_cls()``, validate
      Omni connector runner support, stage ``PrefixCacheConfig``, and run a 2-step
      prefill + prefix-hit lifecycle on ``"tpu"`` hidden-state and mm tensors.
    - Setup check: ``issubclass(worker_cls.model_runner_cls, PrefixCacheRunnerMixin)``
      and ``isinstance(runner.omni_prefix_cache, OmniPrefixCacheManager)`` with the
      configured ``block_size``.
    - Outcome check: step 2 materializes cached prefix slots from host pools and
      matches concatenated step-1 prefix + step-2 tail CPU reference tensors.
    """
    worker_cls_path = current_omni_platform.get_omni_ar_worker_cls()
    validate_worker_omni_connector(worker_cls_path, required=True)
    worker_cls = _resolve_worker_cls(worker_cls_path)
    assert issubclass(worker_cls.model_runner_cls, PrefixCacheRunnerMixin), (
        f"{worker_cls.__name__}.model_runner_cls does not inherit "
        "PrefixCacheRunnerMixin"
    )

    kv_group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=_HIDDEN_DIM,
            dtype=torch.float32,
        )
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=_NUM_BLOCKS,
        kv_cache_groups=[kv_group],
    )
    cache_config = SimpleNamespace(
        enable_prefix_caching=True,
        block_size=block_size,
        prefix_match_unit=None,
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=block_size * 4)
    model_config = SimpleNamespace(
        max_model_len=block_size * 4,
        get_hidden_size=lambda: _HIDDEN_DIM,
        dtype=torch.float32,
    )
    staged_cfg = stage_prefix_cache_config(
        kv_cache_config=kv_cache_config,
        cache_config=cache_config,
        kv_transfer_config=None,
        scheduler_config=scheduler_config,
        model_config=model_config,
        is_pooling_model=False,
    )
    assert staged_cfg is not None, "stage_prefix_cache_config returned None"
    assert staged_cfg.block_size == block_size, (
        f"staged block_size {staged_cfg.block_size} != {block_size}"
    )

    runner = object.__new__(worker_cls.model_runner_cls)
    runner.is_pooling_model = False
    runner._omni_prefix_cache_cfg = staged_cfg
    runner.kv_cache_config = kv_cache_config
    block_table = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(block_tables=[block_table]),
        req_ids=["req-1", "req-2"],
    )
    runner._snapshot_prefix_cache_model_policy(SimpleNamespace())

    torch.manual_seed(0)
    prefix_ref = torch.randn(block_size, _HIDDEN_DIM, dtype=torch.float32)
    tail_ref = torch.randn(block_size, _HIDDEN_DIM, dtype=torch.float32)
    tpu_prefix = prefix_ref.to(device=torch.device("tpu"))
    tpu_tail = tail_ref.to(device=torch.device("tpu"))
    tpu_mm = {"mm_feat": torch.ones(block_size, 4, device=torch.device("tpu"))}

    step1 = PrefixCacheStep(
        events=(
            PrefixCacheRequestEvent(
                req_id="req-1",
                kind=PrefixCacheEventKind.STARTED,
                hit_end=0,
                block_ids=((0,),),
                scheduled_tokens=block_size,
            ),
        ),
        scheduled_tokens=(("req-1", block_size),),
    )
    runner._ensure_omni_prefix_cache()
    assert isinstance(runner.omni_prefix_cache, OmniPrefixCacheManager), (
        "omni_prefix_cache was not initialized as OmniPrefixCacheManager"
    )
    runner.omni_prefix_cache.new_step_starts(step1)
    runner._prefix_cache_step = step1
    step1_id = runner._prefix_cache_save_step(
        tpu_prefix,
        tpu_mm,
        num_tokens_unpadded=block_size,
        num_tokens_padded=block_size,
    )
    runner._prefix_cache_materialize(step1_id, ["req-1"])

    step2 = PrefixCacheStep(
        events=(
            PrefixCacheRequestEvent(
                req_id="req-2",
                kind=PrefixCacheEventKind.STARTED,
                hit_end=block_size,
                block_ids=((0, 2),),
                scheduled_tokens=block_size,
            ),
        ),
        scheduled_tokens=(("req-2", block_size),),
    )
    runner.omni_prefix_cache.new_step_starts(step2)
    runner._prefix_cache_step = step2
    step2_id = runner._prefix_cache_save_step(
        tpu_tail,
        tpu_mm,
        num_tokens_unpadded=block_size,
        num_tokens_padded=block_size,
    )
    hidden_out, _ = runner._prefix_cache_materialize(step2_id, ["req-2"])
    assert hidden_out is not None, "step 2 materialize returned None hidden_states"
    merged = hidden_out["req-2"]
    expected = torch.cat([prefix_ref, tail_ref], dim=0)
    assert merged.shape == expected.shape, (
        f"merged shape {tuple(merged.shape)} != {tuple(expected.shape)}"
    )
    assert_parity(
        merged,
        expected,
        name=f"prefix_cache_bs{block_size}",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
