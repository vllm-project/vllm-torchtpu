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
"""TPU test for Fused-MoE Prefill Warmup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import _resolve_worker_cls

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

_MAX_CUDAGRAPH_SIZE = 8
_MAX_NUM_TOKENS = 64


@pytest.mark.parametrize("warmup_flag", ["1", "0"])
@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_fused_moe_prefill_warmup(
    warmup_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fused-MoE Prefill Warmup

    - TPU path: import ``moe_prefill_warmup_enabled``, ``moe_prefill_warmup_sizes``,
      and ``warm_up_moe_prefill_kernels`` inside the test body, resolve the TPU AR
      worker, and invoke ``warm_up_moe_prefill_kernels`` across
      ``VLLM_OMNI_MOE_PREFILL_WARMUP`` in ``("1", "0")`` on a tiny MoE model.
    - Setup check: ``moe_prefill_warmup_enabled()`` reflects
      ``VLLM_OMNI_MOE_PREFILL_WARMUP`` and ``model_has_fused_moe`` detects the
      ``MoERunner`` layer.
    - Outcome check: ``0`` warms ``[]`` while ``1`` warms every prefill bucket and
      odd neighbour in ``(max_cudagraph_size, max_num_tokens]``.
    """
    from vllm.model_executor.layers.fused_moe import MoERunner
    from vllm_omni.worker.moe_prefill_warmup import (
        model_has_fused_moe,
        moe_prefill_warmup_enabled,
        moe_prefill_warmup_sizes,
        warm_up_moe_prefill_kernels,
    )

    monkeypatch.setenv("VLLM_OMNI_MOE_PREFILL_WARMUP", warmup_flag)
    expected_enabled = warmup_flag == "1"
    assert moe_prefill_warmup_enabled() is expected_enabled, (
        f"moe_prefill_warmup_enabled()={moe_prefill_warmup_enabled()} != "
        f"{expected_enabled}"
    )

    class _TinyMoeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = object.__new__(MoERunner)
            torch.nn.Module.__init__(self.moe)

    model = _TinyMoeModel().to(device=torch.device("tpu"))
    assert model_has_fused_moe(model) is True, (
        "model_has_fused_moe did not detect MoERunner layer"
    )

    worker_cls = _resolve_worker_cls(current_omni_platform.get_omni_ar_worker_cls())
    recorded_sizes: list[int] = []
    runner = object.__new__(worker_cls.model_runner_cls)
    runner.model = model
    runner.max_cudagraph_capture_size = _MAX_CUDAGRAPH_SIZE
    runner.max_num_tokens = _MAX_NUM_TOKENS
    runner._dummy_run = lambda num_tokens, **kwargs: recorded_sizes.append(num_tokens)

    worker = object.__new__(worker_cls)
    worker.model_runner = runner

    warmed = warm_up_moe_prefill_kernels(worker)
    expected_sizes = (
        moe_prefill_warmup_sizes(_MAX_CUDAGRAPH_SIZE, _MAX_NUM_TOKENS)
        if expected_enabled
        else []
    )
    assert warmed == expected_sizes, (
        f"warmed sizes {warmed} != expected {expected_sizes}"
    )
    assert recorded_sizes == expected_sizes, (
        f"recorded dummy_run sizes {recorded_sizes} != expected {expected_sizes}"
    )
    _ = SimpleNamespace()
