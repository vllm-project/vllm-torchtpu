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
"""TPU tests for vllm_omni.platforms detection and OmniTpuPlatform device hooks."""

import pytest
import torch
from vllm_omni.platforms import (
    builtin_omni_platform_plugins,
    current_omni_platform,
    resolve_current_omni_platform_cls_qualname,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_EXPECTED_QUALNAME = "vllm_torchtpu.omni.platform.OmniTpuPlatform"
_BF16_BUDGET_RATIO = 2.0
_BF16_BUDGET_FLOOR = 1e-3


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "torch_tpu max_memory_reserved reports less than max_memory_allocated on TPU"
    ),
)
def test_hardware_execution_platforms() -> None:
    """Hardware Execution Platforms (In-tree Platform Support)

    - TPU path: Exercises ``vllm_omni.platforms`` detection and ``OmniTpuPlatform``
      device, autocast, synchronization, and HBM memory methods on a live TPU matmul.
    - Setup check: Verifies all five in-tree probes return ``None``, qualname resolves
      to ``OmniTpuPlatform``, and ``is_out_of_tree()`` is ``True`` while in-tree
      predicates are ``False``.
    - Outcome check: Runs a TPU bf16 matmul under ``create_autocast_context`` and
      verifies parity against CPU fp32 and valid HBM memory counter bounds.
    """
    probe_results = {
        "cuda": builtin_omni_platform_plugins["cuda"](),
        "rocm": builtin_omni_platform_plugins["rocm"](),
        "npu": builtin_omni_platform_plugins["npu"](),
        "xpu": builtin_omni_platform_plugins["xpu"](),
        "musa": builtin_omni_platform_plugins["musa"](),
    }
    assert probe_results == dict.fromkeys(probe_results), (
        f"built-in probes should all return None on TPU, got {probe_results}"
    )

    qualname = resolve_current_omni_platform_cls_qualname()
    assert qualname == _EXPECTED_QUALNAME, (
        f"resolved qualname {qualname!r} != {_EXPECTED_QUALNAME!r}"
    )
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )
    assert current_omni_platform.is_out_of_tree() is True, (
        "OmniTpuPlatform.is_out_of_tree() must be True"
    )
    in_tree_predicates = {
        "is_cuda": current_omni_platform.is_cuda(),
        "is_rocm": current_omni_platform.is_rocm(),
        "is_npu": current_omni_platform.is_npu(),
        "is_xpu": current_omni_platform.is_xpu(),
        "is_musa": current_omni_platform.is_musa(),
    }
    assert in_tree_predicates == dict.fromkeys(in_tree_predicates, False), (
        f"in-tree predicates must all be False on TPU, got {in_tree_predicates}"
    )

    torch.manual_seed(0)
    a_cpu = torch.randn(128, 128, dtype=torch.float32)
    b_cpu = torch.randn(128, 128, dtype=torch.float32)
    ref_fp32 = a_cpu @ b_cpu
    ref_bf16 = a_cpu.bfloat16() @ b_cpu.bfloat16()
    cpu_bf16_err = float(
        (ref_bf16.double() - ref_fp32.double()).norm() / ref_fp32.double().norm()
    )

    current_omni_platform.reset_peak_memory_stats()
    device = current_omni_platform.get_torch_device(0)
    a_tpu = a_cpu.to(device)
    b_tpu = b_cpu.to(device)
    input_nbytes = a_tpu.numel() * a_tpu.element_size() + (
        b_tpu.numel() * b_tpu.element_size()
    )

    with current_omni_platform.create_autocast_context(
        device_type="tpu", dtype=torch.bfloat16
    ):
        tpu_out = a_tpu @ b_tpu
    current_omni_platform.synchronize()

    max_rel_l2 = max(_BF16_BUDGET_RATIO * cpu_bf16_err, _BF16_BUDGET_FLOOR)
    assert_parity(
        tpu_out,
        ref_fp32,
        name="tpu_platform_bf16_matmul",
        max_rel_l2=max_rel_l2,
    )

    total_mem = current_omni_platform.get_device_total_memory()
    free_mem, _ = current_omni_platform.get_device_memory()
    max_allocated = current_omni_platform.max_memory_allocated()
    max_reserved = current_omni_platform.max_memory_reserved()

    assert 0 < free_mem <= total_mem, (
        f"free HBM {free_mem} must satisfy 0 < free <= total ({total_mem})"
    )
    assert max_allocated >= input_nbytes, (
        f"max_memory_allocated {max_allocated} < input_nbytes {input_nbytes}"
    )
    assert max_reserved >= max_allocated, (
        f"max_memory_reserved {max_reserved} < max_memory_allocated {max_allocated}"
    )
