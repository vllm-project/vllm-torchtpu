# SPDX-License-Identifier: Apache-2.0
"""`_real_traffic` decides from the runner's host-side counts, never the device tensor."""

import types

import numpy as np
import torch

from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv




def test_host_counts_win_over_the_device_tensor():
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(num_computed_tokens_cpu=np.array([10, 900, 5, 7]))
    )
    ctx = torch.tensor([1, 2, 3])  # would say "warmup" if read
    assert drv._real_traffic({"context_lens": ctx}, runner, 2)
    assert not drv._real_traffic({"context_lens": torch.tensor([999])}, runner, 1)


def test_falls_back_to_context_lens_without_a_runner():
    assert drv._real_traffic({"context_lens": torch.tensor([3, 70])})
    assert not drv._real_traffic({"context_lens": torch.tensor([3, 7])})
    assert drv._real_traffic({})
