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
"""Per-step host temporaries are staged until an input fence covers them.

TorchTPU reads the host source of a non-blocking upload when the transfer
runs. A temporary that is freed before that reaches the device as garbage,
so the runner must own it until `_wait_input_staging_fence` has passed a
fence recorded after the upload.
"""

from types import SimpleNamespace

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _staging_runner(**fields):
    return SimpleNamespace(
        device=torch.device("cpu"),
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
        **fields,
    )


def test_sampling_params_are_staged_until_the_fence(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_SAMPLING_PARAMS_REUSE", raising=False)
    assert not envs.VLLM_TPU_SAMPLING_PARAMS_REUSE
    runner = _staging_runner(
        vocab_size=100,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.tensor([0.5, 1.0, 0.7]),
            top_k_cpu_tensor=torch.tensor([5, 200, 7], dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor([0.9, 1.0, 0.8]),
        ),
    )

    temps, top_k, top_p = TPUModelRunner._padded_sampling_params(
        runner, 1, 3, 4, torch.float32, torch.device("cpu")
    )

    assert temps.flatten().tolist() == [1.0, 0.699999988079071, 1.0, 1.0]
    assert top_k.flatten().tolist() == [0, 7, 0, 0]
    staged = [host.flatten().tolist() for host, _ in runner._unfenced_h2d_sources]
    assert staged == [
        [1.0, 0.699999988079071, 1.0, 1.0],
        [0, 7, 0, 0],
        [1.0, 0.800000011920929, 1.0, 1.0],
    ]
    assert top_p.flatten().tolist() == staged[2]
