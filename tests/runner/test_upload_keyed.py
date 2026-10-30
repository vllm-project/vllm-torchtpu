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

from types import SimpleNamespace

import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_upload_keyed_reuses_until_values_change():
    runner = SimpleNamespace(_h2d_reuse={}, device="cpu")
    upload = TPUModelRunner._upload_keyed
    temps = torch.tensor([[0.7], [1.0]], dtype=torch.bfloat16)
    first = upload(runner, "t", temps, temps.float().numpy().tobytes())
    again = upload(runner, "t", temps.clone(), temps.float().numpy().tobytes())
    assert again is first
    changed = torch.tensor([[0.5], [1.0]], dtype=torch.bfloat16)
    third = upload(runner, "t", changed, changed.float().numpy().tobytes())
    assert third is not first and torch.equal(third, changed)


def test_padded_sampling_params_reuse_flag(monkeypatch):
    input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.tensor([0.7, 1.0, 0.9], dtype=torch.float32),
        top_k_cpu_tensor=torch.tensor([20, 1024, 5], dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor([0.95, 1.0, 0.8], dtype=torch.float32),
    )
    runner = SimpleNamespace(
        _h2d_reuse={},
        device=torch.device("cpu"),
        vocab_size=512,
        input_batch=input_batch,
    )
    for name in (
        "_upload_keyed",
        "_upload_unless_unchanged",
        "_padded_sampling_params",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))

    monkeypatch.setenv("VLLM_TPU_SAMPLING_PARAMS_REUSE", "1")
    t1, k1, p1 = runner._padded_sampling_params(
        0, 2, 4, torch.bfloat16, torch.device("cpu")
    )
    t2, k2, p2 = runner._padded_sampling_params(
        0, 2, 4, torch.bfloat16, torch.device("cpu")
    )
    assert t2 is t1
    assert k2 is k1
    assert p2 is p1
    # top_k >= vocab_size clamps to 0; inactive slots pad with (1.0, 0, 1.0)
    assert k1.squeeze(1).tolist() == [20, 0, 0, 0]

    input_batch.temperature_cpu_tensor[0] = 0.3
    t3, k3, p3 = runner._padded_sampling_params(
        0, 2, 4, torch.bfloat16, torch.device("cpu")
    )
    assert t3 is not t1
    assert k3 is k1
    assert p3 is p1

    monkeypatch.setenv("VLLM_TPU_SAMPLING_PARAMS_REUSE", "0")
    t4, k4, p4 = runner._padded_sampling_params(
        0, 2, 4, torch.bfloat16, torch.device("cpu")
    )
    assert t4 is not t3
    assert torch.equal(t4, t3)
