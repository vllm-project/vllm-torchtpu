# SPDX-License-Identifier: Apache-2.0
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
