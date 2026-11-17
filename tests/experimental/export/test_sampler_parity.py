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
"""The exporter's sampler is a copy of the TPU runner's; the two must agree."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.experimental.export import random_sample
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner_sample(*args, **kwargs):
    runner = SimpleNamespace()
    runner._apply_temperature = TPUModelRunner._apply_temperature.__get__(runner)
    return TPUModelRunner.sample_from_logits.__wrapped__(runner, *args, **kwargs)


@pytest.mark.parametrize(
    "all_greedy,apply_topk_topp", [(False, True), (False, False), (True, True)]
)
def test_exporter_sampler_matches_the_tpu_runner(all_greedy, apply_topk_topp):
    generator = torch.Generator().manual_seed(0)
    rows, vocab = 6, 97
    logits = torch.randn(rows, vocab, generator=generator)
    # Greedy rows (0 and below SAMPLING_EPS) next to sampled ones.
    temperatures = torch.tensor([[0.0], [0.5], [1.0], [1.5], [1e-6], [0.8]])
    u = torch.rand(rows, vocab, generator=generator)
    top_k = torch.tensor([[0], [5], [0], [20], [3], [1]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [0.9], [0.5], [1.0], [0.7], [0.95]])
    kwargs = {"all_greedy": all_greedy, "apply_topk_topp": apply_topk_topp}

    expected = _runner_sample(logits, temperatures, u, top_k, top_p, **kwargs)
    actual = random_sample.sample_from_logits(
        logits, temperatures, u, top_k, top_p, **kwargs
    )
    torch.testing.assert_close(actual, expected)


def test_exporter_greedy_threshold_matches_the_tpu_runner():
    assert random_sample.SAMPLING_EPS == tpu_runner.SAMPLING_EPS
