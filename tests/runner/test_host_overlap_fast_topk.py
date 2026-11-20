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
"""Runner pieces behind VLLM_TPU_FAST_TOPK_TOPP and VLLM_TPU_ASYNC_HOST_OVERLAP."""

from types import SimpleNamespace

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(top_k, temps):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 248320
    runner.input_batch = SimpleNamespace(
        top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int64),
        temperature_cpu_tensor=torch.tensor(temps, dtype=torch.float32),
    )
    return runner


def test_fast_topk_cap_off_by_default(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_FAST_TOPK_TOPP", False)
    assert _runner([20, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 0


def test_fast_topk_cap_gating(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_FAST_TOPK_TOPP", True)
    monkeypatch.setattr(envs, "VLLM_TPU_FAST_TOPK_TOPP_MAX_K", 32)
    # Qwen generation_config: top_k = 20 on every sampled row.
    assert _runner([20, 20], [1.0, 0.6])._sampler_fast_topk_cap(0, 2, None) == 32
    # A grammar mask, an empty chunk, or an all-greedy chunk keep the stock path.
    assert _runner([20, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, object()) == 0
    assert _runner([20, 20], [1.0, 1.0])._sampler_fast_topk_cap(1, 1, None) == 0
    assert _runner([20, 20], [0.0, 0.0])._sampler_fast_topk_cap(0, 2, None) == 0
    # Greedy rows do not constrain top_k.
    assert _runner([0, 20], [0.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 32
    # top_k disabled (0 / -1) on a sampled row needs the full vocab.
    assert _runner([0, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 0
    assert _runner([-1, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 0
    # Boundary-tie headroom: top_k <= k_cap - k_cap // 4 = 24 at k_cap = 32.
    assert _runner([24, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 32
    assert _runner([25, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 0
    assert _runner([32, 20], [1.0, 1.0])._sampler_fast_topk_cap(0, 2, None) == 0


def _dummy_runner():
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.uses_mrope = False
    runner._has_mamba_state = False
    runner.mamba_slot_read_offsets = None
    runner._dp_vocab_logits = False
    return runner


def test_dp_lockstep_dummy_tensors_values_and_reuse():
    runner = _dummy_runner()
    (
        input_ids,
        position_ids,
        query_start_loc,
        seq_lens,
        request_distribution,
        mamba_state_indices,
        mamba_request_distribution,
        logits_idx,
    ) = first = runner._get_dp_lockstep_dummy_tensors(8, 4)
    torch.testing.assert_close(input_ids, torch.zeros(8, dtype=torch.int32))
    torch.testing.assert_close(position_ids, torch.zeros(8, dtype=torch.int32))
    torch.testing.assert_close(
        query_start_loc, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(seq_lens, torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(
        request_distribution, torch.tensor([4, 4, 4], dtype=torch.int32)
    )
    assert mamba_state_indices is None and mamba_request_distribution is None
    torch.testing.assert_close(logits_idx, torch.zeros(4, dtype=torch.int32))

    # Same bucket: the cached device tensors, uploaded once.
    second = runner._get_dp_lockstep_dummy_tensors(8, 4)
    assert all(a is b for a, b in zip(first, second))
    # Fewer tokens than requests: only num_tokens requests get a token.
    _, _, qsl, *_ = runner._get_dp_lockstep_dummy_tensors(2, 4)
    torch.testing.assert_close(qsl, torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32))
    assert len(runner._dp_lockstep_dummy_cache) == 2
