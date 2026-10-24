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
"""Tests for streaming DCP decode_query_size=1 normalization and schedule routing."""

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.layers.adapter import cp_attention


def test_dcp_decode_query_size_one_normalizes_and_caches(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kwargs = dict(
        sliding_window=None,
        sm_scale=0.0625,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=4,
        cp_rank=0,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    )
    k_q1 = cp_attention.build_dcp_kernels(**kwargs, decode_query_size=1)
    k_q0 = cp_attention.build_dcp_kernels(**kwargs, decode_query_size=0)
    assert k_q1[0] is k_q0[0]
    assert k_q1[0].keywords["decode_query_size"] == 1
