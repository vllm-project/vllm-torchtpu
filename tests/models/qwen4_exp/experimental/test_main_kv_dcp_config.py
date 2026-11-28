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
"""VLLM_TPU_QSA_MAIN_KV_DCP setup checks: both ranks of a pair share a KV head."""

import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.config import check_main_kv_dcp


@pytest.mark.parametrize(
    ("tp", "kv_heads"),
    [(8, 2), (8, 4), (4, 2), (2, 1)],
)
def test_pairs_sharing_a_kv_head_pass(tp, kv_heads):
    check_main_kv_dcp(tp, kv_heads, unified=True)


@pytest.mark.parametrize(
    ("tp", "kv_heads"),
    # Every rank holds a different KV head (kv >= tp).
    [(8, 8), (4, 4), (8, 16)],
)
def test_pairs_with_different_kv_heads_raise(tp, kv_heads):
    with pytest.raises(ValueError, match=rf"tensor_parallel_size={tp}.*{kv_heads}"):
        check_main_kv_dcp(tp, kv_heads, unified=True)


@pytest.mark.parametrize(("tp", "unified"), [(8, False), (1, True), (3, True)])
def test_layout_and_tp_checks_raise(tp, unified):
    with pytest.raises(ValueError, match="unified pool"):
        check_main_kv_dcp(tp, 1, unified=unified)
