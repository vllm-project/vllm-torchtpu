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
"""The decode carry runs by rule: a Qwen3.5 MTP engine with batch-1 async
decode. Every other engine keeps the uploads and never records or tags."""

import types

import pytest
import torch

from vllm_torchtpu.runner import decode_carry as dc

QWEN35 = ["Qwen3_5MoeForConditionalGeneration"]


def _runner(*, arch=QWEN35, method="mtp", max_num_seqs=1, async_scheduling=True):
    ns = types.SimpleNamespace
    return ns(
        scheduler_config=ns(
            max_num_seqs=max_num_seqs, async_scheduling=async_scheduling
        ),
        vllm_config=ns(
            model_config=ns(architectures=arch),
            speculative_config=None if method is None else ns(method=method),
        ),
        _pre_async_results=ns(),
    )


def test_on_for_a_batch1_async_qwen3_5_mtp_engine():
    assert dc.enabled(_runner()) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arch": ["Qwen3NextForCausalLM"]},
        {"method": None},
        {"method": "eagle3"},
        {"max_num_seqs": 2},
        {"async_scheduling": False},
    ],
)
def test_off_outside_the_rule(kwargs):
    assert dc.enabled(_runner(**kwargs)) is False


def test_off_never_carries_records_or_tags():
    r = _runner(max_num_seqs=8)
    assert (
        dc.check(
            r,
            start_index=0,
            num_reqs=1,
            num_scheduled_tokens_per_req_list=[4],
            total_num_scheduled_tokens=4,
            padded_total_num_scheduled_tokens=4,
            target_num_reqs=4,
            layout_plan=None,
        )
        is None
    )
    assert (
        dc.record(
            r,
            start_index=0,
            num_reqs=1,
            num_scheduled_tokens_per_req_list=[4],
            total_num_scheduled_tokens=4,
            padded_total_num_scheduled_tokens=4,
            target_num_reqs=4,
            sub_pairs=((0, 4),),
            seq_lens=torch.zeros(4),
        )
        is False
    )
    res = types.SimpleNamespace()
    dc.tag_results(r, res)
    assert not hasattr(res, "_decode_carry_seq")
