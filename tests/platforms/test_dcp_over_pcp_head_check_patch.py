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
"""`_patch_vllm_dcp_over_pcp_head_check`: vLLM's GQA DCP head checks assume DCP
reuses the TP ranks; DCP over the PCP axis (dcp == pcp) skips them."""

from types import SimpleNamespace

import pytest
from vllm.config.model import ModelConfig

import vllm_torchtpu


@pytest.fixture
def seen(monkeypatch):
    """The DCP size vLLM's original check sees, per call."""
    calls = []

    def original(self, parallel_config):
        calls.append(parallel_config.decode_context_parallel_size)

    monkeypatch.setattr(ModelConfig, "verify_with_parallel_config", original)
    monkeypatch.setattr(
        ModelConfig, "_tpu_dcp_over_pcp_head_check_patched", False, raising=False
    )
    vllm_torchtpu._patch_vllm_dcp_over_pcp_head_check()
    return calls


def _verify(*, use_mla=False, pcp, dcp):
    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp
    )
    ModelConfig.verify_with_parallel_config(
        SimpleNamespace(use_mla=use_mla), parallel_config
    )
    return parallel_config


def test_dcp_over_the_pcp_axis_skips_the_gqa_head_checks(seen):
    parallel_config = _verify(pcp=4, dcp=4)
    assert seen == [1]
    assert parallel_config.decode_context_parallel_size == 4


@pytest.mark.parametrize(
    ("use_mla", "pcp", "dcp"),
    [
        (False, 1, 2),  # DCP reuses the TP ranks
        (False, 4, 1),  # no DCP
        (False, 2, 8),  # DCP spans TP x PCP
        (True, 4, 4),  # MLA: vLLM skips the head checks itself
    ],
)
def test_other_layouts_keep_vllms_check(seen, use_mla, pcp, dcp):
    _verify(use_mla=use_mla, pcp=pcp, dcp=dcp)
    assert seen == [dcp]
