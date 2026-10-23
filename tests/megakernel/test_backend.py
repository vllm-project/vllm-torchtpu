# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Selecting the megakernel program by model architecture (CPU)."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.megakernel import backend


def _vllm_config(architectures, kv_role="kv_producer", loader=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
        load_config=SimpleNamespace(
            load_format=backend.LOAD_FORMAT,
            model_loader_extra_config=loader or {"presharded": "/weights"},
        ),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=kv_role == "kv_producer",
            is_kv_consumer=kv_role == "kv_consumer",
        ),
    )


def test_each_architecture_selects_its_program_module():
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3

    for architecture in ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"):
        assert backend.program(_vllm_config([architecture])) is k3


def test_an_architecture_without_a_program_is_refused():
    with pytest.raises(ValueError, match="no megakernel program"):
        backend.role(_vllm_config(["LlamaForCausalLM"]))


def test_the_loader_config_takes_the_program_keys_of_the_role():
    k3 = ["KimiK3ForConditionalGeneration"]
    loader = {"presharded": "/weights", "draft": "/draft"}
    assert backend.role(_vllm_config(k3, loader=loader)) == "prefill"
    with pytest.raises(ValueError, match="draft"):
        backend.role(_vllm_config(k3, kv_role="kv_consumer", loader=loader))
