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
"""CPU tests for the model-owned draft proposer registry."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.spec_decode import proposer_registry as reg


def _config(archs):
    draft = None if archs is None else SimpleNamespace(architectures=archs)
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(reg, "_FACTORIES", {})


def test_empty_registry_selects_nothing():
    assert reg.select_draft_proposer(_config(["SomeDraftArch"])) is None
    assert reg.select_draft_proposer(_config(None)) is None
    assert reg.select_draft_proposer(SimpleNamespace()) is None


def test_selects_by_draft_architecture():
    def factory(runner, vllm_config):
        return ("built", runner, vllm_config)

    reg.register_draft_proposer(["DraftA", "DraftB"], factory)
    assert reg.select_draft_proposer(_config(["DraftB"])) is factory
    assert reg.select_draft_proposer(_config(["Other", "DraftA"])) is factory
    assert reg.select_draft_proposer(_config(["Other"])) is None


def test_hf_config_architectures_fallback():
    def factory(runner, vllm_config):
        return None

    reg.register_draft_proposer(["DraftA"], factory)
    draft = SimpleNamespace(hf_config=SimpleNamespace(architectures=["DraftA"]))
    cfg = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))
    assert reg.select_draft_proposer(cfg) is factory


def test_later_registration_replaces_and_unregister_restores_default():
    def first(runner, vllm_config):
        return None

    def second(runner, vllm_config):
        return None

    reg.register_draft_proposer(["DraftA"], first)
    reg.register_draft_proposer(["DraftA"], second)
    assert reg.select_draft_proposer(_config(["DraftA"])) is second
    reg.unregister_draft_proposer(["DraftA"])
    assert reg.select_draft_proposer(_config(["DraftA"])) is None
