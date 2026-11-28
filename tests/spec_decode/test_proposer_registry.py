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

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode import proposer_registry as reg
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def _config(arch, architectures=None):
    """`architecture` is the one vLLM resolved from `architectures`."""
    spec = SimpleNamespace(
        method="mtp",
        use_eagle=lambda: True,
        draft_model_config=SimpleNamespace(
            architecture=arch, architectures=architectures or [arch]
        ),
    )
    return SimpleNamespace(speculative_config=spec, model_config=None)


class _ModelOwnedProposer(Eagle3Proposer):
    def __init__(self, runner, vllm_config):
        self.args = (runner, vllm_config)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(reg, "_FACTORIES", {})


def test_empty_registry_selects_nothing():
    assert reg.select_draft_proposer(_config("SomeDraftArch")) is None


def test_selects_by_draft_architecture():
    def factory(runner, vllm_config):
        return _ModelOwnedProposer(runner, vllm_config)

    reg.register_draft_proposer(["DraftA", "DraftB"], factory)
    assert reg.select_draft_proposer(_config("DraftB")) is factory
    assert reg.select_draft_proposer(_config("DraftA", ["Other", "DraftA"])) is factory
    assert reg.select_draft_proposer(_config("Other")) is None


def test_selects_only_the_resolved_draft_architecture():
    """A listed but unresolved architecture does not select its proposer: the
    proposer must match the draft model vLLM loads."""
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    assert reg.select_draft_proposer(_config("Other", ["Other", "DraftA"])) is None


@pytest.mark.parametrize(
    "architectures, match",
    [("DraftA", "not the string 'DraftA'"), ([], "at least one"), ((), "at least one")],
)
def test_register_rejects_a_string_or_no_architectures(architectures, match):
    with pytest.raises(ValueError, match=match):
        reg.register_draft_proposer(architectures, _ModelOwnedProposer)
    assert reg._FACTORIES == {}


def test_register_accepts_any_iterable_of_names():
    reg.register_draft_proposer(iter(["DraftA", "DraftB"]), _ModelOwnedProposer)
    assert set(reg._FACTORIES) == {"DraftA", "DraftB"}


def test_missing_draft_model_config_fails_loudly():
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    cfg = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=None))
    with pytest.raises(AssertionError, match="draft model config"):
        reg.select_draft_proposer(cfg)


def test_later_registration_replaces():
    def first(runner, vllm_config):
        return _ModelOwnedProposer(runner, vllm_config)

    reg.register_draft_proposer(["DraftA"], first)
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    assert reg.select_draft_proposer(_config("DraftA")) is _ModelOwnedProposer


def _init_speculative_decoding(monkeypatch, cfg):
    monkeypatch.setattr(tpu_runner, "RejectionSampler", lambda *args: None)
    runner = SimpleNamespace(
        speculative_config=cfg.speculative_config,
        _pp_is_last=True,
        vllm_config=cfg,
        device="cpu",
    )
    TPUModelRunner._init_speculative_decoding(runner)
    return runner


def test_runner_builds_the_registered_proposer(monkeypatch):
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    cfg = _config("DraftA")
    runner = _init_speculative_decoding(monkeypatch, cfg)
    assert isinstance(runner.drafter, _ModelOwnedProposer)
    assert runner.drafter.args == (runner, cfg)


def test_registered_proposer_log_names_its_module(monkeypatch):
    logged = []
    monkeypatch.setattr(reg.logger, "info", lambda msg, *args: logged.append(args))
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    _init_speculative_decoding(monkeypatch, _config("DraftA"))
    assert (__name__, "_ModelOwnedProposer") in logged


def test_runner_builds_the_shared_proposer_by_default(monkeypatch):
    built = []
    monkeypatch.setattr(
        reg, "Eagle3Proposer", lambda runner, cfg: built.append(runner) or "shared"
    )
    runner = _init_speculative_decoding(monkeypatch, _config("DraftA"))
    assert runner.drafter == "shared" and built == [runner]


def test_runner_rejects_a_factory_that_does_not_return_an_eagle3_proposer(
    monkeypatch,
):
    def factory(runner, vllm_config):
        return object()

    reg.register_draft_proposer(["DraftA"], factory)
    with pytest.raises(TypeError, match="does not subclass Eagle3Proposer"):
        _init_speculative_decoding(monkeypatch, _config("DraftA"))


def test_a_different_factory_for_the_same_architecture_warns(monkeypatch):
    warned = []
    monkeypatch.setattr(
        reg.logger, "warning", lambda msg, *args: warned.append(args[0])
    )
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)
    reg.register_draft_proposer(["DraftA"], _ModelOwnedProposer)  # same: quiet
    reg.register_draft_proposer(["DraftA"], lambda runner, cfg: None)
    assert warned == ["DraftA"]
