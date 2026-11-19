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
"""The optional model-owned runner class: default unless registered."""

import pytest

from vllm_torchtpu.runner import model_runner_registry
from vllm_torchtpu.runner.model_runner_registry import (
    register_model_runner,
    select_model_runner,
)

ARCH = "ToyRunnerForCausalLM"


class _Default:
    pass


class _Owned(_Default):
    pass


class _Unrelated:
    pass


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(model_runner_registry, "_RUNNERS", {})


def test_no_registration_builds_the_default():
    assert select_model_runner(ARCH, _Default) is _Default


def test_registered_class_serves_only_its_architectures():
    register_model_runner(_Owned, ARCH, "ToyRunnerForConditionalGeneration")
    assert select_model_runner(ARCH, _Default) is _Owned
    assert select_model_runner("ToyRunnerForConditionalGeneration", _Default) is _Owned
    assert select_model_runner("OtherForCausalLM", _Default) is _Default


def test_registered_class_may_be_a_lazy_path():
    register_model_runner(f"{__name__}:_Owned", ARCH)
    assert select_model_runner(ARCH, _Default) is _Owned


def test_registered_class_must_subclass_the_default():
    register_model_runner(_Unrelated, ARCH)
    with pytest.raises(TypeError, match="does not subclass _Default"):
        select_model_runner(ARCH, _Default)


def test_a_different_class_for_the_same_architecture_warns(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        model_runner_registry.logger, "warning", lambda *a: warnings.append(a)
    )
    register_model_runner(_Owned, ARCH)
    register_model_runner(_Owned, ARCH)
    assert warnings == []
    register_model_runner(_Default, ARCH)
    assert len(warnings) == 1
    assert select_model_runner(ARCH, _Default) is _Default


def test_worker_builds_the_selected_runner_class():
    # The worker's only runner construction goes through the registry.
    import inspect

    from vllm_torchtpu.worker import tpu_worker

    source = inspect.getsource(tpu_worker.TPUWorker.init_device)
    assert "select_model_runner(" in source
    assert "self.model_runner_cls(" not in source
