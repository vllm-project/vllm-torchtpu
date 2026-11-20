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


class _Runner:
    def __init__(self, vllm_config, device, *, profiler_rank, profiler_world_size):
        self.args = (vllm_config, device, profiler_rank, profiler_world_size)


class _OwnedRunner(_Runner):
    pass


def _worker(architecture):
    from types import SimpleNamespace as NS

    return NS(
        model_config=NS(architecture=architecture),
        model_runner_cls=_Runner,
        vllm_config="cfg",
        devices=["dev0", "dev1"],
        profile_rank=3,
        profile_world_size=8,
    )


def test_worker_builds_the_default_runner_without_registration():
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    runner = TPUWorker._build_model_runner(_worker(ARCH))
    assert type(runner) is _Runner
    assert runner.args == ("cfg", "dev0", 3, 8)


def test_worker_builds_the_registered_runner_for_its_architecture():
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    register_model_runner(_OwnedRunner, ARCH)
    runner = TPUWorker._build_model_runner(_worker(ARCH))
    assert type(runner) is _OwnedRunner
    assert runner.args == ("cfg", "dev0", 3, 8)
    other = TPUWorker._build_model_runner(_worker("OtherForCausalLM"))
    assert type(other) is _Runner
