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
"""The optional model-owned runner class: default unless registered.

The worker side (`TPUWorker.init_device`) is tested in
tests/worker/test_tpu_worker.py.
"""

import sys

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


def test_registered_class_may_be_a_lazy_path(tmp_path, monkeypatch):
    """Registration does not import the runner module; selection does."""
    module = "toy_lazy_model_runner"
    (tmp_path / f"{module}.py").write_text(
        f"from {__name__} import _Default\n\n\nclass LazyRunner(_Default):\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    register_model_runner(f"{module}:LazyRunner", ARCH)
    assert module not in sys.modules
    selected = select_model_runner(ARCH, _Default)
    assert module in sys.modules
    assert selected is sys.modules[module].LazyRunner
    assert issubclass(selected, _Default)


@pytest.mark.parametrize(
    "cls", ["no_colon", "pkg.mod:", ":Class", "pkg.mod:Outer.Inner", "pkg..mod:C"]
)
def test_register_rejects_a_malformed_path(cls):
    with pytest.raises(ValueError, match="module:Class"):
        register_model_runner(cls, ARCH)
    assert model_runner_registry._RUNNERS == {}


def test_register_rejects_a_class_that_is_not_a_type():
    with pytest.raises(TypeError, match="class or a 'module:Class' path"):
        register_model_runner(_Owned(), ARCH)
    assert model_runner_registry._RUNNERS == {}


def test_register_rejects_no_architectures():
    with pytest.raises(ValueError, match="names no architecture"):
        register_model_runner(_Owned)
    assert model_runner_registry._RUNNERS == {}


@pytest.mark.parametrize(
    "architectures, error",
    [(((ARCH, "Other"),), TypeError), ((ARCH, 3), TypeError), (("",), ValueError)],
)
def test_register_rejects_bad_architectures(architectures, error):
    with pytest.raises(error):
        register_model_runner(_Owned, *architectures)
    assert model_runner_registry._RUNNERS == {}


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
