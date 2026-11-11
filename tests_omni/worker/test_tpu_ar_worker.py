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
"""Unit tests for src/vllm_torchtpu/omni/worker/base.py and tpu_ar_worker.py."""

import ast
import inspect
import textwrap

import pytest
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.worker.base import OmniTPUWorkerBase
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_ar_worker import TPUARWorker
from vllm_torchtpu.worker.tpu_worker import TPUWorker

pytestmark = pytest.mark.omni


def test_tpu_ar_worker_wiring():
    assert resolve_obj_by_qualname(OmniTpuPlatform.get_omni_ar_worker_cls()) is (
        TPUARWorker
    )
    assert TPUARWorker.__bases__ == (OmniWorkerMixin, OmniTPUWorkerBase)
    assert issubclass(OmniTPUWorkerBase, TPUWorker)
    assert TPUARWorker.model_runner_cls is TPUARModelRunner


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("handle_sleep_task", ({"task_id": 1, "level": 1},)),
        ("handle_wake_task", ({"task_id": 2},)),
        ("encoder_loaded", ()),
    ],
)
def test_worker_base_refuses_unimplemented_orchestrator_hooks(method, args):
    worker = OmniTPUWorkerBase.__new__(OmniTPUWorkerBase)
    with pytest.raises(NotImplementedError, match="TPU"):
        getattr(worker, method)(*args)


def test_tpu_worker_builds_runner_from_model_runner_cls():
    tree = ast.parse(textwrap.dedent(inspect.getsource(TPUWorker.init_device)))
    called = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(func, ast.Attribute)
        and func.attr == "model_runner_cls"
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        for func in called
    )
    assert not any(
        isinstance(func, ast.Name) and func.id == "TPUModelRunner" for func in called
    )
