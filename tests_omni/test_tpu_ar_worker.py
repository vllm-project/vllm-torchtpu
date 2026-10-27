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
"""Unit tests for src/vllm_torchtpu/omni/tpu_ar_worker.py."""

import inspect

import pytest
from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.tpu_ar_worker import TPUARWorker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker

pytestmark = pytest.mark.omni


def test_platform_ar_worker_cls():
    assert (
        OmniTpuPlatform.get_omni_ar_worker_cls()
        == "vllm_torchtpu.omni.tpu_ar_worker.TPUARWorker"
    )


def test_tpu_ar_worker_hierarchy_and_runner_cls():
    assert issubclass(TPUARWorker, TPUWorker)
    assert issubclass(TPUARWorker, OmniWorkerMixin)
    assert TPUWorker.model_runner_cls is TPUModelRunner
    assert TPUARWorker.model_runner_cls is OmniTPUARModelRunner


def test_tpu_worker_builds_runner_from_model_runner_cls():
    # init_device() needs a TPU, so check the construction site statically:
    # the worker must build its runner through the overridable class attribute.
    source = inspect.getsource(TPUWorker.init_device)
    assert "self.model_runner_cls(" in source
    assert "= TPUModelRunner(" not in source
