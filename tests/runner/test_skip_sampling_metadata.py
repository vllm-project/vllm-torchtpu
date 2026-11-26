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
"""TPU_SKIP_DEVICE_SAMPLING_METADATA: InputBatch keeps its first
sampling_metadata instead of rebuilding (and uploading) it per batch change."""

from types import SimpleNamespace

import pytest


def test_skip_keeps_first_metadata(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.delenv("TPU_SKIP_DEVICE_SAMPLING_METADATA", raising=False)
    assert envs.TPU_SKIP_DEVICE_SAMPLING_METADATA is False
    builds = []
    batch = SimpleNamespace(is_pooling_model=False, sampling_metadata="first")
    batch._make_sampling_metadata = lambda: builds.append(1) or "rebuilt"
    runner = SimpleNamespace(input_batch=batch)
    TPUModelRunner._skip_device_sampling_metadata(runner)
    assert batch._make_sampling_metadata() == "first"
    assert not builds
    # Idempotent.
    TPUModelRunner._skip_device_sampling_metadata(runner)
    assert batch._make_sampling_metadata() == "first"


@pytest.mark.parametrize("pooling", [True])
def test_pooling_models_untouched(pooling):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    make = lambda: "rebuilt"  # noqa: E731
    batch = SimpleNamespace(is_pooling_model=pooling, _make_sampling_metadata=make)
    TPUModelRunner._skip_device_sampling_metadata(SimpleNamespace(input_batch=batch))
    assert batch._make_sampling_metadata is make
