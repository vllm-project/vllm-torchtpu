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
"""Checks the model-specific patch gates against vLLM-Omni's model registry."""

import pytest
from vllm_omni.diffusion.registry import DiffusionModelRegistry

from vllm_torchtpu.omni.patches import _QWEN_IMAGE_PIPELINES, _WAN_PIPELINES

pytestmark = pytest.mark.omni

_MODELS = "vllm_omni.diffusion.models"


def _pipelines_in(package: str) -> set[str]:
    return {
        arch
        for arch, model in DiffusionModelRegistry.models.items()
        if model.module_name.startswith(f"{_MODELS}.{package}.")
    }


@pytest.mark.parametrize(
    ("pipelines", "package"),
    [(_QWEN_IMAGE_PIPELINES, "qwen_image"), (_WAN_PIPELINES, "wan2_2")],
    ids=["qwen_image", "wan2_2"],
)
def test_patched_pipelines_match_registry(pipelines, package):
    in_package = _pipelines_in(package)
    assert in_package, f"no registry entry under {_MODELS}.{package}"
    assert pipelines - DiffusionModelRegistry.models.keys() == set()
    assert in_package - pipelines == set()
