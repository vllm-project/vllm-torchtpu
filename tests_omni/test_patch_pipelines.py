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

from vllm_torchtpu.omni.patches import (
    _LTX2_PIPELINES,
    _QWEN_IMAGE_PIPELINES,
    _WAN_PIPELINES,
)

pytestmark = pytest.mark.omni

_MODELS = "vllm_omni.diffusion.models"


def _pipelines_in(*packages: str) -> set[str]:
    prefixes = tuple(f"{_MODELS}.{package}." for package in packages)
    return {
        arch
        for arch, model in DiffusionModelRegistry.models.items()
        if model.module_name.startswith(prefixes)
    }


@pytest.mark.parametrize(
    ("pipelines", "packages"),
    [
        (_QWEN_IMAGE_PIPELINES, ("qwen_image",)),
        (_WAN_PIPELINES, ("wan2_2", "causvid_wan")),
        (_LTX2_PIPELINES, ("ltx2",)),
    ],
    ids=["qwen_image", "wan", "ltx2"],
)
def test_patched_pipelines_match_registry(pipelines, packages):
    for package in packages:
        assert _pipelines_in(package), f"no registry entry under {_MODELS}.{package}"
    assert pipelines == _pipelines_in(*packages)
