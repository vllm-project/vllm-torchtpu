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
"""Chip version lookup from the accelerator type (tpu_info)."""

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import tpu_info

CV = pltpu.ChipVersion
KNOWN = [
    ("tpu7x-8", CV.TPU_7X),
    ("TPU7X-16", CV.TPU_7X),
    ("v7x-8", CV.TPU_7X),
    ("v6e-8", CV.TPU_V6E),
    ("v5litepod-4", CV.TPU_V5E),
    ("v5p-8", CV.TPU_V5P),
]
UNKNOWN = ["tpu-unknown-8", "v9z-4", ""]


@pytest.fixture
def accelerator_type(monkeypatch):
    def use(value):
        monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: value)
        tpu_info.find_chip_version.cache_clear()
        tpu_info.get_chip_version.cache_clear()

    yield use
    tpu_info.find_chip_version.cache_clear()
    tpu_info.get_chip_version.cache_clear()


@pytest.mark.parametrize("value, chip", KNOWN)
def test_known_types_resolve_alike(accelerator_type, value, chip):
    accelerator_type(value)
    assert tpu_info.find_chip_version() == chip
    assert tpu_info.get_chip_version() == chip
    assert tpu_info.get_chip_version(value) == chip


@pytest.mark.parametrize("value", UNKNOWN + [None])
def test_find_chip_version_does_not_default_an_unknown_type(accelerator_type, value):
    accelerator_type(value)
    assert tpu_info.find_chip_version() is None


@pytest.mark.parametrize("value", UNKNOWN)
def test_get_chip_version_still_defaults_to_tpu7x(accelerator_type, value, caplog):
    accelerator_type(value)
    with caplog.at_level("WARNING"):
        assert tpu_info.get_chip_version() == CV.TPU_7X
    assert "defaulting to TPU7x" in caplog.text
