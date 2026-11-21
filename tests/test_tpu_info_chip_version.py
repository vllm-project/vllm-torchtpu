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
"""The TPU chip an accelerator type names."""

import logging
import re

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import tpu_info


@pytest.fixture(autouse=True)
def _uncached_chip_version():
    # get_chip_version caches by argument, and a cached answer logs nothing.
    tpu_info.get_chip_version.cache_clear()
    yield
    tpu_info.get_chip_version.cache_clear()


@pytest.mark.parametrize(
    ("accelerator_type", "chip"),
    [
        ("tpu7x", pltpu.ChipVersion.TPU_7X),
        ("tpu7x-8", pltpu.ChipVersion.TPU_7X),
        ("v7x-16", pltpu.ChipVersion.TPU_7X),
        (" TPU7X-8\n", pltpu.ChipVersion.TPU_7X),
        ("v6e", pltpu.ChipVersion.TPU_V6E),
        ("v6e-8", pltpu.ChipVersion.TPU_V6E),
        ("v5litepod-8", pltpu.ChipVersion.TPU_V5E),
    ],
)
def test_an_accelerator_type_names_its_chip(accelerator_type, chip):
    assert tpu_info.parse_chip_version(accelerator_type) is chip
    assert tpu_info.get_chip_version(accelerator_type) is chip


@pytest.mark.parametrize("accelerator_type", ["tpu6e", "v6ee", "tpu7xx", "foo-8"])
def test_a_type_that_names_no_chip(accelerator_type, caplog):
    # The strict lookup raises. get_chip_version takes the type for TPU7x and
    # says so.
    with pytest.raises(ValueError, match=re.escape(repr(accelerator_type))):
        tpu_info.parse_chip_version(accelerator_type)
    with caplog.at_level(logging.WARNING):
        assert tpu_info.get_chip_version(accelerator_type) is pltpu.ChipVersion.TPU_7X
    assert f"{accelerator_type!r}; defaulting to TPU7x" in caplog.text


def test_no_accelerator_type(monkeypatch, caplog):
    # Neither TPU_ACCELERATOR_TYPE nor the GCE metadata names one.
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: None)
    with pytest.raises(ValueError, match="''"):
        tpu_info.parse_chip_version("")
    with caplog.at_level(logging.WARNING):
        assert tpu_info.get_chip_version() is pltpu.ChipVersion.TPU_7X
    assert "defaulting to TPU7x" in caplog.text
