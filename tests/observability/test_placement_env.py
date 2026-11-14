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
"""VLLM_TPU_EMIT_PLACEMENT: accepted spellings and compile-cache neutrality."""

from __future__ import annotations

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler

READ = envs.environment_variables["VLLM_TPU_EMIT_PLACEMENT"]


def test_default_is_on(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_EMIT_PLACEMENT", raising=False)
    assert READ() is True
    monkeypatch.setenv("VLLM_TPU_EMIT_PLACEMENT", "")
    assert READ() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "OFF"])
def test_off_spellings(monkeypatch, value):
    monkeypatch.setenv("VLLM_TPU_EMIT_PLACEMENT", value)
    assert READ() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_on_spellings(monkeypatch, value):
    monkeypatch.setenv("VLLM_TPU_EMIT_PLACEMENT", value)
    assert READ() is True


def test_bad_value_names_the_variable(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_EMIT_PLACEMENT", "maybe")
    with pytest.raises(ValueError, match="VLLM_TPU_EMIT_PLACEMENT"):
        READ()


def test_knob_stays_out_of_the_compile_cache_key():
    assert "VLLM_TPU_EMIT_PLACEMENT" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
