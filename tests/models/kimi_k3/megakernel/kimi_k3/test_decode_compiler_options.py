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

from __future__ import annotations

import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.decode_ops import (
    decode_batch_sizes,
    decode_compiler_options,
)


@pytest.mark.parametrize("attention_dp", [1, 2])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64, 96, 128])
def test_should_disable_hlo_calls_for_all_supported_decode_batches_when_enabled(
    monkeypatch: pytest.MonkeyPatch, attention_dp: int, rows: int
) -> None:
    cfg = KernelConfig(
        attention_dp=attention_dp,
        dcp_group=32 // attention_dp,
        decode_no_hlo_calls=True,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    assert decode_compiler_options(rows) == {
        "xla_tpu_enable_deduplicated_calls": "DISABLED"
    }


@pytest.mark.parametrize("max_batch", [8, 16, 64, 128])
def test_should_cover_every_scheduled_decode_batch_size_at_attention_dp1(
    monkeypatch: pytest.MonkeyPatch, max_batch: int
) -> None:
    cfg = KernelConfig(attention_dp=1, dcp_group=32, decode_no_hlo_calls=True)
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    batches = decode_batch_sizes(max_batch)
    assert batches[0] == 1
    for batch in batches:
        assert decode_compiler_options(batch) == {
            "xla_tpu_enable_deduplicated_calls": "DISABLED"
        }


@pytest.mark.parametrize("unsupported_rows", [-1, 0, 3, 5, 6, 7, 12, 127, 129, 256])
def test_should_return_empty_decode_compiler_options_when_unsupported_or_disabled(
    monkeypatch: pytest.MonkeyPatch, unsupported_rows: int
) -> None:
    cfg_on = KernelConfig(attention_dp=1, decode_no_hlo_calls=True)
    monkeypatch.setattr(k3_config, "active", lambda: cfg_on)
    assert decode_compiler_options(unsupported_rows) == {}

    cfg_off = KernelConfig(attention_dp=1, decode_no_hlo_calls=False)
    monkeypatch.setattr(k3_config, "active", lambda: cfg_off)
    for rows in (1, 2, 4, 8, 16, 32, 64, 96, 128):
        assert decode_compiler_options(rows) == {}
