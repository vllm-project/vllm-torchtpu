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
"""`triangular_block_size` must divide every power-of-two chunk size."""

import types

import pytest

from vllm_torchtpu.kernels.experimental.gdn_pcp_fused.config import (
    GDNConfig as PcpGDNConfig,
)
from vllm_torchtpu.kernels.gdn.v3.config import GDNConfig


def _block_size(config_cls, num_v_heads: int, is_kda: bool) -> int:
    # The property only reads these two fields; skip building a full config.
    cfg = types.SimpleNamespace(num_v_heads=num_v_heads, is_kda=is_kda)
    return config_cls.triangular_block_size.fget(cfg)


@pytest.mark.parametrize("config_cls", [GDNConfig, PcpGDNConfig])
@pytest.mark.parametrize("is_kda", [False, True])
@pytest.mark.parametrize("num_v_heads", range(1, 129))
def test_block_size_divides_chunk(config_cls, is_kda, num_v_heads):
    b = _block_size(config_cls, num_v_heads, is_kda)
    assert 2 <= b <= 16
    assert b & (b - 1) == 0, f"{num_v_heads=} gave non-power-of-two {b}"
    for chunk in (16, 32, 64, 128):
        assert chunk % b == 0


@pytest.mark.parametrize("config_cls", [GDNConfig, PcpGDNConfig])
@pytest.mark.parametrize(
    "num_v_heads, expected",
    [
        (48, 2),  # Qwen3.8-Flash-Next, TP=1.
        (24, 4),  # TP=2: used to be 16 // 3 = 5 and broke chunk_size=64.
        (12, 8),  # TP=4.
        (6, 16),  # TP=8.
    ],
)
def test_gdn_block_size(config_cls, num_v_heads, expected):
    assert _block_size(config_cls, num_v_heads, is_kda=False) == expected
