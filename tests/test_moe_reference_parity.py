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
"""Parsing of the MoE reference-parity `additional_config` settings."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.moe_reference_parity import (
    MoEReferenceParity,
    get_moe_reference_parity,
)


def _config(**additional_config):
    return SimpleNamespace(additional_config=additional_config)


@pytest.mark.parametrize(
    "vllm_config,expected",
    [
        (None, MoEReferenceParity()),
        (_config(), MoEReferenceParity()),
        (_config(tpu_moe_silu_bf16=True), MoEReferenceParity(silu_bf16=True)),
        (
            _config(tpu_moe_topk_weights_float32=True),
            MoEReferenceParity(topk_weights_float32=True),
        ),
        (
            _config(tpu_moe_silu_bf16=True, tpu_moe_topk_weights_float32=True),
            MoEReferenceParity(silu_bf16=True, topk_weights_float32=True),
        ),
    ],
)
def test_get_moe_reference_parity(vllm_config, expected):
    assert get_moe_reference_parity(vllm_config) == expected


@pytest.mark.parametrize("key", ["tpu_moe_silu_bf16", "tpu_moe_topk_weights_float32"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_non_bool_value_raises(key, value):
    with pytest.raises(ValueError, match=key):
        get_moe_reference_parity(_config(**{key: value}))
