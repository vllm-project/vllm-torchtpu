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
"""CPU unit tests for Qwen3.8 serving contract with stock vLLM TorchTPU."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax.experimental.pallas.tpu import ChipVersion

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.batched_rpa.wrapper import get_kv_cache_shape
from vllm_torchtpu.megakernel.errors import MegakernelConfigError, MegakernelLayoutError
from vllm_torchtpu.megakernel.models.qwen38_moe import Qwen38MoeGeometry
from vllm_torchtpu.megakernel.pool.layout import PoolLayout
from vllm_torchtpu.megakernel_runtime.driver import (
    expected_pool_shape,
    validate_pool_shape,
)
from vllm_torchtpu.platforms.tpu_block_size_utils import (
    _align_block_to_backend,
    derive_pooled_gdn_state_layout,
)


def test_stock_qwen38_tp32_block_and_gdn_slot():
    """Verify stock pooled GDN layout and block alignment for Qwen3.8 TP32."""
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=4 * 128 * 128 * 2,
        conv_bytes=3 * 768 * 2,
        token_bytes=512,
    )
    assert layout.ssm_tokens == 256
    assert layout.conv_tokens == 16
    assert layout.required_tokens == 272

    aligned = _align_block_to_backend(layout.required_tokens, [128])
    assert aligned == 384


def test_stock_seq_along_lane_pool_shape_qwen38():
    """Verify stock SEQ_ALONG_LANE pool shape with page_size=384."""
    shape = get_kv_cache_shape(
        10,
        384,
        1,
        256,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
        ChipVersion.TPU_7X,
    )
    assert shape == (10, 2, 64, 4, 384)


def test_megakernel_rejects_stock_qwen38_pool_page():
    """Verify megakernel PoolLayout and driver reject page_size=384 naming 384."""
    layout = PoolLayout(page_size=384)
    with pytest.raises((MegakernelConfigError, MegakernelLayoutError), match="384"):
        layout.validate()

    geom = Qwen38MoeGeometry()
    shape_384 = (10, 2, 64, 4, 384)
    with pytest.raises(MegakernelLayoutError, match="384"):
        validate_pool_shape(shape_384, geom, r_cnt=32, page_size=384)

    # Sanity check: expected_pool_shape produces expected tuple
    expected = expected_pool_shape(geom, r_cnt=32, num_pages=10, page_size=128)
    assert expected == (10, 2, 64, 4, 128)
