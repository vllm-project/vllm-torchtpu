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

import inspect

import pytest
import torch

from vllm_torchtpu import gdn_pool_layout
from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    derive_pooled_gdn_state_layout,
    pooled_gdn_conv_state_bytes,
)
from vllm_torchtpu.layers.core import gdn_attention


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "ssm_tokens", "conv_tokens"),
    [
        pytest.param(512 * 1024, 12 * 1024, 512, 16, id="p4"),
        pytest.param(1024 * 1024, 24 * 1024, 1024, 32, id="d2"),
    ],
)
def test_qwen35_layout_includes_conv_tile_padding(
    ssm_bytes: int,
    conv_bytes: int,
    ssm_tokens: int,
    conv_tokens: int,
):
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=1024,
    )

    assert layout.ssm_tokens == ssm_tokens
    assert layout.conv_tokens == conv_tokens
    assert layout.required_tokens == ssm_tokens + conv_tokens
    assert layout.required_bytes == (ssm_tokens + conv_tokens) * 1024


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "token_bytes", "ssm_tokens", "conv_tokens"),
    [
        # Kimi-Linear TP8: the f32 KDA state (2**18 B) does not divide the
        # padded MLA token row (2**9 * 5 B); the last SSM row is padding.
        pytest.param(262144, 9216, 2560, 103, 4, id="kimi-linear-tp8"),
        # The conv tile no longer needs to divide the SSM region, so it may
        # exceed it in token count.
        pytest.param(3 * 1024, 12 * 1024, 1024, 3, 16, id="conv-wider-than-ssm"),
    ],
)
def test_layout_pads_states_to_token_rows(
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
    ssm_tokens: int,
    conv_tokens: int,
):
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
    )

    assert layout.ssm_tokens == ssm_tokens
    assert layout.conv_tokens == conv_tokens
    assert layout.ssm_tokens * token_bytes >= ssm_bytes
    assert layout.conv_tokens * token_bytes >= conv_bytes
    assert layout.required_tokens == ssm_tokens + conv_tokens


def test_pooled_conv_state_dtype_and_sublane_doc_invariants():
    assert DEFAULT_POOLED_GDN_CONV_STATE_DTYPE == "torch.float32"
    assert pooled_gdn_conv_state_bytes(kernel_size=4, conv_dim=2048) == 3 * 2048 * 4

    for fn in (
        gdn_pool_layout.pooled_gdn_conv_state_bytes,
        gdn_attention.run_jax_gdn_attention_pooled_local,
        gdn_attention.run_jax_gdn_attention_pooled,
    ):
        doc = inspect.getdoc(fn) or ""
        assert "fixed-BF16 Conv" not in doc
        assert "dense bf16 rows" not in doc

    # Verify (-2, -1) sublane unpacking restores exact finite float32 words
    # when adjacent lanes form a quiet-NaN bit pattern under naive row-major view.
    words_i64 = torch.tensor([0x3F800001, 0x40007FC0], dtype=torch.int64)
    expected_f32 = words_i64.to(torch.uint32).view(torch.float32)
    assert torch.isfinite(expected_f32).all()

    pairs_u16 = expected_f32.view(torch.uint16).reshape(-1, 2)
    sublane_packed = pairs_u16.transpose(-2, -1).contiguous().view(torch.bfloat16)
    assert torch.isnan(sublane_packed.contiguous().view(torch.float32)).any()
    restored_f32 = (
        sublane_packed.movedim(-2, -1).contiguous().view(torch.float32).reshape(-1)
    )
    assert torch.equal(restored_f32, expected_f32)
