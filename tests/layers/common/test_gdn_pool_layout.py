# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout


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
