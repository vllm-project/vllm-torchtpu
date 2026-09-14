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


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "token_bytes", "ssm_tokens", "conv_tokens"),
    [
        # Kimi-Linear TP8: the f32 KDA state (2**18 B) does not divide the
        # padded MLA token row (2**9 * 5 B); the last SSM row is padding.
        pytest.param(262144, 9216, 2560, 103, 4, id="kimi-linear-tp8"),
        # The conv tile no longer needs to divide the SSM region, so it may
        # exceed it in token count.
        pytest.param(
            3 * 1024, 12 * 1024, 1024, 3, 16, id="conv-wider-than-ssm"),
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
