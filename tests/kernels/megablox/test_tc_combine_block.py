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
"""tc_combine's token block: the widest that fits the local experts' tiles."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.megablox import tc_combine


@pytest.mark.parametrize(
    "tokens, experts, block",
    [
        (8192, 16, 256),  # EP8 on 128 experts
        (8192, 32, 128),  # 4 x 2 hierarchical EP: 256 tokens' tiles do not fit
        (8192, 64, 0),  # 2 x 4 hierarchical EP: neither block fits
        (8064, 16, 0),  # not a multiple of 128 tokens
    ],
)
def test_token_block(tokens, experts, block):
    assert tc_combine.pick_token_block(tokens, 6144, 4, experts) == block
    assert tc_combine.supported(tokens, 6144, 4, experts, jnp.bfloat16) == (block > 0)
