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
"""Which o_proj shards the token-parallel sequence-row o_proj takes."""

import types

import pytest
import torch

from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

K_SHARD, N_OUT = 128, 256


def _o_proj(
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
    scale_shape: tuple[int, ...] = (N_OUT,),
    flipped: bool = True,
    bias: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    layer = types.SimpleNamespace(
        weight=torch.zeros(K_SHARD, N_OUT, dtype=dtype),
        weight_scale=torch.ones(scale_shape),
        bias=bias,
    )
    if flipped:
        setattr(layer, WEIGHT_FLIPPED_ATTR, True)
    return layer


def test_should_serve_canonical_fp8_per_channel_shard():
    assert dcp_token_parallel_op.serves_sequence_row_oproj(_o_proj())


@pytest.mark.parametrize(
    "layer",
    [
        _o_proj(flipped=False),  # vLLM [n_out, n_in] layout
        _o_proj(dtype=torch.bfloat16),
        _o_proj(scale_shape=(K_SHARD // 128, N_OUT // 128)),  # blockwise
        _o_proj(scale_shape=(K_SHARD,)),  # per input row, not per channel
        _o_proj(bias=torch.zeros(N_OUT)),
    ],
    ids=["vllm_layout", "bf16", "blockwise", "row_scale", "bias"],
)
def test_should_refuse_shard_the_projection_cannot_take(layer):
    assert not dcp_token_parallel_op.serves_sequence_row_oproj(layer)
