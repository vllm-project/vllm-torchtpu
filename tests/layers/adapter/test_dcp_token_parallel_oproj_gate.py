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
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op
from vllm_torchtpu.layers.adapter.linear_common import (
    DCP_ROWS_ATTR,
    WEIGHT_FLIPPED_ATTR,
)

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


def _wrapper_o_proj(token_parallel_layout):
    """The MLA wrapper's o_proj, built with the attention stubbed to `layout`.

    o_proj is a plain namespace, so an attribute the wrapper sets is observable."""

    def fake_attention_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.token_parallel_layout = token_parallel_layout
        self.dcp_q_replicate = True

    mla_modules = MagicMock()
    mla_modules.o_proj = types.SimpleNamespace()
    mla_modules.g_proj = None
    mla_modules.indexer = None
    mla_modules.indexer_rotary_emb = None
    mla_modules.is_sparse = True
    with patch.object(
        mla_attention_op.VllmMLAAttention, "__init__", fake_attention_init
    ):
        wrapper = mla_attention_op.VllmMultiHeadLatentAttentionWrapper(
            hidden_size=1024,
            num_heads=16,
            scale=1.0,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            q_lora_rank=None,
            kv_lora_rank=512,
            mla_modules=mla_modules,
        )
    return wrapper.o_proj


@pytest.mark.parametrize(
    ("layout", "expect_rows"), [(None, False), (object(), True)], ids=["off", "on"]
)
def test_should_request_whole_k_o_proj_only_when_token_parallel_prefill_is_armed(
    layout, expect_rows
):
    o_proj = _wrapper_o_proj(layout)

    assert bool(getattr(o_proj, DCP_ROWS_ATTR, False)) is expect_rows
