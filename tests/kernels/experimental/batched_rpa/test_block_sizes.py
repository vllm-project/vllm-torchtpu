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
"""SEQ_ALONG_LANE KV blocks are whole pages: other sizes are refused."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs


def _cfgs(layout, bkv, page_size=256):
    return configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv, batch_size=8),
        model=configs.ModelConfigs(
            num_q_heads=8, num_kv_heads=4, head_dim=256, mask_value=-1e30
        ),
        serve=configs.ServingConfigs(
            num_seqs=8,
            page_size=page_size,
            total_q_tokens=16,
            num_page_indices=8 * 64,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.bfloat16,
            kv_layout=layout,
        ),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


@pytest.mark.parametrize("bkv", [128, 384, 640])
def test_seq_along_lane_refuses_partial_pages(bkv):
    with pytest.raises(ValueError, match="whole pages"):
        _cfgs(configs.KVLayout.SEQ_ALONG_LANE, bkv)


@pytest.mark.parametrize(
    "layout, bkv",
    [
        (configs.KVLayout.SEQ_ALONG_LANE, 256),
        (configs.KVLayout.SEQ_ALONG_LANE, 768),
        (configs.KVLayout.HEAD_ALONG_SUBLANE, 384),  # tokens on sublanes
    ],
)
def test_whole_pages_and_head_along_sublane_are_accepted(layout, bkv):
    assert _cfgs(layout, bkv).bkv_sz == bkv
