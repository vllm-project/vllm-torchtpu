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
"""The batched RPA MIXED kernel skips the QK mask only where it is all True.

`kernel.block_fully_visible` decides from a step's scalars that
`kernel.generate_mask` would keep every score of the block, for the first q
chunk and so for every later one, on every lane. This checks the two against
each other on the boundary of each term: the causal offset (the block's last
key against the first query row), the lane's valid keys, and the DCP scopes'
`local_k_start` (NEW_TOKENS_ONLY) and `local_k_end` (CACHE_ONLY).
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel

_BQ, _BKV, _NQ = 16, 256, 4
_FULL = configs.AttentionScope.FULL
_NEW = configs.AttentionScope.NEW_TOKENS_ONLY
_CACHE = configs.AttentionScope.CACHE_ONLY
# Each term just inside and just outside its bound.
_CAUSAL = (-_BKV - 1, -_BKV, 1 - _BKV, 2 - _BKV, 0)
_N_VALID = (_BKV - 1, _BKV)
_LOCAL_K_START = (-1, 0, 1)
_LOCAL_K_END = (_BKV - 1, _BKV, _BKV + 1)
# A lane every term passes.
_VISIBLE = (-_BKV, _BKV, 0, _BKV)


def _cfgs(scope, batch_size):
    serve = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=_BQ,
        num_page_indices=64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        attention_scope=scope,
    )
    model = configs.ModelConfigs(
        num_q_heads=_NQ, num_kv_heads=1, head_dim=256, mask_value=-1e9
    )
    block = configs.BlockSizes(
        bq_sz=_BQ, bq_c_sz=_BQ, bkv_sz=_BKV, batch_size=batch_size
    )
    return configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=1 << 26,
    )


def _metadata(scope, lanes):
    """lanes: (causal_offset, n_valid, local_k_start, local_k_end) per lane."""

    def ints(values):
        return [jnp.int32(v) for v in values]

    causal, n_valid, start, end = zip(*lanes)
    stitch = kernel.StitchMetadata(
        bkv_sz_frm_cache=ints(n // 2 for n in n_valid),
        bkv_sz_frm_new=ints(n - n // 2 for n in n_valid),
        new_kv_len_start=ints(0 for _ in lanes),
    )
    step = kernel.ComputeMetadata(
        causal_offset=ints(causal),
        local_k_start=ints(start) if scope == _NEW else None,
        local_k_end=ints(end) if scope == _CACHE else None,
    )
    return stitch, step


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("scope", [_FULL, _NEW, _CACHE])
def test_block_fully_visible_iff_the_mask_keeps_every_score(scope, batch_size):
    cfgs = _cfgs(scope, batch_size)
    shape = (batch_size, 1, _BQ * cfgs.aligned_num_q_heads_per_kv_head, _BKV)
    starts = _LOCAL_K_START if scope == _NEW else (0,)
    ends = _LOCAL_K_END if scope == _CACHE else (_BKV,)
    outcomes = set()
    for lane in itertools.product(_CAUSAL, _N_VALID, starts, ends):
        # With two lanes, the lane under test sits next to a fully visible
        # lane, first and second.
        orders = [[lane]] if batch_size == 1 else [[lane, _VISIBLE], [_VISIBLE, lane]]
        for lanes in orders:
            stitch, step = _metadata(scope, lanes)
            masks = kernel.generate_mask(
                shape, bq_start=0, stitch_meta=stitch, step_meta=step, cfgs=cfgs
            )
            want = all(bool(np.all(np.asarray(m))) for m in masks)
            got = kernel.block_fully_visible(stitch, step, cfgs)
            assert got is not None
            assert bool(got) == want, lanes
            outcomes.add(want)
    assert outcomes == {True, False}
