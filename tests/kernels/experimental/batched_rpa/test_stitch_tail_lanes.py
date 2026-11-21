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
"""What the SEQ_ALONG_LANE stitch writes past a block's last valid token.

The window holds the fetched cache tokens from lane 0 and the fetched new
tokens from the next 128-lane boundary. Every other lane is undefined here
(all bits set, an fp8, bf16 and f32 NaN). The prefill stitch writes the whole
block: cached lanes, then the new tokens, then zeros when V is not masked on
load (`mask_invalid_v` off). With V masked it leaves the lanes past the valid
tokens as the roll fills them. The decode stitch writes only the columns that
hold the stitch boundary: the lanes before it, the new tokens, then zeros.
"""

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, stitch_utils

_LANES = 128
_UNDEF = np.uint32(0xFFFFFFFF)


@pytest.fixture(autouse=True)
def _tpu7x_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)


def _cfgs(bq_sz, bkv_sz, mask_invalid_v=None):
    return configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=bq_sz, bq_c_sz=min(bq_sz, 32), bkv_sz=bkv_sz, batch_size=1
        ),
        model=configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            mask_value=-1e30,
            sm_scale=1.0,
        ),
        serve=configs.ServingConfigs(
            num_seqs=8,
            page_size=128,
            total_q_tokens=512,
            num_page_indices=64,
            dtype_q=jnp.dtype(jnp.bfloat16),
            dtype_kv=jnp.dtype(jnp.bfloat16),
            dtype_out=jnp.dtype(jnp.bfloat16),
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            mask_invalid_v=mask_invalid_v,
        ),
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 << 20,
    )


def _window(cfgs, cache, new, new_tok_offset):
    """A u32 KV window: cached words at [0, cache), the new tokens' words at
    the aligned fetch [cache_end + new_tok_offset, + new), every other lane
    undefined. Returns the window, cache_end and the expected block."""
    words = (
        cfgs.model.num_kv_heads * 2 * cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
    )
    v_len = cfgs.kv_vmem_lanes
    lane = np.arange(v_len, dtype=np.uint32)
    word = np.arange(words, dtype=np.uint32)[:, None] << 16
    window = np.full((words, v_len), _UNDEF, np.uint32)
    window[:, :cache] = (word | lane)[:, :cache]
    cache_end = -(-cache // _LANES) * _LANES
    src = cache_end + new_tok_offset
    window[:, src : src + new] = (word | (lane + 0x8000))[:, :new]
    want = np.zeros((words, v_len), np.uint32)
    want[:, :cache] = window[:, :cache]
    want[:, cache : cache + new] = window[:, src : src + new]
    return window, cache_end, want


# (cache tokens in the block, new tokens, the new tokens' offset in their
# aligned fetch). Cache ending mid-column, on a column boundary, and empty
# (a CACHE_ONLY block when new is 0).
_CASES = [(300, 40, 17), (256, 64, 0), (43, 150, 100), (300, 0, 0), (0, 0, 0)]
# The masked test checks only the valid lanes, so a block without any (the
# empty case) would compare nothing; it stays in the unmasked test, which
# checks that the whole block is zeroed.
_CASES_WITH_TOKENS = [case for case in _CASES if case[0] + case[1] > 0]


def _prefill_stitch(mask_invalid_v, cache, new, offset):
    """The prefill stitch's block for one window, and the expected block."""
    cfgs = _cfgs(bq_sz=256, bkv_sz=512, mask_invalid_v=mask_invalid_v)
    window, cache_end, want = _window(cfgs, cache, new, offset)

    @jax.jit
    def stitch(w, c, e, o, n):
        return stitch_utils._stitch_prefill_lane(
            w, c, e, o, n, cfgs.kv_vmem_lanes, cfgs=cfgs
        )

    got = stitch(
        jnp.asarray(window), jnp.int32(cache), jnp.int32(cache_end), offset, new
    )
    got = np.asarray(got).reshape(window.shape[0], cfgs.bkv_sz)
    return got, want[:, : cfgs.bkv_sz]


@pytest.mark.parametrize(("cache", "new", "offset"), _CASES)
def test_prefill_stitch_zeroes_every_lane_past_the_valid_tokens(cache, new, offset):
    got, want = _prefill_stitch(False, cache, new, offset)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(("cache", "new", "offset"), _CASES_WITH_TOKENS)
def test_prefill_stitch_with_v_masked_places_the_valid_tokens(cache, new, offset):
    """With V masked on load the lanes past the valid tokens keep what the
    roll brings in, so only the cached and new-token lanes are checked."""
    got, want = _prefill_stitch(True, cache, new, offset)
    np.testing.assert_array_equal(got[:, : cache + new], want[:, : cache + new])


@pytest.mark.parametrize("bq_sz", [1, 4])
@pytest.mark.parametrize(("cache", "new"), [(43, 1), (300, 1), (300, 0), (0, 0)])
def test_decode_stitch_zeroes_the_written_columns_past_the_valid_tokens(
    bq_sz, cache, new
):
    cfgs = _cfgs(bq_sz=bq_sz, bkv_sz=512, mask_invalid_v=False)
    offset = 5
    window, cache_end, want = _window(cfgs, cache, new, offset)

    @jax.jit
    def stitch(w, c, e, o, n):
        return stitch_utils._stitch_decode_lane(
            w, c, e, o, n, cfgs.kv_vmem_lanes, cfgs=cfgs
        )

    cols, outer_dim, lanes_per_col, merged = stitch(
        jnp.asarray(window), jnp.int32(cache), jnp.int32(cache_end), offset, new
    )
    # The decode stitch views the window as [-1, 128] rows; column c of the
    # window is rows c, c + lanes_per_col, ... (one per outer row).
    want_rows = want.reshape(outer_dim, lanes_per_col, _LANES)
    assert cols, "the decode stitch writes at least the boundary column"
    for col, vreg in zip(cols, merged):
        col = int(col)
        np.testing.assert_array_equal(
            np.asarray(vreg).reshape(outer_dim, _LANES), want_rows[:, col]
        )
