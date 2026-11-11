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
"""CPU tests for the PV valid-token mask used with `mask_invalid_v`."""

import types

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel

_BKV_SZ = 256


def _cfgs(kv_layout, sliding_window):
    return types.SimpleNamespace(
        serve=types.SimpleNamespace(kv_layout=kv_layout),
        model=types.SimpleNamespace(sliding_window=sliding_window),
    )


def _mask(kv_layout, sliding_window, *, k_id, q_base, frm_cache, frm_new=0):
    stitch_meta = kernel.StitchMetadata(
        bkv_sz_frm_cache=[jnp.int32(frm_cache)],
        bkv_sz_frm_new=[jnp.int32(frm_new)],
        new_kv_len_start=[jnp.int32(0)],
    )
    step_meta = kernel.ComputeMetadata(causal_offset=[jnp.int32(k_id - q_base)])
    token = jnp.arange(_BKV_SZ, dtype=jnp.int32)
    valid = kernel._compute_valid_kv_mask(
        token, 0, stitch_meta, step_meta, _cfgs(kv_layout, sliding_window)
    )
    return np.asarray(valid)


def test_nhd_swa_masks_untrimmed_pre_window_slots():
    """Slots before sw_start are never DMA'd and must be masked out of PV."""
    # Decode at q_pos=599 with W=128: sw_start = 472, i.e. local slot 216 of
    # the chunk starting at k_id=256.
    valid = _mask(
        configs.KVLayout.HEAD_ALONG_SUBLANE,
        128,
        k_id=256,
        q_base=599,
        frm_cache=_BKV_SZ,
    )
    expected = np.zeros(_BKV_SZ, dtype=bool)
    expected[216:] = True
    np.testing.assert_array_equal(valid, expected)


def test_nhd_swa_window_before_chunk_masks_only_tail():
    """A chunk wholly inside the window only masks the post-length tail."""
    # q_pos=599, W=128, chunk [512, 768): sw_start=472 precedes the chunk, so
    # no prefix masking; only tokens >= kv_len (600) are masked.
    valid = _mask(
        configs.KVLayout.HEAD_ALONG_SUBLANE,
        128,
        k_id=512,
        q_base=599,
        frm_cache=88,
    )
    expected = np.zeros(_BKV_SZ, dtype=bool)
    expected[:88] = True
    np.testing.assert_array_equal(valid, expected)


@pytest.mark.parametrize(
    "kv_layout,sliding_window",
    [
        (configs.KVLayout.SEQ_ALONG_LANE, 128),
        (configs.KVLayout.HEAD_ALONG_SUBLANE, None),
    ],
)
def test_no_pre_window_mask_without_intra_page_trim(kv_layout, sliding_window):
    """HND copies whole pages and full attention has no window to trim."""
    valid = _mask(kv_layout, sliding_window, k_id=256, q_base=599, frm_cache=200)
    expected = np.zeros(_BKV_SZ, dtype=bool)
    expected[:200] = True
    np.testing.assert_array_equal(valid, expected)
