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
"""The flash kernel's kv-window table and when it is used (CPU).

The TPU tests compare outputs with the reference attention, which also match
when a window is too wide or the windows are off, so both are checked here.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.flash_attention import kernel
from vllm_torchtpu.kernels.flash_attention.kernel import BlockSizes, SegmentIds

# Two images of 1500 and 2000 rows, then 596 rows of padding with one id per
# 128 rows (TpuMMEncoderAttention's layout): ids 1, 2, then 3..7.
_IDS = np.concatenate(
    [np.full(1500, 1), np.full(2000, 2), 3 + np.arange(596) // 128]
).astype(np.int32)[None]


def test_kv_windows_table():
    # block_q 2048 in row groups of 512: 2 q blocks x 4 groups, [lo, hi) each.
    # 128-column kv blocks: 0-10 hold id 1, 11 ids 1-2, 12-26 id 2, 27 ids
    # 2-3, 28-31 the padding ids 3-7.
    expected = [
        (0, 1536),  # rows 0-511, id 1: kv blocks 0-11
        (0, 1536),  # rows 512-1023, id 1
        (0, 3584),  # rows 1024-1535, ids 1-2: kv blocks 0-27
        (1408, 3584),  # rows 1536-2047, id 2: kv blocks 11-27
        (1408, 3584),  # rows 2048-2559, id 2
        (1408, 3584),  # rows 2560-3071, id 2
        (1408, 3712),  # rows 3072-3583, ids 2-3: kv blocks 11-28
        (3456, 4096),  # rows 3584-4095, ids 3-7: kv blocks 27-31
    ]
    ids = jnp.asarray(_IDS)
    windows = kernel._kv_windows(SegmentIds(q=ids, kv=ids), 2048, 512)
    assert windows.tolist() == [[c for window in expected for c in window]]


@pytest.mark.parametrize(
    ("block_sizes", "with_segment_ids", "kv_windows"),
    [
        (BlockSizes(2048, 4096, 2048, 1, kv_windows=True), True, (1024, 512)),
        # One kv step: the single-step body has no kv loop to window.
        (BlockSizes(2048, 4096, 4096, 1, kv_windows=True), True, None),
        (BlockSizes(2048, 4096, 2048, 1, kv_windows=True), False, None),
        (BlockSizes(2048, 4096, 2048, 1), True, None),
    ],
)
def test_kv_windows_enabled(monkeypatch, block_sizes, with_segment_ids, kv_windows):
    calls = []
    signature = inspect.signature(kernel._flash_attention_impl)

    def fake_impl(*args):
        calls.append(signature.bind(*args).arguments["kv_windows"])

    monkeypatch.setattr(kernel, "_flash_attention_impl", fake_impl)
    x = jnp.zeros((1, 2, 4096, 72), jnp.bfloat16)
    ids = jnp.asarray(_IDS)
    segment_ids = SegmentIds(q=ids, kv=ids) if with_segment_ids else None
    kernel._flash_attention(
        x, x, x, None, segment_ids, False, False, 1.0, block_sizes, None, False
    )
    assert calls == [kv_windows]
