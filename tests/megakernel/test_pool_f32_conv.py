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
"""Split pool I/O with an f32 conv window (`PoolLayout.conv_dtype == f32`).

The serving framework stores the pooled GDN conv state as fp32: one f32 per u32
row element, the `(conv_taps, QKV)` window tap-major as rows of 128 lanes in
the conv page (row `tap * QKV // 128 + j`, lane `c` = channel `j * 128 + c`).
Values here are full f32 (not bf16-representable), so any bf16 round trip
would show.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel import test_pool_gather as tg
from tests.megakernel import test_pool_scatter as ts
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.ops import F32


def _full_f32(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    x = jax.random.normal(key, shape, dtype=jnp.float32) * 0.1
    assert not np.array_equal(
        np.asarray(x), np.asarray(x.astype(jnp.bfloat16).astype(jnp.float32))
    )
    return x


@pytest.fixture(autouse=True)
def _full_f32_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tg, "_bf16_f32", _full_f32)
    monkeypatch.setattr(ts, "_bf16_f32", _full_f32)


def _f32(lay: pool.PoolLayout) -> pool.PoolLayout:
    return replace(lay, conv_dtype=F32)


def test_f32_conv_page_bytes_follow_the_framework_granule_layout():
    """`conv_to_raw_page` writes the f32 window as u32 rows of the page's
    `(rows, 128)` u32 view, one f32 per u32, tap-major; zeros elsewhere."""
    lay = _f32(replace(tg._layout(3), kh=16, vh=64))
    qkv = lay.qkv_per_rank(8)  # 1536 -> 36 rows
    taps = _full_f32(jax.random.key(3), (lay.conv_taps, qkv))
    raw = np.asarray(pool.conv_to_raw_page(taps, lay))
    assert raw.shape == lay.kv_page_shape
    # The TPU u32 view packs the 4 fp8 sublanes (dim 2) of each row into a word.
    u32 = raw.view(np.uint8).transpose(0, 1, 3, 2).copy().view(np.uint32)
    u32 = u32.reshape(lay.kv_page_shape[0] * lay.kv_page_shape[1], 128)
    want = np.asarray(taps, np.float32).reshape(-1, 128).view(np.uint32)
    np.testing.assert_array_equal(u32[: want.shape[0]], want)
    assert not u32[want.shape[0] :].any()
    np.testing.assert_array_equal(
        np.asarray(pool.raw_page_to_conv(raw, qkv, lay)), np.asarray(taps)
    )


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim", [("slots3", 3), ("slots3", 2), ("slots1", 3), ("slots1", 2)]
)
def test_f32_conv_gather_is_exact(name: str, ckpt_ndim: int, T: int):
    lay = _f32(tg._layout(tg.SLOTS[name]))
    lay.validate(tg._exec(R=2, B=2 * T, T=T))
    ex = tg._exec(R=2, B=2 * T, T=T)
    st = tg._random_states(lay, ex, seed=41 + T)
    pt, ckpt, roff = tg._tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(tg._unfold(r, ex.R)[0] for r in raw_pools)
    conv, rec = pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff)
    taps = np.asarray(st["conv"][0])
    exp = np.zeros(conv.shape, np.float32)
    if T == 1:
        exp[:, : 3 * ex.B] = np.swapaxes(taps, 1, 2).reshape(
            taps.shape[0], -1, taps.shape[-1]
        )
    else:
        exp[:, :, :3] = taps
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim", [("slots3", 3), ("slots3", 2), ("slots1", 3), ("slots1", 2)]
)
def test_f32_conv_scatter_is_exact(name: str, ckpt_ndim: int, T: int):
    ts._check_scatter(
        _f32(ts._layout(ts.SLOTS[name])), ts._exec(R=2, B=2 * T, T=T), ckpt_ndim, 61 + T
    )


@pytest.mark.parametrize("T", [2, 4, 5])
def test_f32_conv_direct_and_packed_windows(T: int):
    """12 lane-rows per rank (the Qwen3.5 TP=8 conv width): the direct verify
    window path and the packed fallback both keep f32 exact."""
    lay = _f32(replace(ts._layout(1, num_pools=1), kh=4, vh=4))
    assert lay.qkv_per_rank(1) // 128 == 12
    ts._check_scatter(lay, ts._exec(R=1, B=2 * T, T=T), 3, seed=111 + T)
    ex = tg._exec(R=1, B=2 * T, T=T)
    st = tg._random_states(lay, ex, seed=121 + T)
    pt, ckpt, roff = tg._tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    conv, _ = pool.gather_gdn_states(lay, ex, tuple(raw_pools), ckpt, roff)
    np.testing.assert_array_equal(np.asarray(conv)[:, :, :3], np.asarray(st["conv"][0]))


def test_f32_conv_rejects_partial_conv_page_writes():
    """The GPR partial conv page write stores bf16; f32 fails closed."""
    from vllm_torchtpu.megakernel.errors import MegakernelConfigError

    lay = _f32(ts._layout(1, num_pools=1))
    ex = replace(ts._exec(R=1, B=2, T=2), interpret=False)
    pt, ckpt, _ = ts._tables(lay, ex, 3)
    vh, NS = lay.vh_per_rank(1), ex.NS
    qkv = lay.qkv_per_rank(1)
    with pytest.raises(MegakernelConfigError, match="partial conv page"):
        pool.scatter_pool_states(
            lay,
            ex,
            (jnp.zeros((ts.NUM_PAGES, *lay.kv_page_shape), jnp.float8_e4m3fn),),
            jnp.arange(ex.B, dtype=jnp.int32),
            pt,
            jnp.zeros((1, ex.MR, lay.hd), jnp.float8_e4m3fn),
            jnp.zeros((1, ex.MR, lay.hd), jnp.float8_e4m3fn),
            jnp.zeros((1, NS, 8, qkv), jnp.float32),
            jnp.zeros((1, ex.B * vh, lay.KD, lay.VD), jnp.bfloat16),
            ckpt,
            conv_partial=True,
        )
