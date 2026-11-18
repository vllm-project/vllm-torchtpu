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
"""Unit tests for `PoolLayout` (`pool.layout`)."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.pool import CacheExec, PoolLayout, with_cache

NUM_PAGES = 128
# Recurrent slots per pool: 3-slot, 1-slot, and KV-only (0 slots).
SLOTS = {"slots3": 3, "slots1": 1, "kv_only": 0}
_TINY_TILES = Tiles(bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)


def _layout(slots: int, num_pools: int = 2) -> PoolLayout:
    """Recurrent layouts use kh=2, vh=4; a KV-only layout has no recurrent heads."""
    return PoolLayout(
        num_pools=num_pools,
        slots_per_pool=slots,
        hd=256,
        kvh=1,
        kh=2 if slots else 0,
        vh=4 if slots else 0,
        KD=128,
        VD=128,
    )


def _exec(R: int, B: int, T: int, ctx: int = 256) -> ExecSpec:
    return ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=True,
        interpret=True,
        tiles=_TINY_TILES,
        cache=CacheExec(paged_kv=True, num_pages=NUM_PAGES),
    )


@pytest.mark.parametrize("name", list(SLOTS))
def test_layout_derived_geometry(name: str):
    slots = SLOTS[name]
    lay = _layout(slots)
    ex = _exec(R=2, B=2, T=1)
    lay.validate(ex)
    assert lay.pages_per_slot == 5 and lay.conv_page == 4
    assert lay.has_recurrent == (slots > 0)
    assert lay.num_recurrent_layers == 2 * slots
    assert lay.recurrent_slots_per_pool == max(1, slots)
    assert lay.kv_page_shape == (2, 64, 4, 128)
    assert lay.page_bytes == 65536 and lay.page_rows == 128
    assert lay.conv_page_rows == 256 and lay.rec_u32_rows == 512
    assert lay.reserved_pages == max(1, 5 * slots)
    assert lay.page_base(7, 2, 3) == 35
    assert lay.page_base(7, 2, 2) == 7 * 5 * slots + 10
    assert lay.min_num_pages(ex) >= lay.reserved_pages + ex.NS * (
        ex.ctx // lay.page_size
    )


def test_layout_validate_rejects_unvalidated_geometry():
    """Pages hold whole 128-lane tiles; recurrent slots, sized in pages, are
    validated at 128-token pages only (fused pool I/O of KV-only pages at any
    page size)."""
    for bad in (
        replace(_layout(3), hd=128),
        replace(_layout(3), page_size=64),
        replace(_layout(0), page_size=192),
        replace(_layout(3), page_size=512),
        replace(_layout(3), kvh=2),
        replace(_layout(3), rec_pages_per_slot=2),
        replace(_layout(3), conv_taps=4),
        replace(_layout(3), num_pools=0),
    ):
        with pytest.raises(MegakernelConfigError):
            bad.validate()
    with pytest.raises(MegakernelConfigError, match="page_size"):
        _layout(3).validate(with_cache(_exec(R=2, B=2, T=1), page_size=64))
    ex512 = with_cache(_exec(R=2, B=2, T=1, ctx=1024), page_size=512)
    replace(_layout(0), page_size=512).validate(ex512)
    replace(_layout(0), page_size=512).validate(with_cache(ex512, fused_pool_io=True))
    with pytest.raises(
        MegakernelConfigError,
        match=r"recurrent heads \(kh=2, vh=4\) must be divisible by R=8",
    ):
        _layout(3).validate(_exec(R=8, B=8, T=1))


@pytest.mark.parametrize("page_size", [128, 512])
def test_layout_bf16_kv_pages_are_the_stock_rpa_pages(page_size: int):
    """bf16 hd128 KV-only pages validate and equal the stock batched-RPA
    `SEQ_ALONG_LANE` page at the page size vLLM picks (128, or 512 at 256
    sequences of 262144 tokens), so the stock and megakernel steps share one
    cache, also with fused pool I/O; recurrent slots stay fp8-only."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    lay = PoolLayout(num_pools=2, hd=128, page_size=page_size, page_dtype=jnp.bfloat16)
    ex = _exec(R=8, B=8, T=1, ctx=2 * page_size)
    ex = with_cache(ex, page_size=page_size)
    lay.validate(ex)
    stock = wrapper.get_kv_cache_shape(
        NUM_PAGES, page_size, 1, 128, jnp.bfloat16, configs.KVLayout.SEQ_ALONG_LANE
    )
    assert (NUM_PAGES, *lay.kv_page_shape) == stock == (NUM_PAGES, 2, 64, 2, page_size)
    for bad in (replace(lay, slots_per_pool=1, kh=8, vh=8), replace(lay, hd=256)):
        with pytest.raises(MegakernelConfigError):
            bad.validate()
    lay.validate(with_cache(ex, fused_pool_io=True))


def test_layout_validate_checks_in_kernel_conv_windows():
    """GPR stages the verify conv window in the K half of the conv page (64
    fp8 rows = 128 bf16 rows), fused pool I/O in the whole conv page (256 rows)."""
    lay = replace(_layout(3), kh=8, vh=8)  # 24 conv rows of 128 lanes per rank
    split = with_cache(_exec(R=1, B=6, T=6), paged_kv=True)
    lay.validate(split)  # split path: 3 taps fit
    assert lay.gpr_conv_rows(4, R=1) == 96 and not lay.gpr_conv_fits(4, R=1)
    with pytest.raises(MegakernelConfigError, match="in-kernel gather"):
        lay.validate(
            with_cache(replace(split, B=4, T=4), pool_gather=True)
        )  # 8 * 24 = 192 bf16 rows (96 fp8 page rows > 64)
    narrow = replace(_layout(3), kh=4, vh=4)  # 12 conv rows per rank
    assert narrow.gpr_conv_rows(4, R=1) == 48 and narrow.gpr_conv_fits(4, R=1)
    narrow.validate(
        with_cache(replace(split, B=4, T=4), pool_gather=True)
    )  # 8 * 12 = 96 bf16 rows (48 fp8 page rows <= 64)
    assert narrow.gpr_conv_rows(6, R=1) == 96 and not narrow.gpr_conv_fits(6, R=1)
    with pytest.raises(MegakernelConfigError, match="in-kernel gather"):
        narrow.validate(with_cache(split, pool_gather=True))  # 16 * 12 rows
    narrow.validate(
        with_cache(replace(split, B=4, T=4), fused_pool_io=True)
    )  # 16 * 12 = 192 rows
    with pytest.raises(MegakernelConfigError, match="fused pool I/O"):
        lay.validate(
            with_cache(replace(split, B=1, T=1), fused_pool_io=True)
        )  # 16 * 24 rows


def test_layout_f32_conv_geometry_and_in_kernel_windows():
    """An f32 conv window (the serving framework's pooled conv dtype) keeps the
    slot geometry; its rows are one u32 row each, so a conv page holds 128.
    GPR copies the whole conv page for f32 conv (128 page rows) and fused pool
    I/O stages `round_up(taps + T, 8)` f32 rows, so at Qwen3.5 TP8 both fit up to
    T=5 and fail closed past it."""
    bf16 = replace(_layout(3), kh=16, vh=64)  # Qwen3.5: 12 conv rows per rank at R=8
    f32 = replace(bf16, conv_dtype=jnp.float32)
    assert (bf16.conv_itemsize, f32.conv_itemsize) == (2, 4)
    assert (bf16.conv_tile, f32.conv_tile) == (16, 8)
    assert (bf16.conv_page_rows, f32.conv_page_rows) == (256, 128)
    assert bf16.conv_rows(8) == f32.conv_rows(8) == 36
    assert (bf16.conv_rows_pad(8), f32.conv_rows_pad(8)) == (48, 40)
    assert (bf16.conv_bytes(8), f32.conv_bytes(8)) == (9216, 18432)
    assert f32.pages_per_slot == bf16.pages_per_slot == 5 and f32.conv_page == 4
    split = with_cache(_exec(R=8, B=8, T=4), paged_kv=True)
    f32.validate(split)
    assert (bf16.gpr_conv_row_limit, f32.gpr_conv_row_limit) == (64, 128)
    assert f32.gpr_conv_rows(4, R=8) == 96 and f32.gpr_conv_fits(4, R=8)
    assert bf16.gpr_conv_rows(4, R=8) == 48 and bf16.gpr_conv_fits(4, R=8)
    f32.validate(with_cache(split, pool_gather=True))
    assert (bf16.fused_conv_rows(4, 8), f32.fused_conv_rows(4, 8)) == (192, 96)
    f32.validate(with_cache(split, fused_pool_io=True))  # 8 * 12 <= 128
    wide = with_cache(_exec(R=8, B=8, T=6), paged_kv=True)  # round_up(3 + 6, 8) = 16
    assert f32.gpr_conv_rows(6, R=8) == 192 and not f32.gpr_conv_fits(6, R=8)
    with pytest.raises(MegakernelConfigError, match="in-kernel gather"):
        f32.validate(with_cache(wide, pool_gather=True))
    with pytest.raises(MegakernelConfigError, match="fused pool I/O"):
        f32.validate(with_cache(wide, fused_pool_io=True))  # 16 * 12 > 128
    with pytest.raises(MegakernelConfigError, match="bf16 or f32"):
        replace(bf16, conv_dtype=jnp.float16).validate()
    with pytest.raises(MegakernelConfigError, match="recurrent checkpoints"):
        replace(bf16, rec_dtype=jnp.float32).validate()
