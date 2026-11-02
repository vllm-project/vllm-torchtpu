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

import pytest

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool import CacheExec, PoolLayout, with_cache

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
    for bad in (
        replace(_layout(3), hd=128),
        replace(_layout(3), page_size=64),
        replace(_layout(3), kvh=2),
        replace(_layout(3), rec_pages_per_slot=2),
        replace(_layout(3), conv_taps=4),
        replace(_layout(3), num_pools=0),
    ):
        with pytest.raises(MegakernelConfigError):
            bad.validate()
    with pytest.raises(MegakernelConfigError, match="page_size"):
        _layout(3).validate(with_cache(_exec(R=2, B=2, T=1), page_size=64))
    with pytest.raises(
        MegakernelConfigError,
        match=r"recurrent heads \(kh=2, vh=4\) must be divisible by R=8",
    ):
        _layout(3).validate(_exec(R=8, B=8, T=1))


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
