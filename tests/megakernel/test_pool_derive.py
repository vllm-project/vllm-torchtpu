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
"""Unit tests for `derive_pool_layout` (`pool.derive`) and `from_cache_decls`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_torchtpu.megakernel.decls import KVCacheDecl, RecurrentCacheDecl
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.pool import PoolLayout, derive_pool_layout
from vllm_torchtpu.megakernel.pool.layout import (
    VALIDATED_CONV_PAGES_PER_SLOT,
    VALIDATED_REC_PAGES_PER_SLOT,
)


@dataclass(frozen=True)
class _CacheDeclHolder:
    decl: KVCacheDecl | RecurrentCacheDecl | None

    def cache_decl(self) -> KVCacheDecl | RecurrentCacheDecl | None:
        return self.decl


def test_derive_interleaved_sequence_and_unwrapping():
    """Recurrent decls group onto the next attention decl; `.cache_decl()` unwraps."""
    kv = KVCacheDecl(hd=256, kvh=1, page_size=128)
    rec = RecurrentCacheDecl(kh=16, vh=32, KD=128, VD=128, conv_taps=3)
    seq = [
        None,
        _CacheDeclHolder(None),
        _CacheDeclHolder(rec),
        rec,
        _CacheDeclHolder(rec),
        _CacheDeclHolder(kv),
        None,
    ] * 2

    lay = derive_pool_layout(seq, R=8)
    assert PoolLayout.from_cache_decls(seq, R=8) == lay
    lay.validate()
    assert lay.num_pools == 2
    assert lay.slots_per_pool == 3
    assert lay.hd == 256 and lay.kvh == 1 and lay.page_size == 128
    assert lay.page_dtype == FP8
    assert lay.kh == 16 and lay.vh == 32 and lay.KD == 128 and lay.VD == 128
    assert lay.rec_dtype == BF16 and lay.conv_dtype == BF16
    assert lay.rec_pages_per_slot == VALIDATED_REC_PAGES_PER_SLOT
    assert lay.conv_pages_per_slot == VALIDATED_CONV_PAGES_PER_SLOT


def test_derive_kv_only_and_explicit_overrides():
    """KV-only sequences need no `R`; direct and override forms respect overrides."""
    kv = KVCacheDecl(hd=256)
    rec = RecurrentCacheDecl(kh=16, vh=32)

    kv_only = derive_pool_layout([None, kv, _CacheDeclHolder(kv), None])
    assert PoolLayout.from_cache_decls([kv, kv]) == kv_only
    kv_only.validate()
    assert kv_only.num_pools == 2 and kv_only.slots_per_pool == 0
    assert not kv_only.has_recurrent

    direct = PoolLayout.from_cache_decls(
        _CacheDeclHolder(kv),
        _CacheDeclHolder(rec),
        num_pools=4,
        slots_per_pool=3,
        R=8,
    )
    direct.validate()
    assert direct.num_pools == 4 and direct.slots_per_pool == 3
    assert direct.has_recurrent

    zero_slots = derive_pool_layout(kv, rec, num_pools=3, slots_per_pool=0)
    assert zero_slots.num_pools == 3 and zero_slots.slots_per_pool == 0
    assert not zero_slots.has_recurrent

    overridden = derive_pool_layout(
        [rec, kv, rec, rec], num_pools=5, slots_per_pool=2, R=8
    )
    assert overridden.num_pools == 5 and overridden.slots_per_pool == 2


def test_derive_recurrent_page_sizing_and_explicit_fields():
    """`rec_pages_per_slot=None` scales layout geometry; `validate()` gates >4 pages."""
    kv = KVCacheDecl(hd=256)
    shrunk_rec = RecurrentCacheDecl(kh=2, vh=4, KD=128, VD=128)
    assert (
        derive_pool_layout([shrunk_rec, kv], R=8).rec_pages_per_slot
        == VALIDATED_REC_PAGES_PER_SLOT
    )

    # Layout derivation scales page counts beyond 4 for wider per-rank state,
    # while `PoolLayout.validate()` enforces the currently validated 4-page TPU
    # pool-kernel constraint with an actionable tensor-parallel `R` hint.
    wide_rec = RecurrentCacheDecl(kh=32, vh=128, KD=128, VD=128)
    wide_r8 = derive_pool_layout([wide_rec, kv], R=8)
    wide_r4 = derive_pool_layout([wide_rec, kv], R=4)
    assert wide_r8.rec_pages_per_slot == 8
    assert wide_r4.rec_pages_per_slot == 16
    with pytest.raises(
        MegakernelConfigError,
        match=r"rec_pages_per_slot=8.*increase tensor-parallel R",
    ):
        wide_r8.validate()

    explicit_kv = KVCacheDecl(hd=256, kvh=2, page_size=64, page_dtype=BF16)
    explicit_rec = RecurrentCacheDecl(
        kh=8,
        vh=16,
        KD=64,
        VD=64,
        rec_dtype=F32,
        rec_pages_per_slot=6,
        conv_taps=4,
        conv_dtype=F32,
        conv_pages_per_slot=2,
    )
    custom = derive_pool_layout([explicit_rec, explicit_kv])
    assert custom.page_dtype == BF16
    assert custom.rec_dtype == F32 and custom.conv_dtype == F32
    assert custom.rec_pages_per_slot == 6 and custom.conv_pages_per_slot == 2
    assert custom.conv_taps == 4 and custom.page_size == 64 and custom.kvh == 2


def test_derive_rejects_invalid_declarations():
    """Invalid sequences, mismatched decls, and missing `num_pools`/`R` fail cleanly."""
    kv = KVCacheDecl(hd=256)
    rec = RecurrentCacheDecl(kh=16, vh=32)

    with pytest.raises(MegakernelConfigError, match="unknown cache decl"):
        derive_pool_layout([object(), kv])
    with pytest.raises(MegakernelConfigError, match="at least one KVCacheDecl"):
        derive_pool_layout([None, rec], R=8)
    with pytest.raises(MegakernelConfigError, match="uniform KVCacheDecls"):
        derive_pool_layout([kv, KVCacheDecl(hd=128)])
    with pytest.raises(MegakernelConfigError, match="uniform RecurrentCacheDecls"):
        derive_pool_layout([rec, kv, RecurrentCacheDecl(kh=8, vh=16), kv], R=8)
    with pytest.raises(MegakernelConfigError, match="trailing recurrent"):
        derive_pool_layout([rec, kv, rec], R=8)
    with pytest.raises(MegakernelConfigError, match="uniform recurrent slot counts"):
        derive_pool_layout([rec, kv, rec, rec, kv], R=8)
    with pytest.raises(MegakernelConfigError, match="explicit num_pools"):
        derive_pool_layout(kv)
    with pytest.raises(MegakernelConfigError, match="requires explicit slots_per_pool"):
        derive_pool_layout(kv, rec, num_pools=3)
    with pytest.raises(MegakernelConfigError, match="requires explicit slots_per_pool"):
        derive_pool_layout([kv, kv], rec)
    with pytest.raises(MegakernelConfigError, match="without a RecurrentCacheDecl"):
        derive_pool_layout([kv, kv], slots_per_pool=2)
    with pytest.raises(MegakernelConfigError, match="num_pools >= 1"):
        derive_pool_layout(kv, num_pools=0)
    with pytest.raises(MegakernelConfigError, match="slots_per_pool >= 0"):
        derive_pool_layout(kv, rec, num_pools=1, slots_per_pool=-1, R=8)
    with pytest.raises(MegakernelConfigError, match="requires a KVCacheDecl"):
        derive_pool_layout("not_a_sequence_of_decls", num_pools=1)
    with pytest.raises(
        MegakernelConfigError, match="requires a RecurrentCacheDecl or None"
    ):
        derive_pool_layout(kv, "not_rec", num_pools=1, slots_per_pool=1)
    with pytest.raises(
        MegakernelConfigError, match="requires a RecurrentCacheDecl or None"
    ):
        derive_pool_layout(kv, "not_rec", num_pools=1, slots_per_pool=0)
    for bad_r in (None, 0, -2):
        with pytest.raises(MegakernelConfigError, match="requires explicit R > 0"):
            PoolLayout.from_cache_decls([rec, kv], R=bad_r)
