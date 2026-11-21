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

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.megakernel.core import pool
from vllm_torchtpu.megakernel.core.decls import KVCacheDecl, RecurrentCacheDecl
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.core.pool import (
    CacheExec,
    PoolBackend,
    PoolLayout,
    check_step_pool_metadata,
    derive_pool_layout,
    make_pool_backend,
)
from vllm_torchtpu.megakernel.core.pool.layout import (
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


def test_derive_conv_dtype_override():
    """`conv_dtype` overrides the declared conv storage dtype; nothing else moves."""
    kv = KVCacheDecl(hd=256, kvh=1, page_size=128)
    rec = RecurrentCacheDecl(kh=16, vh=64, KD=128, VD=128, conv_taps=3, conv_dtype=BF16)
    seq = [rec, rec, rec, kv] * 2
    base = derive_pool_layout(seq, R=8)
    f32 = derive_pool_layout(seq, R=8, conv_dtype=F32)
    assert base.conv_dtype == BF16 and f32.conv_dtype == F32
    assert PoolLayout.from_cache_decls(seq, R=8, conv_dtype=F32) == f32
    assert f32 == PoolLayout(**{**base.__dict__, "conv_dtype": F32})
    f32.validate()


def test_pool_backend_with_layout_preserves_callbacks_and_flags():
    """`PoolBackend.with_layout` preserves `check`, `rec_chunk`, `supports_fused_io`,
    and `supports_pool_gather` when rebuilding over a new `PoolLayout`."""
    seen: list[ExecSpec] = []
    lay_bf16 = PoolLayout(
        num_pools=2, slots_per_pool=1, hd=256, kvh=1, kh=2, vh=4, KD=128, VD=128
    )
    lay_f32 = replace(lay_bf16, conv_dtype=F32)

    pb = make_pool_backend(
        lay_bf16,
        check=seen.append,
        rec_chunk=32,
        supports_fused_io=True,
        supports_pool_gather=False,
    )
    rebuilt = pb.with_layout(lay_f32)
    assert rebuilt.layout == lay_f32
    assert rebuilt.supports_fused_io is True
    assert rebuilt.supports_pool_gather is False

    ex = ExecSpec(
        R=1,
        B=2,
        T=1,
        ctx=256,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64),
    )
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay_f32, ex)
    raw = tuple(
        jnp.zeros((64, *lay_f32.kv_page_shape), dtype=FP8)
        for _ in range(lay_f32.num_pools)
    )
    rebuilt.gather_states(ex, raw, ckpt[0], roff[0], "all")
    assert seen == [ex]

    bare = PoolBackend(layout=lay_bf16, kernel_names=("g", "s"))
    with pytest.raises(ValueError, match="not built by make_pool_backend"):
        bare.with_layout(lay_f32)


def test_check_step_pool_metadata_all_error_paths_and_jit():
    """Cover valid input, every `ValueError` branch in `check_step_pool_metadata`,
    and traced execution under `jax.jit` with `force_under_jit=True`."""
    lay = PoolLayout(
        num_pools=2, slots_per_pool=2, hd=256, kvh=1, kh=2, vh=4, KD=128, VD=128
    )
    ex = ExecSpec(
        R=1,
        B=4,
        T=2,
        ctx=256,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=128),
    )
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex)
    pos = jnp.asarray([10, 11, 20, 21], dtype=jnp.int32)

    # 1. Valid input passes eagerly
    check_step_pool_metadata(lay, ex, pt, pos, ckpt_indices=ckpt, read_offsets=roff)

    # 2. Out-of-range pos (< 0 or >= cols * page_size)
    with pytest.raises(ValueError, match="maps to logical page"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos.at[0].set(-1),
            ckpt_indices=ckpt,
            read_offsets=roff,
        )
    with pytest.raises(ValueError, match="maps to logical page"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos.at[0].set(pt.shape[1] * lay.page_size),
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 3. Invalid KV write page (<= 0 or >= num_pages)
    with pytest.raises(ValueError, match="KV write page .* outside valid range"):
        check_step_pool_metadata(
            lay,
            ex,
            pt.at[0, 0].set(0),
            pos,
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 4. Duplicate KV write page across active sequences
    with pytest.raises(ValueError, match="duplicate KV write page"):
        check_step_pool_metadata(
            lay,
            ex,
            pt.at[1, 0].set(pt[0, 0]),
            pos,
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 5. Duplicate KV write (page, lane) within a sequence
    with pytest.raises(ValueError, match="duplicate KV write .* within sequence"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            jnp.asarray([10, 10, 20, 21], dtype=jnp.int32),
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 6. Invalid prior KV read page in active_kv_pages (pos on page 1, page 0 invalid)
    with pytest.raises(ValueError, match="logical page 0 has invalid KV page"):
        check_step_pool_metadata(
            lay,
            ex,
            pt.at[0, 0].set(0),
            jnp.asarray([130, 131, 20, 21], dtype=jnp.int32),
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 7. ckpt_indices K < T
    with pytest.raises(ValueError, match="ckpt_indices K=1 < T=2"):
        check_step_pool_metadata(lay, ex, pt, pos, ckpt_indices=ckpt[..., :1])

    # 8. Non-distinct or non-positive checkpoint write slots for T > 1
    with pytest.raises(ValueError, match="distinct positive checkpoint write slots"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos,
            ckpt_indices=ckpt.at[0, 0, 1].set(ckpt[0, 0, 0]),
        )

    # 9. Out-of-range checkpoint write slot (for T=1)
    ex_t1 = replace(ex, B=2, T=1)
    pt1, ckpt1, roff1, _ = pool.allocate_test_layout(lay, ex_t1)
    pos1 = jnp.asarray([10, 20], dtype=jnp.int32)
    with pytest.raises(ValueError, match="checkpoint write slot .* outside"):
        check_step_pool_metadata(
            lay,
            ex_t1,
            pt1,
            pos1,
            ckpt_indices=ckpt1.at[0, 0, -1].set(999),
        )

    # 10. Duplicate checkpoint write slot pages across sub-layers or sequences
    with pytest.raises(ValueError, match="duplicate checkpoint write slot pages"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos,
            ckpt_indices=ckpt.at[0, 1, 0].set(ckpt[0, 0, 0]),
        )

    # 11. read_offset >= K and out-of-range checkpoint read slot
    with pytest.raises(ValueError, match="read_offset 5 >= K="):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos,
            ckpt_indices=ckpt,
            read_offsets=roff.at[0, 0].set(5),
        )
    extra_slot = jnp.full((lay.slots_per_pool, ex.NS, 1), 999, dtype=jnp.int32)
    ckpt_with_bad_read = jnp.concatenate([extra_slot, ckpt], axis=-1)
    with pytest.raises(ValueError, match="checkpoint read slot 999 .* outside"):
        check_step_pool_metadata(
            lay,
            ex,
            pt,
            pos,
            ckpt_indices=ckpt_with_bad_read,
            read_offsets=jnp.zeros_like(roff),
        )

    # 12. Checkpoint slot pages overlapping active KV pages
    ckpt_page0 = int(lay.page_base(int(ckpt[0, 0, 0]), 0, 3))
    with pytest.raises(ValueError, match="overlap active KV pages"):
        check_step_pool_metadata(
            lay,
            ex,
            pt.at[0, 0].set(ckpt_page0),
            pos,
            ckpt_indices=ckpt,
            read_offsets=roff,
        )

    # 13. Traced call under jax.jit with force_under_jit=True runs the callback
    @jax.jit
    def _traced_check(pt_arg, pos_arg, ck_arg, ro_arg):
        check_step_pool_metadata(
            lay,
            ex,
            pt_arg,
            pos_arg,
            ckpt_indices=ck_arg,
            read_offsets=ro_arg,
            force_under_jit=True,
        )
        return pos_arg + 1

    out = _traced_check(pt, pos, ckpt, roff)
    out.block_until_ready()
    with pytest.raises(
        (ValueError, RuntimeError, jax.errors.JaxRuntimeError),
        match="duplicate KV write",
    ):
        bad_out = _traced_check(
            pt, jnp.asarray([10, 10, 20, 21], dtype=jnp.int32), ckpt, roff
        )
        bad_out.block_until_ready()
