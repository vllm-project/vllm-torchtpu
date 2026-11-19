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
"""Unit tests for `make_pool_backend` and shared pool checks (`pool.backend`)."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool import (
    CacheExec,
    PoolLayout,
    cache_exec,
    make_pool_backend,
    with_cache,
)
from vllm_torchtpu.megakernel.pool.backend import resolve_active_seqs

NUM_PAGES = 128
SLOTS = {"slots3": 3, "slots1": 1, "kv_only": 0}
_TINY_TILES = Tiles(bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _layout(slots: int, num_pools: int = 2) -> PoolLayout:
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
def test_make_pool_backend_derived_properties(name: str):
    slots = SLOTS[name]
    lay = _layout(slots)
    backend = make_pool_backend(lay)
    assert backend.layout is lay and backend.num_pools == 2
    assert backend.has_recurrent_pool == (slots > 0)
    assert backend.recurrent_slots_per_pool == max(1, slots)
    assert backend.ckpt_pages_per_slot == 5
    assert backend.kernel_names == pool.POOL_KERNEL_NAMES


@pytest.mark.parametrize("kh", [2, 4, 6, 8])
@pytest.mark.parametrize("T", [2, 4, 5, 6, 8])
def test_layout_validate_and_can_pool_gather_agree_on_gpr_conv_window(kh: int, T: int):
    lay = replace(_layout(3), kh=kh, vh=kh)
    ex = with_cache(_exec(R=1, B=8 * T, T=T), pool_gather=True)
    fits = lay.gpr_conv_fits(T, R=1)
    assert pool.can_pool_gather(lay, ex, lambda _ex: (8 * T, 2)) == fits
    if fits:
        lay.validate(ex)
    else:
        with pytest.raises(MegakernelConfigError, match="in-kernel gather"):
            lay.validate(ex)


def test_make_pool_backend_supports_pool_gather_only_with_rec_chunk():
    lay = _layout(3)
    assert not make_pool_backend(lay).supports_pool_gather
    assert make_pool_backend(lay, rec_chunk=lambda ex: (8, 2)).supports_pool_gather
    assert not make_pool_backend(
        lay, rec_chunk=lambda ex: (8, 2), supports_pool_gather=False
    ).supports_pool_gather
    with pytest.raises(MegakernelConfigError, match="rec_chunk"):
        make_pool_backend(lay, supports_pool_gather=True)


def test_kv_only_backend_skips_gather():
    lay = _layout(0)
    backend = make_pool_backend(lay)
    assert backend.gather_states(_exec(R=1, B=2, T=1), (), None, None, None) == {}


def test_ckpt_bounds_rejects_out_of_range_and_shared_slots():
    lay = _layout(2, num_pools=3)
    ex = _exec(R=1, B=4, T=2)
    _, ckpt, _, _ = pool.allocate_test_layout(lay, ex)
    pool.check_ckpt_bounds(ckpt, cache_exec(ex).num_pages, lay)
    past_end = cache_exec(ex).num_pages // lay.pages_per_slot
    with pytest.raises(ValueError, match="exceeds pool capacity"):
        pool.check_ckpt_bounds(
            ckpt.at[1, 0, 1].set(past_end), cache_exec(ex).num_pages, lay
        )
    with pytest.raises(ValueError, match="shared between"):
        pool.check_ckpt_bounds(
            ckpt.at[1, 0, 0].set(ckpt[0, 0, 0]), cache_exec(ex).num_pages, lay
        )


def test_resolve_active_seqs_modes():
    assert resolve_active_seqs(None) == ("from_ckpt", None)
    assert resolve_active_seqs("all") == ("all", None)
    mode, mask = resolve_active_seqs(jnp.array([1, 0], dtype=jnp.int32))
    assert mode == "ref" and mask is not None and mask.dtype == jnp.int32
    with pytest.raises(ValueError, match="Unsupported active_seqs"):
        resolve_active_seqs("invalid")
