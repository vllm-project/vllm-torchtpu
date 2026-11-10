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
"""Split pool I/O on a 5-page slot whose conv window follows a smaller state.

TorchTPU's vLLM packs a GDN checkpoint slot from the start: the recurrent
state, then the conv window (`vllm_torchtpu.gdn_pool_layout`).
`PoolLayout.with_packed_slots` lays a slot out the same way, so 4 value heads
per rank (Qwen3.6-35B-A3B at R=8) give 2 recurrent pages and the conv window on
page 2, against 4 + 1 for 8 heads (Qwen3.5-397B). These run the gather and
scatter kernels in interpret mode on that geometry against the reference packer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.core import test_pool_scatter as ts
from vllm_torchtpu.megakernel.core import pool
from vllm_torchtpu.megakernel.core.decls import KVCacheDecl, RecurrentCacheDecl
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.ops import BF16, F32


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _derived(
    R: int, kh_per_rank: int, vh_per_rank: int, slots: int = 3, conv: Any = BF16
) -> pool.PoolLayout:
    rec = RecurrentCacheDecl(kh=kh_per_rank * R, vh=vh_per_rank * R)
    decls = ([rec] * slots + [KVCacheDecl(hd=256)]) * 2
    return pool.derive_pool_layout(decls, R=R, conv_dtype=conv)


def _packed(R: int, slots: int = 3, conv: Any = BF16) -> pool.PoolLayout:
    """2 key and 4 value heads per rank, packed: 2 + 3 pages per slot."""
    lay = _derived(R, 2, 4, slots, conv).with_packed_slots(R)
    assert (lay.rec_pages_per_slot, lay.conv_page, lay.pages_per_slot) == (2, 2, 5)
    return lay


@pytest.mark.parametrize(
    "vh_per_rank,rec_pages", [(1, 1), (2, 1), (3, 2), (4, 2), (6, 3), (8, 4)]
)
def test_with_packed_slots_shrinks_the_state_region(vh_per_rank: int, rec_pages: int):
    """The state region becomes the pages the state fills (two 128 x 128 bf16
    value heads a page) and the conv region the rest of the 5-page slot."""
    R = 8
    derived = _derived(R, 1, vh_per_rank)
    assert (derived.rec_pages_per_slot, derived.conv_pages_per_slot) == (4, 1)
    lay = derived.with_packed_slots(R)
    assert lay.rec_pages_per_slot == lay.conv_page == rec_pages
    assert lay.pages_per_slot == 5
    rest = 5 - rec_pages
    assert lay == replace(
        derived, rec_pages_per_slot=rec_pages, conv_pages_per_slot=rest
    )
    assert lay.with_packed_slots(R) == lay
    assert lay.state_fills_whole_pages(R) == (vh_per_rank % 2 == 0)
    lay.validate(ts._exec(R=R, B=8, T=1))
    kv_only = pool.derive_pool_layout([KVCacheDecl(hd=256)] * 2)
    assert kv_only.with_packed_slots(R) == kv_only
    assert kv_only.state_fills_whole_pages(R)


def test_validate_accepts_state_packed_five_page_slots():
    ex = ts._exec(R=1, B=2, T=1)
    _packed(1).validate(ex)
    ts._layout(3).validate(ex)  # 4 + 1
    for rec, conv in ((2, 1), (2, 4), (5, 0), (0, 5)):
        bad = replace(_packed(1), rec_pages_per_slot=rec, conv_pages_per_slot=conv)
        with pytest.raises(MegakernelConfigError, match="5-page slots"):
            bad.validate(ex)
    with pytest.raises(MegakernelConfigError, match="per-rank recurrent state"):
        replace(_packed(1), kh=4, vh=8).validate(ex)  # 8 heads need 4 pages


def _check_gather(lay: pool.PoolLayout, ex: Any, ckpt_ndim: int, seed: int) -> None:
    """Gather rank 0's conv window and state; compare with the packed states."""
    st = ts._random_states(lay, ex, seed=seed)
    pt, ckpt, roff = ts._tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(ts._unfold(r, ex.R)[0] for r in raw_pools)

    conv_out, rec = pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff)
    B = ex.B
    taps = np.asarray(st["conv"][0])  # (ngdn, NS, 3, QKV)
    exp = np.zeros(conv_out.shape, np.float32)
    if ex.T == 1:
        exp[:, : 3 * B] = np.swapaxes(taps, 1, 2).reshape(taps.shape[0], 3 * B, -1)
    else:
        exp[:, :, :3] = taps
    np.testing.assert_array_equal(np.asarray(conv_out), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))


@pytest.mark.parametrize("conv", [BF16, F32])
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("ckpt_ndim", [3, 2])
def test_gather_reads_the_packed_slot(ckpt_ndim: int, T: int, conv: Any):
    _check_gather(_packed(2, conv=conv), ts._exec(R=2, B=2 * T, T=T), ckpt_ndim, 71 + T)


@pytest.mark.parametrize("conv", [BF16, F32])
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("ckpt_ndim", [3, 2])
def test_scatter_writes_the_packed_slot(ckpt_ndim: int, T: int, conv: Any):
    lay = _packed(1, conv=conv)
    ts._check_scatter(lay, ts._exec(R=1, B=2 * T, T=T), ckpt_ndim, 81 + T)


def test_scatter_cold_start_on_the_packed_slot():
    ts._check_scatter(_packed(1), ts._exec(R=1, B=2, T=1), 3, 91, cold_start=True)



def _scatter_states(lay: pool.PoolLayout, ex: Any, seed: int) -> dict:
    """Scatter random new KV, conv and state on rank 0 of 1; read the pool back."""
    NS, B, T = ex.NS, ex.B, ex.T
    st = ts._random_states(lay, ex, seed=seed)
    pt, ckpt, roff = ts._tables(lay, ex, 3)
    raw = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    k = jax.random.split(jax.random.key(seed + 10), 4)
    pos = jnp.asarray(
        [10 + 129 * (s % 2) + 3 * (s // 2) + t for s in range(NS) for t in range(T)],
        dtype=jnp.int32,
    )
    knew = (jax.random.normal(k[0], (lay.num_pools, ex.MR, lay.hd)) * 0.3).astype(
        jnp.float8_e4m3fn
    )
    vnew = (jax.random.normal(k[1], (lay.num_pools, ex.MR, lay.hd)) * 0.3).astype(
        jnp.float8_e4m3fn
    )
    ngdn, vh, qkv = lay.num_recurrent_layers, lay.vh_per_rank(1), lay.qkv_per_rank(1)
    conv_shape = (
        (ngdn, 8 * ((3 * B + 7) // 8), qkv)
        if T == 1
        else (ngdn, NS, 8 * ((3 + T + 7) // 8), qkv)
    )
    conv = ts._bf16_f32(k[2], conv_shape)
    rec = (jax.random.normal(k[3], (ngdn, B * vh, lay.KD, lay.VD)) * 0.1).astype(
        jnp.bfloat16
    )
    out = pool.scatter_pool_states(
        lay, ex, tuple(raw), pos, pt, knew, vnew, conv, rec, ckpt
    )
    return pool.pool_to_states(lay, ex, tuple(out), pt, ckpt)


@pytest.mark.tpu_test
@pytest.mark.parametrize("conv", [BF16, F32])
@pytest.mark.parametrize("T", [1, 2])
def test_gather_and_scatter_on_the_packed_slot_on_tpu(T: int, conv: Any):
    """On TPU (`interpret=False`), the lowered gather kernel reads the 2 + 3
    slot, and the lowered scatter kernel (recurrent and conv page writes)
    leaves the same states in it as in the 4 + 1 slot."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("Requires TPU device for Mosaic lowering")
    ex = replace(ts._exec(R=1, B=2 * T, T=T), interpret=False)
    with jax.default_device(tpu_devs[0]):
        _check_gather(_packed(1, conv=conv), ex, 3, 171 + T)
        packed = _scatter_states(_packed(1, conv=conv), ex, 181 + T)
        unpacked = _scatter_states(_derived(1, 2, 4, conv=conv), ex, 181 + T)
    assert packed.keys() == unpacked.keys()
    for key, val in unpacked.items():
        np.testing.assert_array_equal(ts._bytes(packed[key]), ts._bytes(val))
