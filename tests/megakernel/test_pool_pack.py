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
"""Unit tests for host / test pool packing (`pool.pack`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool import CacheExec, PoolLayout, cache_exec

FP8 = jnp.float8_e4m3fn
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


def _bf16_f32(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    x = jax.random.normal(key, shape, dtype=jnp.float32) * 0.1
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _random_states(lay: PoolLayout, ex: ExecSpec, seed: int) -> dict[str, jax.Array]:
    R, NS = ex.R, ex.NS
    k = jax.random.split(jax.random.key(seed), 4)
    kv_shape = (R, lay.num_pools, NS, ex.ctx, lay.hd)
    st = {
        "kc": (jax.random.normal(k[0], kv_shape) * 0.3).astype(FP8),
        "vc": (jax.random.normal(k[1], kv_shape) * 0.3).astype(FP8),
    }
    if lay.has_recurrent:
        ngdn, vh = lay.num_recurrent_layers, lay.vh_per_rank(R)
        qkv = lay.qkv_per_rank(R)
        st["conv"] = _bf16_f32(k[2], (R, ngdn, NS, lay.conv_taps, qkv))
        st["rec"] = (
            jax.random.normal(k[3], (R, ngdn, NS * vh, lay.KD, lay.VD)) * 0.1
        ).astype(jnp.bfloat16)
    return st


def _tables(
    lay: PoolLayout, ex: ExecSpec, ckpt_ndim: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    pt, ckpt, roff, min_pages = pool.allocate_test_layout(lay, ex)
    assert cache_exec(ex).num_pages >= min_pages
    assert (
        int(jnp.min(pt[:, : ex.ctx // cache_exec(ex).page_size])) >= lay.reserved_pages
    )
    if lay.has_recurrent and ckpt_ndim == 2:
        ckpt, roff = ckpt[0], roff[0]
    return pt, ckpt, roff


def _bytes(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


@pytest.mark.parametrize("name", list(SLOTS))
def test_page_pack_unpack_roundtrip(name: str):
    lay = _layout(SLOTS[name])
    R = 2
    k = jax.random.split(jax.random.key(3), 4)
    kp = (jax.random.normal(k[0], (lay.page_size, lay.hd)) * 0.3).astype(FP8)
    vp = (jax.random.normal(k[1], (lay.page_size, lay.hd)) * 0.3).astype(FP8)
    raw = pool.kv_to_raw_page(kp, vp)
    assert raw.shape == lay.kv_page_shape
    k_rt, v_rt = pool.raw_page_to_kv(raw)
    np.testing.assert_array_equal(_bytes(k_rt), _bytes(kp))
    np.testing.assert_array_equal(_bytes(v_rt), _bytes(vp))

    vh, qkv = lay.vh_per_rank(R), lay.qkv_per_rank(R)
    rec = (jax.random.normal(k[2], (vh, lay.KD, lay.VD)) * 0.1).astype(jnp.bfloat16)
    raw_rec = pool.ssm_to_raw_pages(rec, lay)
    assert raw_rec.shape == (lay.rec_pages_per_slot, *lay.kv_page_shape)
    rec_rt = pool.raw_pages_to_ssm(raw_rec, vh, lay.KD, lay.VD, lay)
    np.testing.assert_array_equal(np.asarray(rec_rt), np.asarray(rec))

    taps = _bf16_f32(k[3], (lay.conv_taps, qkv))
    raw_conv = pool.conv_to_raw_page(taps, lay)
    assert raw_conv.shape == lay.kv_page_shape
    np.testing.assert_array_equal(
        np.asarray(pool.raw_page_to_conv(raw_conv, qkv, lay)), np.asarray(taps)
    )


@pytest.mark.parametrize(
    "name,ckpt_ndim",
    [("slots3", 3), ("slots3", 2), ("slots1", 3), ("slots1", 2), ("kv_only", 3)],
)
def test_states_to_pool_roundtrip(name: str, ckpt_ndim: int):
    lay = _layout(SLOTS[name])
    ex = _exec(R=2, B=2, T=1)
    st = _random_states(lay, ex, seed=11 + SLOTS[name])
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    assert len(raw_pools) == lay.num_pools
    assert raw_pools[0].shape == (
        ex.R * cache_exec(ex).num_pages,
        *lay.kv_page_shape,
    )
    out = pool.pool_to_states(lay, ex, raw_pools, pt, ckpt)
    np.testing.assert_array_equal(_bytes(out["kc"]), _bytes(st["kc"]))
    np.testing.assert_array_equal(_bytes(out["vc"]), _bytes(st["vc"]))
    if not lay.has_recurrent:
        assert "rec" not in out and "conv" not in out
        return
    np.testing.assert_array_equal(np.asarray(out["rec"]), np.asarray(st["rec"]))
    np.testing.assert_array_equal(
        np.asarray(out["conv"][:, :, :, 0]), np.asarray(st["conv"])
    )


def test_pack_rejects_ctx_not_divisible_by_page_size():
    lay = _layout(0)
    ex = _exec(R=1, B=1, T=1, ctx=256)
    st = _random_states(lay, ex, seed=19)
    pt, _, _ = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt)

    bad_ex = _exec(R=1, B=1, T=1, ctx=200)
    with pytest.raises(AssertionError, match="multiple of page_size"):
        pool.allocate_test_layout(lay, bad_ex)
    with pytest.raises(AssertionError, match="multiple of page_size"):
        pool.states_to_pool(lay, bad_ex, st, pt)
    with pytest.raises(AssertionError, match="multiple of page_size"):
        pool.pool_to_states(lay, bad_ex, raw_pools, pt)
