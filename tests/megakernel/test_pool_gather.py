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
"""Unit tests for the Pallas pool gather kernel (`pool.gather`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool import (
    CacheExec,
    PoolLayout,
    cache_exec,
    make_pool_backend,
)

FP8 = jnp.float8_e4m3fn
NUM_PAGES = 128
SLOTS = {"slots3": 3, "slots1": 1}
_TINY_TILES = Tiles(bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)


def _unfold(x: jax.Array, R: int) -> jax.Array:
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


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
    ngdn, vh = lay.num_recurrent_layers, lay.vh_per_rank(R)
    qkv = lay.qkv_per_rank(R)
    return {
        "kc": (jax.random.normal(k[0], kv_shape) * 0.3).astype(FP8),
        "vc": (jax.random.normal(k[1], kv_shape) * 0.3).astype(FP8),
        "conv": _bf16_f32(k[2], (R, ngdn, NS, lay.conv_taps, qkv)),
        "rec": (
            jax.random.normal(k[3], (R, ngdn, NS * vh, lay.KD, lay.VD)) * 0.1
        ).astype(jnp.bfloat16),
    }


def _tables(
    lay: PoolLayout, ex: ExecSpec, ckpt_ndim: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    pt, ckpt, roff, min_pages = pool.allocate_test_layout(lay, ex)
    assert cache_exec(ex).num_pages >= min_pages
    if ckpt_ndim == 2:
        ckpt, roff = ckpt[0], roff[0]
    return pt, ckpt, roff


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim", [("slots3", 3), ("slots3", 2), ("slots1", 3), ("slots1", 2)]
)
def test_gather_matches_packed_states(name: str, ckpt_ndim: int, T: int):
    lay = _layout(SLOTS[name])
    ex = _exec(R=2, B=2 * T, T=T)
    st = _random_states(lay, ex, seed=21 + T)
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(_unfold(r, ex.R)[0] for r in raw_pools)

    conv, rec = pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff)
    NS, B = ex.NS, ex.B
    taps = np.asarray(st["conv"][0])  # (ngdn, NS, 3, QKV)
    if T == 1:
        exp = np.zeros(conv.shape, np.float32)
        exp[:, : 3 * B] = np.swapaxes(taps, 1, 2).reshape(taps.shape[0], 3 * B, -1)
    else:
        exp = np.zeros(conv.shape, np.float32)
        exp[:, :, :3] = taps
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))
    assert rec.shape[1] == NS * lay.vh_per_rank(ex.R)

    backend = make_pool_backend(lay)
    got = backend.gather_states(ex, raw_r0, ckpt, roff, None)
    np.testing.assert_array_equal(np.asarray(got["rec"]), np.asarray(rec))
