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

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool import (
    CacheExec,
    PoolLayout,
    cache_exec,
    make_pool_backend,
    with_cache,
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


def test_gather_long_verify_window_falls_back_when_direct_conv_exceeds_page():
    """`kh=vh=8, R=1, T=6` has `HR * qkv_128 = 384 > conv_page_rows = 256`,
    so `gather_gdn_states` takes the packed `conv_rows_pad` path without crashing."""
    lay = replace(_layout(1, num_pools=1), kh=8, vh=8)
    ex = _exec(R=1, B=12, T=6)
    st = _random_states(lay, ex, seed=201)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    conv, rec = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff)
    exp = np.zeros(conv.shape, np.float32)
    exp[:, :, :3] = np.asarray(st["conv"][0])
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))


@pytest.mark.parametrize("NS", [1, 3, 5])
def test_gather_nonzero_roff_active_modes_and_odd_ns(NS: int):
    """Cover `NS=1, 3, 5`, `num_pools=3`, `read_offsets > 0`, cold-start `roff=-1`,
    and `active_seqs` modes (`"all"`, `"from_ckpt"`, and `"ref"` mask)."""
    lay = _layout(2, num_pools=3)
    ex = with_cache(_exec(R=1, B=NS * 2, T=2), num_pages=2 * NUM_PAGES)
    st = _random_states(lay, ex, seed=301 + NS)
    pt, ckpt, _ = _tables(lay, ex, 3)
    roff = jnp.ones((lay.slots_per_pool, NS), dtype=jnp.int32)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)

    conv, rec = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff, "all")
    exp = np.zeros(conv.shape, np.float32)
    exp[:, :, :3] = np.asarray(st["conv"][0])
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))

    if NS >= 3:
        vh = lay.vh_per_rank(ex.R)
        mask = jnp.asarray([1, 0] + [1] * (NS - 2), dtype=jnp.int32)
        conv_m, rec_m = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff, mask)
        rec_seq = np.asarray(rec_m).reshape(lay.num_recurrent_layers, NS, vh, 128, 128)
        assert not np.any(rec_seq[:, 1])
        assert not np.any(np.asarray(conv_m[:, 1]))
        np.testing.assert_array_equal(
            rec_seq[:, 0],
            np.asarray(st["rec"][0]).reshape(
                lay.num_recurrent_layers, NS, vh, 128, 128
            )[:, 0],
        )

        # Negative read_offset (cold start) on seq 0 outputs zeros for seq 0
        roff_cold = roff.at[:, 0].set(-1)
        conv_c, rec_c = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff_cold)
        rec_c_seq = np.asarray(rec_c).reshape(
            lay.num_recurrent_layers, NS, vh, 128, 128
        )
        assert not np.any(rec_c_seq[:, 0])
        assert not np.any(np.asarray(conv_c[:, 0]))


def test_gather_input_validation_and_bounds_checks():
    lay = _layout(1, num_pools=2)
    ex = _exec(R=2, B=4, T=2)
    st = _random_states(lay, ex, seed=401)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(_unfold(r, ex.R)[0] for r in raw_pools)

    # 1. KV-only layout is rejected
    kv_only = _layout(0, num_pools=1)
    with pytest.raises(ValueError, match="requires a layout with recurrent state"):
        pool.gather_gdn_states(kv_only, ex, raw_r0[:1], ckpt, roff)

    # 2. Rank-folded or under-sized pool is rejected
    with pytest.raises(ValueError, match=r"raw_pools\[0\] has shape"):
        pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff)
    with pytest.raises(ValueError, match=r"raw_pools\[0\] has shape"):
        pool.gather_gdn_states(lay, ex, (raw_r0[0][:-1], raw_r0[1]), ckpt, roff)

    # 3. Invalid ckpt_indices, read_offsets, or active_seqs mask shapes are rejected
    with pytest.raises(ValueError, match="ckpt_indices has shape"):
        pool.gather_gdn_states(lay, ex, raw_r0, ckpt[:, :1], roff)
    with pytest.raises(ValueError, match="read_offsets has shape"):
        pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff[:, :1])
    with pytest.raises(ValueError, match="active_seqs mask has shape"):
        pool.gather_gdn_states(
            lay, ex, raw_r0, ckpt, roff, jnp.ones((1,), dtype=jnp.int32)
        )

    # 4. Out-of-bounds read_offsets >= K fails eagerly and inside the kernel
    roff_bad = roff.at[0, 0].set(ckpt.shape[-1])
    with pytest.raises(ValueError, match="read_offsets=2 .* out of bounds"):
        pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff_bad)
    with (
        pl.enable_debug_checks(True),
        pytest.raises(
            (RuntimeError, jax.errors.JaxRuntimeError),
            match="GDN gather read_offsets outside",
        ),
    ):
        c_bad, _ = jax.jit(
            lambda ro: pool.gather_gdn_states(lay, ex, raw_r0, ckpt, ro)
        )(roff_bad)
        c_bad.block_until_ready()

    # 5. Tiles.bounds_checks is respected in CompilerParams
    ex_bc = replace(ex, interpret=False, tiles=replace(_TINY_TILES, bounds_checks=True))
    jaxpr = jax.make_jaxpr(
        lambda r0, ck, ro: pool.gather_gdn_states(lay, ex_bc, r0, ck, ro)
    )(raw_r0, ckpt, roff)
    assert "disable_bounds_checks=False" in str(jaxpr)
