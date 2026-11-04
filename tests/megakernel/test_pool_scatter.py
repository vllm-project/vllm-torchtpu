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
"""Unit tests for the Pallas pool scatter kernel (`pool.scatter`)."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

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

FP8 = jnp.float8_e4m3fn
NUM_PAGES = 128
SLOTS = {"slots3": 3, "slots1": 1, "kv_only": 0}
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
    if lay.has_recurrent and ckpt_ndim == 2:
        ckpt, roff = ckpt[0], roff[0]
    return pt, ckpt, roff


def _bytes(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


def _check_scatter(
    lay: PoolLayout,
    ex: ExecSpec,
    ckpt_ndim: int,
    seed: int,
    *,
    cold_start: bool = False,
) -> None:
    """Scatter on rank 0 of `ex.R`; check every rank's KV and checkpoints."""
    R, NS, B, T, NATT = ex.R, ex.NS, ex.B, ex.T, lay.num_pools
    st = _random_states(lay, ex, seed=seed)
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    before = pool.pool_to_states(lay, ex, raw_pools, pt, ckpt)
    ckpt_in = ckpt
    if cold_start and lay.has_recurrent:
        fresh = int(jnp.max(ckpt)) + 1 + jnp.arange(ckpt[..., 0].size, dtype=ckpt.dtype)
        ckpt_in = jnp.concatenate([fresh.reshape(ckpt[..., :1].shape), ckpt], axis=-1)
        pool.check_ckpt_bounds(ckpt_in, cache_exec(ex).num_pages, lay)
        conv0, rec0 = pool.gather_gdn_states(
            lay,
            ex,
            tuple(_unfold(r, R)[0] for r in raw_pools),
            ckpt_in,
            jnp.full_like(roff, -1),
        )
        assert not np.any(np.asarray(conv0)) and not np.any(np.asarray(rec0))
    k = jax.random.split(jax.random.key(seed + 10), 4)
    pos = jnp.asarray(
        [10 + 129 * (s % 2) + 3 * (s // 2) + t for s in range(NS) for t in range(T)],
        dtype=jnp.int32,
    )
    knew = (jax.random.normal(k[0], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    vnew = (jax.random.normal(k[1], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    conv = rec = None
    if lay.has_recurrent:
        ngdn, vh, qkv = (
            lay.num_recurrent_layers,
            lay.vh_per_rank(R),
            lay.qkv_per_rank(R),
        )
        conv_shape = (
            (ngdn, 8 * ((3 * B + 7) // 8), qkv)
            if T == 1
            else (ngdn, NS, 8 * ((3 + T + 7) // 8), qkv)
        )
        conv = _bf16_f32(k[2], conv_shape)
        rec = (jax.random.normal(k[3], (ngdn, B * vh, lay.KD, lay.VD)) * 0.1).astype(
            jnp.bfloat16
        )
    out_r0 = pool.scatter_pool_states(
        lay,
        ex,
        tuple(_unfold(r, R)[0] for r in raw_pools),
        pos,
        pt,
        knew,
        vnew,
        conv,
        rec,
        ckpt_in if lay.has_recurrent else None,
    )
    out_pools = tuple(
        _unfold(r, R).at[0].set(o).reshape(r.shape) for r, o in zip(raw_pools, out_r0)
    )
    out = pool.pool_to_states(lay, ex, out_pools, pt, ckpt)

    for key, val in before.items():
        np.testing.assert_array_equal(_bytes(out[key][1:]), _bytes(val[1:]))
    kc_exp = np.array(_bytes(before["kc"][0]))
    vc_exp = np.array(_bytes(before["vc"][0]))
    for b in range(B):
        s = b // T
        kc_exp[:, s, int(pos[b])] = _bytes(knew[:, b])
        vc_exp[:, s, int(pos[b])] = _bytes(vnew[:, b])
    np.testing.assert_array_equal(_bytes(out["kc"][0]), kc_exp)
    np.testing.assert_array_equal(_bytes(out["vc"][0]), vc_exp)
    if not lay.has_recurrent:
        return
    np.testing.assert_array_equal(np.asarray(out["rec"][0]), np.asarray(rec))
    conv_np = np.asarray(conv)
    for s in range(NS):
        for t in range(T):
            if T == 1:
                exp_taps = conv_np[:, [j * B + s for j in range(3)]]
            else:
                exp_taps = conv_np[:, s, 1 + t : 4 + t]
            np.testing.assert_array_equal(np.asarray(out["conv"][0, :, s, t]), exp_taps)


def test_scatter_direct_conv_window_when_the_caller_conv_is_short():
    """At `T=5` the caller's `rup(3 + T, 8)` conv rows are one short of the
    `taps + 1 + T` rows of the exact-window direct path (12 lane-rows per rank):
    the scatter falls back to the packed copy instead of failing."""
    lay = replace(_layout(1, num_pools=1), kh=4, vh=4)
    assert lay.qkv_per_rank(1) // 128 == 12
    _check_scatter(lay, _exec(R=1, B=10, T=5), 3, seed=96)


@pytest.mark.parametrize("T", [2, 4, 5])
def test_scatter_direct_conv_window_traces_for_24_lane_rows(T: int):
    """24 lane-rows per rank has no exact `taps + 1` row window of its 80
    padded conv rows, so the direct path reads `taps` rows. Traced for TPU
    (`interpret=False`) and verified against reference (`interpret=True`)."""
    lay = replace(_layout(1, num_pools=1), kh=8, vh=8)
    qkv = lay.qkv_per_rank(1)
    assert qkv // 128 == 24
    _check_scatter(lay, _exec(R=1, B=2 * T, T=T), 3, seed=101 + T)
    ex = replace(_exec(R=1, B=2 * T, T=T), interpret=False)
    pt, ckpt, _ = _tables(lay, ex, 3)
    vh, NS = lay.vh_per_rank(1), ex.NS
    args = (
        (jnp.zeros((NUM_PAGES, *lay.kv_page_shape), FP8),),
        jnp.arange(ex.B, dtype=jnp.int32),
        pt,
        jnp.zeros((1, ex.MR, lay.hd), FP8),
        jnp.zeros((1, ex.MR, lay.hd), FP8),
        jnp.zeros((1, NS, 8 * ((3 + T + 7) // 8), qkv), jnp.float32),
        jnp.zeros((1, ex.B * vh, lay.KD, lay.VD), jnp.bfloat16),
    )
    jaxpr = jax.make_jaxpr(
        lambda *a: pool.scatter_pool_states(lay, ex, *a, ckpt_indices=ckpt)
    )(*args)
    assert "qwen3_5_scatter_pool_states" in str(jaxpr)


@pytest.mark.parametrize("R,kh,vh", [(1, 8, 8), (4, 32, 32)])
@pytest.mark.parametrize("T", [2, 4, 5])
def test_scatter_direct_conv_window_lowers_for_24_lane_rows_on_tpu(
    R: int, kh: int, vh: int, T: int
):
    """On TPU, lower and run the `qkv_128=24` direct conv scatter path
    (`conv_tail_row=False`, padded from 72 to 80 lane-rows)."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("Requires TPU device for Mosaic lowering")
    lay = replace(_layout(1, num_pools=1), kh=kh, vh=vh)
    qkv = lay.qkv_per_rank(R)
    assert qkv // 128 == 24
    ex = replace(_exec(R=R, B=2 * T, T=T), interpret=False)
    with jax.default_device(tpu_devs[0]):
        pt, ckpt, _ = _tables(lay, ex, 3)
        vh_r, NS = lay.vh_per_rank(R), ex.NS
        args = (
            (jnp.zeros((NUM_PAGES, *lay.kv_page_shape), FP8),),
            jnp.arange(ex.B, dtype=jnp.int32),
            pt,
            jnp.zeros((1, ex.MR, lay.hd), FP8),
            jnp.zeros((1, ex.MR, lay.hd), FP8),
            jnp.ones((1, NS, 8 * ((3 + T + 7) // 8), qkv), jnp.float32),
            jnp.ones((1, ex.B * vh_r, lay.KD, lay.VD), jnp.bfloat16),
        )
        fn = jax.jit(
            lambda *a: pool.scatter_pool_states(lay, ex, *a, ckpt_indices=ckpt)
        )
        out = fn(*args)
        jax.block_until_ready(out)
        assert out[0].shape == (NUM_PAGES, *lay.kv_page_shape)


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim",
    [("slots3", 3), ("slots3", 2), ("slots1", 3), ("slots1", 2), ("kv_only", 3)],
)
def test_scatter_writes_kv_and_checkpoints(name: str, ckpt_ndim: int, T: int):
    _check_scatter(_layout(SLOTS[name]), _exec(R=1, B=2 * T, T=T), ckpt_ndim, 31 + T)


@pytest.mark.parametrize("NS", [1, 3, 4])
def test_gather_and_scatter_sequence_counts(NS: int):
    """A 2-slot, 3-pool layout with 1, 3 and 4 sequences (T=2, 3D ckpt)."""
    lay, T = _layout(2, num_pools=3), 2
    ex = _exec(R=1, B=NS * T, T=T)
    st = _random_states(lay, ex, seed=51 + NS)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    conv, rec = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff)
    exp = np.zeros(conv.shape, np.float32)
    exp[:, :, :3] = np.asarray(st["conv"][0])
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))
    _check_scatter(lay, ex, 3, seed=61 + NS)


@pytest.mark.parametrize("name,ckpt_ndim", [("slots1", 3), ("slots3", 2)])
def test_scatter_on_one_rank_of_two(name: str, ckpt_ndim: int):
    """Rank-local scatter (`R=2`): rank 0 pools change, rank 1 pools don't."""
    _check_scatter(_layout(SLOTS[name]), _exec(R=2, B=4, T=2), ckpt_ndim, seed=71)


def test_cold_start_ckpt_k_plus_one():
    """`K = T + 1` checkpoint table with `read_offsets = -1` (no prior state)."""
    lay = _layout(2, num_pools=3)
    ex = with_cache(_exec(R=1, B=6, T=2), num_pages=2 * NUM_PAGES)
    _check_scatter(lay, ex, 3, seed=81, cold_start=True)


def test_scatter_cross_page_active_modes_k_plus_two_and_backend():
    """Cover cross-page `T > 1` KV scatter, `active_seqs` (`"all"`, `"from_ckpt"`,
    `"ref"` mask), `K = T + 2` last-`T` column writes, and `make_pool_backend`."""
    lay = _layout(1, num_pools=2)
    ex = _exec(R=1, B=4, T=2)
    st = _random_states(lay, ex, seed=211)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    before = pool.pool_to_states(lay, ex, raw_pools, pt, ckpt)

    # Sequence 0 crosses page boundary (pos 127 -> 128); sequence 1 stays on page 0.
    pos = jnp.asarray([127, 128, 10, 11], dtype=jnp.int32)
    k = jax.random.split(jax.random.key(212), 4)
    knew = (jax.random.normal(k[0], (lay.num_pools, ex.MR, lay.hd)) * 0.3).astype(FP8)
    vnew = (jax.random.normal(k[1], (lay.num_pools, ex.MR, lay.hd)) * 0.3).astype(FP8)
    ngdn, vh, qkv = lay.num_recurrent_layers, lay.vh_per_rank(1), lay.qkv_per_rank(1)
    conv = _bf16_f32(k[2], (ngdn, ex.NS, 8, qkv))
    rec = (jax.random.normal(k[3], (ngdn, ex.B * vh, lay.KD, lay.VD)) * 0.1).astype(
        jnp.bfloat16
    )

    # 1. Cross-page scatter + active_seqs="ref" mask (seq 0 active, seq 1 inactive)
    mask = jnp.asarray([1, 0], dtype=jnp.int32)
    out_ref = pool.scatter_pool_states(
        lay, ex, raw_pools, pos, pt, knew, vnew, conv, rec, ckpt, mask
    )
    after_ref = pool.pool_to_states(lay, ex, out_ref, pt, ckpt)
    for b_idx in (0, 1):
        np.testing.assert_array_equal(
            _bytes(after_ref["kc"][0, :, 0, int(pos[b_idx])]),
            _bytes(knew[:, b_idx]),
        )
        np.testing.assert_array_equal(
            _bytes(after_ref["vc"][0, :, 0, int(pos[b_idx])]),
            _bytes(vnew[:, b_idx]),
        )
    # Inactive sequence 1 KV and checkpoints remain unchanged
    np.testing.assert_array_equal(
        _bytes(after_ref["kc"][0, :, 1]), _bytes(before["kc"][0, :, 1])
    )
    np.testing.assert_array_equal(
        np.asarray(after_ref["rec"][0, :, 2 * vh : 4 * vh]),
        np.asarray(before["rec"][0, :, 2 * vh : 4 * vh]),
    )

    # 2. active_seqs="from_ckpt" with seq 1 slot 0 == 0 matches the ref mask
    ckpt_inact = ckpt.at[:, 1, :].set(0)
    out_fc = pool.scatter_pool_states(
        lay, ex, raw_pools, pos, pt, knew, vnew, conv, rec, ckpt_inact, "from_ckpt"
    )
    after_fc = pool.pool_to_states(lay, ex, out_fc, pt, ckpt)
    np.testing.assert_array_equal(_bytes(after_fc["kc"]), _bytes(after_ref["kc"]))

    # 3. K = T + 2 writes to the last T columns [2, 3]
    extra = (
        int(jnp.max(ckpt)) + 1 + jnp.arange(ckpt.shape[0] * ex.NS * 2, dtype=jnp.int32)
    )
    ckpt_k4 = jnp.concatenate([extra.reshape(ckpt.shape[0], ex.NS, 2), ckpt], axis=-1)
    out_k4 = pool.scatter_pool_states(
        lay, ex, raw_pools, pos, pt, knew, vnew, conv, rec, ckpt_k4, "all"
    )
    after_k4_full = pool.pool_to_states(lay, ex, out_k4, pt, ckpt_k4)
    after_k4_tail = pool.pool_to_states(lay, ex, out_k4, pt, ckpt)
    np.testing.assert_array_equal(
        np.asarray(after_k4_full["rec"]), np.asarray(after_k4_tail["rec"])
    )
    np.testing.assert_array_equal(np.asarray(after_k4_tail["rec"][0]), np.asarray(rec))

    # 4. make_pool_backend.scatter_states wiring for verify (T=2) and decode (T=1)
    pb = make_pool_backend(lay)
    out_pb = pb.scatter_states(
        ex,
        raw_pools,
        pos,
        pt,
        {"knew": knew, "vnew": vnew, "conv_ckpt": conv, "rec_ckpt": rec},
        ckpt,
        "all",
    )
    for r_a, r_b in zip(out_pb, out_k4):
        np.testing.assert_array_equal(_bytes(r_a), _bytes(r_b))

    ex_dec = _exec(R=1, B=2, T=1)
    pos_dec = jnp.asarray([15, 25], dtype=jnp.int32)
    conv_dec = _bf16_f32(k[2], (ngdn, 8, qkv))
    rec_dec = rec[:, : 2 * vh]
    out_pb_dec = pb.scatter_states(
        ex_dec,
        raw_pools,
        pos_dec,
        pt,
        {"knew": knew, "vnew": vnew, "conv": conv_dec, "rec": rec_dec},
        ckpt[..., :1],
        "all",
    )
    out_direct_dec = pool.scatter_pool_states(
        lay,
        ex_dec,
        raw_pools,
        pos_dec,
        pt,
        knew,
        vnew,
        conv_dec,
        rec_dec,
        ckpt[..., :1],
        "all",
    )
    for r_a, r_b in zip(out_pb_dec, out_direct_dec):
        np.testing.assert_array_equal(_bytes(r_a), _bytes(r_b))


def test_scatter_input_validation_and_pos_debug_check():
    lay = _layout(1, num_pools=2)
    ex = _exec(R=2, B=4, T=2)
    st = _random_states(lay, ex, seed=311)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(_unfold(r, ex.R)[0] for r in raw_pools)
    pos = jnp.asarray([10, 11, 20, 21], dtype=jnp.int32)
    knew = jnp.zeros((lay.num_pools, ex.MR, lay.hd), FP8)
    vnew = jnp.zeros((lay.num_pools, ex.MR, lay.hd), FP8)
    ngdn, vh, qkv = (
        lay.num_recurrent_layers,
        lay.vh_per_rank(ex.R),
        lay.qkv_per_rank(ex.R),
    )
    conv = jnp.zeros((ngdn, ex.NS, 8, qkv), jnp.float32)
    rec = jnp.zeros((ngdn, ex.B * vh, lay.KD, lay.VD), jnp.bfloat16)

    # 1. Rank-folded or under-sized raw_pools is rejected
    with pytest.raises(ValueError, match=r"raw_pools\[0\] has shape"):
        pool.scatter_pool_states(
            lay, ex, raw_pools, pos, pt, knew, vnew, conv, rec, ckpt
        )

    # 2. KV-only layout with active_seqs="from_ckpt" is rejected
    kv_only = _layout(0, num_pools=2)
    with pytest.raises(ValueError, match="active_seqs='from_ckpt' requires a layout"):
        pool.scatter_pool_states(
            kv_only, ex, raw_r0, pos, pt, knew, vnew, active_seqs="from_ckpt"
        )

    # 3. Short 4D conv at T > 1 (shape[2] < taps + T) is rejected
    conv_short = jnp.zeros((ngdn, ex.NS, 3, qkv), jnp.float32)
    with pytest.raises(ValueError, match="conv has shape .* expected 4D"):
        pool.scatter_pool_states(
            lay, ex, raw_r0, pos, pt, knew, vnew, conv_short, rec, ckpt
        )

    # 4. ckpt_indices with K < T is rejected
    with pytest.raises(ValueError, match="ckpt_indices last dim K=1 must be >= 2"):
        pool.scatter_pool_states(
            lay, ex, raw_r0, pos, pt, knew, vnew, conv, rec, ckpt[..., :1]
        )

    # 4b. Missing conv/rec/ckpt_indices with slots_per_pool > 0 raises ValueError,
    # and page_size < MR raises MegakernelConfigError.
    with pytest.raises(ValueError, match="requires conv, rec, and ckpt_indices"):
        pool.scatter_pool_states(lay, ex, raw_r0, pos, pt, knew, vnew, None, rec, ckpt)
    ex_big_mr = replace(ex)
    object.__setattr__(ex_big_mr, "B", 256)
    with pytest.raises(
        MegakernelConfigError,
        match=r"scatter_pool_states requires CacheExec\.page_size",
    ):
        pool.scatter_pool_states(kv_only, ex_big_mr, raw_r0, pos, pt, knew, vnew)

    # 5. pos beyond page_table capacity triggers kernel_debug_check
    pos_bad = jnp.asarray([261, 262, 20, 21], dtype=jnp.int32)
    with (
        pl.enable_debug_checks(True),
        pytest.raises(
            (RuntimeError, jax.errors.JaxRuntimeError),
            match="KV scatter pos outside",
        ),
    ):
        out_bad = pool.scatter_pool_states(
            lay, ex, raw_r0, pos_bad, pt, knew, vnew, conv, rec, ckpt
        )
        out_bad[0].block_until_ready()

    # 6. Tiles.bounds_checks is respected in CompilerParams
    ex_bc = replace(ex, interpret=False, tiles=replace(_TINY_TILES, bounds_checks=True))
    jaxpr = jax.make_jaxpr(
        lambda *a: pool.scatter_pool_states(lay, ex_bc, *a, ckpt_indices=ckpt)
    )(raw_r0, pos, pt, knew, vnew, conv, rec)
    assert "disable_bounds_checks=False" in str(jaxpr)
