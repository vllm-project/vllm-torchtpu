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
"""CPU unit tests for `vllm_torchtpu.kernels.glm53_megakernel.fastpath_ops`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.glm53_megakernel.fastpath_ops import (
    copy_subpages_local,
    dequantize_kv_latent_fp8,
    flush_latents_local,
    kda_pool_regions,
    kda_state_from_pool_local,
    latent_write_local,
    quantize_kv_latent_fp8,
)
from vllm_torchtpu.kernels.glm53_megakernel.kernel import KV_GROUP, kv_bytes


def test_quantize_dequantize_kv_latent_fp8_roundtrip() -> None:
    rng = np.random.default_rng(0)
    lat = jnp.asarray(rng.standard_normal((16, 512), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    c_q, scale = quantize_kv_latent_fp8(lat)
    assert c_q.shape == (16, 2, 512)
    assert c_q.dtype == jnp.float8_e4m3fn
    assert scale.shape == (16, 2)
    assert scale.dtype == jnp.float32
    rec = dequantize_kv_latent_fp8(c_q, scale, out_dtype=jnp.float32)
    rel_err = float(
        jnp.linalg.norm(rec - lat.astype(jnp.float32))
        / jnp.linalg.norm(lat.astype(jnp.float32))
    )
    assert rel_err < 2e-3


def test_latent_write_and_flush_latents_local() -> None:
    rng = np.random.default_rng(1)
    nl, nkp, page, kv, world = 3, 4, 16, 64, 2
    prow = 32
    kprow = KV_GROUP * prow
    kvq0 = jnp.zeros((1, nl, nkp, kprow // 4, 2 * kv), dtype=jnp.int32)
    kvs0 = jnp.zeros((1, nl, nkp, 8, kprow), dtype=jnp.float32)
    stash = jnp.asarray(rng.standard_normal((nl, 8, kv), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    tok_slot = jnp.array([0, 0, 1, -1, 0, 1, 0, 1], dtype=jnp.int32)
    tok_pos = jnp.array([0, 1, 0, 5, 16, 16, 32, 33], dtype=jnp.int32)
    kpt = jnp.array([[1, 2], [3, 0]], dtype=jnp.int32)
    rank = jnp.array([0], dtype=jnp.int32)

    # Contiguous flush via vmap vs per-layer latent_write_local.
    q_vmap, s_vmap = flush_latents_local(
        kvq0,
        kvs0,
        stash,
        tok_slot,
        tok_pos,
        kpt,
        rank,
        layers=(0, 1, 2),
        n_tokens=8,
        page_size=page,
        world=world,
    )
    q_ref, s_ref = kvq0, kvs0
    for li in (0, 1, 2):
        q_ref, s_ref = latent_write_local(
            q_ref,
            s_ref,
            stash[li],
            tok_slot,
            tok_pos,
            kpt,
            rank,
            li,
            page_size=page,
            world=world,
        )
    np.testing.assert_array_equal(np.asarray(q_vmap), np.asarray(q_ref))
    np.testing.assert_allclose(np.asarray(s_vmap), np.asarray(s_ref))

    # Non-contiguous layer tuple exercises the loop fallback.
    q_noncontig, s_noncontig = flush_latents_local(
        kvq0,
        kvs0,
        stash,
        tok_slot,
        tok_pos,
        kpt,
        rank,
        layers=(0, 2),
        n_tokens=8,
        page_size=page,
        world=world,
    )
    np.testing.assert_array_equal(
        np.asarray(q_noncontig[:, 0]), np.asarray(q_ref[:, 0])
    )
    np.testing.assert_array_equal(
        np.asarray(q_noncontig[:, 2]), np.asarray(q_ref[:, 2])
    )
    assert int(jnp.abs(q_noncontig[:, 1]).sum()) == 0


def test_copy_subpages_local() -> None:
    nl, nkp, prow, kv = 2, 3, 32, 32
    kprow = KV_GROUP * prow
    rng = np.random.default_rng(2)
    kvq = jnp.asarray(
        rng.integers(-1000, 1000, size=(1, nl, nkp, kprow // 4, 2 * kv), dtype=np.int32)
    )
    kvs = jnp.asarray(rng.standard_normal((1, nl, nkp, 8, kprow), dtype=np.float32))
    # Copy subpage 2 -> subpage 9, and drop invalid (-1 -> -1).
    pairs = jnp.array([[2, -1], [9, -1]], dtype=jnp.int32)
    q_out, s_out = copy_subpages_local(kvq, kvs, pairs, kv_group=KV_GROUP)

    prow4 = (kprow // 4) // KV_GROUP
    q_sub_in = kvq[0].reshape(nl, nkp * KV_GROUP, prow4, 2 * kv)
    q_sub_out = q_out[0].reshape(nl, nkp * KV_GROUP, prow4, 2 * kv)
    np.testing.assert_array_equal(
        np.asarray(q_sub_out[:, 9]), np.asarray(q_sub_in[:, 2])
    )
    assert kv_bytes(q_out).dtype == jnp.float8_e4m3fn


def test_kda_state_from_pool_local_float32_conv(monkeypatch: Any) -> None:
    """Verify `kda_state_from_pool_local` reads 4-byte `float32` conv state."""
    from vllm_torchtpu.kernels import pool_adapters

    num_heads, head_dim, kernel_size = 8, 128, 4
    conv_dim = 3 * num_heads * head_dim
    pool_shape = (4, 136, 1, 512)
    layout, split = kda_pool_regions(
        pool_shape,
        jnp.uint8,
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        pool_block_tokens=1088,
    )
    assert split == 8
    assert layout.ssm_tokens > 0 and layout.conv_tokens > 0

    rng = np.random.default_rng(3)
    conv_src = rng.standard_normal((kernel_size - 1, conv_dim), dtype=np.float32)
    rec_src = rng.standard_normal((num_heads, head_dim, head_dim), dtype=np.float32)

    seen_calls: list[tuple[int, int, Any]] = []

    def _fake_gather_region(
        pool: jax.Array,
        ids: jax.Array,
        *,
        tok0: int,
        ntok: int,
        out_dtype: Any,
        out_lanes: int | None = None,
        split: int = 1,
    ) -> jax.Array:
        del pool, ids, out_lanes, split
        seen_calls.append((tok0, ntok, jnp.dtype(out_dtype)))
        if tok0 == 0:
            return jnp.asarray(rec_src.reshape(1, num_heads * head_dim, head_dim))
        return jnp.asarray(conv_src.reshape(1, -1, 1))

    monkeypatch.setattr(pool_adapters, "gather_region", _fake_gather_region)

    pool = jnp.zeros(pool_shape, dtype=jnp.uint8)
    conv_buf = jnp.zeros((1, 1, 4, kernel_size - 1, conv_dim), dtype=jnp.bfloat16)
    rec_buf = jnp.zeros((1, 1, 4, num_heads, head_dim, head_dim), dtype=jnp.float32)
    block_ids = jnp.array([[2]], dtype=jnp.int32)
    slots = jnp.array([1], dtype=jnp.int32)

    conv_out, rec_out = kda_state_from_pool_local(
        [pool],
        block_ids,
        conv_buf,
        rec_buf,
        slots,
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        pool_block_tokens=1088,
    )
    assert seen_calls == [
        (0, layout.ssm_tokens, jnp.dtype(jnp.float32)),
        (layout.ssm_tokens, layout.conv_tokens, jnp.dtype(jnp.float32)),
    ]
    expected_conv = jnp.asarray(conv_src).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(conv_out[0, 0, 1]), np.asarray(expected_conv)
    )
    np.testing.assert_allclose(np.asarray(rec_out[0, 0, 1]), rec_src, rtol=0, atol=0)


def test_flush_latents_local_partial_layers_avoids_pool_slice_copy() -> None:
    """Flushing a subset of layers (`0..10` out of `12` under MTP spec decode)
    must scatter directly into donated `kvq` without slicing `kvq` across layers."""
    kvq_s = jax.ShapeDtypeStruct((1, 12, 314, 320, 1024), jnp.int32)
    kvs_s = jax.ShapeDtypeStruct((1, 12, 314, 8, 1280), jnp.float32)
    stash_s = jax.ShapeDtypeStruct((12, 4096, 512), jnp.bfloat16)
    sp_s = jax.ShapeDtypeStruct((3, 16), jnp.int32)
    kpt_s = jax.ShapeDtypeStruct((16, 16), jnp.int32)
    rank_s = jax.ShapeDtypeStruct((1,), jnp.int32)

    def body(
        kvq: jax.Array,
        kvs: jax.Array,
        stash: jax.Array,
        sp: jax.Array,
        kpt: jax.Array,
        rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return flush_latents_local(
            kvq,
            kvs,
            stash,
            sp[0],
            sp[1],
            kpt,
            rank,
            layers=tuple(range(11)),
            n_tokens=16,
            page_size=144,
            world=8,
            tok_idx=sp[2],
        )

    jit_fn = jax.jit(body, donate_argnums=(0, 1))
    mlir = jax.export.export(jit_fn, platforms=("tpu",))(
        kvq_s, kvs_s, stash_s, sp_s, kpt_s, rank_s
    ).mlir_module()
    assert "tf.aliasing_output = 0 : i32" in mlir
    assert "tf.aliasing_output = 1 : i32" in mlir
    # Must never allocate an 11-layer slice of the 4.6 GiB `kvq` pool.
    assert "tensor<11x314x320x1024xi32>" not in mlir


def test_quantize_kv_latent_fp8_sanitizes_non_finite_inputs() -> None:
    """Non-finite padded rows must never write NaN/Inf scales into `kvs`."""
    raw = jnp.array(
        [[1.0, -2.0, 0.5, -0.25] * 128, [jnp.nan, jnp.inf, -jnp.inf, 0.0] * 128],
        dtype=jnp.bfloat16,
    )
    c_q, scale = quantize_kv_latent_fp8(raw)
    assert bool(jnp.all(jnp.isfinite(c_q.astype(jnp.float32))))
    assert bool(jnp.all(jnp.isfinite(scale)))
