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
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla


def _encode_block_reversed_lse(
    lse_f32: np.ndarray,
    send_world: int,
    send_lanes: int,
    pad_bits_u32: int = 0xFF800000,
) -> np.ndarray:
    num_tokens, num_heads = lse_f32.shape
    send_per = num_heads // send_world
    buf = np.full((num_tokens, send_lanes), np.uint32(pad_bits_u32), dtype=np.uint32)
    lse_u32 = lse_f32.view(np.uint32)
    for dest in range(send_world):
        src_slice = slice(dest * send_per, (dest + 1) * send_per)
        dst_col = (send_world - 1 - dest) * send_per
        buf[:, dst_col : dst_col + send_per] = lse_u32[:, src_slice]
    return buf.view(np.float32)


def _reference_packed_words(
    lse_f32: np.ndarray,
    send_world: int,
    send_lanes: int,
    *,
    skip_lo: bool = False,
) -> np.ndarray:
    num_tokens, num_heads = lse_f32.shape
    send_per = num_heads // send_world
    send_rows = num_tokens * send_per
    bits = (
        lse_f32.view(np.int32)
        .reshape(num_tokens, send_world, send_per)
        .transpose(1, 0, 2)
        .reshape(send_world, send_rows)
    )
    words = np.zeros((send_world, send_lanes), dtype=np.int32)
    words[:, :send_rows] = bits >> 16
    if not skip_lo:
        words[:, send_rows : 2 * send_rows] = ((bits & 0xFFFF) ^ 0x8000) - 0x8000
    return words


def _run_pack_send_lse_words(
    x_lse: np.ndarray,
    *,
    num_q_heads: int,
    send_world: int,
    fault: str | None = None,
) -> np.ndarray:
    num_tokens, send_lanes = x_lse.shape

    def body(x_ref, out_ref):
        out_ref[...] = sparse_mla.pack_send_lse_words(
            x_ref[...],
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            send_world=send_world,
            send_lanes=send_lanes,
            fault=fault,
        )

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((send_world, send_lanes), jnp.int32),
        interpret=True,
    )(jnp.asarray(x_lse, dtype=jnp.float32))
    return np.asarray(out, dtype=np.int32)


def _distinct_lse_bits(num_tokens: int, num_heads: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = (rng.normal(size=(num_tokens, num_heads)) * 12.0).astype(np.float32)
    bits = vals.view(np.uint32)
    # Mix signed high/low 16-bit halfwords across every token and head.
    tok_idx = np.arange(num_tokens, dtype=np.uint32)[:, None]
    head_idx = np.arange(num_heads, dtype=np.uint32)[None, :]
    anti_lo = ((tok_idx * 97 + head_idx * 17 + 0x8001) & 0xFFFF).astype(np.uint32)
    bits = (bits & np.uint32(0xFFFF0000)) | anti_lo
    lse = bits.view(np.float32)
    # Row 0 holds -inf (0xFF800000) to catch lo-half contamination turning it into NaN.
    lse[0, :] = -np.inf
    return lse


@pytest.mark.parametrize(
    "world,heads,tokens",
    [
        (16, 64, 4),
        (16, 64, 8),
        (8, 64, 4),
        (8, 32, 4),
        (16, 64, 16),
        (16, 64, 32),
        (8, 64, 8),
        (8, 64, 16),
        (8, 32, 8),
        (8, 32, 16),
    ],
)
def test_should_pack_exact_hi_and_lo_lse_words_across_small_and_large_token_counts(
    world: int, heads: int, tokens: int
) -> None:
    lanes = sparse_mla.send_layout_lanes(tokens, heads, world)
    assert sparse_mla.send_layout_supported(tokens, heads, world, 256, tokens)
    lse = _distinct_lse_bits(tokens, heads, seed=1000 + world * 100 + tokens)
    # Non-zero sentinel past num_q_heads catches out-of-range lane reads.
    x_in = _encode_block_reversed_lse(lse, world, lanes, pad_bits_u32=0xDEADFACE)

    got = _run_pack_send_lse_words(x_in, num_q_heads=heads, send_world=world)
    want = _reference_packed_words(lse, world, lanes)

    np.testing.assert_array_equal(got, want)
    send_rows = tokens * (heads // world)
    repacked_u32 = (
        (got[:, :send_rows] << 16) | (got[:, send_rows : 2 * send_rows] & 0xFFFF)
    ).astype(np.int32)
    repacked_f32 = repacked_u32.view(np.float32)
    send_per = heads // world
    assert np.all(np.isneginf(repacked_f32[:, :send_per]))
    assert not np.any(np.isnan(repacked_f32))
    assert not np.any(got[:, 2 * send_rows :])


def test_should_match_reference_for_every_supported_send_layout_geometry() -> None:
    checked = 0
    for world in (2, 4, 8, 16, 32):
        for heads in (16, 32, 64, 128):
            for tokens in (1, 2, 4, 8, 16, 32, 64):
                if not sparse_mla.send_layout_supported(
                    tokens, heads, world, 512, tokens
                ):
                    continue
                lanes = sparse_mla.send_layout_lanes(tokens, heads, world)
                lse = _distinct_lse_bits(
                    tokens, heads, seed=world * 31 + heads + tokens
                )
                x_in = _encode_block_reversed_lse(
                    lse, world, lanes, pad_bits_u32=0x7F81BEEF
                )
                got = _run_pack_send_lse_words(
                    x_in, num_q_heads=heads, send_world=world
                )
                want = _reference_packed_words(lse, world, lanes)
                np.testing.assert_array_equal(got, want)
                checked += 1
    assert checked >= 50


@pytest.mark.parametrize("world,heads,tokens", [(16, 64, 4), (16, 64, 8), (16, 64, 16)])
def test_should_zero_lo_halfwords_when_send_lse_lo_skip_fault_injected(
    world: int, heads: int, tokens: int
) -> None:
    lanes = sparse_mla.send_layout_lanes(tokens, heads, world)
    lse = _distinct_lse_bits(tokens, heads, seed=2026 + tokens)
    x_in = _encode_block_reversed_lse(lse, world, lanes)

    got = _run_pack_send_lse_words(
        x_in, num_q_heads=heads, send_world=world, fault="send_lse_lo_skip"
    )
    want_skip = _reference_packed_words(lse, world, lanes, skip_lo=True)
    want_clean = _reference_packed_words(lse, world, lanes, skip_lo=False)

    np.testing.assert_array_equal(got, want_skip)
    assert not np.array_equal(got, want_clean)


def _build_received_slabs(
    partials_bf16: jnp.ndarray,
    lse_f32: np.ndarray,
    *,
    slab_row0: int,
) -> jnp.ndarray:
    """Build `[world, SEND_SLAB_ROW0 + send_rows, o_dim]` int16 received slabs."""
    world, send_rows, o_dim = partials_bf16.shape
    slab_rows = sparse_mla.SEND_SLAB_ROW0 + send_rows
    # Poison unused header/tail rows with NaN bit-patterns (0x7FC1) so out-of-slab
    # slicing is caught immediately.
    slabs = np.full((world, slab_rows, o_dim), np.int16(0x7FC1), dtype=np.int16)
    bits = lse_f32.view(np.int32)
    words = np.zeros((world, o_dim), dtype=np.int32)
    words[:, :send_rows] = bits >> 16
    words[:, send_rows : 2 * send_rows] = ((bits & 0xFFFF) ^ 0x8000) - 0x8000
    # Row `slab_row0 - 1` holds the signed int16 LSE halfwords.
    slabs[:, slab_row0 - 1, :] = words.astype(np.int16)
    part_i16 = np.asarray(
        jax.lax.bitcast_convert_type(partials_bf16, jnp.int16), dtype=np.int16
    )
    slabs[:, slab_row0 : slab_row0 + send_rows, :] = part_i16
    return jnp.asarray(slabs, dtype=jnp.int16)


def _reference_merge_slabs(
    partials_bf16: jnp.ndarray,
    lse_f32: np.ndarray,
    *,
    live_rows: int,
) -> jnp.ndarray:
    part_f32 = np.asarray(partials_bf16.astype(jnp.float32), dtype=np.float32)
    world, send_rows, o_dim = part_f32.shape
    finite = np.isfinite(lse_f32)
    masked = np.where(finite, lse_f32, -np.inf)
    shift = np.max(masked, axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    weights = np.where(finite, np.exp(lse_f32 - shift[None, :]), 0.0).astype(np.float32)
    denom = np.sum(weights, axis=0)
    denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32)
    acc_hi = np.zeros((send_rows, o_dim), dtype=np.float32)
    acc_lo = np.zeros((send_rows, o_dim), dtype=np.float32)
    row_mask = np.arange(send_rows)[:, None] < live_rows
    for w in range(world):
        wt = (weights[w, :, None] / denom[:, None]).astype(np.float32)
        w_hi = np.asarray(
            jnp.asarray(wt).astype(jnp.bfloat16).astype(jnp.float32), dtype=np.float32
        )
        p = np.where(row_mask & ~np.isnan(part_f32[w]), part_f32[w], 0.0)
        acc_hi += p * w_hi
        acc_lo += p * (wt - w_hi)
    return jnp.asarray(acc_hi + acc_lo, dtype=jnp.float32).astype(jnp.bfloat16)


@pytest.mark.parametrize(
    "world,heads,tokens,slab_row0,live_tokens",
    [
        # Compact merge slab (send_per == 4, slab_row0 == 4 < SEND_SLAB_ROW0 == 8).
        (16, 64, 4, 4, 4),
        (16, 64, 4, 4, 3),
        (16, 64, 8, 4, 5),
        # Standard merge slab (slab_row0 == SEND_SLAB_ROW0 == 8).
        (8, 64, 4, 8, 4),
        (8, 64, 2, 8, 1),
    ],
)
def test_should_merge_received_slabs_for_compact_and_standard_slab_offsets(
    world: int,
    heads: int,
    tokens: int,
    slab_row0: int,
    live_tokens: int,
) -> None:
    o_dim = 512
    send_per = heads // world
    send_rows = tokens * send_per
    live_rows = live_tokens * send_per
    rng = np.random.default_rng(4200 + world * 100 + tokens * 10 + slab_row0)
    lse = (rng.normal(size=(world, send_rows)) * 4.0).astype(np.float32)
    # Include an all-(-inf) row and a partial NaN sender to exercise masking.
    lse[:, 0] = -np.inf
    lse[0, 1] = -np.inf
    raw_part = jnp.asarray(
        rng.normal(size=(world, send_rows, o_dim)).astype(np.float32)
    ).astype(jnp.bfloat16)
    raw_part = raw_part.at[0, 1, :].set(jnp.nan)
    slabs = _build_received_slabs(raw_part, lse, slab_row0=slab_row0)
    assert slabs.shape == (world, sparse_mla.SEND_SLAB_ROW0 + send_rows, o_dim)

    def body(recv_v_ref, live_ref, out_ref):
        out_ref[...] = sparse_mla.merge_received_slabs(
            recv_v_ref,
            send_world=world,
            send_rows=send_rows,
            slab_row0=slab_row0,
            o_dim=o_dim,
            live_rows=live_ref[0],
        )

    got = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((send_rows, o_dim), jnp.bfloat16),
        interpret=True,
    )(slabs, jnp.asarray([live_rows], dtype=jnp.int32))
    want = _reference_merge_slabs(raw_part, lse, live_rows=live_rows)

    assert np.all(np.asarray(got[0].astype(jnp.float32)) == 0.0)
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(want.astype(jnp.float32)),
        rtol=8e-3,
        atol=1e-4,
    )
    assert np.all(np.asarray(got[live_rows:].astype(jnp.float32)) == 0.0)
