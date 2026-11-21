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
