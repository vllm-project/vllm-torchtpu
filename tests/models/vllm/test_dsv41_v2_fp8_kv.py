"""CPU checks for the DeepSeek V4.1 FP8 main-KV record and RoPE companion."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import compact_records
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    NOPE_DIM,
    PagedCacheRows,
    encode_fp8_kv,
    scatter_cache_values,
)


def _latent(n, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, 512)) * rng.uniform(0.01, 30, (n, 1))
    return jnp.asarray(values, jnp.bfloat16)


def _decode(words, rope):
    """Dequantize records back to float32 ``[n, 512]``."""
    record = np.asarray(csa_cache_layout.words_to_slabs(words)).reshape(-1, 512)
    values = record[:, :NOPE_DIM].view(jnp.float8_e4m3fn).astype(np.float32)
    scales = record[:, NOPE_DIM:].view(jnp.float8_e8m0fnu).astype(np.float32)
    nope = values * np.tile(scales, NOPE_DIM // 64)
    bits = np.asarray(rope).view(np.uint32)
    halves = np.concatenate((bits & 0xFFFF, bits >> 16), axis=-1).astype(np.uint16)
    return np.concatenate((nope, halves.view(jnp.bfloat16).astype(np.float32)), -1)


def test_encode_matches_exact_lane_periodic_quantizer():
    """V4's lane-periodic UE8M0 quantizer, with exact power-of-two scales."""
    latent = _latent(37)
    # Exact e4m3 ties (bf16 inputs make them common) and zero lanes.
    latent = latent.at[0, :NOPE_DIM].set(0).at[1, :NOPE_DIM].set(76 * 2.0**-19)
    words, rope = encode_fp8_kv(latent)
    assert words.shape == (37, 128) and words.dtype == jnp.int32
    assert rope.shape == (37, 32) and rope.dtype == jnp.int32
    nope = np.asarray(latent, np.float64)[:, :NOPE_DIM]
    amax = np.maximum(np.abs(nope).reshape(37, -1, 64).max(axis=1), 1e-4)
    exponent = np.ceil(np.log2(amax / 448)).astype(np.int64)
    q = (nope / np.tile(2.0**exponent, NOPE_DIM // 64)).astype(np.float32)
    expected = np.concatenate(
        (
            q.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
            (exponent + 127).astype(np.uint8),
        ),
        axis=-1,
    )
    got = np.asarray(csa_cache_layout.words_to_slabs(words)).reshape(37, 512)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        np.asarray(rope), np.asarray(csa_cache_layout.rope_words(latent[:, NOPE_DIM:]))
    )


def test_encode_round_trip_error():
    latent = _latent(64, seed=1)
    decoded = _decode(*encode_fp8_kv(latent))
    reference = np.asarray(latent, np.float32)
    np.testing.assert_array_equal(decoded[:, NOPE_DIM:], reference[:, NOPE_DIM:])
    # e4m3 keeps 3 mantissa bits; power-of-two scales add at most one more.
    nope = reference[:, :NOPE_DIM]
    lane_max = np.tile(np.abs(nope).reshape(64, -1, 64).max(axis=1), NOPE_DIM // 64)
    assert np.all(np.abs(decoded[:, :NOPE_DIM] - nope) <= lane_max / 8 + 1e-6)


def test_companion_scatter_places_slot_by_row():
    rows = 16
    companion = jnp.zeros((rows * 32 // 128, 128), jnp.int32)
    records = jnp.asarray([5, -1, 0, 15], jnp.int32)
    values = jnp.arange(4 * 32, dtype=jnp.int32).reshape(4, 32) + 1
    got = np.asarray(scatter_cache_values(companion, records, values)).reshape(-1, 32)
    expected = np.zeros((rows, 32), np.int32)
    for record, value in zip(np.asarray(records), np.asarray(values)):
        if record >= 0:
            expected[record] = value
    np.testing.assert_array_equal(got, expected)


def test_paged_cache_rows_fp8_cpu_write():
    pages, block = 3, 4
    raw = torch.zeros((pages * block * 2, 128), dtype=torch.int32)
    rope = torch.zeros((raw.shape[0] // 4, 128), dtype=torch.int32)
    # Second half of each 2-row page stride belongs to this layer.
    layout = ((pages, block, 512), torch.uint8, 2 * block * 512, block * 512)
    cache = PagedCacheRows(raw, layout, rope)
    latent = _latent(3, seed=2)
    slots = torch.tensor([0, -1, 9])
    values = torch.from_numpy(np.asarray(latent, np.float32)).to(torch.bfloat16)
    cache.write(slots, values)
    words, rope_words = encode_fp8_kv(latent)
    for index, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        row = int(cache.row_indices(torch.tensor(slot)))
        np.testing.assert_array_equal(raw[row].numpy(), np.asarray(words[index]))
        np.testing.assert_array_equal(
            rope.view(-1, 32)[row].numpy(), np.asarray(rope_words[index])
        )
    written = {int(cache.row_indices(torch.tensor(s))) for s in (0, 9)}
    untouched = [r for r in range(raw.shape[0]) if r not in written]
    assert not raw[untouched].any()
    assert not rope.view(-1, 32)[untouched].any()


def test_paged_cache_rows_pages_match_row_indices():
    pages, block = 3, 4
    raw = torch.zeros((pages * block * 2, 128), dtype=torch.int32)
    rope = torch.zeros((raw.shape[0] // 4, 128), dtype=torch.int32)
    layout = ((pages, block, 512), torch.uint8, 2 * block * 512, block * 512)
    cache = PagedCacheRows(raw, layout, rope)
    table = torch.tensor([[2, 0], [1, 2]])
    physical = cache.pages(table)
    for request in range(2):
        for logical in range(2 * block):
            block_id = table[request, logical // block]
            expected = cache.row_indices(block_id * block + logical % block)
            got = physical[request, logical // block] * block + logical % block
            assert int(got) == int(expected)
    # A layer offset that is not a whole page cannot be expressed as pages.
    odd = ((pages, block, 512), torch.uint8, 2 * block * 512, 2 * 512)
    assert PagedCacheRows(raw, odd, rope).pages(table) is None


def test_compact_records_moves_invalid_to_tail():
    records = jnp.asarray([[-1, 7, -1, 3, 9], [-1, -1, -1, -1, -1]], jnp.int32)
    got = np.asarray(compact_records(records))
    assert got.shape == (2, 128)
    np.testing.assert_array_equal(got[0, :3], [3, 7, 9])
    assert np.all(got[0, 3:] == -1) and np.all(got[1] == -1)
