# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""device_decode.rng draws vLLM Model Runner V2's sampling uniforms
(`murmur3_uniform32` in vllm/v1/worker/gpu/sample/gumbel.py, v0.33.0) bit for
bit, and a draw depends only on its (seed, position, token id) key."""

import struct

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.device_decode import rng

# Qwen3.5's padded vocabulary; its last token id is 248319.
VOCAB = 248320
SEEDS = [
    0,
    1,
    -1,
    2**63 - 1,
    -(2**63),
    0xFFFFFFFF,
    -(2**32),
    0x0123456789ABCDEF,
    -0x0123456789ABCDEF,
    2**32,
    2**31,
    -(2**31),
]
POSITIONS = [0, 1, 543, 544, 10239, 2**24 + 1, 2**31 - 1]

# (seed, pos, token, murmur3_hash32, murmur3_uniform32 bits), computed by
# MRV2's own murmur3_hash32 and murmur3_uniform32, run on the Triton CPU
# interpreter (TRITON_INTERPRET=1) from vLLM v0.33.0's gumbel.py.
KNOWN_ANSWERS = [
    (0, 0, 0, 0x8134CDF8, 0x3F0134CE),
    (1, 0, 0, 0x92228D1B, 0x3F12228D),
    (-1, 0, 0, 0x3E003DC7, 0x3E7800F7),
    (2**63 - 1, 2**31 - 1, 248319, 0x43A991AD, 0x3E875323),
    (-(2**63), 2**31 - 1, 248319, 0xF7AA2840, 0x3F77AA28),
    (0x0123456789ABCDEF, 544, 151935, 0x96AF4AC1, 0x3F16AF4B),
    (-0x0123456789ABCDEF, 10239, 65535, 0x08C811DA, 0x3D0C811E),
    (2**32, 1, 65536, 0x14BEBCAE, 0x3DA5F5E5),
    (0xFFFFFFFF, 2, 127, 0x18AF3DF4, 0x3DC579F0),
    (-(2**32), 543, 128, 0x5EDD39E8, 0x3EBDBA74),
    (2**31, 2**24 + 1, 1023, 0xE1CC0E84, 0x3F61CC0F),
    (-(2**31), 2**31 - 1, 1024, 0x946D6EE7, 0x3F146D6F),
]


# MRV2's murmur3_hash32 and murmur3_uniform32, line by line, on numpy uint32.
def _np_rotl(x, r):
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def _np_mix(h, key):
    key = key * np.uint32(0xCC9E2D51)
    key = _np_rotl(key, 15)
    key = key * np.uint32(0x1B873593)
    h = h ^ key
    h = _np_rotl(h, 13)
    return h * np.uint32(5) + np.uint32(0xE6546B64)


def _np_fmix32(h):
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13))
    h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def mrv2_hash32(seed, pos, token) -> np.ndarray:
    seed = np.asarray(seed, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    token = np.asarray(token).astype(np.uint32)
    with np.errstate(over="ignore"):
        h = np.zeros(np.broadcast(seed, pos, token).shape, dtype=np.uint32)
        h = _np_mix(h, (seed & 0xFFFFFFFF).astype(np.uint32))
        h = _np_mix(h, ((seed >> 32) & 0xFFFFFFFF).astype(np.uint32))
        h = _np_mix(h, (pos & 0xFFFFFFFF).astype(np.uint32))
        h = _np_mix(h, token)
        return _np_fmix32(h ^ np.uint32(16))


def mrv2_uniform32(seed, pos, token) -> np.ndarray:
    r = mrv2_hash32(seed, pos, token)
    hi16 = (r >> np.uint32(16)).astype(np.float32)
    lo16 = (r & np.uint32(0xFFFF)).astype(np.float32)
    return hi16 * np.float32(2.0**-16) + (lo16 + np.float32(0.5)) * np.float32(2.0**-32)


def _seed_words(seeds) -> tuple[torch.Tensor, torch.Tensor]:
    """The int32 bit patterns of each int64 seed's low and high words."""
    s = np.asarray(seeds, dtype=np.int64)
    lo = (s & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    hi = ((s >> 32) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    return torch.from_numpy(lo.copy()), torch.from_numpy(hi.copy())


def _uniform(seeds, positions, vocab: int) -> torch.Tensor:
    lo, hi = _seed_words(seeds)
    return rng.counter_uniform(
        lo, hi, torch.tensor(positions, dtype=torch.int32), vocab
    )


def _bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.int32)


def test_known_answers_of_the_mrv2_kernel():
    for seed, pos, token, want_hash, want_bits in KNOWN_ANSWERS:
        assert int(mrv2_hash32(seed, pos, token)) == want_hash
        assert int(_bits(mrv2_uniform32(seed, pos, token)).view(np.uint32)) == (
            want_bits
        )
        got = _uniform([seed], [pos], VOCAB)[0, token]
        assert int(_bits(got.numpy()).view(np.uint32)) == want_bits, (seed, pos)


def _murmur3_x86_32(data: bytes, seed: int) -> int:
    """Textbook MurmurHash3_x86_32 of a key of whole 32-bit words."""
    mask = 0xFFFFFFFF
    h = seed
    for (k,) in struct.iter_unpack("<I", data):
        k = (k * 0xCC9E2D51) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * 0x1B873593) & mask
        h ^= k
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    return h ^ (h >> 16)


def test_hash_is_murmur3_x86_32_of_the_16_byte_key():
    # Published MurmurHash3_x86_32 vectors check the textbook version first.
    assert _murmur3_x86_32(b"", 1) == 0x514E28B7
    assert _murmur3_x86_32(b"\0\0\0\0", 0) == 0x2362F9DE
    assert _murmur3_x86_32(b"aaaa", 0x9747B28C) == 0x5A97808A
    assert _murmur3_x86_32(b"abcd", 0x9747B28C) == 0xF0478627
    for seed in SEEDS:
        for pos in POSITIONS:
            for token in (0, 1, 128, 151935, 248319):
                key = struct.pack(
                    "<IIII", seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, pos, token
                )
                assert int(mrv2_hash32(seed, pos, token)) == _murmur3_x86_32(key, 0)


def test_counter_uniform_matches_mrv2_over_the_whole_vocabulary():
    # One position per seed over the full vocabulary, every (seed, position)
    # pair over its first 1024 tokens.
    positions = [POSITIONS[i % len(POSITIONS)] for i in range(len(SEEDS))]
    got = _uniform(SEEDS, positions, VOCAB).numpy()
    tokens = np.arange(VOCAB)
    for row, (seed, pos) in enumerate(zip(SEEDS, positions)):
        want = mrv2_uniform32(seed, pos, tokens)
        np.testing.assert_array_equal(_bits(got[row]), _bits(want))

    pairs = [(s, p) for s in SEEDS for p in POSITIONS]
    got = _uniform([s for s, _ in pairs], [p for _, p in pairs], 1024).numpy()
    want = mrv2_uniform32(
        np.array([s for s, _ in pairs])[:, None],
        np.array([p for _, p in pairs])[:, None],
        np.arange(1024)[None, :],
    )
    np.testing.assert_array_equal(_bits(got), _bits(want))


def test_uniform_is_the_hash_midpoint_rounded_once():
    bits = np.array(
        [0, 1, 0xFFFF, 0x10000, 0x12345678, 0x7FFFFFFF, 0x80000000]
        + [0xFFFFFF7F, 0xFFFFFF80, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    got = rng._uniform_from_bits(torch.from_numpy(bits.view(np.int32))).numpy()
    want = ((bits.astype(np.float64) + 0.5) * 2.0**-32).astype(np.float32)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    # The smallest draw keeps all 32 bits; the largest rounds up to 1.0, which
    # the sampler's clamp keeps finite.
    assert got[0] == 2.0**-33
    assert got[-1] == 1.0


def test_a_draw_depends_only_on_its_key():
    seeds, positions = SEEDS[:6], [5, 5, 9, 2**31 - 1, 0, 77]
    batch = _uniform(seeds, positions, 4096)
    assert torch.equal(batch, _uniform(seeds, positions, 4096))
    for row in range(len(seeds)):
        alone = _uniform([seeds[row]], [positions[row]], 4096)
        assert torch.equal(alone[0], batch[row])
    order = [3, 0, 5, 1, 4, 2]
    reordered = _uniform([seeds[i] for i in order], [positions[i] for i in order], 4096)
    assert torch.equal(reordered, batch[order])
    # The low columns of a wider vocabulary are the same draws.
    assert torch.equal(_uniform(seeds, positions, VOCAB)[:, :4096], batch)
    # Each key word changes every draw of the row.
    base = _uniform([7], [100], 4096)[0]
    for seed, pos in ((8, 100), (7 + 2**32, 100), (7, 101)):
        other = _uniform([seed], [pos], 4096)[0]
        assert (other != base).float().mean() > 0.99, (seed, pos)


def test_counter_uniform_for_rows_reads_seeds_by_slot():
    lo, hi = _seed_words([11, -5, 2**40 + 3, 0])
    # Rows as columns of a [rows, 4] plan, as the stock path stages them.
    plan = torch.tensor(
        [[2, 9, 1, 10], [0, 0, 1, 1], [2, 4, 0, 5], [3, 0, 0, 0]], dtype=torch.int32
    )
    got = rng.counter_uniform_for_rows(lo, hi, plan[:, 0], plan[:, 1], 512)
    want = rng.counter_uniform(lo[[2, 0, 2, 3]], hi[[2, 0, 2, 3]], plan[:, 1], 512)
    assert torch.equal(got, want)


def test_wider_integers_are_rejected():
    seeds = torch.zeros(2, dtype=torch.int64)
    pos = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(TypeError, match="seed_lo must be int32"):
        rng.counter_uniform(seeds, seeds.int(), pos, 8)


@pytest.mark.tpu_test
def test_counter_uniform_bits_on_tpu(device):
    """The compiled TPU programs draw MRV2's bits over the whole vocabulary."""
    positions = [POSITIONS[i % len(POSITIONS)] for i in range(len(SEEDS))]
    lo, hi = _seed_words(SEEDS)
    pos = torch.tensor(positions, dtype=torch.int32)
    seeds = np.array(SEEDS, dtype=np.int64)[:, None]
    tokens = np.arange(VOCAB)[None, :]

    got = rng.counter_uniform(lo.to(device), hi.to(device), pos.to(device), VOCAB)
    want = mrv2_uniform32(seeds, np.array(positions)[:, None], tokens)
    mismatches = int((_bits(got.cpu().numpy()) != _bits(want)).sum())
    assert mismatches == 0, f"{mismatches} of {want.size} uniforms differ"

    # The stock-path entry, with rows as columns of a [rows, 4] plan.
    order = torch.arange(len(SEEDS)).flip(0)
    plan = torch.zeros((len(SEEDS), 4), dtype=torch.int32)
    plan[:, 0], plan[:, 1] = order, pos
    plan = plan.to(device)
    got = rng.counter_uniform_for_rows(
        lo.to(device), hi.to(device), plan[:, 0], plan[:, 1], VOCAB
    )
    want = mrv2_uniform32(seeds[order.numpy()], np.array(positions)[:, None], tokens)
    mismatches = int((_bits(got.cpu().numpy()) != _bits(want)).sum())
    assert mismatches == 0, f"{mismatches} of {want.size} uniforms differ"
