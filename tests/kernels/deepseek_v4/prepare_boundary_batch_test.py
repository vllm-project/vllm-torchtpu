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
"""prepare_boundary_batch against a token-by-token reference of its layout."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import kernel as compress_kernel

KV_BLOCK_SIZE = 64
MAX_KV_BLOCKS = 40


def reference_batch(pos, req, kv, is_boundary, is_decode, ratio, num_reqs, pack=4):
    """[decode boundaries | pad to 4 | one tile of `pack` per prefill kv row | pad]."""
    num_tokens = len(pos)
    size = num_tokens + pack * (num_tokens // ratio + 1 + num_reqs)
    size = -(-size // compress_kernel.TILE_N) * compress_kernel.TILE_N
    out = [
        np.zeros(size, np.int32),
        np.zeros(size, np.int32),
        np.full(size, -1, np.int32),
    ]
    packed = [i for i in range(num_tokens) if is_boundary[i]]
    decode = [i for i in packed if is_decode[i]]
    dest = {i: j for j, i in enumerate(decode)}
    slot, prev_row = -(-len(decode) // 4) * 4 - pack, None
    for i in packed:
        if is_decode[i]:
            prev_row = None
            continue
        row = kv[i] // pack
        slot = slot + pack - slot % pack if row != prev_row else slot + 1
        dest[i], prev_row = slot, row
    for i, d in dest.items():
        out[0][d], out[1][d], out[2][d] = pos[i], req[i], kv[i]
    return out


def random_batch(rng, num_tokens, num_reqs, ratio):
    """A vLLM-shaped step: decode requests first, then prefill chunks, then padding."""
    num_active = int(rng.integers(1, num_reqs + 1))
    num_decode = int(rng.integers(0, num_active + 1))
    lens, budget = [], num_tokens
    for r in range(num_active):
        if r < num_decode:
            n = int(rng.integers(1, 4))
        else:
            n = int(rng.integers(0, budget // (num_active - r) + 1))
        n = min(n, budget)
        budget -= n
        lens.append(n)
    lens += [0] * (num_reqs - num_active)
    qsl = np.concatenate([[0], np.cumsum(lens)])
    req = np.repeat(np.arange(num_reqs), lens)
    req = np.concatenate([req, np.zeros(num_tokens - len(req), int)])
    pos = np.zeros(num_tokens, int)
    for r in range(num_reqs):
        pos[qsl[r] : qsl[r + 1]] = rng.integers(0, 2000) + np.arange(lens[r])
    # Each request owns its compressed-KV pages.
    pages = rng.permutation(num_reqs * MAX_KV_BLOCKS).reshape(num_reqs, -1)
    kv_idx = pos // ratio
    kv = pages[req, kv_idx // KV_BLOCK_SIZE] * KV_BLOCK_SIZE + kv_idx % KV_BLOCK_SIZE
    token = np.arange(num_tokens)
    is_boundary = ((pos + 1) % ratio == 0) & (token < qsl[num_active])
    kv = np.where(is_boundary, kv, -1)
    is_decode = token < qsl[num_decode]
    return [a.astype(np.int32) for a in (pos, req, kv)] + [is_boundary, is_decode]


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("num_tokens, num_reqs", [(16, 4), (128, 256), (1024, 32)])
def test_prepare_boundary_batch_matches_reference(ratio, num_tokens, num_reqs):
    rng = np.random.default_rng(num_tokens * ratio + num_reqs)
    for _ in range(10):
        pos, req, kv, is_boundary, is_decode = random_batch(
            rng, num_tokens, num_reqs, ratio
        )
        got = compressor_v1.prepare_boundary_batch(
            positions=jnp.asarray(pos),
            token_to_req_indices=jnp.asarray(req),
            kv_slot_mapping=jnp.asarray(kv),
            is_boundary=jnp.asarray(is_boundary),
            is_decode_token=jnp.asarray(is_decode),
            compress_ratio=ratio,
            num_reqs=num_reqs,
        )
        want = reference_batch(pos, req, kv, is_boundary, is_decode, ratio, num_reqs)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(np.asarray(g), w)
