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
"""`msa_dcp.merge_candidates` against a sequential top-k reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_select


def _reference(candidates: np.ndarray, topk: int) -> np.ndarray:
    """Pick the best (score desc, block asc) real candidate topk times."""
    dcp, num_tokens, _ = candidates.shape
    out = np.full((num_tokens, topk), -1, np.int32)
    for t in range(num_tokens):
        pool = [
            (candidates[r, t, k], int(candidates[r, t, topk + k]))
            for r in range(dcp)
            for k in range(topk)
            if candidates[r, t, k] > -np.inf
        ]
        pool.sort(key=lambda c: (-c[0], c[1]))
        for k, (_, block) in enumerate(pool[:topk]):
            out[t, k] = block
    return out


def _candidates(rng, dcp, num_tokens, topk, num_blocks):
    """Each rank's local top-k as `local_candidates` returns it: distinct
    blocks ``c * dcp + rank`` in selection order, ``(-inf, -1)`` padded."""
    packed = np.zeros((dcp, num_tokens, 2 * topk), np.float32)
    for r in range(dcp):
        for t in range(num_tokens):
            cols = num_blocks // dcp
            real = rng.integers(0, min(topk, cols) + 1)
            picked = rng.choice(cols, size=real, replace=False)
            # Coarse scores make ties common; forced blocks score highest.
            scores = rng.integers(-3, 3, size=real).astype(np.float32)
            scores[picked == 0] = msa_select._FORCED_INIT_SCORE
            scores[picked == cols - 1] = msa_select._FORCED_LOCAL_SCORE
            order = np.lexsort((picked, -scores))
            s = np.full(topk, -np.inf, np.float32)
            b = np.full(topk, -1, np.float32)
            s[:real] = scores[order]
            b[:real] = picked[order] * dcp + r
            packed[r, t] = np.concatenate([s, b])
    return packed


@pytest.mark.parametrize("dcp,num_tokens,topk", [(2, 8, 16), (2, 37, 16), (4, 5, 8)])
def test_merge_candidates_matches_sequential_topk(dcp, num_tokens, topk):
    rng = np.random.default_rng(dcp * 100 + num_tokens)
    packed = _candidates(rng, dcp, num_tokens, topk, num_blocks=24 * dcp)
    got = np.asarray(msa_dcp.merge_candidates(jnp.asarray(packed), topk))
    np.testing.assert_array_equal(got, _reference(packed, topk))


@pytest.mark.parametrize("num_heads,num_tokens", [(1, 37), (4, 300)])
def test_kernel_merge_matches_sequential_topk(num_heads, num_tokens):
    """The DCP 8 merge kernel (`_merge_kernel`), per head, head-major packing."""
    dcp, topk = 8, 16
    rng = np.random.default_rng(num_heads * 1000 + num_tokens)
    heads = [
        _candidates(rng, dcp, num_tokens, topk, num_blocks=24 * dcp)
        for _ in range(num_heads)
    ]
    packed = np.concatenate(heads, axis=-1)  # [dcp, T, G * 2 * topk]
    got = np.asarray(
        msa_dcp.merge_head_candidates(jnp.asarray(packed), topk, interpret=True)
    )
    for g, head in enumerate(heads):
        np.testing.assert_array_equal(got[:, g], _reference(head, topk))


@pytest.mark.parametrize("num_heads,num_tokens", [(4, 512), (4, 8192), (1, 8192)])
def test_kernel_merge_lowers_for_tpu(num_heads, num_tokens):
    arg = jax.ShapeDtypeStruct((8, num_tokens, num_heads * 32), jnp.float32)
    merge = functools.partial(msa_dcp.merge_head_candidates, topk=16)
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(merge).trace(arg).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
