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


@pytest.mark.parametrize("num_heads,num_tokens", [(1, 64), (3, 37)])
def test_pair_merge_matches_sequential_topk_as_sets(num_heads, num_tokens):
    """`merge_pair_candidates` (DCP 2): the reference's blocks, any order,
    whichever rank comes first."""
    topk = 16
    rng = np.random.default_rng(num_heads * 10 + num_tokens)
    heads = [_candidates(rng, 2, num_tokens, topk, 48) for _ in range(num_heads)]
    # Rank 1 lists its picks in reverse, as `local_candidates`.
    odd = [
        np.concatenate([h[1, :, topk - 1 :: -1], h[1, :, : topk - 1 : -1]], -1)
        for h in heads
    ]
    packed = np.stack(
        [np.concatenate([h[0] for h in heads], -1), np.concatenate(odd, -1)]
    )  # [2, T, G * 2 * topk]
    for order in ([0, 1], [1, 0]):
        got = np.asarray(
            msa_dcp.merge_pair_candidates(jnp.asarray(packed[order]), topk)
        )
        for g, head in enumerate(heads):
            np.testing.assert_array_equal(
                np.sort(got[:, g], axis=1), np.sort(_reference(head, topk), axis=1)
            )


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


def _partials_reference(parts, lse, hadamard=None):
    """`ops.msa.merge_head_partials`' weights (torch.logsumexp over ranks),
    summed in f32 and rounded once."""
    peak = lse.max(axis=0)
    peak = np.where(np.isinf(peak), 0.0, peak)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = peak + np.log(np.exp(lse - peak).sum(axis=0))
        w = np.exp(lse - total).transpose(0, 2, 1)[..., None]  # [dcp, Hl, T, 1]
        terms = np.where(w > 0, parts.astype(np.float32) * w, 0.0)
    merged = np.asarray(jnp.asarray(terms.sum(axis=0)).astype(jnp.bfloat16))
    if hadamard is not None:
        d = parts.shape[-1]
        merged = np.asarray(
            (
                jnp.dot(
                    jnp.asarray(merged), hadamard, preferred_element_type=jnp.float32
                )
                * (1.0 / d)
            ).astype(jnp.bfloat16)
        )
    return merged.astype(np.float32).transpose(1, 0, 2).reshape(parts.shape[2], -1)


def _merge(pair, parts, lse, hadamard=None, **kwargs):
    """`merge_partials`, or `merge_pair_partials` on head-major LSEs."""
    if pair:
        return msa_dcp.merge_pair_partials(
            parts, jnp.swapaxes(lse, 1, 2), hadamard, **kwargs
        )
    return msa_dcp.merge_partials(parts, lse, hadamard, **kwargs)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("num_tokens,block_tokens", [(8, 256), (300, 128)])
def test_merge_partials_matches_reference(num_tokens, block_tokens, rotated, pair):
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    dcp, num_heads, head_dim = 2, 8, 128
    rng = np.random.default_rng(num_tokens)
    parts = rng.normal(size=(dcp, num_heads, num_tokens, head_dim)).astype(np.float32)
    lse = rng.normal(0, 4, size=(dcp, num_tokens, num_heads)).astype(np.float32)
    # A rank with nothing selected: lse -inf and a zero partial (as the
    # attention kernel writes it); some tokens attended by no rank at all.
    gone = rng.random((dcp, num_tokens, num_heads)) < 0.2
    lse[gone] = -np.inf
    parts[gone.transpose(0, 2, 1)] = 0.0
    parts = np.asarray(jnp.asarray(parts).astype(jnp.bfloat16))
    hadamard = (
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16) if rotated else None
    )
    got = _merge(
        pair,
        jnp.asarray(parts, jnp.bfloat16),
        jnp.asarray(lse),
        hadamard,
        block_tokens=block_tokens,
        interpret=True,
    )
    assert got.shape == (num_tokens, num_heads * head_dim)
    want = _partials_reference(parts, lse, hadamard)
    got = np.asarray(got, np.float32)
    assert np.isfinite(got).all()
    # Within a bf16 rounding of the reference (exp / log differ by an ulp,
    # which the rotation spreads over a head).
    np.testing.assert_allclose(got, want, rtol=2**-7, atol=1e-4)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("num_tokens", [8, 8192])
def test_merge_partials_lowers_for_tpu(num_tokens, rotated, pair):
    args = [
        jax.ShapeDtypeStruct((2, 8, num_tokens, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, num_tokens, 8), jnp.float32),
    ]
    if rotated:
        args.append(jax.ShapeDtypeStruct((128, 128), jnp.bfloat16))
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(_merge, pair))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("rotated", [False, True])
def test_split_merge_matches_pair_merge(rotated):
    """`merge_split_partials` of the own partial and the peer's is the pair
    merge of the two, bit for bit, over several blocks and a partial one."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    num_heads, num_tokens, head_dim = 8, 1100, 128
    rng = np.random.default_rng(5)
    out = jnp.asarray(
        rng.normal(size=(2 * num_heads, num_tokens, head_dim)), jnp.bfloat16
    )
    peer = jnp.asarray(rng.normal(size=(num_heads, num_tokens, head_dim)), jnp.bfloat16)
    lse = jnp.asarray(rng.normal(0, 4, size=(2 * num_heads, num_tokens)), jnp.float32)
    peer_lse = jnp.asarray(rng.normal(0, 4, size=(num_heads, num_tokens)), jnp.float32)
    hadamard = (
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16) if rotated else None
    )
    got = msa_dcp.merge_split_partials(
        out, peer, lse, peer_lse, hadamard, interpret=True
    )
    want = msa_dcp.merge_pair_partials(
        jnp.stack([out[:num_heads], peer]),
        jnp.stack([lse[:num_heads], peer_lse]),
        hadamard,
        block_tokens=128,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want, np.float32)
    )
