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
"""DCP local top-k candidates (interpret mode) against a numpy reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_select
from vllm_torchtpu.models.minimax_m3.kernels.msa_dcp import local_candidates

BLOCK = 128
TOPK = 16


def _reference(scores, pos, rank, dcp, init_blocks, local_blocks):
    """Per row: this rank's top-k (forced blocks first, then score, ties to
    the lower block) as (scores, blocks), (-inf, -1) padded."""
    num_tokens, num_cols = scores.shape
    out_s = np.full((num_tokens, TOPK), -np.inf, np.float32)
    out_b = np.full((num_tokens, TOPK), -1, np.int32)
    for t in range(num_tokens):
        visible = pos[t] // BLOCK + 1
        cand = []
        for c in range(num_cols):
            b = c * dcp + rank
            if b >= visible:
                continue
            s = scores[t, c]
            if b >= visible - local_blocks:
                s = msa_select._FORCED_LOCAL_SCORE
            if b < init_blocks:
                s = msa_select._FORCED_INIT_SCORE
            cand.append((-s, b, s))
        cand.sort()
        for i, (_, b, s) in enumerate(cand[:TOPK]):
            out_s[t, i], out_b[t, i] = s, b
    return out_s, out_b


@pytest.mark.parametrize("pretransposed", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_cols,dcp,rank,max_pos,init_blocks,local_blocks",
    [
        (300, 512, 2, 1, 40_000, 0, 1),  # 128-row blocks, few chunks visible
        (300, 512, 2, 0, 130_000, 1, 1),  # every chunk visible in some block
        (5, 256, 8, 3, 9_000, 0, 1),  # decode-sized: one padded 128-row block
        (40, 96, 4, 2, 20_000, 1, 2),  # one padded 128-row block, < 128 columns
        (300, 384, 2, 0, 60_000, 0, 1),  # 1, 2 or 3 column chunks per block
        (128, 512, 2, 1, 50_000, 0, 1),
        (129, 512, 2, 0, 50_000, 0, 1),  # one block of all its rows
        (1032, 256, 2, 1, 30_000, 0, 1),  # past decode-sized: 256-row blocks
        (300, 1040, 2, 1, 270_000, 0, 1),  # a width of whole chunks plus a part
    ],
)
def test_candidates_match_reference(
    num_tokens,
    num_cols,
    dcp,
    rank,
    max_pos,
    init_blocks,
    local_blocks,
    pretransposed,
):
    rng = np.random.default_rng(num_tokens + num_cols)
    # Few distinct values so ties are common.
    scores = rng.integers(0, 6, (num_tokens, num_cols)).astype(np.float32)
    pos = np.sort(rng.integers(0, max_pos, num_tokens)).astype(np.int32)
    got_s, got_b = local_candidates(
        # [P, T] as the large tile class scores, or [T, 1, P]
        jnp.asarray(scores).T if pretransposed else jnp.asarray(scores)[:, None, :],
        jnp.asarray(pos),
        block_size=BLOCK,
        topk=TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=True,
    )
    want_s, want_b = _reference(scores, pos, rank, dcp, init_blocks, local_blocks)
    if rank % 2:  # odd ranks list their picks in reverse
        want_s, want_b = want_s[:, ::-1], want_b[:, ::-1]
    np.testing.assert_array_equal(np.asarray(got_b), want_b)
    np.testing.assert_array_equal(np.asarray(got_s), want_s)


def _kept_scores(scores, num_cols):
    """The scores `msa_dcp`'s packed keys keep: the order-preserving int32
    code shifted right one bit, its low `msa_dcp._key_bits` bits cleared."""
    low = (1 << msa_dcp._key_bits(num_cols)) - 1
    code = scores.astype(np.float32).view(np.int32)
    code = np.where(code < 0, code ^ 0x7FFFFFFF, code)
    code = ((code >> 1) & ~low) << 1
    return np.where(code < 0, code ^ 0x7FFFFFFF, code).astype(np.int32).view(np.float32)


@pytest.mark.parametrize("pretransposed", [False, True])
@pytest.mark.parametrize("num_tokens,num_cols,rank", [(8, 1024, 0), (300, 520, 1)])
def test_candidates_order_kept_scores(num_tokens, num_cols, rank, pretransposed):
    """Real-valued scores with near-ties below the kept precision: the picks
    are the reference's on the kept scores (ties to the lower block), and
    return those scores."""
    rng = np.random.default_rng(num_cols)
    base = rng.normal(0, 4, (num_tokens, num_cols)).astype(np.float32)
    # Neighbours that differ only in bits the keys drop.
    base[:, 1::2] = np.nextafter(base[:, ::2][:, : num_cols // 2], np.inf)
    pos = np.full((num_tokens,), 2 * 128 * num_cols - 1, np.int32)
    got_s, got_b = local_candidates(
        jnp.asarray(base).T if pretransposed else jnp.asarray(base)[:, None, :],
        jnp.asarray(pos),
        block_size=BLOCK,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        rank=rank,
        dcp=2,
        interpret=True,
    )
    kept = _kept_scores(base, num_cols)
    want_s, want_b = _reference(kept, pos, rank, 2, 0, 1)
    if rank % 2:
        want_s, want_b = want_s[:, ::-1], want_b[:, ::-1]
    np.testing.assert_array_equal(np.asarray(got_b), want_b)
    np.testing.assert_array_equal(np.asarray(got_s), want_s)


@pytest.mark.parametrize("num_tokens", [256, 384, 300, 129])
def test_row_blocks_stay_inside_the_scores(num_tokens):
    """Without `live_rows` every row block runs: the scores the kernel reads
    must hold whole row blocks (a block past them halted the c48 decode twin
    at 384 tokens, j-433e936a)."""
    scores = jnp.zeros((num_tokens, 1, 1024), jnp.float32)
    pos = jnp.zeros((num_tokens,), jnp.int32)
    jaxpr = jax.make_jaxpr(
        lambda s, p: local_candidates(
            s,
            p,
            block_size=BLOCK,
            topk=TOPK,
            init_blocks=0,
            local_blocks=1,
            rank=1,
            dcp=2,
            interpret=True,
        )
    )(scores, pos)
    (call,) = _pallas_calls(jaxpr.jaxpr)
    rows = call.invars[1].aval.shape[0]
    block_rows = call.params["grid_mapping"].block_mappings[1].block_aval.shape[0]
    assert rows % block_rows == 0, (rows, block_rows)
    assert call.params["grid_mapping"].grid[0] * block_rows == rows


def _pallas_calls(jaxpr):
    """The pallas_call equations of `jaxpr`, inside nested jaxprs too."""
    calls = []
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            calls.append(e)
        for p in e.params.values():
            inner = getattr(p, "jaxpr", None)
            if inner is not None and e.primitive.name != "pallas_call":
                calls += _pallas_calls(getattr(inner, "jaxpr", inner))
    return calls
