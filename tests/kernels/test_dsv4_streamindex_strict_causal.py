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
"""streamindex_topk `strict_causal` (TPU_STREAMIDX_STRICT_CAUSAL) at DSv4-Pro
served shapes (ratio 4, k 1024, 1 + 6 verify windows).

Compressed group g holds tokens [4g, 4g + 4) and is complete at position 4g + 3.
A token at position p may select only the (p + 1) // 4 groups complete at p --
what single-token decode at p sees (kv_len = seq_len // 4 = (p + 1) // 4), and
the bound the HCA layers already use (`(positions + 1) // compress_ratio`).

The old bound, p // 4 + 1, lets an earlier row of a multi-token forward (verify
window, prefill chunk) select the group a later token of the same forward
completes: the verifier sees the draft. The leak is deterministic when a
sequence has fewer than k compressed positions (every visible one is selected),
so the short cases below must show it without the flag and never with it,
through every served variant: the verify pass with and without verify_bd, the
kv split (c4/c8 TP8) and the token split (c16 TP4).

    python3 -m pytest tests/kernels/test_dsv4_streamindex_strict_causal.py
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_dsv4_indexer_kv_split import K, RATIO, VQ, _case, _kw
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    merge_kv_split_topk,
    streamindex_topk,
    token_window_metadata,
)

pytestmark = [pytest.mark.tpu_test]

_HAS_FLAG = "strict_causal" in inspect.signature(streamindex_topk).parameters


def _flag(strict):
    # A tree without the flag runs the old bound: the strict test then fails
    # on the leak itself, which is the before/after evidence.
    return {"strict_causal": strict} if _HAS_FLAG else {}


# (q_len, seq_len, kind), verify windows first, and the token bucket.
CASES = {
    # one window crossing a group boundary: rows 996..998 can leak group 249
    "v1@1003": ([(VQ, 1003, "verify")], 8),
    # four windows, every alignment mod 4
    "v4_align": ([(VQ, 2000 + i, "verify") for i in range(4)], 32),
    # verify windows next to a fresh prefill chunk (T512 mixed step, short)
    "v2_c495@fresh": ([(VQ, 1007, "verify"), (VQ, 3001, "verify"),
                       (495, 495, "chunk")], 512),
    # chunk continuing a prefix
    "c250@2k": ([(250, 2250, "chunk")], 256),
}
# long sequences: the bound must hold; the leak is only probabilistic here
LONG_CASES = {
    "v2_c495@127k": ([(VQ, 40_007, "verify"), (VQ, 90_005, "verify"),
                      (495, 127_495, "chunk")], 512),
}


def _positions(inp, seqs):
    cu = np.asarray(inp["cu_q_lens"])
    sl = np.asarray(inp["seq_lens"])
    pos = []
    for i, (q_len, _, _) in enumerate(seqs):
        assert cu[i + 1] - cu[i] == q_len
        pos.extend(range(sl[i] - q_len, sl[i]))
    return np.asarray(pos)


def _violations(idx, pos):
    """Rows that selected a group not complete at their own position."""
    n = len(pos)
    bound = (pos + 1) // RATIO  # valid indices are < bound
    return np.nonzero(np.any(idx[:n] >= bound[:, None], axis=1))[0]


def _counts_ok(idx, pos):
    """Short rows (< k visible): exactly the (p + 1) // 4 complete groups."""
    n = len(pos)
    got = np.sum(idx[:n] >= 0, axis=1)
    return np.array_equal(got, np.minimum((pos + 1) // RATIO, K))


def _plain(inp, strict, verify_bd):
    return np.asarray(
        jax.jit(
            lambda i: streamindex_topk(**i, **_kw(verify_bd=verify_bd, **_flag(strict)))
        )(inp)
    )


def _kv_split(inp, strict, nsplit=8):
    share = jax.jit(
        lambda i, r: jnp.concatenate(
            streamindex_topk(
                **i,
                **_kw(kv_split=nsplit, kv_split_rank=r, verify_bd=True, **_flag(strict)),
            ),
            axis=1,
        )
    )
    return np.asarray(
        merge_kv_split_topk(
            jnp.stack([share(inp, jnp.int32(r)) for r in range(nsplit)]), K
        )
    )


def _token_split(inp, strict, tp=4):
    T = inp["q"].shape[0]
    w = T // tp

    @jax.jit
    def window(i, rank):
        off = (rank * w).astype(jnp.int32)
        seq_w, pages_w, cu_w, dist_w = token_window_metadata(
            i["seq_lens"], i["page_indices"], i["cu_q_lens"], i["distribution"],
            off, window=w, compression_ratio=RATIO,
        )  # fmt: skip
        return streamindex_topk(
            q=jax.lax.dynamic_slice_in_dim(i["q"], off, w, 0),
            indexer_weights=jax.lax.dynamic_slice_in_dim(
                i["indexer_weights"], off, w, 0
            ),
            cache_kv=i["cache_kv"],
            seq_lens=seq_w,
            page_indices=pages_w,
            cu_q_lens=cu_w,
            distribution=dist_w,
            row_window=(i["seq_lens"], i["cu_q_lens"], i["distribution"], off),
            row_window_tokens=T,
            **_kw(verify_bd=True, **_flag(strict)),
        )

    return np.concatenate(
        [np.asarray(window(inp, jnp.int32(r))) for r in range(tp)], axis=0
    )


VARIANTS = {
    "verify": lambda inp, s: _plain(inp, s, False),
    "verify_bd": lambda inp, s: _plain(inp, s, True),
    "kv_split8": _kv_split,
    "token_split4": _token_split,
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("name", list(CASES))
def test_rows_see_only_complete_groups(name, variant):
    seqs, T = CASES[name]
    if variant == "token_split4" and T % 4:
        pytest.skip("token split needs T % tp == 0")
    inp = _case(seqs, T)
    pos = _positions(inp, seqs)
    idx = VARIANTS[variant](inp, True)
    bad = _violations(idx, pos)
    assert bad.size == 0, (
        f"{bad.size} rows select an incomplete group, first row {bad[0]} at "
        f"pos {pos[bad[0]]}: max idx {idx[bad[0]].max()} >= {(pos[bad[0]] + 1) // RATIO}"
    )
    assert _counts_ok(idx, pos)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("name", list(CASES))
def test_old_bound_leaks_the_draft(name, variant):
    """Positive control: the check above catches the old bound."""
    seqs, T = CASES[name]
    if variant == "token_split4" and T % 4:
        pytest.skip("token split needs T % tp == 0")
    inp = _case(seqs, T)
    pos = _positions(inp, seqs)
    idx = VARIANTS[variant](inp, False)
    # A row leaks when its own group is incomplete at its position but a later
    # token of the same forward completes it (it is inside seq_len // 4).
    seq_end = np.concatenate(
        [np.full(q_len, seq_len) for q_len, seq_len, _ in seqs]
    )
    leaking = int(np.sum(((pos + 1) % RATIO != 0) & (pos // RATIO < seq_end // RATIO)))
    assert _violations(idx, pos).size == leaking > 0


@pytest.mark.parametrize("variant", ["verify_bd", "kv_split8", "token_split4"])
@pytest.mark.parametrize("name", list(LONG_CASES))
def test_long_rows_bound(name, variant):
    seqs, T = LONG_CASES[name]
    inp = _case(seqs, T)
    pos = _positions(inp, seqs)
    idx = VARIANTS[variant](inp, True)
    assert _violations(idx, pos).size == 0
    assert np.all(np.sum(idx[: len(pos)] >= 0, axis=1) == K)
