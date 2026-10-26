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
"""streamindex_topk `verify_bd` (TPU_STREAMIDX_VERIFY_BD): the verify pass with a
block-diagonal q and a packed bf16 head reduction, at DSv4-Pro served shapes.

* verify_bd=True selects exactly what the mixed pass selects for the same
  verify windows (same rounding points, same head order, f32 accumulation):
  same indices, same score bits.
* verify_bd=False is the default path.
* under kv_split the merged shares equal the unsplit top-k, with verify_bd on.

    python3 -m pytest tests/kernels/test_dsv4_streamindex_verify_bd.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    merge_kv_split_topk,
    streamindex_topk,
)

from tests.kernels.test_dsv4_indexer_kv_split import VQ, K, _case, _kw

pytestmark = [pytest.mark.tpu_test]

# (q_len, seq_len, kind) with every verify window first, and the bucket.
VERIFY_CASES = {
    # one window: 4 full KV blocks + a last block more than half full (FULL)
    "v1@300k": ([(VQ, 300_007, "verify")], 8),
    # last block at most half full (HALF branch), and one key past a block
    "v1@479k": ([(VQ, 479_000, "verify")], 8),
    "v1@blk+1": ([(VQ, 16_385 * 4 + 1, "verify")], 8),
    # ragged windows, padding rows in the bucket
    "v3_ragged": ([(VQ, 4_003, "verify"), (VQ, 70_001, "verify"),
                   (VQ, 200_005, "verify")], 32),
    "v4@83k": ([(VQ, 83_000 + 97 * i, "verify") for i in range(4)], 32),
}


_verify = jax.jit(
    lambda inp, verify_bd: streamindex_topk(
        **inp, **_kw(return_scores=True, verify_bd=verify_bd)
    ),
    static_argnums=(1,),
)


@jax.jit
def _mixed(inp):
    # The same windows scored by the mixed pass (no uniform_q_len segment).
    n = inp["distribution"][2]
    inp = dict(inp, distribution=jnp.stack([0, 0, n]).astype(jnp.int32))
    kw = _kw(return_scores=True)
    kw["uniform_q_len"] = None
    return streamindex_topk(**inp, **kw)


def _rows(inp, n_seqs):
    return int(np.asarray(inp["cu_q_lens"])[n_seqs])


@pytest.mark.parametrize("name", list(VERIFY_CASES))
def test_verify_bd_matches_mixed_pass(name):
    seqs, T = VERIFY_CASES[name]
    inp = _case(seqs, T)
    n = _rows(inp, len(seqs))
    idx_bd, sc_bd = (np.asarray(x)[:n] for x in _verify(inp, True))
    idx_mx, sc_mx = (np.asarray(x)[:n] for x in _mixed(inp))
    np.testing.assert_array_equal(idx_bd, idx_mx)
    np.testing.assert_array_equal(sc_bd, sc_mx)


@pytest.mark.parametrize("name", ["v1@300k", "v3_ragged"])
def test_verify_bd_off_is_the_default_path(name):
    seqs, T = VERIFY_CASES[name]
    inp = _case(seqs, T)
    n = _rows(inp, len(seqs))
    off = [np.asarray(x)[:n] for x in _verify(inp, False)]
    default = [
        np.asarray(x)[:n]
        for x in jax.jit(lambda i: streamindex_topk(**i, **_kw(return_scores=True)))(
            inp
        )
    ]
    for a, b in zip(off, default):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nsplit", [8])
def test_verify_bd_kv_split_matches_unsplit(nsplit):
    # c4/c8 TP8 kv split from 128 tokens: verify windows ride the split too.
    seqs, T = ([(VQ, 40_007, "verify"), (VQ, 290_007, "verify"),
                (495, 127_495, "chunk")], 512)
    inp = _case(seqs, T)
    ref = np.asarray(
        jax.jit(lambda i: streamindex_topk(**i, **_kw(verify_bd=True)))(inp)
    )
    share = jax.jit(
        lambda i, r: jnp.concatenate(
            streamindex_topk(
                **i, **_kw(kv_split=nsplit, kv_split_rank=r, verify_bd=True)
            ),
            axis=1,
        )
    )
    merged = np.asarray(
        merge_kv_split_topk(
            jnp.stack([share(inp, jnp.int32(r)) for r in range(nsplit)]), K
        )
    )
    n = _rows(inp, len(seqs))
    assert int(np.sum(np.any(ref[:n] != merged[:n], axis=1))) == 0
