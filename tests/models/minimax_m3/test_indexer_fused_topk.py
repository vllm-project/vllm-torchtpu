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
"""TPU_MINIMAX_M3_INDEXER_FUSED_TOPK, interpret mode: the DCP prefill
indexer's large tile class picking its local top-k inside the block-score
kernel gives the same candidates (and caches) bit for bit as the separate
`msa_dcp_candidates` call: 4-bit (folded scales) and fp8 caches, one and two
KV heads per rank, tiles of one and several page groups, forced init / local
blocks and exact score ties (pages with identical keys)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as t
from vllm_torchtpu import envs
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_indexer

ENV = "TPU_MINIMAX_M3_INDEXER_FUSED_TOPK"
WIDE = "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q"


def test_default_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert envs.TPU_MINIMAX_M3_INDEXER_FUSED_TOPK is False
    monkeypatch.setenv(ENV, "1")
    assert envs.TPU_MINIMAX_M3_INDEXER_FUSED_TOPK is True


def _run(monkeypatch, *, dcp, g, dtype, seed, wide=None, ties=False):
    rng, num_tokens, meta = t._setup(seed, dcp)
    layer = t._layer(rng, num_tokens, dcp, g, dtype)
    if ties:
        # Pages backing several context blocks get the keys of one page: the
        # same scores, so picks among them are decided by the column order.
        tables = np.asarray(meta[1]).reshape(t.MAX_SEQS, -1)
        dup = tables[0, 1:6]
        layer["index"] = [
            idx.at[dup].set(jnp.broadcast_to(idx[tables[0, 0]], idx[dup].shape))
            for idx in layer["index"]
        ]
    if wide:
        monkeypatch.setenv(WIDE, str(wide))
    else:
        monkeypatch.delenv(WIDE, raising=False)
    monkeypatch.delenv(ENV, raising=False)
    ref_caches, ref = t._full(layer, meta, dcp)
    monkeypatch.setenv(ENV, "1")
    caches, cands = t._full(layer, meta, dcp)
    num_real = sum(t.Q_LENS)
    ref, cands = np.asarray(ref), np.asarray(cands)
    assert np.isfinite(ref[:, :num_real]).any()
    # Every real token's candidates, as bits (scores and ids).
    np.testing.assert_array_equal(
        cands[:, :num_real].view(np.uint32), ref[:, :num_real].view(np.uint32)
    )
    for r in range(dcp):
        for a, b in zip(caches[r], ref_caches[r]):
            if a is not None:
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    return ref[:, :num_real]


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
@pytest.mark.parametrize("wide", [72, 128])
def test_fp4_bitwise(dcp, g, wide, monkeypatch):
    """A 300-token chunk (the large class) in `wide`-row tiles, as
    test_kv4_indexer_large_block_q."""
    monkeypatch.setattr(t, "Q_LENS", [300, 1, 3])
    monkeypatch.setattr(t, "CTX_LENS", [1300, 40, 300])
    _run(monkeypatch, dcp=dcp, g=g, dtype="fp4", seed=40 + g, wide=wide)


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
def test_fp4_page_groups_ties(dcp, g, monkeypatch):
    """Tiles of several page groups (4 pages per grid step over ~3K tokens of
    context), two large requests, forced init and local blocks, and ties."""
    monkeypatch.setattr(t, "Q_LENS", [300, 200, 3])
    monkeypatch.setattr(t, "CTX_LENS", [3000, 1700, 300])
    monkeypatch.setattr(t, "PAGES", 64)
    monkeypatch.setattr(t, "TOPK", 5)
    monkeypatch.setattr(t, "CONFIG", dict(init_blocks=2, local_blocks=2))
    monkeypatch.setattr(msa_indexer, "DEFAULT_PAGES_PER_STEP", 4)
    ref = _run(monkeypatch, dcp=dcp, g=g, dtype="fp4", seed=7 + g, wide=72, ties=True)
    # Picks past the forced ones exist (the ties are exercised).
    topk = t.TOPK
    ids = ref[..., topk : 2 * topk]
    assert (ids >= 0).sum() > ref.shape[0] * ref.shape[1] * 3


def test_fp8_bitwise(monkeypatch):
    """fp8 pages (default 512-row large tiles)."""
    monkeypatch.setattr(t, "Q_LENS", [300, 1, 3])
    monkeypatch.setattr(t, "CTX_LENS", [1300, 40, 300])
    _run(monkeypatch, dcp=2, g=1, dtype=jnp.float8_e4m3fn, seed=11)


def test_fused_needs_v3(monkeypatch):
    """No silent fallback: a fused select on a kernel it does not support raises."""
    sel = msa_indexer.FusedSelect(
        topk=2, block_size=128, init_blocks=0, local_blocks=1, dcp=2,
        rank=msa_dcp.rank_array(0), pos=None,
        widths=lambda n: [n], unrolled_width=512, pick_unroll=2,
        block_rows=256, forced_local=1e29, forced_init=1e30,
    )  # fmt: skip
    q = jnp.zeros((300, 1, 128), jnp.bfloat16)
    cache = jnp.zeros((4, 32, 4, 128), jnp.float8_e4m3fn)
    seq_lens = jnp.asarray([400, 0], jnp.int32)
    tables = jnp.asarray([1, 2, 0, 0], jnp.int32)
    cu = jnp.asarray([0, 300, 300], jnp.int32)
    with pytest.raises(NotImplementedError):
        msa_indexer._class_scores(
            q, cache, seq_lens, tables, cu, jnp.asarray([True, False]),
            block_q=256, max_tiles=2, pages_per_step=32, kernel="v2",
            interpret=True, select=sel,
        )  # fmt: skip
