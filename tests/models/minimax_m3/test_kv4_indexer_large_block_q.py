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
"""TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q, interpret mode: the 4-bit MSA
indexer's large (prefill) class scored in tiles of another row count gives
the same caches and candidates bit for bit (each page maximum is one dot of
the same keys and query row), at one KV head per rank (DCP 2) and two (DCP 4,
folded index heads); fp8 caches and short steps keep the default tiles."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as t
from vllm_torchtpu import envs
from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

ENV = "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q"


def test_large_block_q_selection(monkeypatch):
    fp4 = jnp.zeros((2, 17, 4, 128), jnp.uint8)
    fp8 = jnp.zeros((2, 32, 4, 128), jnp.float8_e4m3fn)
    monkeypatch.delenv(ENV, raising=False)
    assert envs.TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q == 0
    assert msa_indexer.kv4_large_block_q(fp4, 8192, 512) == 512
    monkeypatch.setenv(ENV, "1024")
    assert msa_indexer.kv4_large_block_q(fp4, 8192, 512) == 1024
    assert msa_indexer.kv4_large_block_q(fp4, 2047, 512) == 512  # short step
    assert msa_indexer.kv4_large_block_q(fp8, 8192, 512) == 512  # fp8 cache
    for bad in ("100", "64"):  # not a multiple of 8; transposed-size tiles
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError):
            msa_indexer.kv4_large_block_q(fp4, 8192, 512)


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
@pytest.mark.parametrize("wide", [72, 128])
def test_candidates_bitwise(dcp, g, wide, monkeypatch):
    """A 300-token chunk is the large class: one 512-row tile by default,
    several `wide`-row tiles with the knob (served use widens the tiles,
    this narrows them to keep interpret mode small; the path is the same
    above 64 rows); same candidates and caches."""
    monkeypatch.setattr(t, "Q_LENS", [300, 1, 3])
    monkeypatch.setattr(t, "CTX_LENS", [1300, 40, 300])
    rng, num_tokens, meta = t._setup(40 + g, dcp)
    layer = t._layer(rng, num_tokens, dcp, g, "fp4")
    monkeypatch.delenv(ENV, raising=False)
    ref_caches, ref = t._full(layer, meta, dcp)
    monkeypatch.setenv(ENV, str(wide))
    assert msa_indexer.kv4_large_block_q(layer["index"][0], g * num_tokens, 512) == wide
    caches, cands = t._full(layer, meta, dcp)
    assert np.isfinite(np.asarray(ref)).any()
    np.testing.assert_array_equal(np.asarray(cands), np.asarray(ref))
    for r in range(dcp):
        for a, b in zip(caches[r], ref_caches[r]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
