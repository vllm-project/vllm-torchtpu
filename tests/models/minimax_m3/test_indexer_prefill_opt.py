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
"""TPU_MINIMAX_M3_INDEXER_PREFILL_OPT, interpret mode: the 4-bit MSA
indexer's large (prefill) class scored with unpaired pages or another chunk
of pages per dot gives the same candidates and caches bit for bit, at one KV
head per rank (DCP 2) and two (DCP 4, folded index heads); default off."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as t
from vllm_torchtpu import envs
from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

ENV = "TPU_MINIMAX_M3_INDEXER_PREFILL_OPT"
BQ = "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q"


def test_parse(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert envs.TPU_MINIMAX_M3_INDEXER_PREFILL_OPT == ""
    assert msa_indexer.prefill_opt() == msa_indexer.PrefillOpt()
    monkeypatch.setenv(ENV, "unpair+chunk16")
    assert msa_indexer.prefill_opt() == msa_indexer.PrefillOpt(unpair=True, chunk=16)
    for bad in ("chunk3", "pair", "chunk128"):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError):
            msa_indexer.prefill_opt()


def test_chunk_override():
    assert msa_indexer._v3_chunk(64, 1024, 128) == 8
    assert msa_indexer._v3_chunk(64, 1024, 128, 16) == 16
    assert msa_indexer._v3_chunk(32, 1024, 128, 64) == 32


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
@pytest.mark.parametrize("opt", ["unpair", "chunk4", "unpair,chunk16"])
def test_candidates_bitwise(dcp, g, opt, monkeypatch):
    """A 300-token chunk in 128-row large tiles (the served path above 64
    rows), 32 pages per step: same candidates and caches."""
    monkeypatch.setattr(t, "Q_LENS", [300, 1, 3])
    monkeypatch.setattr(t, "CTX_LENS", [1300, 40, 300])
    rng, num_tokens, meta = t._setup(40 + g, dcp)
    layer = t._layer(rng, num_tokens, dcp, g, "fp4")
    monkeypatch.setenv(BQ, "128")
    monkeypatch.delenv(ENV, raising=False)
    ref_caches, ref = t._full(layer, meta, dcp)
    monkeypatch.setenv(ENV, opt)
    caches, cands = t._full(layer, meta, dcp)
    assert np.isfinite(np.asarray(ref)).any()
    np.testing.assert_array_equal(np.asarray(cands), np.asarray(ref))
    for r in range(dcp):
        for a, b in zip(caches[r], ref_caches[r]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
