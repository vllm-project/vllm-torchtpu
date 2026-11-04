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
"""TPU_SC_TOPK_COOP_V2 and TPU_SC_TOPK_COOP_MERGE are bitwise identical to the
default SparseCore top-k paths: same indices in the same order, same keys.

* ``sparsecore_topk.sparsecore_topk(coop_v2=True)``: the level loop and the
  winner gather (prefix scatter at P >= 4, overlapped peer copies at P = 2) on
  single-wave calls; multi-wave calls keep the default path.
* ``sparsecore_topk.sparsecore_topk_dcp_merge_coop``: the DCP merge split
  over P subcores by rank segment, against
  ``sparsecore_topk.sparsecore_topk_dcp_merge(rank_major=True)``.

Rows carry heavy ties (the tie split across subcores and the owner's prior ties
are the paths most likely to differ), -inf entries, -1 local slots and short /
empty rows. `test_comparison_detects_a_corrupted_row` is the positive control.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_topk

pytestmark = pytest.mark.tpu_test

K = 2048
N = 131072  # one DCP8 rank's width at --max-model-len 1M


def _require_sparsecore():
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:  # noqa: S110
        pass
    pytest.skip("needs SparseCore")


def _same(a, b) -> str | None:
    """None if the outputs are bit-identical, else where they first differ."""
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return f"output {i}: shape {x.shape} != {y.shape}"
        d = np.argwhere(x != y)
        if d.size:
            r, c = d[0]
            return (
                f"output {i}: {len(set(d[:, 0]))} rows differ, "
                f"first [{r},{c}] {x[r, c]} != {y[r, c]}"
            )
    return None


def _local_case(b, kind, seed):
    rng = np.random.default_rng(seed)
    s = np.full((b, N), -np.inf, np.float32)
    if kind == "short":
        lens = np.array(
            [
                0,
                1,
                17,
                2047,
                2048,
                2049,
                12900,
                47600,
                74500,
                130000,
                N,
                15,
                16,
                4095,
                4097,
                9400,
            ]
        )
    else:
        lens = np.array(
            [
                12900,
                12900,
                12900,
                12901,
                15100,
                15100,
                15100,
                15101,
                9400,
                9400,
                9400,
                9401,
                130000,
                74500,
                47600,
                13600,
            ]
        )
    lens = lens[np.arange(b) % lens.size].astype(np.int32)
    for r, ln in enumerate(lens):
        if kind == "ties":
            v = np.round(rng.standard_normal(ln) * 16) / 8
        elif kind == "short":
            v = rng.standard_normal(ln) * 3
        else:  # indexer-like: non-negative sums of relu terms
            v = np.maximum(rng.standard_normal((ln, 8)), 0).sum(-1)
        s[r, :ln] = v.astype(np.float32)
        s[r, ln : ln + 64] = 1e30  # would win if a row were read past its length
    if b > 1:
        s[1, : min(64, int(lens[1]))] = -np.inf  # in-range -inf
    return jnp.asarray(s), jnp.asarray(lens)


@pytest.mark.parametrize("b", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("kind", ["relu", "ties", "short"])
def test_coop_v2_local_is_bitwise(b, kind):
    _require_sparsecore()
    scores, lens = _local_case(b, kind, seed=b * 7 + len(kind))
    want = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=False)
    got = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=True)
    assert _same(got, want) is None, _same(got, want)


def test_coop_v2_engages_on_single_wave_only():
    _require_sparsecore()
    for b, engaged in (
        (1, False),
        (2, False),
        (4, True),
        (16, True),
        (32, False),
        (64, False),
    ):
        scores = jnp.zeros((b, N), jnp.float32)
        lens = jnp.full((b,), 4096, jnp.int32)
        txt = (
            jax.jit(
                lambda s, ln: sc_topk.sparsecore_topk(
                    s, K, ln, return_scores=True, coop_v2=True
                )
            )
            .lower(scores, lens)
            .as_text()
        )
        assert (
            bool(re.search(r"sc_topk_direct_b\d+_n\d+_k\d+_p\d+_v2", txt)) is engaged
        ), b


# The a2a merge's stage 2 (TPU_DCP_INDEXER_A2A_MERGE): rows of dcp * k = 8 * 2048.
N_A2A = 16384


def _a2a_case(b, kind, seed):
    """Stage-2 rows as `streamindex_topk_dcp(a2a_merge=True)` builds them: full width
    (row_lengths = n), every rank's k candidates, -inf for unfilled slots; plus one
    padding-token row that is -inf end to end and one whole-row tie, both spanning
    every subcore's part, and a short negative row (rows < k select everything)."""
    rng = np.random.default_rng(seed)
    if kind == "ties":
        s = (np.round(rng.standard_normal((b, N_A2A)) * 16) / 8).astype(np.float32)
    else:
        s = np.abs(rng.standard_normal((b, N_A2A))).astype(np.float32)
    fill = rng.integers(K // 4, K + 1, size=(b, 8))
    for r in range(b):
        for q in range(8):
            s[r, q * K + fill[r, q] : (q + 1) * K] = -np.inf
    s[0, :] = -np.inf  # padding token: no candidate anywhere
    s[1, :] = 0.75  # whole-row tie across all p parts
    s[2, :] = -np.inf
    s[2, : K - 300] = -np.abs(rng.standard_normal(K - 300)).astype(np.float32) * 3
    return jnp.asarray(s), jnp.full((b,), N_A2A, jnp.int32)


@pytest.mark.parametrize("b", [8, 16])
@pytest.mark.parametrize("kind", ["relu", "ties"])
def test_coop_v2_a2a_stage2_is_bitwise(b, kind):
    _require_sparsecore()
    scores, lens = _a2a_case(b, kind, seed=b + len(kind))
    want = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=False)
    got = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=True)
    assert _same(got, want) is None, _same(got, want)
    assert (np.asarray(got[0])[0] == -1).all()  # the padding row selects nothing
    # The comparison sees a fault INSIDE the kernel: an in-kernel "-inf" key moved up
    # to the key of +0.0 drops the all-negative row's candidates in the kernel only.
    key_neg_inf = sc_topk.KEY_NEG_INF
    try:
        jax.clear_caches()
        sc_topk.KEY_NEG_INF = 0
        bad = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=True)
    finally:
        sc_topk.KEY_NEG_INF = key_neg_inf
        jax.clear_caches()
    assert _same(bad, want) is not None, "in-kernel fault not detected"


def test_coop_v2_engages_through_the_a2a_dispatcher(monkeypatch):
    """The a2a merge's stage 2 reaches this kernel through
    sparsecore_topk.sparsecore_topk with coop_v2 read from the env: engaged
    exactly on the allow-listed shapes."""
    _require_sparsecore()
    monkeypatch.setattr(envs, "TPU_SC_TOPK_COOP_V2", True)
    for b, engaged in ((8, True), (16, True), (32, False), (4, False)):
        scores = jnp.zeros((b, N_A2A), jnp.float32)
        lens = jnp.full((b,), N_A2A, jnp.int32)
        txt = (
            jax.jit(
                lambda s, ln: sc_topk.sparsecore_topk.__wrapped__(
                    s, K, ln, write_empty_rows=True
                )
            )
            .lower(scores, lens)
            .as_text()
        )
        assert (
            bool(re.search(r"sc_topk_direct_b\d+_n\d+_k\d+_p\d+_v2", txt)) is engaged
        ), b


def _merge_case(b, dcp, k, seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "ties":
        vals = rng.integers(-64, 64, size=(dcp, b, k)).astype(np.float32) / 8.0
    else:
        vals = np.abs(rng.standard_normal((dcp, b, k))).astype(np.float32)
    fill = rng.integers(k // 4, k + 1, size=(dcp, b))
    vals[np.arange(k)[None, None, :] >= fill[..., None]] = -np.inf
    if b > 1:
        vals[:, 1, :] = -np.inf  # a row with no candidates anywhere
    idxs = rng.integers(0, 1 << 20, size=(b, k)).astype(np.int32)
    idxs[rng.random((b, k)) < 0.1] = -1
    return jnp.asarray(vals.view(np.int32).reshape(-1)), jnp.asarray(idxs)


@pytest.mark.parametrize("b", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dcp,k", [(8, 2048), (16, 2048), (2, 256), (8, 512)])
@pytest.mark.parametrize("kind", ["ties", "relu"])
def test_coop_merge_is_bitwise(b, dcp, k, kind):
    _require_sparsecore()
    if not sc_topk.coop_merge_fits(b, dcp, k):
        pytest.skip("shape outside the coop merge")
    flat, idxs = _merge_case(b, dcp, k, seed=b * 131 + dcp + k, kind=kind)
    for rank in sorted({0, dcp // 2, dcp - 1}):
        r = jnp.int32(rank)
        want = sc_topk.sparsecore_topk_dcp_merge(
            flat, idxs, k, dcp_rank=r, dcp_size=dcp, rank_major=True
        )
        got = sc_topk.sparsecore_topk_dcp_merge_coop(
            flat, idxs, k, dcp_rank=r, dcp_size=dcp
        )
        assert _same((got,), (want,)) is None, (rank, _same((got,), (want,)))


def test_coop_merge_route(monkeypatch):
    _require_sparsecore()
    monkeypatch.setattr(envs, "TPU_SC_TOPK_COOP_MERGE", True)
    for b, routed in ((4, True), (16, True), (32, False)):
        flat, idxs = _merge_case(b, 8, K, seed=b, kind="ties")
        txt = (
            jax.jit(
                lambda f, i: sc_topk.sparsecore_topk_dcp_merge.__wrapped__(
                    f, i, K, dcp_rank=jnp.int32(3), dcp_size=8, rank_major=True
                )
            )
            .lower(flat, idxs)
            .as_text()
        )
        assert (
            bool(re.search(r"sc_topk_direct_b\d+_n\d+_k\d+_p\d+_coop", txt)) is routed
        ), b
    assert sc_topk.coop_merge_fits(16, 8, K) and not sc_topk.coop_merge_fits(32, 8, K)
    assert not sc_topk.coop_merge_fits(4, 6, K)


def test_comparison_detects_a_corrupted_row():
    _require_sparsecore()
    scores, lens = _local_case(4, "ties", seed=3)
    want = sc_topk.sparsecore_topk(scores, K, lens, return_scores=True, coop_v2=False)
    bad = (want[0].at[2, 100].set(want[0][2, 101]), want[1])
    assert _same(bad, want) is not None
    flat, idxs = _merge_case(4, 8, K, seed=5, kind="ties")
    m = sc_topk.sparsecore_topk_dcp_merge_coop(
        flat, idxs, K, dcp_rank=jnp.int32(5), dcp_size=8
    )
    m_wrong_rank = sc_topk.sparsecore_topk_dcp_merge_coop(
        flat, idxs, K, dcp_rank=jnp.int32(4), dcp_size=8
    )
    assert _same((m,), (m_wrong_rank,)) is not None
