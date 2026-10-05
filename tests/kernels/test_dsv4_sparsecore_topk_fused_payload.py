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
"""`sparsecore_topk(fused_payload=True)` must return the unfused arrays exactly.

The fused path changes no selection: it only replaces what the kernel's emit
writes (a payload value instead of a position), removing the XLA gathers that
mapped positions afterwards. So the bar here is `array_equal` on the indices --
slot order and `-1` padding included -- not the multiset comparison the
selection tests use. Ties are deliberately heavy: if the payload were applied
to a different slot than the position, ties are where it would still look
plausible.

`test_comparison_detects_a_misplaced_window` is the positive control: the same
comparison against a payload window shifted by one chunk must fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    LANES,
    _pick_partition,
    payload_fits,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import dcp_sc_compact

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = 20261001
K = 2048


def _require_sparsecore():
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs a TPU with SparseCore")


def _scores(kind, b, n, rng):
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
        return rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(np.float32)
    if kind == "strict_desc":
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    raise ValueError(kind)


def _np(x):
    return np.asarray(jax.block_until_ready(x))


# (b, n, p): the GLM-5.2 DCP16 decode shapes first.
SHAPES = [
    # Per-rank local top-k at T=16 over 65536 local positions.
    pytest.param(16, 65536, 2, id="decode-local-b16-n65536"),
    # The cross-rank merge at T=16: 16 ranks x k=2048 gathered scores.
    pytest.param(16, 16 * K, 2, id="decode-merge-b16-n32768"),
    pytest.param(16, 133120, 8, id="p8-b16-n133120"),
    pytest.param(64, 10240, 1, id="p1-b64-n10240"),
    # Multi-wave single stage: double-buffered pipeline with the payload.
    pytest.param(256, 16 * K, 1, id="multiwave-b256-n32768"),
]


def test_payload_fits_budget():
    # Decode merge stage 2, decode merge stage 1 window, mixed-step merge.
    assert payload_fits(16, 4096, K, 4096)
    assert payload_fits(32, 16384, K, K)
    assert payload_fits(4096, 32768, K, K)
    # A full-width payload on a double-buffered 32K slice does not fit.
    assert not payload_fits(4096, 32768, K, 32768)
    assert not payload_fits(16, 4096, K, 4096 + 8)


@pytest.mark.parametrize("b,n,p", SHAPES)
@pytest.mark.parametrize("kind", ["random", "heavy_ties", "strict_desc"])
@pytest.mark.parametrize("ragged", [False, True])
def test_fused_indices_equal_unfused(b, n, p, kind, ragged):
    _require_sparsecore()
    assert _pick_partition(b, n, K) == p
    rng = np.random.default_rng(SEED)
    scores = jnp.asarray(_scores(kind, b, n, rng))
    lengths = (
        rng.integers(0, n + 1, size=b).astype(np.int32)
        if ragged
        else np.full(b, n, np.int32)
    )
    lengths = jnp.asarray(lengths)
    want_i, want_s = sparsecore_topk(scores, K, lengths, return_scores=True)
    got_i, got_s = sparsecore_topk(
        scores, K, lengths, return_scores=True, fused_payload=True
    )
    np.testing.assert_array_equal(_np(got_i), _np(want_i))
    np.testing.assert_array_equal(_np(got_s), _np(want_s))


def test_fused_removes_the_map_back_gather():
    """At the decode shape, the fused program has no gather left at all."""
    scores = jax.ShapeDtypeStruct((16, 65536), jnp.float32)
    lengths = jax.ShapeDtypeStruct((16,), jnp.int32)
    if jax.default_backend() != "tpu":
        pytest.skip("lowering needs the SparseCore geometry")
    unfused = sparsecore_topk.lower(scores, K, lengths).as_text()
    fused = sparsecore_topk.lower(scores, K, lengths, fused_payload=True).as_text()
    assert "gather" in unfused, "the reference path no longer gathers"
    assert "gather" not in fused


def _owned_case(t, dcp, kind, seed):
    rng = np.random.default_rng(seed)
    scores = _scores(kind, t, dcp * K, rng)
    # Each rank's candidate list: distinct local indices, with -inf slots
    # (unfilled) carrying -1, exactly as the local top-k pads.
    # Distinct within a row, as a real candidate list is.
    local = np.stack(
        [rng.permutation(1 << 20)[: t * K].reshape(t, K) for _ in range(dcp)]
    ).astype(np.int32)
    dead = rng.random((dcp, t, K)) < 0.05
    local[dead] = -1
    s = scores.reshape(t, dcp, K)
    s[np.transpose(dead, (1, 0, 2))] = -np.inf
    return scores, local


@pytest.mark.parametrize("t", [16, 256])
@pytest.mark.parametrize("kind", ["random", "heavy_ties"])
def test_fused_resolve_equals_resolve_owned_winners(t, kind):
    """The DCP rank-local resolve, for every rank of a DCP16 group."""
    _require_sparsecore()
    dcp = 16
    scores, local = _owned_case(t, dcp, kind, SEED + t)
    scores = jnp.asarray(scores)
    full = jnp.full((t,), dcp * K, jnp.int32)
    slots = sparsecore_topk(scores, K, full)
    for r in range(dcp):
        want = dcp_sc_compact.resolve_owned_winners(
            jnp.asarray(local[r]), slots, K, jnp.int32(r)
        )
        owned = sparsecore_topk(
            scores,
            K,
            full,
            payload=jnp.asarray(local[r]),
            payload_offset=jnp.full((t,), r * K, jnp.int32),
            fused_payload=True,
        )
        got = dcp_sc_compact.pack_nonnegative(owned)
        np.testing.assert_array_equal(_np(got), _np(want), err_msg=f"rank {r}")


@pytest.mark.parametrize("b,n", [(16, 16 * K), (4096, 16 * K)])
def test_fused_payload_equals_gather_fallback(b, n):
    """Payload with per-row windows: fused kernel vs the XLA gather fallback.

    (4096, 32768) is the mixed-step merge shape (single stage, 128 waves).
    """
    _require_sparsecore()
    rng = np.random.default_rng(SEED)
    scores = jnp.asarray(_scores("heavy_ties", b, n, rng))
    payload = jnp.asarray(rng.integers(0, 1 << 30, size=(b, K), dtype=np.int32))
    offs = jnp.asarray(rng.integers(0, n // K, size=b).astype(np.int32) * K)
    full = jnp.full((b,), n, jnp.int32)
    kw = dict(payload=payload, payload_offset=offs)
    want = sparsecore_topk(scores, K, full, **kw)
    got = sparsecore_topk(scores, K, full, fused_payload=True, **kw)
    np.testing.assert_array_equal(_np(got), _np(want))
    assert (_np(got) >= 0).any(), "no row selected inside its window"


def test_comparison_detects_a_misplaced_window():
    _require_sparsecore()
    t, dcp, r = 16, 16, 3
    scores, local = _owned_case(t, dcp, "random", SEED)
    scores = jnp.asarray(scores)
    full = jnp.full((t,), dcp * K, jnp.int32)
    slots = sparsecore_topk(scores, K, full)
    want = _np(
        dcp_sc_compact.resolve_owned_winners(
            jnp.asarray(local[r]), slots, K, jnp.int32(r)
        )
    )
    wrong = sparsecore_topk(
        scores,
        K,
        full,
        payload=jnp.asarray(local[r]),
        payload_offset=jnp.full((t,), r * K + LANES, jnp.int32),
        fused_payload=True,
    )
    assert not np.array_equal(_np(dcp_sc_compact.pack_nonnegative(wrong)), want)
