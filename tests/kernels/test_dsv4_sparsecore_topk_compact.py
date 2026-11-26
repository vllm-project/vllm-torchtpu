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

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_topk
from vllm_torchtpu.kernels.deepseek_v4 import (
    sparsecore_topk_compact as sc_compact,
)


def _require_sparsecore() -> None:
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except (RuntimeError, NotImplementedError, ValueError):
        pass
    pytest.skip("needs SparseCore")


def _reference_mismatch(
    scores: np.ndarray,
    idxs: np.ndarray,
    k: int,
    row_lengths: np.ndarray,
) -> str | None:
    b, n = scores.shape
    if idxs.shape != (b, k):
        return f"shape {idxs.shape} != {(b, k)}"
    for r in range(b):
        rl = int(np.clip(row_lengths[r], 0, n))
        row_scores = scores[r, :rl]
        finite_pos = np.flatnonzero(row_scores > -np.inf)
        want_n = min(k, finite_pos.size)
        got = idxs[r]
        if not np.all(got[want_n:] == -1):
            return f"row {r}: suffix past {want_n} not -1"
        if want_n == 0:
            continue
        sel = got[:want_n]
        if np.any((sel < 0) | (sel >= rl)):
            return f"row {r}: out-of-range index in [0, {rl})"
        if np.unique(sel).size != want_n:
            return f"row {r}: duplicate indices"
        got_vals = np.sort(scores[r, sel])
        want_vals = np.sort(row_scores[finite_pos])[-want_n:]
        if not np.array_equal(got_vals, want_vals):
            return f"row {r}: score multisets differ"
    return None


@pytest.mark.parametrize(
    ("b", "n", "k", "return_scores", "expected"),
    [
        (16, 81920, 2048, False, None),
        (32, 32768, 2048, False, None),
        (128, 49136, 2048, False, None),
        (128, 81920, 2048, True, None),
        (32, 49152, 2048, False, "resident"),
        (128, 65536, 2048, False, "resident"),
        (128, 98304, 2048, False, "resident"),
        (128, 114688, 2048, False, "resident"),
        (128, 122880, 2048, False, "streaming"),
        (128, 131072, 2048, False, "streaming"),
        (128, 163840, 2048, False, "streaming"),
        (128, 1048576, 2048, False, "streaming"),
        (128, 114688, 4096, False, "streaming"),
        (131072, 131072, 2048, False, None),
    ],
)
def test_should_select_compact_p1_mode_across_shape_and_vmem_boundaries(
    b: int, n: int, k: int, return_scores: bool, expected: str | None
) -> None:
    got = sc_compact.select_compact_p1_mode(b, n, k, return_scores=return_scores)
    assert got == expected


def test_should_enable_dcp16_decode_shapes_in_coop_v2() -> None:
    expected = {
        (4, 16384, 8),
        (4, 10240, 4),
        (8, 10240, 4),
        (16, 12288, 2),
        (16, 10240, 2),
        (4, 32768, 8),
    }
    assert expected.issubset(sc_topk.COOP_V2_TESTED_SHAPES)
    assert (2, 16384, 16) not in sc_topk.COOP_V2_TESTED_SHAPES


def test_should_route_wide_batch_indices_only_to_compact_p1_with_group_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_compact(
        scores: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        mode: str,
        write_empty: bool = True,
        scheduling_group_id: int | None = None,
    ) -> jax.Array:
        calls.append(
            {
                "shape": scores.shape,
                "k": k,
                "mode": mode,
                "write_empty": write_empty,
                "scheduling_group_id": scheduling_group_id,
            }
        )
        return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

    monkeypatch.setattr(sc_topk, "sparsecore_topk_compact_p1", fake_compact)

    res = sc_topk.sparsecore_topk.__wrapped__(
        jnp.zeros((128, 65536), dtype=jnp.float32),
        2048,
        jnp.full((128,), 65536, dtype=jnp.int32),
        write_empty_rows=False,
        scheduling_group_id=7,
        return_scores=False,
    )
    assert res.shape == (128, 2048)

    str_out = sc_topk.sparsecore_topk.__wrapped__(
        jnp.zeros((128, 131072), dtype=jnp.float32),
        2048,
        jnp.full((128,), 131072, dtype=jnp.int32),
        write_empty_rows=True,
        scheduling_group_id=9,
        return_scores=False,
    )
    assert str_out.shape == (128, 2048)
    assert calls == [
        {
            "shape": (128, 65536),
            "k": 2048,
            "mode": "resident",
            "write_empty": False,
            "scheduling_group_id": 7,
        },
        {
            "shape": (128, 131072),
            "k": 2048,
            "mode": "streaming",
            "write_empty": True,
            "scheduling_group_id": 9,
        },
    ]


def test_should_reject_corrupted_output_in_reference_oracle() -> None:
    scores = np.arange(64, dtype=np.float32).reshape(2, 32)
    row_lengths = np.array([32, 10], dtype=np.int32)
    good = np.array(
        [
            list(range(16, 32)),
            list(range(10)) + [-1] * 6,
        ],
        dtype=np.int32,
    )
    assert _reference_mismatch(scores, good, 16, row_lengths) is None

    bad = good.copy()
    bad[0, 0] = 0
    assert _reference_mismatch(scores, bad, 16, row_lengths) is not None

    bad_pad = good.copy()
    bad_pad[1, 12] = 0
    assert _reference_mismatch(scores, bad_pad, 16, row_lengths) is not None


@pytest.mark.tpu_test
@pytest.mark.parametrize("n", [65536, 98304, 131072, 163840])
def test_should_match_exact_topk_on_adversarial_inputs_in_compact_p1(n: int) -> None:
    _require_sparsecore()
    b, k = 64, 2048
    rng = np.random.default_rng(1870 + n)
    scores = np.maximum(rng.standard_normal((b, n)).astype(np.float32) * 2.0 + 3.0, 0.0)
    row_lengths = np.full((b,), n, dtype=np.int32)
    # Row 0: empty, Row 1: short < k, Row 2: non-16-aligned + poison past row_len,
    # Row 3: all -inf, Row 4: all negative finite, Row 5: tie storm (> cand_cap),
    # Row 6: 9000 ties at boundary forcing fallback, Row 7: sparse finite + -inf.
    row_lengths[0] = 0
    row_lengths[1] = 137
    row_lengths[2] = n - 19
    scores[2, n - 19 :] = 1e30
    scores[3, :] = -np.inf
    scores[4, :] = -np.abs(rng.standard_normal(n).astype(np.float32)) - 1.0
    scores[5, :] = 1.25
    scores[6, :9000] = 5.0
    scores[6, 9000:] = 1.0
    scores[7, :] = -np.inf
    scores[7, :53] = -np.arange(53, dtype=np.float32) - 0.5

    out = np.asarray(
        jax.block_until_ready(
            sc_topk.sparsecore_topk(
                jnp.asarray(scores),
                k,
                jnp.asarray(row_lengths),
                scheduling_group_id=5,
                return_scores=False,
            )
        )
    )
    assert _reference_mismatch(scores, out, k, row_lengths) is None


@pytest.mark.tpu_test
@pytest.mark.parametrize(("n", "tag"), [(65536, "_p1_c11"), (131072, "_p1_s12")])
def test_should_emit_compact_kernel_and_scheduling_group_in_hlo(
    n: int, tag: str
) -> None:
    _require_sparsecore()
    scores = jnp.zeros((128, n), dtype=jnp.float32)
    row_lengths = jnp.full((128,), n, dtype=jnp.int32)
    txt = sc_topk.sparsecore_topk.lower(
        scores, 2048, row_lengths, scheduling_group_id=4, return_scores=False
    ).as_text()
    assert tag in txt
    assert '_scheduling_group_id = "4"' in txt
