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
"""TPU_SC_TOPK_ONE_SUBCORE_CAPACITY is bitwise identical to the default
SparseCore top-k: same indices in the same order, same keys.

A call whose row width splits every row over P > 1 subcores also holds the
P = 1 body and runs it when no row is longer than the capacity. The rows below
reach the capacity exactly (the P = 1 body, its slice buffer full) or pass it
by one word (the P > 1 body), and carry ties, empty rows and a fence of 1e30
after each row's end that would win if a row were read past its length.
Which calls hold both bodies is decided before any kernel is built, so that
part runs on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_topk

K = 2048
N = 131072  # one DCP8 rank's width at --max-model-len 1M
CAPACITY = 2 * sc_topk.MAX_SLICE_WORDS


@pytest.mark.parametrize(
    "b,n,capacity,expected",
    [
        (256, N, CAPACITY, CAPACITY),
        (32, N, CAPACITY, CAPACITY),
        (17, N, CAPACITY, CAPACITY),
        (16, N, CAPACITY, 0),
        (256, N // 2, CAPACITY, 0),
        (256, N, 0, 0),
    ],
    ids=["prefill-chunk", "one-wave", "17-rows", "16-rows", "already-p1", "off"],
)
def test_which_calls_hold_the_one_subcore_body(b, n, capacity, expected):
    assert sc_topk._one_subcore_capacity(b, n, K, capacity) == expected


@pytest.mark.parametrize("capacity", [-sc_topk.LANES, 8, CAPACITY + sc_topk.LANES])
def test_capacity_must_fit_one_subcore(capacity):
    with pytest.raises(ValueError):
        sc_topk._one_subcore_capacity(256, N, K, capacity)


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
        d = np.argwhere(x != y)
        if d.size:
            r, c = d[0]
            return (
                f"output {i}: {len(set(d[:, 0]))} rows differ, "
                f"first [{r},{c}] {x[r, c]} != {y[r, c]}"
            )
    return None


def _case(b, longest, seed):
    rng = np.random.default_rng(seed)
    lens = rng.integers(0, longest + 1, size=b).astype(np.int32)
    lens[:6] = [0, 1, K - 1, K, K + 1, longest]
    s = np.full((b, N), -np.inf, np.float32)
    for r, ln in enumerate(lens):
        s[r, :ln] = np.round(np.abs(rng.standard_normal(ln)) * 8) / 8
        s[r, ln : ln + 64] = 1e30
    return jnp.asarray(s), jnp.asarray(lens)


@pytest.mark.tpu_test
@pytest.mark.parametrize("b", [32, 48, 256])
@pytest.mark.parametrize("longest", [CAPACITY, CAPACITY + 1])
def test_one_subcore_capacity_is_bitwise(b, longest):
    _require_sparsecore()
    scores, lens = _case(b, longest, seed=b + longest)
    want = sc_topk.sparsecore_topk(
        scores, K, lens, return_scores=True, one_subcore_capacity=0
    )
    got = sc_topk.sparsecore_topk(
        scores, K, lens, return_scores=True, one_subcore_capacity=CAPACITY
    )
    assert _same(got, want) is None, _same(got, want)


@pytest.mark.tpu_test
def test_kernel_name_shows_the_one_subcore_body():
    _require_sparsecore()
    scores = jnp.zeros((256, N), jnp.float32)
    lens = jnp.full((256,), 4096, jnp.int32)

    def text(capacity):
        return (
            jax.jit(
                lambda s, ln: sc_topk.sparsecore_topk(
                    s, K, ln, one_subcore_capacity=capacity
                )
            )
            .lower(scores, lens)
            .as_text()
        )

    assert f"sc_topk_direct_b256_n{N}_k{K}_p2_p1upto{CAPACITY}" in text(CAPACITY)
    assert "p1upto" not in text(0)
