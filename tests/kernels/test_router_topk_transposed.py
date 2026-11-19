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
"""TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS: the transposed router top-k kernel
picks the row kernel's experts and weights bit for bit (interpret)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import router_topk

ENV = "TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS"


def _sel(monkeypatch, scores, topk, env):
    monkeypatch.setenv(ENV, env)
    w, i = router_topk.select(jnp.asarray(scores), topk, interpret=True)
    return np.asarray(w), np.asarray(i)


@pytest.mark.parametrize(
    "rows,experts,topk", [(128, 128, 4), (1024, 128, 4), (640, 128, 8), (256, 512, 10)]
)
def test_transposed_matches_row_kernel(monkeypatch, rows, experts, topk):
    rng = np.random.default_rng(rows + experts)
    scores = np.round(rng.random((rows, experts), dtype=np.float32), 2)  # ties
    scores[3] = np.nan
    scores[5, :7] = -np.inf
    scores[7] = -np.inf
    rw, ri = _sel(monkeypatch, scores, topk, "0")
    tw, ti = _sel(monkeypatch, scores, topk, "128")
    np.testing.assert_array_equal(ti, ri)
    np.testing.assert_array_equal(tw, rw)  # NaN rows compare equal


def test_rows_not_multiple_of_128_keep_row_kernel(monkeypatch):
    rng = np.random.default_rng(1)
    scores = rng.random((200, 128), dtype=np.float32)
    rw, ri = _sel(monkeypatch, scores, 4, "0")
    tw, ti = _sel(monkeypatch, scores, 4, "128")
    np.testing.assert_array_equal(ti, ri)
    np.testing.assert_array_equal(tw, rw)


@pytest.mark.parametrize("rows,experts,topk", [(256, 128, 4), (640, 128, 8)])
def test_biased_matches_add_topk_gather(monkeypatch, rows, experts, topk):
    """select_biased == top-k of scores + bias, then gather the scores."""
    rng = np.random.default_rng(rows * 3 + topk)
    scores = np.round(rng.random((rows, experts), dtype=np.float32), 2)
    scores[3] = np.nan
    scores[7] = -np.inf
    bias = np.round(rng.standard_normal(experts).astype(np.float32) * 0.1, 2)
    monkeypatch.setenv(ENV, "0")
    _, ri = router_topk.select(jnp.asarray(scores + bias[None, :]), topk, interpret=True)
    rw = np.take_along_axis(scores, np.asarray(ri), axis=1)
    tw, ti = router_topk.select_biased(
        jnp.asarray(scores), jnp.asarray(bias), topk, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(ti), np.asarray(ri))
    np.testing.assert_array_equal(np.asarray(tw), rw)
