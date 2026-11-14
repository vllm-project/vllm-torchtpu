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
"""``combine.weighted_sum``'s in-place read of the real tokens' pairs
(``top_k`` given, what Config.combine_real_tokens passes), in TPU interpret
mode on the CPU: the rows may run past ``tokens * top_k`` (a gather padded to
whole blocks), only the real pairs are read, and the sum is the one the
pair-order call gives on exactly those rows."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import combine

pytestmark = pytest.mark.cpu_test

LANES = 128
TOKENS, TOP_K, HIDDEN = 128, 4, 1024


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        yield


def _inputs(extra_rows, fp8_rows):
    rng = np.random.default_rng(7)
    n = TOKENS * TOP_K
    rows = rng.standard_normal((n + extra_rows, HIDDEN)).astype(np.float32)
    # Rows past the real pairs hold garbage the in-place read must not touch.
    rows[n:] = np.nan
    weights = rng.uniform(0.05, 1.0, (TOKENS, TOP_K)).astype(np.float32)
    if fp8_rows:
        finite = np.nan_to_num(rows, nan=1.0)
        scales = (np.abs(finite).max(axis=1) / 448.0).astype(np.float32)
        q = jnp.asarray(np.nan_to_num(rows, nan=0.0) / scales[:, None]).astype(
            jnp.float8_e4m3fn
        )
        if extra_rows:
            q = q.at[n:].set(jnp.nan)
        return q, jnp.asarray(scales), jnp.asarray(weights)
    return jnp.asarray(rows).astype(jnp.bfloat16), None, jnp.asarray(weights)


def _sum(rows, scales, weights, **kwargs):
    out = combine.weighted_sum(
        rows.reshape(rows.shape[0], HIDDEN // LANES, LANES),
        scales,
        weights,
        tokens_per_shard=TOKENS,
        hidden=HIDDEN,
        out_dtype=jnp.bfloat16,
        **kwargs,
    )
    return np.asarray(out.astype(jnp.float32))


@pytest.mark.parametrize("fp8_rows", [False, True], ids=["bf16_rows", "fp8_rows"])
@pytest.mark.parametrize("extra_rows", [0, 64], ids=["exact", "padded_past_pairs"])
def test_in_place_read_of_the_real_pairs_is_the_pair_order_sum(
    interpret, fp8_rows, extra_rows
):
    rows, scales, weights = _inputs(extra_rows, fp8_rows)
    n = TOKENS * TOP_K
    want = _sum(rows[:n], None if scales is None else scales[:n], weights)
    got = _sum(rows, scales, weights, top_k=TOP_K)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, want)


def test_in_place_read_refuses_too_few_rows():
    rows, _, weights = _inputs(0, False)
    with pytest.raises(ValueError, match="fewer than"):
        _sum(rows[: TOKENS * TOP_K - 8], None, weights, top_k=TOP_K)
