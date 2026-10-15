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
"""Exact CSA decode TopK with short rows and poisoned cache capacity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import tuned_block_sizes


@pytest.mark.parametrize(
    "length", [0, 1, 511, 512, 513, 4607, 4608, 4609, 16384, 16640, 18432]
)
@pytest.mark.parametrize("tied", [False, True])
def test_exact_decode_selection(length: int, tied: bool):
    rng = np.random.default_rng(121110)
    scores = rng.standard_normal((1, 18432), dtype=np.float32)
    if tied:
        scores = np.floor(scores * 2)
    scores[:, length:] = 1e9
    result = np.asarray(
        sparsecore_topk(jnp.asarray(scores), 512, jnp.array([length], jnp.int32))
    )[0]
    valid = result[result >= 0]
    assert len(valid) == min(length, 512)
    assert np.all(result[len(valid) :] == -1)
    assert len(valid) == len(np.unique(valid))
    assert np.all(valid < length)
    expected = np.sort(scores[0, :length])[-512:]
    np.testing.assert_array_equal(np.sort(scores[0, valid]), expected)


def test_decode_tuning_is_present_in_compiled_hlo():
    if jax.devices()[0].device_kind != "TPU7x":
        pytest.skip("Measured tuning targets TPU7x")
    compiled = (
        jax.jit(sparsecore_topk, static_argnums=(1,))
        .lower(
            jax.ShapeDtypeStruct((1, 18432), jnp.float32),
            512,
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        .compile()
    )
    assert "sc_topk_direct_b1_n18432_k512_p4" in compiled.as_text()


def test_other_hardware_retains_fallback(monkeypatch):
    monkeypatch.setattr(tuned_block_sizes, "get_device_name", lambda: "TPU v6e")
    assert tuned_block_sizes.get_topk_partition(1, 18432, 512, default=16) == 16


@pytest.mark.parametrize(
    "capacity,divisor,width,batch",
    [
        (256, 4, 18432, 1),
        (16, 4, 18432, 1),
        (1, 4, 18432, 1),
        (256, 1, 18432, 1),
        (256, 8, 18432, 1),
        (256, 4, 4096, 1),
        (256, 4, 4096, 2),
    ],
)
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.tpu_test
def test_scaled_lengths_ignore_capacity_and_poisoned_tail(
    capacity: int, divisor: int, width: int, batch: int, tied: bool
):
    """Use changing raw lengths; unused metadata and score tails are poisoned."""
    rng = np.random.default_rng(121334)
    values = rng.standard_normal((batch, width), dtype=np.float32)
    if tied:
        values = np.floor(values * 2)
    raw_cases = [
        0,
        1,
        divisor - 1,
        divisor,
        divisor * 511 - 1,
        divisor * 512,
        divisor * 512 + 1,
        divisor * 513,
        divisor * (width // 4) - 1,
        divisor * (width // 4),
        divisor * (width // 4) + 1,
        divisor * (width - 1),
        divisor * width,
        divisor * width + 1,
    ]
    for raw in raw_cases:
        lengths = np.full(capacity, np.iinfo(np.int32).max, np.int32)
        lengths[:batch] = raw + np.arange(batch)
        effective = np.minimum(lengths[:batch] // divisor, width)
        scores = values.copy()
        for row in range(batch):
            scores[row, effective[row] :] = 1e9
        expected = sparsecore_topk(
            jnp.asarray(scores), 512, jnp.asarray(effective), return_scores=True
        )
        actual = sparsecore_topk(
            jnp.asarray(scores),
            512,
            jnp.asarray(lengths),
            row_length_divisor=divisor,
            return_scores=True,
        )
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        indices = np.asarray(actual[0])
        for row in range(batch):
            valid = indices[row][indices[row] >= 0]
            assert len(valid) == min(effective[row], 512)
            assert len(valid) == len(np.unique(valid))
            assert np.all(valid < effective[row])
            np.testing.assert_array_equal(
                np.sort(scores[row, valid]),
                np.sort(scores[row, : effective[row]])[-512:],
            )


@pytest.mark.tpu_test
def test_scaled_length_contract_rejects_invalid_metadata():
    scores = jnp.zeros((1, 18432), jnp.float32)
    with pytest.raises(ValueError, match="must be positive"):
        sparsecore_topk(scores, 512, jnp.ones(1, jnp.int32), row_length_divisor=0)
    with pytest.raises(ValueError, match="requires explicit"):
        sparsecore_topk(scores, 512, row_length_divisor=4)
    with pytest.raises(ValueError, match="at least"):
        sparsecore_topk(scores, 512, jnp.ones((1, 1), jnp.int32))
