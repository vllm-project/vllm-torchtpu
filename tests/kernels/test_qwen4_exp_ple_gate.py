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
"""CPU interpretation of the production PLE gate's BF16/reduction boundaries."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple_gate import normalized_gate_dot

STREAMS, HIDDEN = 4, 2560
BF16 = ml_dtypes.bfloat16


def _fixture(tokens):
    rng = np.random.default_rng(1968)
    key, query = (
        rng.normal(size=(tokens, STREAMS * HIDDEN)).astype(BF16) for _ in range(2)
    )
    weights = [rng.normal(0, 0.1, STREAMS * HIDDEN).astype(BF16) for _ in range(2)]
    # RMS statistics are computed independently in NumPy.
    inverses = [
        1
        / np.sqrt(
            np.mean(
                raw.astype(np.float32).reshape(tokens, STREAMS, HIDDEN) ** 2, axis=-1
            )
            + np.float32(1e-6)
        )
        for raw in (key, query)
    ]
    return key, query, *inverses, *weights


def _reference(args, *, round_norm=True, round_product=True, reassociate=False):
    key, query, key_inv, query_inv, key_weight, query_weight = args
    operands = []
    for raw, inverse, weight in (
        (key, key_inv, key_weight),
        (query, query_inv, query_weight),
    ):
        norm = raw.astype(np.float32).reshape(-1, STREAMS, HIDDEN) * inverse[..., None]
        norm = norm * (
            np.float32(1) + weight.astype(np.float32).reshape(STREAMS, HIDDEN)
        )
        operands.append(norm.astype(BF16).astype(np.float32) if round_norm else norm)
    product = operands[0] * operands[1]
    if round_product:
        product = product.astype(BF16).astype(np.float32)
    total = np.zeros(product.shape[:2], np.float32)
    for block in range(HIDDEN // 512):
        vectors = product[..., block * 512 : (block + 1) * 512].reshape(
            -1, STREAMS, 4, 128
        )
        if reassociate:
            lanes = (
                (vectors[..., 0, :] + vectors[..., 2, :]) + vectors[..., 1, :]
            ) + vectors[..., 3, :]
        else:
            lanes = (
                (vectors[..., 0, :] + vectors[..., 1, :]) + vectors[..., 2, :]
            ) + vectors[..., 3, :]
        total = total + np.sum(lanes, axis=-1, dtype=np.float32)
    return total.astype(BF16)


def _run(args):
    return np.asarray(
        normalized_gate_dot(
            *(jnp.asarray(value) for value in args), streams=STREAMS, interpret=True
        )
    )


@pytest.mark.parametrize("tokens", [0, 1, 7, 8, 9, 128, 129, 1024])
def test_gate_interpret_matches_independent_bf16_oracle(tokens):
    args = _fixture(tokens)
    actual = _run(args)
    assert actual.shape == (tokens, STREAMS)
    assert actual.dtype == BF16
    np.testing.assert_array_equal(actual, _reference(args))


@pytest.mark.parametrize("boundary", ["norm", "product"])
def test_gate_oracle_detects_missing_bf16_rounding(boundary):
    args = _fixture(7)
    expected = _reference(args)
    wrong = _reference(
        args, round_norm=boundary != "norm", round_product=boundary != "product"
    )
    assert np.any(expected != wrong), "fixture must detect the missing BF16 boundary"
    np.testing.assert_array_equal(_run(args), expected)


def test_gate_oracle_detects_reassociated_reduction():
    raw = np.ones((1, STREAMS * HIDDEN), BF16)
    key_weight = np.zeros((STREAMS, HIDDEN), BF16)
    query_weight = np.full((STREAMS, HIDDEN), -1, BF16)
    # Sequential accumulation gives 0; cancelling the large terms first gives 1.
    query_weight[:, 0] = 2**24
    query_weight[:, 128] = 0
    query_weight[:, 256] = -(2**24)
    inverses = np.ones((1, STREAMS), np.float32)
    args = (
        raw,
        raw,
        inverses,
        inverses,
        key_weight.reshape(-1),
        query_weight.reshape(-1),
    )
    expected, wrong = _reference(args), _reference(args, reassociate=True)
    np.testing.assert_array_equal(expected, np.zeros((1, STREAMS)))
    np.testing.assert_array_equal(wrong, np.ones((1, STREAMS)))
    np.testing.assert_array_equal(_run(args), expected)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_gate_nonfinite_propagation(value):
    args = list(_fixture(1))
    args[0][0, 0] = value
    with np.errstate(invalid="ignore", over="ignore"):
        expected = _reference(args)
        actual = _run(args)
    np.testing.assert_array_equal(
        actual.astype(np.float32), expected.astype(np.float32)
    )


@pytest.mark.parametrize(
    "argument", ["query_shape", "inverse_shape", "raw_dtype", "weight_shape"]
)
def test_gate_rejects_incompatible_operands(argument):
    args = list(map(jnp.asarray, _fixture(1)))
    if argument == "query_shape":
        args[1] = args[1][:, :-1]
    elif argument == "inverse_shape":
        args[2] = args[2][:, :-1]
    elif argument == "raw_dtype":
        args[0] = args[0].astype(jnp.float32)
    else:
        args[4] = args[4][:-1]
    with pytest.raises(ValueError):
        normalized_gate_dot(*args, streams=STREAMS, interpret=True)
