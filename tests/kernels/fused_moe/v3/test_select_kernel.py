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
"""select_top_k: the documented contract against lax.top_k, tie order,
and a row of NaN logits."""

import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.fused_moe.v3 import device
from vllm_torchtpu.kernels.fused_moe.v3.select_kernel import (
    UNROUTABLE_LOGIT,
    select_top_k,
)

SERVED_EXPERTS, SERVED_TOP_K, SERVED_SHARDS = 512, 10, 8


@contextlib.contextmanager
def interpret_pallas():
    """Runs pallas_call under interpret for the block, so a test needs no
    TPU."""
    real = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs.setdefault("interpret", True)
        return real(*args, **kwargs)

    with mock.patch.object(pl, "pallas_call", interpreted):
        yield


def reference_top_k(scores, top_k):
    weights, indices = jax.lax.top_k(scores, top_k)
    return np.asarray(weights), np.asarray(indices).astype(np.int32)


def softmax_scores(logits):
    return jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)


def select(scores, **kwargs):
    """select_top_k under jit, as the layer calls it (its operand pin only
    lowers there)."""
    return jax.jit(lambda s: select_top_k(s, **kwargs))(scores)


@pytest.mark.cpu_test
class SelectTopKTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("one_block", 32, 64, 4, 32),
        ("multi_block", 64, 32, 6, 16),
        ("top_k_one", 16, 128, 1, 16),
    )
    def test_matches_lax_top_k(self, rows, experts, top_k, block_rows):
        rng = np.random.default_rng(0)
        scores = softmax_scores(rng.normal(size=(rows, experts)))
        with interpret_pallas():
            weights, indices = select(scores, top_k=top_k, block_rows=block_rows)
        want_w, want_i = reference_top_k(scores, top_k)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_ties_break_on_the_lowest_index(self):
        rng = np.random.default_rng(1)
        rows, experts, top_k = 32, 24, 8
        logits = rng.choice(
            np.array([-1.0, 0.0, 1.0], np.float32), size=(rows, experts)
        )
        scores = softmax_scores(logits)
        self.assertLess(len(np.unique(np.asarray(scores)[0])), experts)
        with interpret_pallas():
            weights, indices = select(scores, top_k=top_k, block_rows=rows)
        want_w, want_i = reference_top_k(scores, top_k)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_nan_row_indices_stay_inside_the_expert_range(self):
        rng = np.random.default_rng(2)
        rows, experts, top_k = 32, 64, 6
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        logits[7, 3] = np.inf
        scores = softmax_scores(logits)
        self.assertTrue(np.isnan(np.asarray(scores)[7]).all())
        with interpret_pallas():
            _, indices = select(scores, top_k=top_k, block_rows=rows)
        indices = np.asarray(indices)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < experts).all())

    def test_nan_row_selects_the_lowest_expert_with_the_unroutable_logit(self):
        rng = np.random.default_rng(3)
        rows, experts, top_k = 16, 32, 5
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        logits[4, 0] = np.inf
        scores = softmax_scores(logits)
        with interpret_pallas():
            weights, indices = select(scores, top_k=top_k, block_rows=rows)
        np.testing.assert_array_equal(
            np.asarray(indices)[4], np.zeros((top_k,), np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(weights)[4], np.full((top_k,), UNROUTABLE_LOGIT, np.float32)
        )

    def test_nan_row_leaves_every_other_row_bitwise_unchanged(self):
        rng = np.random.default_rng(4)
        rows, experts, top_k = 32, 64, 6
        clean = rng.normal(size=(rows, experts)).astype(np.float32)
        poisoned = clean.copy()
        poisoned[11, 5] = np.inf
        with interpret_pallas():
            w_clean, i_clean = select(
                softmax_scores(clean), top_k=top_k, block_rows=rows
            )
            w_bad, i_bad = select(
                softmax_scores(poisoned), top_k=top_k, block_rows=rows
            )
        keep = [r for r in range(rows) if r != 11]
        np.testing.assert_array_equal(
            np.asarray(i_bad)[keep], np.asarray(i_clean)[keep]
        )
        np.testing.assert_array_equal(
            np.asarray(w_bad)[keep], np.asarray(w_clean)[keep]
        )

    def test_renormalized_weights_are_a_softmax_of_the_selected_logits(self):
        rng = np.random.default_rng(6)
        rows, experts, top_k = 32, 64, 8
        logits = jnp.asarray(rng.normal(size=(rows, experts)), jnp.float32)
        with interpret_pallas():
            weights, indices = select(
                logits, top_k=top_k, block_rows=rows, renormalize=True
            )
        top, want_i = jax.lax.top_k(logits, top_k)
        np.testing.assert_array_equal(np.asarray(indices), np.asarray(want_i))
        np.testing.assert_allclose(
            np.asarray(weights),
            np.asarray(jax.nn.softmax(top, axis=-1)),
            rtol=1e-6,
            atol=1e-7,
        )


class SelectTopKDeviceTest(parameterized.TestCase):
    def test_served_shape_matches_lax_top_k(self):
        if jax.default_backend() != "tpu" or not device.supported_on(jax.devices()[0]):
            self.skipTest("the served layout runs on the supported TPU")
        rng = np.random.default_rng(5)
        rows = 8192 // SERVED_SHARDS
        scores = softmax_scores(rng.normal(size=(rows, SERVED_EXPERTS)))
        weights, indices = select(scores, top_k=SERVED_TOP_K, block_rows=256)
        want_w, want_i = reference_top_k(scores, SERVED_TOP_K)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)


if __name__ == "__main__":
    absltest.main()
