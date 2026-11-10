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
"""Tests for fused_moe.v3.combine weighted_sum, combine, and padded-row handling."""

import re
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src.pallas.mosaic import tpu_info as tpu_info_mod
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.fused_moe.v3 import combine


@pytest.mark.multichip
class CombineTest(absltest.TestCase):
    def setUp(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("the combine runs on TPU only")

    def test_a_zero_weight_route_contributes_nothing(self):
        tokens, top_k, lane_blocks = 128, 4, 8
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        rows = jax.random.normal(
            jax.random.PRNGKey(0), (tokens, top_k, lane_blocks, lanes), jnp.float32
        )
        weights = jax.random.uniform(
            jax.random.PRNGKey(1), (tokens, top_k), jnp.float32, 0.1, 1.0
        )
        # Slot 1 of every token: weight zero, and a row of NaN and Inf as a
        # row nobody wrote might hold.
        weights = weights.at[:, 1].set(0.0)
        poison = jnp.where(jnp.arange(lanes) % 2 == 0, jnp.nan, jnp.inf)
        rows = rows.at[:, 1].set(poison).astype(jnp.bfloat16)
        out = combine.weighted_sum(
            rows.reshape(tokens * top_k, lane_blocks, lanes),
            None,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.float32,
        )
        clean = rows.astype(jnp.float32).at[:, 1].set(0.0)
        expected = jnp.einsum("tkbl,tk->tbl", clean, weights).reshape(tokens, hidden)
        self.assertTrue(bool(jnp.isfinite(out).all()))
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(expected), rtol=1e-2, atol=1e-2
        )

    def test_padded_rows_and_scales_ignore_trailing_padding(self):
        tokens, top_k, padded_rows, lane_blocks = 40, 10, 512, 16
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        num_pairs = tokens * top_k
        valid_rows = jax.random.normal(
            jax.random.PRNGKey(2), (num_pairs, lane_blocks, lanes), jnp.float32
        )
        poison_rows = jnp.full(
            (padded_rows - num_pairs, lane_blocks, lanes), jnp.nan, jnp.float32
        )
        weights = jax.random.uniform(
            jax.random.PRNGKey(3), (tokens, top_k), jnp.float32, 0.05, 1.0
        )

        # BF16 rows without scales: NaN padding past num_pairs must never be read.
        bf16_valid = valid_rows.astype(jnp.bfloat16)
        bf16_padded = jnp.concatenate(
            [bf16_valid, poison_rows.astype(jnp.bfloat16)], axis=0
        )
        expected_bf16 = combine.weighted_sum(
            bf16_valid,
            None,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.bfloat16,
            tokens_per_tile=8,
        )
        actual_bf16 = combine.weighted_sum(
            bf16_padded,
            None,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.bfloat16,
            tokens_per_tile=8,
        )
        self.assertTrue(bool(jnp.isfinite(actual_bf16).all()))
        np.testing.assert_array_equal(
            np.asarray(actual_bf16), np.asarray(expected_bf16)
        )

        # FP8 rows with padded float32 scales: both trailing rows and scales hold NaN.
        fp8_valid = jnp.clip(valid_rows, -4.0, 4.0).astype(jnp.float8_e4m3fn)
        fp8_padded = jnp.concatenate(
            [
                fp8_valid,
                jnp.full(
                    (padded_rows - num_pairs, lane_blocks, lanes),
                    jnp.nan,
                    jnp.float8_e4m3fn,
                ),
            ],
            axis=0,
        )
        valid_scales = jax.random.uniform(
            jax.random.PRNGKey(4), (num_pairs,), jnp.float32, 0.2, 1.5
        )
        padded_scales = jnp.concatenate(
            [valid_scales, jnp.full((padded_rows - num_pairs,), jnp.nan, jnp.float32)],
            axis=0,
        )
        expected_fp8 = combine.weighted_sum(
            fp8_valid,
            valid_scales,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.bfloat16,
            tokens_per_tile=8,
        )
        actual_fp8 = combine.weighted_sum(
            fp8_padded,
            padded_scales,
            weights,
            tokens_per_shard=tokens,
            hidden=hidden,
            out_dtype=jnp.bfloat16,
            tokens_per_tile=8,
        )
        self.assertTrue(bool(jnp.isfinite(actual_fp8).all()))
        np.testing.assert_array_equal(np.asarray(actual_fp8), np.asarray(expected_fp8))

    def test_combine_with_padded_arrival_rows_matches_exact_pairs(self):
        tokens, top_k, num_arrivals, padded_pairs, lane_blocks = 128, 4, 640, 1024, 16
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        num_pairs = tokens * top_k
        arrivals_f32 = jax.random.normal(
            jax.random.PRNGKey(5), (num_arrivals, lane_blocks, lanes), jnp.float32
        )
        arrivals_fp8 = jnp.clip(arrivals_f32, -4.0, 4.0).astype(jnp.float8_e4m3fn)
        # Keep row `num_arrivals - 1` poisoned so any read of padded pair indices fails.
        arrivals_fp8 = arrivals_fp8.at[-1].set(
            jnp.full((lane_blocks, lanes), jnp.nan, jnp.float8_e4m3fn)
        )
        arrival_scales = (
            jax.random.uniform(
                jax.random.PRNGKey(6), (num_arrivals,), jnp.float32, 0.2, 1.5
            )
            .at[-1]
            .set(jnp.nan)
        )
        valid_pair_rows = jax.random.randint(
            jax.random.PRNGKey(7), (num_pairs,), 0, num_arrivals - 1, jnp.int32
        )
        padded_pair_rows = jnp.concatenate(
            [
                valid_pair_rows,
                jnp.full((padded_pairs - num_pairs,), num_arrivals - 1, jnp.int32),
            ],
            axis=0,
        )
        weights = jax.random.uniform(
            jax.random.PRNGKey(8), (tokens, top_k), jnp.float32, 0.05, 1.0
        )

        expected = combine.combine(
            arrivals_fp8,
            arrival_scales,
            valid_pair_rows,
            weights,
            jnp.bfloat16,
            hidden=hidden,
            block_rows=16,
            tokens_per_tile=8,
        )
        actual = combine.combine(
            arrivals_fp8,
            arrival_scales,
            padded_pair_rows,
            weights,
            jnp.bfloat16,
            hidden=hidden,
            block_rows=16,
            tokens_per_tile=8,
        )
        self.assertTrue(bool(jnp.isfinite(actual).all()))
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class CombineLoweringAndValidationTest(parameterized.TestCase):
    def test_validation_rejects_short_rows_or_scales(self):
        lanes = combine.LANES
        weights = jax.ShapeDtypeStruct((16, 4), jnp.float32)
        short_rows = jax.ShapeDtypeStruct((63, 8, lanes), jnp.bfloat16)
        with self.assertRaisesRegex(ValueError, "fewer than"):
            jax.jit(
                lambda r, w: combine.weighted_sum(
                    r,
                    None,
                    w,
                    tokens_per_shard=16,
                    hidden=8 * lanes,
                    out_dtype=jnp.bfloat16,
                    tokens_per_tile=8,
                )
            ).trace(short_rows, weights)

        padded_rows = jax.ShapeDtypeStruct((128, 8, lanes), jnp.float8_e4m3fn)
        short_scales = jax.ShapeDtypeStruct((63,), jnp.float32)
        with self.assertRaisesRegex(ValueError, "scales"):
            jax.jit(
                lambda r, s, w: combine.weighted_sum(
                    r,
                    s,
                    w,
                    tokens_per_shard=16,
                    hidden=8 * lanes,
                    out_dtype=jnp.bfloat16,
                    tokens_per_tile=8,
                )
            ).trace(padded_rows, short_scales, weights)

        bad_weights = jax.ShapeDtypeStruct((15, 4), jnp.float32)
        with self.assertRaisesRegex(ValueError, "weights"):
            jax.jit(
                lambda r, w: combine.weighted_sum(
                    r,
                    None,
                    w,
                    tokens_per_shard=16,
                    hidden=8 * lanes,
                    out_dtype=jnp.bfloat16,
                    tokens_per_tile=8,
                )
            ).trace(padded_rows, bad_weights)

        short_arrival_rows = jax.ShapeDtypeStruct((63,), jnp.int32)
        with self.assertRaisesRegex(ValueError, "arrival rows"):
            jax.jit(
                lambda r, s, a, w: combine.combine(
                    r, s, a, w, jnp.bfloat16, hidden=8 * lanes, tokens_per_tile=8
                )
            ).trace(padded_rows, None, short_arrival_rows, weights)

    @parameterized.parameters(
        (384, 10, 4096, 32, 128, 4096, True, jnp.float32),
        (384, 10, 4096, 32, 128, 4096, False, jnp.bfloat16),
        (128, 10, 1536, 32, 128, 4096, True, jnp.bfloat16),
        (128, 10, 1536, 24, 128, 2944, True, jnp.float32),
        (40, 10, 512, 16, 8, 2048, True, jnp.float32),
    )
    def test_tpu_lowering_avoids_slicing_padded_rows(
        self,
        tokens_per_shard,
        top_k,
        padded_rows,
        lane_blocks,
        tokens_per_tile,
        hidden,
        with_scales,
        acc_dtype,
    ):
        lanes = combine.LANES
        row_dtype = jnp.float8_e4m3fn if with_scales else jnp.bfloat16
        abstract_mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )

        def run(rows, scales, weights):
            return combine.weighted_sum(
                rows,
                scales,
                weights,
                tokens_per_shard=tokens_per_shard,
                hidden=hidden,
                out_dtype=jnp.bfloat16,
                tokens_per_tile=tokens_per_tile,
                dtype=acc_dtype,
            )

        rows_aval = jax.ShapeDtypeStruct((padded_rows, lane_blocks, lanes), row_dtype)
        scales_aval = (
            jax.ShapeDtypeStruct((padded_rows,), jnp.float32) if with_scales else None
        )
        weights_aval = jax.ShapeDtypeStruct((tokens_per_shard, top_k), jnp.float32)
        with use_abstract_mesh(abstract_mesh):
            hlo = (
                jax.jit(run)
                .trace(rows_aval, scales_aval, weights_aval)
                .lower(lowering_platforms=("tpu",))
                .as_text("hlo")
            )
        # No 3D slice of `[tokens * top_k, lane_blocks, lanes]` may appear on `rows`.
        num_pairs = tokens_per_shard * top_k
        row_slice_pattern = rf"\[{num_pairs},{lane_blocks},{lanes}\].*slice\("
        self.assertIsNone(
            re.search(row_slice_pattern, hlo),
            msg=f"Unexpected 3D row slice found in lowered HLO:\n{hlo}",
        )

    @parameterized.parameters((True,), (False,))
    def test_combine_tpu_lowering_avoids_slicing_gathered_rows(self, with_scales):
        tokens_per_shard, top_k, padded_pairs, lane_blocks = 384, 10, 4096, 32
        lanes = combine.LANES
        hidden = lane_blocks * lanes
        row_dtype = jnp.float8_e4m3fn if with_scales else jnp.bfloat16
        info = tpu_info_mod.get_tpu_info_for_chip(tpu_info_mod.ChipVersion.TPU_7X, 1)
        abstract_mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        arrivals = jax.ShapeDtypeStruct((4480, lane_blocks, lanes), row_dtype)
        arrival_scales = (
            jax.ShapeDtypeStruct((4480,), jnp.float32) if with_scales else None
        )
        pair_index = jax.ShapeDtypeStruct((padded_pairs,), jnp.int32)
        weights = jax.ShapeDtypeStruct((tokens_per_shard, top_k), jnp.float32)
        with (
            use_abstract_mesh(abstract_mesh),
            mock.patch.object(pltpu, "get_tpu_info", return_value=info),
        ):
            hlo = (
                jax.jit(
                    lambda a, s, p, w: combine.combine(
                        a, s, p, w, jnp.bfloat16, hidden=hidden
                    )
                )
                .trace(arrivals, arrival_scales, pair_index, weights)
                .lower(lowering_platforms=("tpu",))
                .as_text("hlo")
            )
        num_pairs = tokens_per_shard * top_k
        row_slice_pattern = rf"\[{num_pairs},{lane_blocks},{lanes}\].*slice\("
        self.assertIsNone(
            re.search(row_slice_pattern, hlo),
            msg=f"Unexpected 3D row slice found in combine() HLO:\n{hlo}",
        )


if __name__ == "__main__":
    absltest.main()
