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
"""The weight-format table, the routing message, and the private jax
surfaces the package binds."""

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import formats, layer
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

from vllm_torchtpu.kernels.fused_moe.v3.formats import (  # isort: skip
    FP4,
    WEIGHT_FORMAT_NAMES,
    WEIGHT_FORMS,
    WeightFormat,
    weight_form,
)

pytestmark = pytest.mark.cpu_test


class WeightFormatTableTest(parameterized.TestCase):
    def test_the_two_fp8_formats_differ_only_in_scale_layout(self):
        plain = WEIGHT_FORMS[WeightFormat.FP8]
        block = WEIGHT_FORMS[WeightFormat.FP8_BLOCK]
        self.assertEqual(jnp.dtype(plain.weight_dtype), jnp.dtype(block.weight_dtype))
        self.assertFalse(plain.block_scaled)
        self.assertTrue(block.block_scaled)

    def test_a_format_outside_the_table_raises(self):
        for name in WEIGHT_FORMAT_NAMES:
            self.assertIs(weight_form(name), WEIGHT_FORMS[name])
        with self.assertRaises(ValueError) as caught:
            weight_form("fp6")
        message = str(caught.exception)
        self.assertIn("fp6", message)
        for name in WEIGHT_FORMAT_NAMES:
            self.assertIn(str(name), message)

    def test_the_scale_layout_follows_the_format(self):
        self.assertFalse(WEIGHT_FORMS[WeightFormat.FP8].block_scaled)
        self.assertTrue(WEIGHT_FORMS[WeightFormat.FP4].block_scaled)
        self.assertEqual(formats.PACKED_ROW_TILE, 64)

    @parameterized.parameters((WeightFormat.FP8, FP4), (WeightFormat.FP4, FP8))
    def test_the_layer_checks_the_weights_against_the_format(
        self, weight_format, wrong_dtype
    ):
        struct = jax.ShapeDtypeStruct
        with self.assertRaisesRegex(ValueError, "expert weights"):
            layer.fused_ep_moe(
                struct((64, 256), jnp.bfloat16),
                struct((4, 256, 512), wrong_dtype),
                struct((4, 256, 512), wrong_dtype),
                struct((4, 512), jnp.float32),
                struct((4, 256), jnp.float32),
                struct((64, 4), jnp.float32),
                top_k=2,
                renormalize=True,
                mesh=None,
                tile_rows=128,
                weight_format=weight_format,
            )


class PrivateJaxSurfacesTest(absltest.TestCase):
    def test_the_ref_level_bitcast_is_still_there(self):
        from jax._src.state import types as state_types

        self.assertTrue(hasattr(state_types.AbstractRef, "bitcast"))

    def test_the_core_map_lowering_is_still_there(self):
        from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel

        self.assertEqual(len(row_gather_kernel._jax_private()), 5)


if __name__ == "__main__":
    absltest.main()
