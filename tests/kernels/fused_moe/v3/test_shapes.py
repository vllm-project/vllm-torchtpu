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
"""The layer at real model shapes, on the device: the per-shard geometry of
eight served models at every batch from 64 to 8192.

Each model keeps its true top_k, hidden, inter, weight format, activation
and biases. The expert count is cut to a few experts per shard so the
sweep runs in minutes: the kernels see a shard's experts one at a time and
the tables see the count only as their width, and one full-count case per
model covers the width.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe import v3 as fused_ep
from vllm_torchtpu.kernels.fused_moe.v3 import Config, WeightFormat, device, layout
from vllm_torchtpu.kernels.fused_moe.v3.reference import fused_ep_moe_reference

from . import test_layer  # the sibling module, importable only as part of this package

pytestmark = pytest.mark.multichip

NUM_SHARDS = test_layer.NUM_SHARDS
# Every batch, or the subset FUSED_EP_TEST_BATCHES names (the per-push
# tier runs "64,8192", where the reference checks are; the full suite is
# the release gate).
BATCHES = tuple(
    int(b)
    for b in os.environ.get(
        "FUSED_EP_TEST_BATCHES", "64,128,256,512,1024,2048,4096,8192"
    ).split(",")
)
TILE_ROWS = 128
# The per-shard geometry of each served model. `experts` is the reduced
# count the sweep runs at; `full_experts` is the model's. `activation_block`
# is the widest block (512 is the production layer's) both of the model's
# widths divide into, for the block-quantization run.
MODELS = {
    "qwen3_5_397b": dict(
        top_k=10,
        hidden=4096,
        inter=1024,
        experts=64,
        full_experts=512,
        weight_format=WeightFormat.FP8,
        activation="silu",
        bias=False,
        activation_block=512,
    ),
    "qwen3_30b": dict(
        top_k=8,
        hidden=2048,
        inter=768,
        experts=64,
        full_experts=128,
        weight_format=WeightFormat.FP8,
        activation="silu",
        bias=False,
        activation_block=256,
    ),
    # The served width: 2880 padded to 2944 (23 lane blocks) by the serving
    # adapter. Whole-expert fp8 buffers of 2944 x 2944 fit two weight slots;
    # at 3072 they do not (REFUSED_MODELS below).
    "gpt_oss_20b_fp8": dict(
        top_k=4,
        hidden=2944,
        inter=2944,
        experts=32,
        full_experts=32,
        weight_format=WeightFormat.FP8,
        activation="swigluoai",
        bias=True,
        activation_block=128,
    ),
    # Whole-expert fp8 buffers of 4096 x 1536 need 63.0 MiB at three weight
    # slots against the 62.7 MiB budget, so the layer runs this shape at two
    # (test_weight_slots_fit).
    "qwen3_235b": dict(
        top_k=8,
        hidden=4096,
        inter=1536,
        experts=64,
        full_experts=128,
        weight_format=WeightFormat.FP8,
        activation="silu",
        bias=False,
        activation_block=512,
    ),
    "qwen3_next_80b": dict(
        top_k=10,
        hidden=2048,
        inter=512,
        experts=64,
        full_experts=512,
        weight_format=WeightFormat.FP8,
        activation="silu",
        bias=False,
        activation_block=512,
    ),
    "deepseek_v2_lite": dict(
        top_k=6,
        hidden=2048,
        inter=1408,
        experts=64,
        full_experts=64,
        weight_format=WeightFormat.BF16,
        activation="silu",
        bias=False,
        activation_block=128,
    ),
}
# Unsupported shapes and the operands identified by their errors.
# Every refusal below is the VMEM budget of whole-expert weight buffers.
REFUSED_MODELS = {
    # 3072 x 3072 four-bit experts: the whole-expert buffers are over the
    # VMEM budget, and the streamed form cannot take them either (a streamed
    # block covers a whole number of SUBLANES scale rows, and 3072 in blocks
    # of 512 has 6), so the layer refuses both ways
    # (test_streamed_block_scaled_experts_need_whole_scale_tiles).
    "gpt_oss_20b_fp4": (
        dict(
            top_k=4,
            hidden=3072,
            inter=3072,
            experts=32,
            weight_format=WeightFormat.FP4,
            activation="swigluoai",
            bias=True,
        ),
        "VMEM",
    ),
    "gpt_oss_120b_fp4": (
        dict(
            top_k=4,
            hidden=3072,
            inter=3072,
            experts=64,
            weight_format=WeightFormat.FP4,
            activation="swigluoai",
            bias=True,
        ),
        "VMEM",
    ),
    "deepseek_v3": (
        dict(
            top_k=8,
            hidden=7168,
            inter=2048,
            experts=64,
            weight_format=WeightFormat.FP8,
            activation="silu",
            bias=False,
        ),
        "VMEM",
    ),
    # Whole-expert fp8 weight buffers of this width exceed the VMEM budget
    # (the model serves at 2944, MODELS above).
    "gpt_oss_20b_fp8_3072": (
        dict(
            top_k=4,
            hidden=3072,
            inter=3072,
            experts=32,
            weight_format=WeightFormat.FP8,
            activation="swigluoai",
            bias=True,
        ),
        "VMEM",
    ),
    "kimi_k2": (
        dict(
            top_k=8,
            hidden=7168,
            inter=2048,
            experts=64,
            weight_format=WeightFormat.FP8,
            activation="silu",
            bias=False,
        ),
        "VMEM",
    ),
    # One expert per shard whose bf16 matrices are far past the VMEM budget.
    "mixtral_8x7b": (
        dict(
            top_k=2,
            hidden=4096,
            inter=14336,
            experts=8,
            weight_format=WeightFormat.BF16,
            activation="silu",
            bias=False,
        ),
        "VMEM",
    ),
    "mixtral_8x22b": (
        dict(
            top_k=2,
            hidden=6144,
            inter=16384,
            experts=8,
            weight_format=WeightFormat.BF16,
            activation="silu",
            bias=False,
        ),
        "VMEM",
    ),
    # top-k of 1 is outside the kernel's contract.
    "llama4_scout": (
        dict(
            top_k=1,
            hidden=5120,
            inter=8192,
            experts=16,
            weight_format=WeightFormat.BF16,
            activation="silu",
            bias=False,
        ),
        "VMEM",
    ),
}
REFERENCE_BATCHES = (64, 8192)
REFERENCE_REL_L2_MAX = test_layer.REFERENCE_REL_L2_MAX


def shape_of(model, batch, experts=None):
    return dict(
        tokens=batch,
        top_k=model["top_k"],
        hidden=model["hidden"],
        inter=model["inter"],
        tile_rows=TILE_ROWS,
        experts=experts or model["experts"],
    )


class ShapeSweepTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("a TPU is needed")
        if not device.supported_on(jax.devices()[0]):
            self.skipTest(
                f"TPU generation {device.chip_generation()} is not "
                f"in {device.SUPPORTED_GENERATIONS}"
            )
        if len(jax.devices()) < NUM_SHARDS:
            self.skipTest(f"{NUM_SHARDS} devices are needed")
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:NUM_SHARDS]), (layout.MESH_AXIS,)
        )

    def run_layer(self, model, shape, config, seed=0):
        block = 512 if model["weight_format"] == WeightFormat.FP4 else None
        if config.activation_block and block:
            block = config.activation_block
        no_gate = model.get("no_gate", False)
        op = test_layer.operands(
            np.random.default_rng(seed),
            shape,
            weight_format=model["weight_format"],
            bias=model["bias"],
            weight_block=block,
            no_gate=no_gate,
        )
        out = fused_ep.fused_ep_moe(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            op["router_logits"],
            op["w1_bias"],
            op["w2_bias"],
            top_k=shape["top_k"],
            renormalize=True,
            mesh=self.mesh,
            tile_rows=shape["tile_rows"],
            weight_format=model["weight_format"],
            weight_block=block,
            activation=model["activation"],
            no_gate=no_gate,
            config=config,
        )
        return np.asarray(out, np.float32), op, block

    @parameterized.parameters(*[(name, batch) for name in MODELS for batch in BATCHES])
    def test_runs(self, name, batch):
        model = MODELS[name]
        out, op, block = self.run_layer(model, shape_of(model, batch), Config())
        self.assertTrue(np.isfinite(out).all(), f"{name} B{batch}")
        if batch in REFERENCE_BATCHES:
            ref = fused_ep_moe_reference(
                op["x"],
                op["w1"],
                op["w2"],
                op["w1_scales"],
                op["w2_scales"],
                op["router_logits"],
                op["w1_bias"],
                op["w2_bias"],
                top_k=model["top_k"],
                renormalize=True,
                weight_format=model["weight_format"],
                weight_block=block,
                activation=model["activation"],
                no_gate=model.get("no_gate", False),
            )
            error = test_layer.rel_l2(out, ref)
            self.assertLess(
                error, REFERENCE_REL_L2_MAX, f"{name} B{batch}: rel_l2 {error:.2e}"
            )

    @parameterized.parameters(
        *[(name, rows) for name in MODELS for rows in ("fp8", "bf16")]
    )
    def test_activation_block_at_model_shapes(self, name, result_rows):
        """Each shape at its activation block with either result-row form
        (bf16 with block 512 is the production layer's rounding), 1024
        tokens, against the reference at the same rounding."""
        model = MODELS[name]
        block = model["activation_block"]
        shape = shape_of(model, 1024)
        config = Config(activation_block=block, result_rows=result_rows)
        out, op, wblock = self.run_layer(model, shape, config)
        ref = fused_ep_moe_reference(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            op["router_logits"],
            op["w1_bias"],
            op["w2_bias"],
            top_k=model["top_k"],
            renormalize=True,
            weight_format=model["weight_format"],
            weight_block=wblock,
            activation=model["activation"],
            no_gate=model.get("no_gate", False),
            activation_block=block,
            result_rows=result_rows,
        )
        error = test_layer.rel_l2(out, ref)
        self.assertLess(
            error,
            REFERENCE_REL_L2_MAX,
            f"{name} block {block} {result_rows}: rel_l2 {error:.2e}",
        )

    def test_weight_slots_fit(self):
        """Qwen3-235B's whole-expert buffers fit two weight slots, not the
        default three; the layer takes two on its own."""
        model = MODELS["qwen3_235b"]
        shape = shape_of(model, 1024)
        fit = fused_ep.layer.expert_buffers_that_fit(
            model["experts"],
            shape["tile_rows"],
            model["hidden"],
            model["inter"],
            config=Config(),
            weight_format=model["weight_format"],
            weight_block=None,
            has_w1_bias=False,
            has_w2_bias=False,
            no_gate=False,
        )
        self.assertEqual(fit, (2, 1, 0, 0))
        model = MODELS["qwen3_5_397b"]
        fit = fused_ep.layer.expert_buffers_that_fit(
            model["experts"],
            shape["tile_rows"],
            model["hidden"],
            model["inter"],
            config=Config(),
            weight_format=model["weight_format"],
            weight_block=None,
            has_w1_bias=False,
            has_w2_bias=False,
            no_gate=False,
        )
        self.assertEqual(fit, (3, 2, 0, 0))

    @parameterized.parameters(*MODELS)
    def test_full_expert_count(self, name):
        model = MODELS[name]
        batch = 8192
        shape = shape_of(model, batch, experts=model["full_experts"])
        out, _, _ = self.run_layer(model, shape, Config())
        self.assertTrue(np.isfinite(out).all(), name)

    def run_refused(self, model, config):
        """Call the layer on zero operands of a refused model's shape."""
        shape = shape_of(model, 512)
        E, H, inter = shape["experts"], shape["hidden"], shape["inter"]
        K = shape["top_k"]
        form = fused_ep.weight_form(model["weight_format"])
        cols = 2 * inter
        if form.block_scaled:
            scales = (
                jnp.ones((E, H // 512, cols), jnp.float32),
                jnp.ones((E, inter // 512, H), jnp.float32),
            )
        elif form.has_scales:
            scales = (jnp.ones((E, cols), jnp.float32), jnp.ones((E, H), jnp.float32))
        else:
            scales = (None, None)
        w1 = jnp.zeros((E, H, cols), form.weight_dtype)
        w2 = jnp.zeros((E, inter, H), form.weight_dtype)
        x = jnp.zeros((512, H), jnp.bfloat16)
        logits = jnp.zeros((512, E), jnp.float32)
        return fused_ep.fused_ep_moe(
            x,
            w1,
            w2,
            scales[0],
            scales[1],
            logits,
            top_k=K,
            renormalize=True,
            mesh=self.mesh,
            tile_rows=TILE_ROWS,
            weight_format=model["weight_format"],
            weight_block=512 if form.block_scaled else None,
            activation=model["activation"],
            config=config,
        )

    @parameterized.parameters(*REFUSED_MODELS)
    def test_refused_shapes_raise_by_operand(self, name):
        model, names = REFUSED_MODELS[name]
        with self.assertRaisesRegex(ValueError, names):
            self.run_refused(model, Config())

    def test_streamed_block_scaled_experts_need_whole_scale_tiles(self):
        """A streamed block-scaled expert copies its scale rows one sublane
        tile at a time: 3072 in blocks of 512 has 6 scale rows, so no
        stream block works and the layer refuses with that reason, in the
        whole form's message and for a caller's own stream block."""
        model, _ = REFUSED_MODELS["gpt_oss_20b_fp4"]
        with self.assertRaisesRegex(ValueError, "scale rows"):
            self.run_refused(model, Config(activation_block=512))
        with self.assertRaisesRegex(ValueError, "scale rows"):
            self.run_refused(model, Config(activation_block=512, stream_block=1536))


if __name__ == "__main__":
    absltest.main()
