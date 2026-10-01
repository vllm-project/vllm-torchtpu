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
"""fused_ep_moe on device: the layer against the reference at every
configurable value, the refusals."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe import v3 as fused_ep
from vllm_torchtpu.kernels.fused_moe.v3 import (
    Config,
    WeightFormat,
    device,
    layout,
    transport,
)
from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather
from vllm_torchtpu.kernels.fused_moe.v3.config import sorted_rows_by_shape
from vllm_torchtpu.kernels.fused_moe.v3.reference import fused_ep_moe_reference
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

pytestmark = pytest.mark.multichip

# Against the plain-jnp reference. Both quantize the same rows to fp8 with
# the same scales, but the roundings do not all land on the same side: the
# kernel's fp8 rows and the reference's differ by one fp8 step on a share
# of the elements, and that share sets the gap (measured 5.2e-2 at these
# shapes. A wrong expert, a wrong scale or a dropped row reads far past
# 1e-1).
REFERENCE_REL_L2_MAX = 8e-2
# The smallest shape: 8 shards. An expert count that is a whole number of
# lane blocks (the tables kernel's one-hot). Hidden a multiple of 1024 (the
# gather moves fp8 rows as 32-bit words and the sum stores eight lane
# blocks at a time). Any token count: the layer pads to 8-row blocks.
SHAPE = dict(tokens=2048, top_k=4, experts=128, hidden=1024, inter=512, tile_rows=128)
NUM_SHARDS = 8


def operands(rng, shape, *, weight_format, bias, no_gate=False, weight_block=None):
    E, H, inter = shape["experts"], shape["hidden"], shape["inter"]
    cols = inter if no_gate else 2 * inter
    x = jnp.asarray(rng.standard_normal((shape["tokens"], H)) * 0.5, jnp.bfloat16)
    logits = jnp.asarray(rng.standard_normal((shape["tokens"], E)), jnp.float32)
    w1 = rng.standard_normal((E, H, cols)).astype(np.float32) * 0.05
    w2 = rng.standard_normal((E, inter, H)).astype(np.float32) * 0.05
    if weight_format == WeightFormat.BF16:
        w1_scales = w2_scales = None
        w1 = jnp.asarray(w1, jnp.bfloat16)
        w2 = jnp.asarray(w2, jnp.bfloat16)
    elif weight_format == WeightFormat.FP8:
        w1_scales = np.abs(w1).max(axis=1) / 448.0
        w2_scales = np.abs(w2).max(axis=1) / 448.0
        w1 = jnp.asarray(w1 / w1_scales[:, None, :], FP8)
        w2 = jnp.asarray(w2 / w2_scales[:, None, :], FP8)
    elif weight_format == WeightFormat.INT4:
        b = weight_block
        w1_scales = np.abs(w1.reshape(E, H // b, b, cols)).max(axis=2) / 7.0
        w2_scales = np.abs(w2.reshape(E, inter // b, b, H)).max(axis=2) / 7.0
        w1 = jnp.asarray(
            np.round(w1.reshape(E, H // b, b, cols) / w1_scales[:, :, None, :]).reshape(
                E, H, cols
            ),
            jnp.int4,
        )
        w2 = jnp.asarray(
            np.round(
                w2.reshape(E, inter // b, b, H) / w2_scales[:, :, None, :]
            ).reshape(E, inter, H),
            jnp.int4,
        )
    elif weight_format == WeightFormat.FP8_BLOCK:
        b = weight_block
        w1_scales = np.abs(w1.reshape(E, H // b, b, cols)).max(axis=2) / 448.0
        w2_scales = np.abs(w2.reshape(E, inter // b, b, H)).max(axis=2) / 448.0
        w1 = jnp.asarray(
            (w1.reshape(E, H // b, b, cols) / w1_scales[:, :, None, :]).reshape(
                E, H, cols
            ),
            FP8,
        )
        w2 = jnp.asarray(
            (w2.reshape(E, inter // b, b, H) / w2_scales[:, :, None, :]).reshape(
                E, inter, H
            ),
            FP8,
        )
    else:
        b = weight_block
        w1_scales = np.abs(w1.reshape(E, H // b, b, cols)).max(axis=2) / 6.0
        w2_scales = np.abs(w2.reshape(E, inter // b, b, H)).max(axis=2) / 6.0
        w1 = jnp.asarray(
            (w1.reshape(E, H // b, b, cols) / w1_scales[:, :, None, :]).reshape(
                E, H, cols
            ),
            jnp.float4_e2m1fn,
        )
        w2 = jnp.asarray(
            (w2.reshape(E, inter // b, b, H) / w2_scales[:, :, None, :]).reshape(
                E, inter, H
            ),
            jnp.float4_e2m1fn,
        )
    if w1_scales is not None:
        w1_scales = jnp.asarray(w1_scales, jnp.float32)
        w2_scales = jnp.asarray(w2_scales, jnp.float32)
    w1_bias = w2_bias = None
    if bias:
        w1_bias = jnp.asarray(rng.standard_normal((E, 1, cols)) * 0.1, jnp.float32)
        w2_bias = jnp.asarray(rng.standard_normal((E, 1, H)) * 0.1, jnp.float32)
    return dict(
        x=x,
        w1=w1,
        w2=w2,
        w1_scales=w1_scales,
        w2_scales=w2_scales,
        router_logits=logits,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


# The default configuration's output per shape, which every other
# configuration has to reproduce bit for bit.
_BASELINES = {}


def rel_l2(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))


class LayerTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("the layer runs on TPU only")
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

    def run_layer(
        self,
        shape,
        config,
        *,
        weight_format=WeightFormat.FP8,
        bias=False,
        activation="silu",
        no_gate=False,
        renormalize=True,
        weight_block=None,
        drop_nan_rows=False,
        seed=0,
        mesh=None,
        swiglu_limit=fused_ep.ffn.SWIGLU_LIMIT_DEFAULT,
    ):
        rng = np.random.default_rng(seed)
        op = operands(
            rng,
            shape,
            weight_format=weight_format,
            bias=bias,
            no_gate=no_gate,
            weight_block=weight_block,
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
            renormalize=renormalize,
            mesh=mesh or self.mesh,
            tile_rows=shape["tile_rows"],
            weight_format=weight_format,
            weight_block=weight_block,
            activation=activation,
            swiglu_limit=swiglu_limit,
            drop_nan_rows=drop_nan_rows,
            no_gate=no_gate,
            config=config,
        )
        return np.asarray(out, np.float32), op

    def check_reference(self, shape, config, **kwargs):
        out, op = self.run_layer(shape, config, **kwargs)
        ref = fused_ep_moe_reference(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            op["router_logits"],
            op["w1_bias"],
            op["w2_bias"],
            top_k=shape["top_k"],
            renormalize=kwargs.get("renormalize", True),
            weight_format=kwargs.get("weight_format", WeightFormat.FP8),
            weight_block=kwargs.get("weight_block"),
            activation=kwargs.get("activation", "silu"),
            swiglu_limit=kwargs.get("swiglu_limit", fused_ep.ffn.SWIGLU_LIMIT_DEFAULT),
            no_gate=kwargs.get("no_gate", False),
            activation_block=config.activation_block,
            token_block=config.token_block,
            result_rows=config.result_rows,
            elementwise_dtype=config.elementwise_dtype,
            intermediate=config.intermediate,
            combine_dtype=config.combine_dtype,
            token_rows=config.token_rows,
        )
        error = rel_l2(out, ref)
        self.assertLess(
            error,
            REFERENCE_REL_L2_MAX,
            f"rel_l2 {error:.2e} against the reference, {config}",
        )
        return out

    # ---- against the reference ----
    def test_intermediate_bf16(self):
        """The bf16 intermediate matches its own reference, and against the
        exact reference (float32 rows against float32 weights between the
        matmuls) it errs LESS than the fp8 intermediate: the numerics gate
        of Config.intermediate. Both errors are printed for the record."""
        out16 = self.check_reference(SHAPE, Config(intermediate="bf16"))
        out8, op = self.run_layer(SHAPE, Config())
        exact = fused_ep_moe_reference(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            op["router_logits"],
            op["w1_bias"],
            op["w2_bias"],
            top_k=SHAPE["top_k"],
            renormalize=True,
            weight_format=WeightFormat.FP8,
            intermediate="exact",
        )
        err8, err16 = rel_l2(out8, exact), rel_l2(out16, exact)
        self.assertLess(
            err16,
            err8,
            f"the bf16 intermediate should err less: fp8 {err8:.3e} bf16 {err16:.3e}",
        )

    @parameterized.named_parameters(
        ("fp8", WeightFormat.FP8, False),
        ("fp8_bias", WeightFormat.FP8, True),
        ("fp4", WeightFormat.FP4, False),
        ("fp4_bias", WeightFormat.FP4, True),
        ("bf16", WeightFormat.BF16, False),
        ("bf16_bias", WeightFormat.BF16, True),
    )
    def test_formats(self, weight_format, bias):
        block = 64 if weight_format == WeightFormat.FP4 else None
        self.check_reference(
            SHAPE, Config(), weight_format=weight_format, bias=bias, weight_block=block
        )

    @parameterized.named_parameters(
        ("swigluoai", "swigluoai", False),
        ("gelu", "gelu", False),
        ("no_gate_silu", "silu", True),
        ("no_gate_gelu", "gelu", True),
    )
    def test_activations(self, activation, no_gate):
        self.check_reference(SHAPE, Config(), activation=activation, no_gate=no_gate)

    def test_swiglu_limit_is_the_models(self):
        """The clamped swiglu's clip is a parameter (GPT-OSS ships 7.0);
        another value against the reference at that value."""
        self.check_reference(SHAPE, Config(), activation="swigluoai", swiglu_limit=4.0)

    def test_routing_handed_in(self):
        """The caller's routing (indices and weights) in place of the
        kernel's top-k: the reference's own selection handed in matches
        the logits path; a sigmoid-scored routing the select kernel cannot
        compute matches the reference at that routing; an index outside
        the expert range routes nowhere."""
        from vllm_torchtpu.kernels.fused_moe.v3 import reference

        shape = SHAPE
        out_logits, op = self.run_layer(shape, Config())
        weights, indices = reference.select(
            op["router_logits"], top_k=shape["top_k"], renormalize=True
        )

        def run(routing):
            return np.asarray(
                fused_ep.fused_ep_moe(
                    op["x"],
                    op["w1"],
                    op["w2"],
                    op["w1_scales"],
                    op["w2_scales"],
                    None,
                    top_k=shape["top_k"],
                    renormalize=True,
                    mesh=self.mesh,
                    tile_rows=shape["tile_rows"],
                    routing=routing,
                    config=Config(),
                ),
                np.float32,
            )

        handed = run((indices, weights))
        self.assertLess(rel_l2(handed, out_logits), 1e-5)
        scores = jax.nn.sigmoid(op["router_logits"])
        top, sig_indices = jax.lax.top_k(scores, shape["top_k"])
        sig_weights = top / jnp.sum(top, axis=-1, keepdims=True)
        sig_out = run((sig_indices.astype(jnp.int32), sig_weights))
        ref = fused_ep_moe_reference(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            None,
            top_k=shape["top_k"],
            renormalize=True,
            weight_format=WeightFormat.FP8,
            routing=(sig_indices.astype(jnp.int32), sig_weights),
        )
        error = rel_l2(sig_out, ref)
        self.assertLess(error, REFERENCE_REL_L2_MAX, f"sigmoid: {error:.2e}")
        nowhere = indices.at[0].set(-1)
        out = run((nowhere, weights))
        self.assertTrue(bool((out[0] == 0).all()))
        np.testing.assert_array_equal(out[1:], handed[1:])

    def test_full_softmax_weights(self):
        self.check_reference(SHAPE, Config(), renormalize=False)

    def test_drop_nan_rows(self):
        self.check_reference(SHAPE, Config(), drop_nan_rows=True)

    @parameterized.parameters(64, 200, 1600, 3000)
    def test_small_and_odd_batches(self, tokens):
        """Tokens pad to 8-row blocks and the routed pairs to whole tables
        steps; shards of 8, 25, 200 and 375 tokens against the reference."""
        self.check_reference(dict(SHAPE, tokens=tokens), Config())

    def test_top_k_one(self):
        self.check_reference(dict(SHAPE, top_k=1), Config())

    # ---- the configurable values: the pipeline depths change no arithmetic,
    # so every setting has to reproduce the default's output bit for bit ----
    def baseline(self, shape):
        key = (shape["tokens"], shape["hidden"], shape["inter"])
        if key not in _BASELINES:
            _BASELINES[key] = self.check_reference(shape, Config())
        return _BASELINES[key]

    def check_bitwise(self, shape, config):
        out, _ = self.run_layer(shape, config)
        np.testing.assert_array_equal(out, self.baseline(shape), err_msg=str(config))

    @parameterized.parameters(
        *[(slots, ahead) for slots in (2, 3, 4) for ahead in range(1, slots)]
    )
    def test_result_slots(self, result_slots, row_fetch_ahead):
        self.check_bitwise(
            SHAPE, Config(result_slots=result_slots, row_fetch_ahead=row_fetch_ahead)
        )

    @parameterized.parameters(
        *[(slots, ahead) for slots in (2, 3, 4) for ahead in range(1, slots)]
    )
    def test_weight_slots(self, weight_slots, weight_prefetch):
        self.check_bitwise(
            SHAPE, Config(weight_slots=weight_slots, weight_prefetch=weight_prefetch)
        )

    def test_bounds_checks_build(self):
        self.check_bitwise(SHAPE, Config(bounds_checks=True))

    def test_tables_blocks_per_step_choice(self):
        """Tokens pad to 8-row blocks; the routed pairs pad on their own to
        whole tables steps (1024 pairs up to 8 blocks a step) and gather
        blocks, with pairs that route nowhere. At the default 0 the layer
        takes the largest step whose pair padding is no larger than the
        smallest step's, so 8; an explicit count is taken as given. Reads
        the chip's subcore count."""
        layer = fused_ep.layer
        self.assertEqual(layer.pallas_token_multiple(8), 8)
        self.assertEqual(layer.pallas_pair_multiple(8), 1024)
        self.assertEqual(layer.pallas_pair_multiple(16), 2048)
        self.assertEqual(layer.pallas_geometry(8, 10, 8, 8), (8, 128, 2048))
        self.assertEqual(layer.pallas_geometry(200, 10, 8, 8), (200, 256, 3072))
        self.assertEqual(layer.pallas_geometry(375, 10, 8, 8), (376, 384, 4096))
        self.assertEqual(layer.pallas_geometry(512, 10, 8, 8), (512, 512, 5120))
        choose = layer.tables_blocks_per_step_for
        self.assertEqual(choose(64, 10, 8, Config()), 8)
        self.assertEqual(choose(1024, 10, 8, Config()), 8)
        self.assertEqual(choose(64, 10, 8, Config(tables_blocks_per_step=2)), 2)

    @parameterized.parameters(4, 16)
    def test_tables_blocks_per_step(self, blocks_per_step):
        """The tables kernels' blocks per step is a setting: the tables, and
        so the layer output, come out the same at 4 and 16 blocks per step
        as at the default 8 (16 pads the shard's pairs to whole steps)."""
        self.check_bitwise(SHAPE, Config(tables_blocks_per_step=blocks_per_step))

    def test_bf16_row_scales(self):
        """A numerics option: bf16-rounded row scales move the output by
        fp8 re-quantization noise, well past the default bound."""
        out, op = self.run_layer(SHAPE, Config(row_scale_dtype=jnp.bfloat16))
        ref = fused_ep_moe_reference(
            op["x"],
            op["w1"],
            op["w2"],
            op["w1_scales"],
            op["w2_scales"],
            op["router_logits"],
            top_k=SHAPE["top_k"],
            renormalize=True,
            weight_format=WeightFormat.FP8,
        )
        self.assertLess(rel_l2(out, ref), 2e-1)

    # ---- widths ----
    @parameterized.parameters(1, 2, 4)
    def test_other_widths(self, width):
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:width]), (layout.MESH_AXIS,))
        self.check_reference(SHAPE, Config(), mesh=mesh)

    def test_refuses_widths_that_are_not_powers_of_two(self):
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:6]), (layout.MESH_AXIS,))
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            self.run_layer(dict(SHAPE, tokens=1536, experts=24), Config(), mesh=mesh)

    def test_generation_scope(self):
        self.assertIn(device.chip_generation(), device.SUPPORTED_GENERATIONS)
        transport.check_width(NUM_SHARDS)

    # ---- the rounding settings ----
    @parameterized.parameters(256, 512)
    def test_activation_block(self, block):
        """One fp8 scale per `block` values of a row, for the token rows and
        the intermediate (the production layer's rounding at 512), carried
        as one scale plane per block through the transport, the row tables
        and the FFN kernel: at a full quantization tile and at a partial one
        (64 tokens per shard), with fp8 and with bf16 weights."""
        for tokens in (SHAPE["tokens"], 512):
            shape = dict(SHAPE, tokens=tokens, inter=1024)
            self.check_reference(shape, Config(activation_block=block))
        self.check_reference(
            dict(SHAPE, inter=1024),
            Config(activation_block=block),
            weight_format=WeightFormat.BF16,
        )

    def test_activation_block_with_four_bit_weights(self):
        """With four-bit weights the activation block is the weight block."""
        shape = dict(SHAPE, inter=1024)
        self.check_reference(
            shape,
            Config(activation_block=256),
            weight_format=WeightFormat.FP4,
            weight_block=256,
        )
        with self.assertRaisesRegex(ValueError, "activation block"):
            self.run_layer(
                shape,
                Config(activation_block=512),
                weight_format=WeightFormat.FP4,
                weight_block=256,
            )

    def test_int4_block_scaled_weights(self):
        """int4 weights with one scale per block of the contraction (the
        released form of Kimi K2), through the four-bit path with the
        compiler's widening, against the reference: a whole-expert build
        and a streamed one."""
        shape = dict(SHAPE, inter=1024)
        self.check_reference(
            shape,
            Config(activation_block=256),
            weight_format=WeightFormat.INT4,
            weight_block=256,
        )
        # A streamed block-scaled expert copies eight scale rows (one sublane
        # tile) per stream block: 2,048 columns of 256-wide blocks.
        self.check_reference(
            dict(SHAPE, inter=4096),
            Config(activation_block=256, stream_block=2048),
            weight_format=WeightFormat.INT4,
            weight_block=256,
        )

    def test_fp8_block_scaled_weights(self):
        """fp8 weights with one scale per 128-row block of the contraction
        (the DeepSeek-family checkpoints' form) against the reference, a
        whole-expert build and a streamed one."""
        shape = dict(SHAPE, inter=1024)
        self.check_reference(
            shape,
            Config(activation_block=128),
            weight_format=WeightFormat.FP8_BLOCK,
            weight_block=128,
        )
        # Eight 128-wide scale blocks per stream block: 1,024 columns.
        self.check_reference(
            dict(SHAPE, inter=2048),
            Config(activation_block=128, stream_block=1024),
            weight_format=WeightFormat.FP8_BLOCK,
            weight_block=128,
        )

    def test_token_block(self):
        """The token rows' own rounding block: None follows the activation
        block bit for bit; 0 keeps one scale per row into the up projection
        while the intermediate stays blocked (a streamed expert needs the
        blocked intermediate and nothing else), for fp8, bf16 and four-bit
        weights; a token block that does not divide hidden is refused."""
        shape = dict(SHAPE, inter=2048)
        blocked, _ = self.run_layer(shape, Config(activation_block=512))
        same, _ = self.run_layer(shape, Config(activation_block=512, token_block=512))
        np.testing.assert_array_equal(same, blocked)
        self.check_reference(shape, Config(activation_block=512, token_block=0))
        self.check_reference(
            shape, Config(activation_block=512, token_block=0, stream_block=1024)
        )
        self.check_reference(
            shape,
            Config(activation_block=512, token_block=0),
            weight_format=WeightFormat.BF16,
        )
        self.check_reference(
            shape,
            Config(activation_block=256, token_block=0),
            weight_format=WeightFormat.FP4,
            weight_block=256,
        )
        with self.assertRaisesRegex(ValueError, "token block"):
            self.run_layer(shape, Config(activation_block=512, token_block=384))

    def test_streamed_expert_takes_per_row_token_scales(self):
        """A streamed expert with no token_block of its own takes one scale
        per token row (the layer's choice), bit for bit the explicit
        setting; a caller's own token block is kept."""
        shape = dict(SHAPE, inter=2048)
        chosen, _ = self.run_layer(
            shape, Config(activation_block=512, stream_block=1024)
        )
        explicit, _ = self.run_layer(
            shape, Config(activation_block=512, stream_block=1024, token_block=0)
        )
        np.testing.assert_array_equal(chosen, explicit)
        kept = self.check_reference(
            shape, Config(activation_block=512, stream_block=1024, token_block=512)
        )
        self.assertFalse(np.array_equal(kept, chosen))

    def test_activation_block_refusals(self):
        with self.assertRaisesRegex(ValueError, "activation block"):
            self.run_layer(dict(SHAPE, inter=768), Config(activation_block=512))

    def test_vector_math_settings(self):
        """The scaling product in float32 (the bit-defined form) and the
        combine in bf16, each against the reference at the same settings.
        With the default bf16 product the compiler chooses where the
        second rounding falls, so the bound is the reference tolerance."""
        shape = dict(SHAPE, inter=1024)
        for config in (
            Config(elementwise_dtype="float32"),
            Config(combine_dtype="bfloat16"),
            Config(
                activation_block=512,
                result_rows="bf16",
                elementwise_dtype="float32",
                combine_dtype="bfloat16",
            ),
        ):
            self.check_reference(shape, config)

    def test_replicas(self):
        """Expert parallelism over part of the mesh and replicas over the
        rest: a two-axis mesh, the experts split over the expert-parallel
        axis and copied over the other, every width against the reference
        (width 1 is data parallelism alone; each core pair of a width-2
        mesh is one chip)."""
        devices = np.array(jax.devices()[:NUM_SHARDS])
        for replicas, width in ((4, 2), (2, 4), (8, 1)):
            mesh = jax.sharding.Mesh(
                devices.reshape(replicas, width), ("dp", layout.MESH_AXIS)
            )
            self.check_reference(SHAPE, Config(), mesh=mesh)

    def test_bf16_token_rows(self):
        """The token rows dispatched as bf16, no rounding (Config.token_rows):
        against the reference with bf16 weights (the up matmul as it is)
        and with fp8 weights (on a per-expert bf16 copy of w1), with the
        bf16 intermediate too; the block-scaled forms are refused."""
        shape = dict(SHAPE, inter=1024)
        self.check_reference(
            shape, Config(token_rows="bf16"), weight_format=WeightFormat.BF16
        )
        self.check_reference(shape, Config(token_rows="bf16"))
        self.check_reference(shape, Config(token_rows="bf16", intermediate="bf16"))
        with self.assertRaisesRegex(ValueError, "bf16 token rows"):
            self.run_layer(
                shape,
                Config(token_rows="bf16", activation_block=256),
                weight_format=WeightFormat.FP4,
                weight_block=256,
            )

    def test_bf16_result_rows(self):
        """Expert outputs travel between shards as bf16 rows: against the
        reference in that form, and with the production rounding (block 512
        and bf16 rows) together."""
        shape = dict(SHAPE, inter=1024)
        for config in (
            Config(result_rows="bf16"),
            Config(activation_block=512, result_rows="bf16"),
        ):
            self.check_reference(shape, config)

    # ---- streamed experts ----
    @parameterized.named_parameters(
        ("fp8", WeightFormat.FP8, False, None),
        ("fp8_bias", WeightFormat.FP8, True, None),
        ("bf16", WeightFormat.BF16, False, None),
    )
    def test_streamed_expert_is_the_whole_expert_build(
        self, weight_format, bias, weight_block
    ):
        """An expert streamed in column blocks the width of the activation
        block computes bit for bit what the whole-expert build computes at
        that activation block: the same blocked contractions, summed in the
        same order, one block's weights at a time."""
        block = weight_block or 512
        shape = dict(SHAPE, inter=2048)
        kwargs = dict(weight_format=weight_format, bias=bias, weight_block=weight_block)
        whole, _ = self.run_layer(shape, Config(activation_block=block), **kwargs)
        # A streamed expert takes per-row token scales on its own; the
        # invariant is at one rounding rule, so the block is pinned here.
        streamed = self.check_reference(
            shape,
            Config(activation_block=block, stream_block=block, token_block=block),
            **kwargs,
        )
        np.testing.assert_array_equal(streamed, whole)

    def test_streamed_expert_in_wider_blocks_and_small_groups(self):
        """Two activation blocks per stream block reassociate the f32 sum,
        which flips the fp8 rounding of a few result values (1e-4 measured
        against 5e-2 for a wrong rounding scheme); 128-row groups make
        every expert several groups with a partial last one, still bit
        for bit."""
        shape = dict(SHAPE, tokens=8192, inter=2048)
        whole, _ = self.run_layer(shape, Config(activation_block=512))
        streamed = self.check_reference(
            shape, Config(activation_block=512, stream_block=1024, token_block=512)
        )
        error = rel_l2(streamed, whole)
        self.assertLess(error, 1e-3, f"wider blocks: {error:.2e}")
        streamed = self.check_reference(
            shape,
            Config(
                activation_block=512, stream_block=512, stream_rows=128, token_block=512
            ),
        )
        np.testing.assert_array_equal(streamed, whole)

    def test_streamed_four_bit_expert_in_eight_block_streams(self):
        """A block-scaled expert streams at least eight weight blocks per
        stream block (its scale rows travel one sublane tile at a time), so
        its stream block is wider than its activation block and the f32 sums
        reassociate, within the wider-blocks tolerance above."""
        shape = dict(SHAPE, inter=4096)
        kwargs = dict(weight_format=WeightFormat.FP4, weight_block=256)
        whole, _ = self.run_layer(shape, Config(activation_block=256), **kwargs)
        streamed = self.check_reference(
            shape,
            Config(activation_block=256, stream_block=2048, token_block=256),
            **kwargs,
        )
        error = rel_l2(streamed, whole)
        self.assertLess(error, 1e-3, f"eight-block streams: {error:.2e}")

    def sorts(self, shape, config, **kwargs):
        """The layer's output and whether the sparse-core sort ran. The
        gather is traced once per layer program, so a program another test
        built for the same shape and config would hide the call: the
        program cache is emptied first."""
        sorted_calls = []
        original = gather.gather_rows_prefix

        def counted(*args, **kw):
            sorted_calls.append(1)
            return original(*args, **kw)

        with fused_ep.layer._LAYER_CACHE_LOCK:
            fused_ep.layer._LAYER_CACHE.clear()
        with mock.patch.object(gather, "gather_rows_prefix", counted):
            out, _ = self.run_layer(shape, config, **kwargs)
        return out, bool(sorted_calls)

    def test_sorted_rows(self):
        """The routed rows sorted into expert order on the sparse cores and
        fetched one copy per tile: the same bytes reach the same tiles, so
        the output is bit for bit the per-row fetch's, and against the
        reference. A streamed build keeps the per-row fetch under either
        setting, bit for bit."""
        out = self.check_reference(SHAPE, Config(sorted_rows=True))
        per_row, sorted_ran = self.sorts(SHAPE, Config(sorted_rows=False))
        self.assertFalse(sorted_ran)
        np.testing.assert_array_equal(out, per_row)
        streamed = dict(
            activation_block=512, stream_block=512, stream_rows=128, token_block=512
        )
        shape = dict(SHAPE, inter=2048)
        sorted_setting, sorted_ran = self.sorts(
            shape, Config(sorted_rows=True, **streamed)
        )
        self.assertFalse(sorted_ran)
        per_row_setting, _ = self.run_layer(
            shape, Config(sorted_rows=False, **streamed)
        )
        np.testing.assert_array_equal(sorted_setting, per_row_setting)

    def test_sorted_rows_by_shape(self):
        """The default (sorted_rows=None) chooses by the shard's shape: the
        test shape's 256 tokens a shard sorts, 512 a shard with its 16
        experts fetches per row; either output is bit for bit the forced
        form's."""
        experts_per_shard = SHAPE["experts"] // NUM_SHARDS
        self.assertTrue(
            sorted_rows_by_shape(experts_per_shard, SHAPE["tokens"] // NUM_SHARDS)
        )
        out, sorted_ran = self.sorts(SHAPE, Config())
        self.assertTrue(sorted_ran)
        forced, _ = self.run_layer(SHAPE, Config(sorted_rows=True))
        np.testing.assert_array_equal(out, forced)
        shape = dict(SHAPE, tokens=4096)
        self.assertFalse(sorted_rows_by_shape(experts_per_shard, 4096 // NUM_SHARDS))
        out, sorted_ran = self.sorts(shape, Config())
        self.assertFalse(sorted_ran)
        forced, _ = self.run_layer(shape, Config(sorted_rows=False))
        np.testing.assert_array_equal(out, forced)
        out, sorted_ran = self.sorts(shape, Config(sorted_rows=True))
        self.assertTrue(sorted_ran)
        np.testing.assert_array_equal(out, forced)

    def test_sort_program_size(self):
        """A width whose rows do not move as one copy per block (2560: 20
        lane blocks, 640 words) at a table whose sort program is over the
        sparse core's program size: the choice by shape fetches per row,
        and a forced sort is refused with the estimate, not aborted by the
        compiler."""
        shape = dict(SHAPE, tokens=8192, top_k=8, hidden=2560, experts=256)
        self.assertFalse(gather.row_words_whole_copy(20))
        # 32 experts a shard: the choice by shape would sort but for the size
        self.assertTrue(sorted_rows_by_shape(256 // NUM_SHARDS, 8192 // NUM_SHARDS))
        out, sorted_ran = self.sorts(shape, Config())
        self.assertFalse(sorted_ran)
        forced, _ = self.run_layer(shape, Config(sorted_rows=False))
        np.testing.assert_array_equal(out, forced)
        with self.assertRaisesRegex(ValueError, "sort program .* bundles"):
            self.run_layer(shape, Config(sorted_rows=True))
        # A smaller table of the same width sorts (2048 tokens: 4658
        # bundles compiled), bit for bit the per-row fetch.
        shape = dict(SHAPE, tokens=2048, top_k=8, hidden=2560, experts=256)
        out, sorted_ran = self.sorts(shape, Config())
        self.assertTrue(sorted_ran)
        forced, _ = self.run_layer(shape, Config(sorted_rows=False))
        np.testing.assert_array_equal(out, forced)

    def test_sorted_rows_with_routing_handed_in(self):
        """The served path hands the stack's routing in; the sort sits
        after the tables, so the output stays bit for bit the default's."""
        from vllm_torchtpu.kernels.fused_moe.v3 import reference

        out_default, op = self.run_layer(SHAPE, Config())
        weights, indices = reference.select(
            op["router_logits"], top_k=SHAPE["top_k"], renormalize=True
        )
        out = np.asarray(
            fused_ep.fused_ep_moe(
                op["x"],
                op["w1"],
                op["w2"],
                op["w1_scales"],
                op["w2_scales"],
                None,
                top_k=SHAPE["top_k"],
                renormalize=True,
                mesh=self.mesh,
                tile_rows=SHAPE["tile_rows"],
                routing=(indices, weights),
                config=Config(),
            ),
            np.float32,
        )
        self.assertLess(rel_l2(out, out_default), 1e-5)

    def test_streamed_expert_refusals(self):
        shape = dict(SHAPE, inter=2048)
        with self.assertRaisesRegex(ValueError, "activation_block"):
            self.run_layer(shape, Config(stream_block=512))
        with self.assertRaisesRegex(ValueError, "dividing inter"):
            self.run_layer(shape, Config(activation_block=128, stream_block=768))
        with self.assertRaisesRegex(ValueError, "activation blocks"):
            self.run_layer(shape, Config(activation_block=512, stream_block=256))

    def test_big_expert_streams_on_its_own(self):
        """An ungated 4096 x 16384 fp8 expert (128 MB) never fits whole:
        the layer streams it with no setting but the activation block, and
        refuses without one."""
        shape = dict(
            tokens=2048, top_k=2, experts=8, hidden=4096, inter=16384, tile_rows=128
        )
        chosen = fused_ep.layer.expert_buffers_that_fit(
            1,
            128,
            4096,
            16384,
            config=Config(activation_block=512),
            weight_format=WeightFormat.FP8,
            weight_block=None,
            has_w1_bias=False,
            has_w2_bias=False,
            no_gate=True,
            activation_block=512,
        )
        self.assertGreater(chosen[2], 0, chosen)
        with self.assertRaisesRegex(ValueError, "activation_block"):
            self.run_layer(shape, Config(), no_gate=True)
        self.check_reference(shape, Config(activation_block=512), no_gate=True)

    # ---- widths of the hidden dimension ----
    def test_hidden_over_4096(self):
        self.check_reference(dict(SHAPE, hidden=5120, inter=256), Config())
        self.check_reference(dict(SHAPE, hidden=4608, inter=256), Config())

    @parameterized.named_parameters(
        ("fp8_rows", "fp8", 2944),
        ("bf16_rows", "bf16", 2944),
        ("nine_blocks", "fp8", 1152),
    )
    def test_hidden_not_a_multiple_of_1024(self, result_rows, hidden):
        """A width of any whole number of lane blocks: the result rows carry
        zero blocks up to a whole number of eight (2944 is 23 blocks staged
        as 24; 1152 is 9 staged as 16), the combine writes the true width.
        GPT-OSS-20B fp8 serves at 2944."""
        self.check_reference(
            dict(SHAPE, hidden=hidden, inter=256), Config(result_rows=result_rows)
        )

    def test_hidden_must_be_whole_lane_blocks(self):
        with self.assertRaisesRegex(ValueError, "lane blocks"):
            self.run_layer(dict(SHAPE, hidden=2880, inter=256), Config())

    def test_hidden_over_8192(self):
        """No width cap: 9216 (72 lane blocks) runs, and a row too wide for
        the VMEM budget is refused by the estimate."""
        self.check_reference(dict(SHAPE, hidden=9216, inter=128), Config())
        with self.assertRaisesRegex(ValueError, "VMEM"):
            self.run_layer(dict(SHAPE, hidden=65536, inter=128), Config())


if __name__ == "__main__":
    absltest.main()
