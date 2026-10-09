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

"""Tests of fused_gdn: the GDN prologue."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.experimental.fused_qkv import models, reference
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs, gdn

MODEL = models.QWEN3_5_GDN_TP8
OUTPUTS = ("mixed_qkv", "z", "b", "a", "residual")
TOKENS = (128, 1024, 16384)
NORM_DTYPES = (jnp.bfloat16, jnp.float32)
# test_accuracy: how far the kernel may fall below vLLM's SNR.
SNR_TOLERANCE_DB = 0.15
# The same for b, a and their activations, by norm_dtype. The worst measured
# kernel - vLLM SNR on TPU7x is -1.43 dB with f32 and -3.46 dB with bf16, both
# on sigmoid(b).
BA_SNR_TOLERANCE_DB = {jnp.bfloat16: 4.0, jnp.float32: 1.5}
# vLLM (torch) rounds every bf16 op; XLA must not elide those roundings.
strict_reference = jax.jit(
    reference.gdn_reference,
    static_argnums=0,
    static_argnames=("numerics",),
    compiler_options={"xla_allow_excess_precision": False},
)
Numerics = reference.Numerics


def run_kernel(
    config: configs.GdnConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    **kwargs: Any,
) -> tuple[jax.Array | None, ...]:
    """fused_gdn on weights as loaded."""
    weights = gdn.prepare_weights(config, weights)
    return gdn.fused_gdn(config, x, residual, weights, **kwargs)


def snr_db(got: jax.Array, want: jax.Array) -> float:
    """10 log10(|want|^2 / |got - want|^2), in f64."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return 10 * np.log10(np.sum(want**2) / np.sum((got - want) ** 2))


def _bf16(outs: tuple[jax.Array | None, ...]) -> tuple[jax.Array | None, ...]:
    """Rounds the f32 outputs to bf16, as vLLM's bf16 linear layers do."""
    return tuple(None if o is None else o.astype(jnp.bfloat16) for o in outs)


class FusedGdnTest(parameterized.TestCase):
    def assert_exact(
        self,
        label: str,
        got: tuple[jax.Array | None, ...],
        want: tuple[jax.Array | None, ...],
    ) -> None:
        self.assertLen(got, len(OUTPUTS))
        for out, g, w in zip(OUTPUTS, got, want):
            self.assertEqual(g is None, w is None, out)
            if g is None:
                continue
            self.assertEqual(g.dtype, w.dtype, out)
            g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
            self.assertEqual(g.shape, w.shape, out)
            diff = np.abs(g - w)
            first = np.unravel_index(diff.argmax(), diff.shape)
            print(
                f"{label} {out}: max_abs_diff={diff.max():.3e} "
                f"mismatches={(diff > 0).sum()}/{diff.size} "
                f"worst@{first} got={g[first]} want={w[first]}",
                flush=True,
            )
            self.assertLess(diff.max(), 1e-5, out)

    @parameterized.product(t=TOKENS, norm_dtype=NORM_DTYPES)
    def test_kernel_matches_reference(self, t: int, norm_dtype: jnp.dtype) -> None:
        """The kernel computes exactly Numerics.KERNEL."""
        config = dataclasses.replace(MODEL.config, norm_dtype=norm_dtype)
        args = models.make_gdn_inputs(MODEL, t, exact=True)
        self.assert_exact(
            f"T={t} norm_dtype={jnp.dtype(norm_dtype)}",
            run_kernel(config, *args),
            strict_reference(config, *args),
        )

    @parameterized.parameters((512, 128), (1024, 256))
    def test_multi_tile(self, t: int, tile: int) -> None:
        """Explicit tile sizes: several pipeline steps, each of 1-2 row groups."""
        config = MODEL.config
        args = models.make_gdn_inputs(MODEL, t, exact=True)
        weights = gdn.prepare_weights(config, args[2])
        tiling_ = gdn.choose_tiling(config, t, weights, tile_size=tile)
        self.assert_exact(
            f"T={t} tile={tile}",
            gdn.fused_gdn(config, *args[:2], weights, tiling=tiling_),
            strict_reference(config, *args),
        )

    def test_kernel_numerics_match_vllm_when_exact(self) -> None:
        """Numerics.KERNEL and VLLM agree exactly on exact inputs, in bf16.

        With eps = 0, 1/rms is exactly 1 and nothing rounds. KERNEL keeps the
        projections in f32 where vLLM rounds them to bf16, so both are compared
        after that rounding.
        """
        config = dataclasses.replace(MODEL.config, eps=0.0)
        args = models.make_gdn_inputs(MODEL, 1024, exact=True)
        kernel_numerics = strict_reference(config, *args, numerics=Numerics.KERNEL)
        vllm = strict_reference(config, *args, numerics=Numerics.VLLM)
        self.assert_exact("eps=0", _bf16(kernel_numerics), _bf16(vllm))

    @parameterized.product(t=TOKENS, norm_dtype=NORM_DTYPES)
    def test_accuracy(self, t: int, norm_dtype: jnp.dtype) -> None:
        """On random data, the kernel's SNR vs. UNQUANTIZED is close to vLLM's.

        b and a are also compared after the GDN kernel's activations, sigmoid(b)
        and softplus(a + dt_bias). The residual must match vLLM exactly.
        """
        config = dataclasses.replace(MODEL.config, norm_dtype=norm_dtype)
        nv = config.num_v_heads
        args = models.make_gdn_inputs(MODEL, t)
        got = run_kernel(config, *args)
        vllm = strict_reference(config, *args, numerics=Numerics.VLLM)
        truth = strict_reference(config, *args, numerics=Numerics.UNQUANTIZED)
        dt_bias = jax.random.normal(jax.random.key(1), (nv,), jnp.float32)

        def activated(
            outs: tuple[jax.Array | None, ...],
        ) -> tuple[jax.Array, jax.Array]:
            b, a = outs[2][:, 0, :nv], outs[3][:, 0, :nv]
            return jax.nn.sigmoid(b), jax.nn.softplus(a + dt_bias)

        names = (*OUTPUTS[:4], "sigmoid(b)", "softplus(a+dt_bias)")
        rows = zip(
            names,
            (*got[:4], *activated(got)),
            (*vllm[:4], *activated(vllm)),
            (*truth[:4], *activated(truth)),
        )
        failures = []
        for out, g, v, e in rows:
            if out in ("b", "a"):  # Pad lanes are exact zeros in all three.
                self.assertFalse(np.asarray(g[:, 0, nv:]).any(), out)
                g, v, e = g[:, 0, :nv], v[:, 0, :nv], e[:, 0, :nv]
            snr_kernel, snr_vllm = snr_db(g, e), snr_db(v, e)
            print(
                f"T={t} norm_dtype={jnp.dtype(norm_dtype)} {out}: SNR vs unquantized"
                f" kernel={snr_kernel:.2f} dB vllm={snr_vllm:.2f} dB"
                f" delta={snr_kernel - snr_vllm:+.2f} dB",
                flush=True,
            )
            tolerance = (
                SNR_TOLERANCE_DB
                if out in OUTPUTS[:2]
                else BA_SNR_TOLERANCE_DB[norm_dtype]
            )
            if snr_kernel < snr_vllm - tolerance:
                failures.append(out)
        np.testing.assert_array_equal(
            np.asarray(got[4], np.float32), np.asarray(vllm[4], np.float32)
        )
        self.assertEmpty(failures)

    def test_output_layouts(self) -> None:
        """The kernel's outputs have the GDN kernel's tiled HBM layouts."""
        config, t = MODEL.config, 1024
        x, residual, weights = models.make_gdn_inputs(MODEL, t)
        weights = gdn.prepare_weights(config, weights)
        hlo = (
            jax.jit(gdn.fused_gdn, static_argnums=0)
            .lower(config, x, residual, weights)
            .compile()
            .as_text()
        )
        calls = [
            line
            for line in hlo.splitlines()
            if "custom-call(" in line and "fused_qkv_gdn" in line
        ]
        self.assertLen(calls, 1, hlo)
        print(calls[0][:1000])
        result = calls[0].split(" = ", 1)[1].split(" custom-call(")[0]
        for want in (
            rf"f32\[{t},1,{config.qkv_cols}\]\{{2,1,0:T\(1,128\)",
            rf"bf16\[{t},{config.num_v_heads},128\]\{{2,1,0:T\(8,128\)\(2,1\)",
            rf"f32\[{t},1,128\]\{{2,1,0:T\(1,128\)",
        ):
            self.assertRegex(result, want)


def _distinct(shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Random values: a misplaced row, lane or head is a mismatch."""
    key = jax.random.key(np.prod(shape))
    return jax.random.normal(key, shape, jnp.float32).astype(dtype)


def _assert_equal(got: jax.Array, want: jax.Array) -> None:
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want, np.float32)
    )


class GdnLayoutTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = configs.GdnConfig(num_k_heads=2, num_v_heads=8)

    def test_prepare_weights(self) -> None:
        k, nv = 2 * self.config.act_quant_block, self.config.num_v_heads
        side_w = _distinct((k, 2 * nv), jnp.bfloat16)
        norm_w = _distinct((1, k), jnp.bfloat16)
        w = gdn.prepare_weights(
            self.config,
            configs.QkvWeights(None, input_norm_w=norm_w, side_w=side_w),
        )
        got = np.asarray(w.ba_w, np.float32)
        self.assertEqual(got.shape, (k, 128))
        _assert_equal(got[:, :nv], side_w[:, :nv])
        _assert_equal(got[:, 64 : 64 + nv], side_w[:, nv:])
        got[:, :nv] = got[:, 64 : 64 + nv] = 0
        self.assertFalse(got.any(), "pad columns must be zero")
        self.assertEqual(w.norm_w.dtype, jnp.uint32)  # norm_dtype bf16.
        low = jax.lax.bitcast_convert_type(
            (w.norm_w & 0xFFFF).astype(jnp.uint16), jnp.bfloat16
        )
        want = (norm_w.astype(jnp.float32) + 1.0).astype(jnp.bfloat16)
        _assert_equal(low.reshape(1, k), want)
        np.testing.assert_array_equal(w.norm_w >> 16, w.norm_w & 0xFFFF)
        w32 = gdn.prepare_weights(
            dataclasses.replace(self.config, norm_dtype=jnp.float32),
            configs.QkvWeights(None, input_norm_w=norm_w, side_w=side_w),
        )
        self.assertEqual(w32.norm_w.dtype, jnp.float32)
        _assert_equal(w32.norm_w.reshape(1, k), norm_w.astype(jnp.float32) + 1.0)

    def test_reference_layout(self) -> None:
        c, t = self.config, 16
        qkvz = _distinct((t, c.num_cols))
        ba = _distinct((t, c.side_cols))
        mixed, z, b, a = reference.gdn_layout(c, qkvz, ba)
        _assert_equal(mixed[:, 0], qkvz[:, : c.qkv_cols])
        self.assertEqual(mixed.dtype, jnp.float32)
        self.assertEqual(z.dtype, jnp.bfloat16)
        _assert_equal(z.reshape(t, -1), qkvz[:, c.z_col :].astype(jnp.bfloat16))
        nv = c.num_v_heads
        for got, want in ((b, ba[:, :nv]), (a, ba[:, nv:])):
            self.assertEqual(got.shape, (t, 1, 128))
            _assert_equal(got[:, 0, :nv], want)
            self.assertFalse(np.asarray(got[:, 0, nv:]).any())


if __name__ == "__main__":
    absltest.main()
