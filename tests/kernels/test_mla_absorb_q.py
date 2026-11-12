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
"""`absorb_q` vs the head-major batched matmul it replaces."""

from unittest import mock

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.absorb_q import _absorb_q_impl, absorb_q


def _bits16(x: jax.Array) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))


def _bits32(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [4, 200, 256, 300, 4096])
def test_matches_head_major_bmm(tokens):
    """Decode sizes take XLA's einsum and prefill sizes the kernel (with a
    partial last block at 300); both equal the bmm the call site ran."""
    heads, nope, rank = 64, 192, 512
    k1, k2 = jax.random.split(jax.random.key(tokens))
    q_nope = jax.random.normal(k1, (tokens, heads, nope)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    bmm = jnp.matmul(jnp.swapaxes(q_nope, 0, 1), w, preferred_element_type=jnp.float32)
    want = jnp.swapaxes(bmm, 0, 1).astype(q_nope.dtype)
    np.testing.assert_array_equal(
        np.asarray(absorb_q(q_nope, w), np.float32), np.asarray(want, np.float32)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [4, 200, 256, 300, 4096])
def test_flat_2d_matches_3d_bit_for_bit(tokens):
    """Flat 2D `[T, H * 256]` q matches 3D `[T, H, 192]` bit-for-bit and ignores
    the trailing `q_pe` lanes (`192:256`)."""
    heads, nope, stride, rank = 64, 192, 256, 512
    k1, k2 = jax.random.split(jax.random.key(1000 + tokens))
    q_raw = jax.random.normal(k1, (tokens, heads, stride)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    want = absorb_q(q_raw[:, :, :nope], w)
    got = absorb_q(q_raw.reshape(tokens, heads * stride), w)
    np.testing.assert_array_equal(_bits16(got), _bits16(want))

    q_poison = q_raw.at[:, :, nope:].set(jnp.nan).reshape(tokens, heads * stride)
    np.testing.assert_array_equal(_bits16(absorb_q(q_poison, w)), _bits16(want))


@pytest.mark.parametrize("tokens", [256, 272, 300])
def test_flat_2d_cpu_interpret_and_poison(tokens):
    """CPU Pallas interpret check: `_kernel_flat` matches `_kernel` to 0 ULP,
    ignores poisoned `q_pe` lanes (`192:256`), and propagates `q_nope` NaNs."""
    heads, nope, stride, rank = 16, 192, 256, 512
    rng = np.random.default_rng(2000 + tokens)
    q_raw = jnp.asarray(
        rng.normal(size=(tokens, heads, stride)).astype(np.float32), jnp.bfloat16
    )
    w = jnp.asarray(
        (rng.normal(size=(heads, nope, rank)) * 0.05).astype(np.float32), jnp.bfloat16
    )
    ref = _absorb_q_impl(q_raw[:, :, :nope], w, interpret=True)
    got = _absorb_q_impl(q_raw.reshape(tokens, heads * stride), w, interpret=True)
    np.testing.assert_array_equal(_bits16(got), _bits16(ref))

    q_poison_pe = q_raw.at[:, :, nope:].set(jnp.nan).reshape(tokens, heads * stride)
    got_poison = _absorb_q_impl(q_poison_pe, w, interpret=True)
    np.testing.assert_array_equal(_bits16(got_poison), _bits16(ref))

    # Positive control: NaN in q_nope lanes [0:192] must reach every output head.
    q_poison_nope = q_raw.at[:, :, :nope].set(jnp.nan).reshape(tokens, heads * stride)
    got_bad = _absorb_q_impl(q_poison_nope, w, interpret=True)
    assert np.isnan(np.asarray(got_bad, np.float32)).all()


@pytest.mark.parametrize("tokens", [4, 200])
def test_flat_2d_decode_fallback_cpu(tokens):
    """Below `MIN_TOKENS`, 2D `q_nope` reshapes + slices onto the exact einsum."""
    heads, nope, stride, rank = 16, 192, 256, 512
    rng = np.random.default_rng(3000 + tokens)
    q_raw = jnp.asarray(rng.normal(size=(tokens, heads, stride)).astype(np.float32))
    w = jnp.asarray((rng.normal(size=(heads, nope, rank)) * 0.05).astype(np.float32))
    ref = _absorb_q_impl(q_raw[:, :, :nope], w, interpret=True)
    got = _absorb_q_impl(q_raw.reshape(tokens, heads * stride), w, interpret=True)
    np.testing.assert_array_equal(_bits32(got), _bits32(ref))


def test_flat_2d_validation():
    w = jnp.zeros((8, 192, 512), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="divisible by heads"):
        absorb_q(jnp.zeros((256, 8 * 256 + 1), dtype=jnp.bfloat16), w)
    with pytest.raises(ValueError, match="head_stride"):
        absorb_q(jnp.zeros((256, 8 * 128), dtype=jnp.bfloat16), w)
    with pytest.raises(ValueError, match="incompatible"):
        absorb_q(jnp.zeros((256, 8, 128), dtype=jnp.bfloat16), w)
    with pytest.raises(ValueError, match="2D or 3D"):
        absorb_q(jnp.zeros((256,), dtype=jnp.bfloat16), w)


def test_flat_2d_tpu7x_mosaic_export_layout():
    """TPU7x Mosaic lowering constrains 2D `q_nope` to `dense<[1, 0]>` with no
    intermediate reshape on `%arg0`."""
    info_7x = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    with (
        mock.patch.object(tpu_info, "get_tpu_info", return_value=info_7x),
        mock.patch.object(pltpu, "get_tpu_info", return_value=info_7x),
    ):
        exp = jax.export.export(jax.jit(absorb_q), platforms=("tpu",))(
            jax.ShapeDtypeStruct((4096, 64 * 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((64, 192, 512), jnp.bfloat16),
        )
    mlir_txt = exp.mlir_module()
    custom_calls = [
        ln for ln in mlir_txt.splitlines() if "custom_call @tpu_custom_call" in ln
    ]
    assert len(custom_calls) == 1
    assert "tpu_custom_call(%arg0, %arg1)" in custom_calls[0]
    expected_layouts = (
        "operand_layouts = [dense<[1, 0]> : tensor<2xindex>, "
        "dense<[2, 1, 0]> : tensor<3xindex>]"
    )
    assert expected_layouts in custom_calls[0]
